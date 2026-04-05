"""
FRED (Federal Reserve Economic Data) API client for macro feature ingestion.

Fetches time-series data from the St. Louis Fed public REST API and upserts
monthly observations into the ``fred_macro_monthly`` SQLite table.

FRED is a free public API — fetches do not count against the AV or FMP daily
budgets.  An API key is required (free registration at fred.stlouisfed.org).

Series fetched in v3.0 (FRED_SERIES_MACRO from config.py):
  T10Y2Y         — 10Y-2Y yield curve spread (daily, business days)
  GS5            — 5-Year Treasury CMT Rate (monthly)
  GS2            — 2-Year Treasury CMT Rate (monthly)
  GS10           — 10-Year Treasury CMT Rate (monthly)
  T10YIE         — 10-Year Breakeven Inflation Rate (daily, business days)
  BAA10Y         — Baa Corp Bond minus 10Y Treasury spread (daily)
  BAMLH0A0HYM2   — ICE BofA HY OAS (daily, business days)
  NFCI           — Chicago Fed NFCI (weekly, Fridays)
  VIXCLS         — CBOE VIX (daily, business days)

Series added in v3.1 / v4.5 (FRED_SERIES_PGR from config.py):
  TRFVOLUSM227NFWA   — Vehicle miles traveled NSA (monthly)
  CUSR0000SETA02     — Used car & truck CPI (auto total-loss severity; v4.5)
  CUSR0000SAM2       — Medical care CPI (bodily injury / PIP severity; v4.5)
  NOTE: CUSR0000SETC01 (motor vehicle insurance CPI) removed 2026-03-24 —
        series does not exist in FRED (400 Bad Request). Re-add when valid ID found.

All series are resampled to month-end frequency using the last available
observation in the month.  Forward-fill is applied for up to 5 business days
to handle end-of-month reporting gaps.

Usage:
    import sqlite3
    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    conn = sqlite3.connect("data/pgr_financials.db")
    df = fetch_all_fred_macro(config.FRED_SERIES_MACRO)
    n = upsert_fred_to_db(conn, df)
    print(f"Upserted {n} FRED rows.")
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd
import requests

import config
from src.database.db_client import upsert_fred_macro
from src.ingestion.http_utils import build_retry_session


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_id: str,
    observation_start: str = "2008-01-01",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Fetch a single FRED series via the public observations endpoint.

    Args:
        series_id:         FRED series identifier (e.g. ``"T10Y2Y"``).
        observation_start: ISO date string; earliest observation to retrieve.
                           Default ``"2008-01-01"`` ensures sufficient history
                           for WFO training windows starting from 2009.
        dry_run:           If True, return an empty DataFrame without making
                           any HTTP calls.

    Returns:
        DataFrame with a DatetimeIndex and a single column named ``series_id``.
        FRED's missing-value sentinel ``'.'`` is converted to NaN.
        Index is in ascending date order.

    Raises:
        RuntimeError: If ``config.FRED_API_KEY`` is None and dry_run is False.
        requests.HTTPError: On non-2xx HTTP responses.
    """
    if dry_run:
        return pd.DataFrame(columns=[series_id])

    if config.FRED_API_KEY is None:
        raise RuntimeError(
            "FRED_API_KEY is not set. Register at fred.stlouisfed.org and add "
            "FRED_API_KEY to your .env file."
        )

    params: dict[str, Any] = {
        "series_id":         series_id,
        "observation_start": observation_start,
        "api_key":           config.FRED_API_KEY,
        "file_type":         "json",
    }

    session = build_retry_session()
    resp = session.get(config.FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=[series_id])

    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    # FRED uses '.' as the missing value sentinel
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").sort_index()
    df.columns = [series_id]

    return df


# ---------------------------------------------------------------------------
# Multi-series fetch + monthly resampling
# ---------------------------------------------------------------------------

def fetch_all_fred_macro(
    series_ids: list[str],
    observation_start: str = "2008-01-01",
    dry_run: bool = False,
    apply_publication_lags: bool = True,
) -> pd.DataFrame:
    """
    Fetch multiple FRED series and join them into a single month-end DataFrame.

    Each series is individually fetched, then resampled to month-end frequency
    using the last available observation on the final business day of each month.
    A forward-fill
    of up to 5 periods is applied to handle series that are not available on the
    last calendar day of the month (e.g. daily yield curve data).

    Args:
        series_ids:              List of FRED series identifiers.  Typically
                                 ``config.FRED_SERIES_MACRO`` or
                                 ``config.FRED_SERIES_PGR``.
        observation_start:       ISO date string passed to each ``fetch_fred_series``
                                 call.
        dry_run:                 If True, return an empty DataFrame without HTTP calls.
        apply_publication_lags:  If True (default), shift each series by its
                                 configured publication lag from
                                 ``config.FRED_SERIES_LAGS`` /
                                 ``config.FRED_DEFAULT_LAG_MONTHS`` to prevent
                                 look-ahead bias from FRED data revisions (v4.1).

    Returns:
        DataFrame with a DatetimeIndex (month-end, last business day) and one
        column per series_id.  Missing observations are NaN.  Index is sorted
        ascending.
    """
    if dry_run:
        return pd.DataFrame(columns=series_ids)

    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        try:
            df = fetch_fred_series(sid, observation_start=observation_start)
        except Exception as exc:  # noqa: BLE001 — log and continue
            print(f"  [fred_loader] WARNING: failed to fetch {sid}: {exc}")
            continue

        if df.empty:
            continue

        # Resample to the last business day of month; take the final observation.
        monthly = df.resample("BME").last()
        # Forward-fill to handle occasional end-of-month data gaps
        monthly = monthly.ffill(limit=5)
        frames.append(monthly)

    if not frames:
        return pd.DataFrame(columns=series_ids)

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()

    # v4.1: apply per-series publication lags to prevent look-ahead bias
    if apply_publication_lags:
        for sid in combined.columns:
            lag = config.FRED_SERIES_LAGS.get(sid, config.FRED_DEFAULT_LAG_MONTHS)
            if lag > 0:
                combined[sid] = combined[sid].shift(lag)

    return combined


# ---------------------------------------------------------------------------
# Database upsert helper
# ---------------------------------------------------------------------------

def upsert_fred_to_db(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
) -> int:
    """
    Upsert a wide FRED macro DataFrame into the ``fred_macro_monthly`` table.

    Converts the wide DataFrame (DatetimeIndex × series_id columns) to the
    long format expected by ``db_client.upsert_fred_macro()``.

    Args:
        conn: Open SQLite connection.
        df:   Output of ``fetch_all_fred_macro()`` — DatetimeIndex (month-end),
              one column per series_id.

    Returns:
        Total number of rows upserted (series × months).
    """
    if df.empty:
        return 0

    records: list[dict] = []
    for series_id in df.columns:
        series = df[series_id].dropna()
        for month_end, value in series.items():
            records.append(
                {
                    "series_id": series_id,
                    "month_end": month_end.strftime("%Y-%m-%d"),
                    "value":     float(value),
                }
            )

    return upsert_fred_macro(conn, records)
