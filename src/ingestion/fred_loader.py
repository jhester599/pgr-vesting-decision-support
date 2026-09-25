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
  PCU5241265241261   — PPI: private passenger auto insurance (v4.5)
  Both lists are refreshed by the weekly and monthly jobs via
  ``production_fred_series()``.
  NOTE: CUSR0000SETC01 (motor vehicle insurance CPI) removed 2026-03-24 —
        series does not exist in FRED (400 Bad Request). Re-add when valid ID found.

All series are resampled to one row per calendar month, labelled with the
month's last business day and holding the last observation in the month.
Values are stored raw: no publication lag and no forward fill (review F06).
``feature_engineering.build_feature_matrix_from_db`` applies each series'
publication lag exactly once, by calendar month, when features are built.

Usage:
    import sqlite3
    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    conn = sqlite3.connect("data/pgr_financials.db")
    df = fetch_all_fred_macro(config.FRED_SERIES_MACRO)
    n = upsert_fred_to_db(conn, df)
    print(f"Upserted {n} FRED rows.")
"""

from __future__ import annotations

import logging
import sqlite3
from io import StringIO
from typing import Any

import pandas as pd
import requests

import config
from src.database.db_client import upsert_fred_macro
from src.ingestion.http_utils import build_retry_session


logger = logging.getLogger(__name__)


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


FREDGRAPH_CSV_URL: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series_csv(
    series_id: str,
    observation_start: str = "2008-01-01",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a FRED series from the public ``fredgraph.csv`` endpoint.

    Same observations as :func:`fetch_fred_series` (the current vintage),
    but no API key is needed. Used by ``scripts/rebuild_fred_macro.py`` when
    ``FRED_API_KEY`` is not set, and by the v19 research loader.

    Returns:
        DataFrame with a DatetimeIndex and one column named ``series_id``,
        in ascending date order. Missing values are NaN.
    """
    http = session or build_retry_session()
    resp = http.get(FREDGRAPH_CSV_URL, params={"id": series_id}, timeout=60)
    resp.raise_for_status()
    return parse_fredgraph_csv(resp.text, series_id, observation_start)


def parse_fredgraph_csv(
    text: str,
    series_id: str,
    observation_start: str = "2008-01-01",
) -> pd.DataFrame:
    """Parse a ``fredgraph.csv`` body into a one-column observation frame."""
    raw = pd.read_csv(StringIO(text))
    date_col = raw.columns[0]
    value_col = raw.columns[-1]
    df = pd.DataFrame(
        {
            series_id: pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(pd.to_datetime(raw[date_col], errors="coerce")),
    )
    df = df.loc[df.index.notna()].sort_index()
    return df.loc[df.index >= pd.Timestamp(observation_start)]


def to_monthly_observations(observations: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Collapse raw observations to one value per calendar month.

    Each month holds its last non-missing observation and is labelled with
    the month's last business day, the label ``fred_macro_monthly`` uses.
    Months with no observation are dropped rather than filled, so stored rows
    are raw FRED values only.
    """
    frame = observations.to_frame() if isinstance(observations, pd.Series) else observations
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    monthly = frame.sort_index().resample("BME").last()
    return monthly.dropna(how="all")


# ---------------------------------------------------------------------------
# Multi-series fetch + monthly resampling
# ---------------------------------------------------------------------------

def production_fred_series() -> list[str]:
    """Return every FRED series the production jobs must keep fresh.

    This is ``FRED_SERIES_MACRO`` followed by ``FRED_SERIES_PGR``, without
    duplicates. The PGR-specific series feed live model features (for example
    ``rate_adequacy_gap_yoy`` in the GBT), so the weekly and monthly jobs must
    refresh them too, not only the yearly bootstrap.
    """
    series: list[str] = []
    for sid in [*config.FRED_SERIES_MACRO, *config.FRED_SERIES_PGR]:
        if sid not in series:
            series.append(sid)
    return series


def fetch_all_fred_macro(
    series_ids: list[str],
    observation_start: str = "2008-01-01",
    dry_run: bool = False,
    apply_publication_lags: bool = False,
) -> pd.DataFrame:
    """
    Fetch multiple FRED series and join them into a single month-end DataFrame.

    Each series is fetched and collapsed to one row per calendar month (the
    last observation in the month, labelled with the month's last business
    day; see :func:`to_monthly_observations`). Nothing is forward-filled.

    Args:
        series_ids:              List of FRED series identifiers.  Typically
                                 ``config.FRED_SERIES_MACRO`` or
                                 ``config.FRED_SERIES_PGR``.
        observation_start:       ISO date string passed to each ``fetch_fred_series``
                                 call.
        dry_run:                 If True, return an empty DataFrame without HTTP calls.
        apply_publication_lags:  Default False: return raw observations, which
                                 is what ``fred_macro_monthly`` stores. The
                                 feature builder applies the lags once (review
                                 F06). True shifts each series by its configured
                                 lag, by calendar month, for ad-hoc analysis;
                                 never store that output.

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
            logger.exception(
                "Failed to fetch FRED series %s; continuing with remaining series. Error=%r",
                sid,
                exc,
            )
            continue

        if df.empty:
            continue

        frames.append(to_monthly_observations(df))

    if not frames:
        return pd.DataFrame(columns=series_ids)

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()

    if apply_publication_lags:
        from src.processing.feature_engineering import _apply_fred_lags

        combined = _apply_fred_lags(combined)

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
        df:   Raw (unlagged) output of ``fetch_all_fred_macro()`` —
              DatetimeIndex (month-end), one column per series_id. Each
              timestamp is stored under its month's last business day.

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
