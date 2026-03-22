"""
Multi-ticker DRIP total return computation for the v2 relative return pipeline.

Generalises the v1 build_monthly_returns() from PGR-only to any ticker held in
the v2 SQLite database.  The primary output is build_relative_return_targets(),
which computes PGR-minus-ETF forward returns for all 20 benchmark ETFs and
upserts the results into the monthly_relative_returns table.

Design notes:
  - No new DRIP logic — delegates entirely to the tested build_position_series()
    and compute_total_return() in total_return.py.
  - The AV DIVIDENDS endpoint stores amounts in the ``amount`` column; this
    module renames it to ``dividend`` before calling build_position_series().
  - ETF split history is typically empty; build_position_series() handles an
    empty split DataFrame correctly (no events applied).
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

import config
from src.database import db_client
from src.processing.total_return import build_position_series, compute_total_return


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ticker_data(
    conn: sqlite3.Connection,
    ticker: str,
    exclude_proxy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prices, dividends, and splits for one ticker from the v2 DB.

    Returns DataFrames in the format expected by build_position_series():
      - prices:    DatetimeIndex (date), ``close`` column
      - dividends: DatetimeIndex (ex_date), ``dividend`` column (renamed from ``amount``)
      - splits:    DatetimeIndex (split_date), ``split_ratio`` column (may be empty)

    Args:
        conn:          Open SQLite connection.
        ticker:        Ticker symbol.
        exclude_proxy: If True, omit rows where ``proxy_fill = 1`` from prices.
    """
    prices = db_client.get_prices(conn, ticker, exclude_proxy=exclude_proxy)

    dividends_raw = db_client.get_dividends(conn, ticker)
    if dividends_raw.empty:
        # Ensure DatetimeIndex so build_position_series can compare against Timestamps.
        dividends = pd.DataFrame(
            columns=["dividend", "source"],
            index=pd.DatetimeIndex([], name="ex_date"),
        )
    else:
        dividends = dividends_raw.rename(columns={"amount": "dividend"})

    splits = db_client.get_splits(conn, ticker)
    if splits.empty:
        # Same guard for the split_history table.
        splits = pd.DataFrame(
            columns=["split_ratio", "numerator", "denominator"],
            index=pd.DatetimeIndex([], name="split_date"),
        )

    return prices, dividends, splits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_etf_monthly_returns(
    conn: sqlite3.Connection,
    ticker: str,
    forward_months: int,
    exclude_proxy: bool = False,
) -> pd.Series:
    """Compute monthly forward DRIP total returns for any ticker in the v2 DB.

    Loads price, dividend, and split data from the SQLite database and calls
    the existing build_position_series() / compute_total_return() machinery
    from total_return.py.  The result uses the same month-end convention and
    forward-return NaN treatment as the v1 build_monthly_returns().

    Args:
        conn:           Open SQLite connection with v2 schema.
        ticker:         Ticker symbol (e.g. ``"PGR"`` or ``"VTI"``).
        forward_months: Number of months in the forward return window (6 or 12).
        exclude_proxy:  If True, omit proxy-filled rows before computing returns.
                        Useful to measure model sensitivity to backfilled data.

    Returns:
        Series indexed by month-end date with forward total return values.
        Name: ``f"{ticker}_{forward_months}m_return"``.
        Observations where the forward window extends beyond available data
        are NaN (no look-ahead leakage).  Returns an empty Series if no
        price data exists for ``ticker``.
    """
    prices, dividends, splits = _load_ticker_data(conn, ticker, exclude_proxy)

    if prices.empty:
        return pd.Series(name=f"{ticker}_{forward_months}m_return", dtype=float)

    full_position = build_position_series(
        prices, dividends, splits, initial_shares=1.0
    )

    monthly_dates = full_position.resample("BME").last().index
    data_end = full_position.index.max()

    returns: dict[pd.Timestamp, float] = {}
    for t in monthly_dates:
        t_end = t + pd.DateOffset(months=forward_months)
        if t_end > data_end:
            returns[t] = np.nan
            continue
        try:
            returns[t] = compute_total_return(full_position, t, t_end)
        except (ValueError, KeyError):
            returns[t] = np.nan

    result = pd.Series(returns, name=f"{ticker}_{forward_months}m_return")
    result.index.name = "date"
    return result


def build_relative_return_targets(
    conn: sqlite3.Connection,
    forward_months: int,
    exclude_proxy: bool = False,
    upsert: bool = True,
) -> pd.DataFrame:
    """Compute PGR-minus-ETF relative returns for all benchmark ETFs.

    For each ETF in ``config.ETF_BENCHMARK_UNIVERSE``:
      1. Computes the ETF's forward DRIP total return series.
      2. Subtracts from PGR's forward return to produce a relative return.
      3. Upserts the rows into ``monthly_relative_returns`` (if ``upsert=True``).

    Only date-rows where both PGR and ETF returns are non-NaN are included
    in the output (and persisted to the DB).

    Args:
        conn:           Open SQLite connection with v2 schema.
        forward_months: Target horizon in months (6 or 12).
        exclude_proxy:  If True, exclude proxy-filled rows from both PGR and
                        ETF return calculations.
        upsert:         If True, persist results to the ``monthly_relative_returns``
                        table.  Pass False for read-only / test scenarios.

    Returns:
        DataFrame with index = month-end date (DatetimeIndex), columns = ETF
        ticker symbols, values = relative return (PGR return − ETF return).
        Columns for ETFs with no overlapping data with PGR will be absent.
    """
    pgr_returns = build_etf_monthly_returns(conn, "PGR", forward_months, exclude_proxy)

    result_cols: dict[str, pd.Series] = {}

    for etf in config.ETF_BENCHMARK_UNIVERSE:
        etf_returns = build_etf_monthly_returns(conn, etf, forward_months, exclude_proxy)

        # Inner join: keep only dates where both PGR and ETF return are non-NaN.
        aligned = pd.DataFrame({
            "pgr": pgr_returns,
            "etf": etf_returns,
        }).dropna()

        if aligned.empty:
            continue

        relative = aligned["pgr"] - aligned["etf"]
        relative.name = etf
        result_cols[etf] = relative

        if upsert:
            records: list[dict[str, Any]] = [
                {
                    "date":             idx.strftime("%Y-%m-%d"),
                    "benchmark":        etf,
                    "target_horizon":   forward_months,
                    "pgr_return":       float(aligned.at[idx, "pgr"]),
                    "benchmark_return": float(aligned.at[idx, "etf"]),
                    "relative_return":  float(rel_val),
                    "proxy_fill":       0,
                }
                for idx, rel_val in relative.items()
                if not pd.isna(rel_val)
            ]
            if records:
                db_client.upsert_relative_returns(conn, records)

    if not result_cols:
        return pd.DataFrame()

    df = pd.DataFrame(result_cols)
    df.index.name = "date"
    return df


def load_relative_return_matrix(
    conn: sqlite3.Connection,
    benchmark: str,
    forward_months: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    """Load the pre-computed relative return series for one benchmark/horizon.

    Wraps ``db_client.get_relative_returns()`` for a consistent interface used
    by the WFO engine and backtest engine.

    Args:
        conn:           Open SQLite connection.
        benchmark:      ETF ticker (e.g. ``"VTI"``).
        forward_months: Target horizon in months (6 or 12).
        start_date:     Optional ISO date lower bound (``"YYYY-MM-DD"``).
        end_date:       Optional ISO date upper bound.

    Returns:
        Series indexed by date (DatetimeIndex), values = relative return.
        Name = ``f"{benchmark}_{forward_months}m"``.
        Returns an empty Series if no rows exist in the DB.
    """
    return db_client.get_relative_returns(
        conn, benchmark, forward_months,
        start_date=start_date,
        end_date=end_date,
    )
