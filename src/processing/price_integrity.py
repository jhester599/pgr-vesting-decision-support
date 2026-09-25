"""
Price-table integrity guards (review 2026-09-25, F03/F05/F22).

* :func:`find_unexplained_price_jumps` flags weekly close-to-close ratios
  outside ``[low, high]`` that no ``split_history`` row within ``window_days``
  explains.  A missing split shows up exactly like this (VOO 2013-10-25 at
  2.02x, VGT 2026-04-24 at 0.13x), and DRIP targets that span it are off by
  100+ return points.
* :func:`find_duplicate_week_bars` lists ticker-ISO-weeks holding more than one
  bar (partial-week bars left behind by ``TIME_SERIES_WEEKLY``).

Both are read-only and are used by the DB-integrity tests and by
``scripts/weekly_fetch.py``, which skips the target rebuild when a jump is
unexplained.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

import config

JUMP_LOW: float = 0.6
JUMP_HIGH: float = 1.7
JUMP_WINDOW_DAYS: int = 7


def _load_closes(conn: sqlite3.Connection, tickers: list[str] | None) -> pd.DataFrame:
    sql = "SELECT ticker, date, close FROM daily_prices"
    params: list[Any] = []
    if tickers:
        sql += f" WHERE ticker IN ({','.join('?' * len(tickers))})"
        params = list(tickers)
    sql += " ORDER BY ticker, date"
    return pd.read_sql_query(sql, conn, params=params)


def find_unexplained_price_jumps(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
    low: float = JUMP_LOW,
    high: float = JUMP_HIGH,
    window_days: int = JUMP_WINDOW_DAYS,
    allowlist: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return consecutive-bar close ratios outside ``[low, high]`` with no split nearby.

    Args:
        conn:        Open SQLite connection (read-only is fine).
        tickers:     Restrict the scan to these tickers (default: all).
        low, high:   Accepted range for ``close_t / close_{t-1}``.
        window_days: A ``split_history`` row for the same ticker within this
                     many days of the jump bar explains it.
        allowlist:   Reviewed genuine moves as ``{"ticker", "date"}`` dicts.
                     Defaults to ``config.KNOWN_PRICE_JUMPS``.

    Returns:
        DataFrame with ``ticker``, ``prev_date``, ``date``, ``prev_close``,
        ``close`` and ``ratio``; empty when every jump is explained.
    """
    if allowlist is None:
        allowlist = config.KNOWN_PRICE_JUMPS
    allowed = {(a["ticker"], a["date"]) for a in allowlist}

    prices = _load_closes(conn, tickers)
    columns = ["ticker", "prev_date", "date", "prev_close", "close", "ratio"]
    if prices.empty:
        return pd.DataFrame(columns=columns)

    grouped = prices.groupby("ticker", sort=False)
    prices["prev_date"] = grouped["date"].shift(1)
    prices["prev_close"] = grouped["close"].shift(1)
    prices["ratio"] = prices["close"] / prices["prev_close"]
    jumps = prices[(prices["ratio"] < low) | (prices["ratio"] > high)]
    if jumps.empty:
        return pd.DataFrame(columns=columns)

    splits = pd.read_sql_query("SELECT ticker, split_date FROM split_history", conn)
    split_dates: dict[str, list[pd.Timestamp]] = {
        t: [pd.Timestamp(d) for d in g["split_date"]]
        for t, g in splits.groupby("ticker")
    }
    window = pd.Timedelta(days=window_days)

    def explained(row: pd.Series) -> bool:
        if (row["ticker"], row["date"]) in allowed:
            return True
        bar = pd.Timestamp(row["date"])
        return any(abs(d - bar) <= window for d in split_dates.get(row["ticker"], []))

    mask = jumps.apply(explained, axis=1)
    return jumps.loc[~mask, columns].reset_index(drop=True)


def find_duplicate_week_bars(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Return every bar that shares its ticker and ISO week with another bar."""
    prices = _load_closes(conn, tickers)
    if prices.empty:
        return prices.assign(iso_year=[], iso_week=[])
    iso = pd.to_datetime(prices["date"]).dt.isocalendar()
    prices["iso_year"] = iso["year"].to_numpy()
    prices["iso_week"] = iso["week"].to_numpy()
    dupes = prices[prices.duplicated(["ticker", "iso_year", "iso_week"], keep=False)]
    return dupes.reset_index(drop=True)
