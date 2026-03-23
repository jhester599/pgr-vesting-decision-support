"""
SQLite database client for the PGR Vesting Decision Support v2 engine.

Responsibilities:
  - Schema initialization (idempotent CREATE TABLE IF NOT EXISTS)
  - Bulk upsert operations for prices, dividends, splits, and fundamentals
  - API rate-limit tracking (replaces the v1 JSON counter file)
  - Typed query helpers returning pandas DataFrames

All connections use WAL journal mode for concurrent-write safety (GitHub Actions
commits happen while local reads may be in progress) and enable foreign key
enforcement.

Usage:
    import config
    from src.database.db_client import get_connection, initialize_schema

    conn = get_connection(config.DB_PATH)
    initialize_schema(conn)
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with WAL mode and FK enforcement enabled.

    Args:
        db_path: Path to the SQLite file. Defaults to ``config.DB_PATH``.
            Parent directories are created automatically if missing.

    Returns:
        An open ``sqlite3.Connection`` with row_factory set to
        ``sqlite3.Row`` for dict-like column access.
    """
    path = db_path or config.DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Execute schema.sql to create all tables if they do not yet exist.

    This operation is idempotent: running it on an already-initialised
    database is a safe no-op.

    Args:
        conn: An open SQLite connection returned by :func:`get_connection`.
    """
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def upsert_prices(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Bulk-insert or replace price records.

    Args:
        conn: Open connection.
        records: List of dicts with keys matching ``daily_prices`` columns.
            Required keys: ``ticker``, ``date``, ``close``.
            Optional keys: ``open``, ``high``, ``low``, ``volume``,
            ``source``, ``proxy_fill`` (default 0).

    Returns:
        Number of rows written.
    """
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO daily_prices
            (ticker, date, open, high, low, close, volume, source, proxy_fill)
        VALUES
            (:ticker, :date, :open, :high, :low, :close, :volume, :source, :proxy_fill)
    """
    # Apply defaults for optional fields
    normalised = [
        {
            "ticker":     r["ticker"],
            "date":       r["date"],
            "open":       r.get("open"),
            "high":       r.get("high"),
            "low":        r.get("low"),
            "close":      r["close"],
            "volume":     r.get("volume"),
            "source":     r.get("source"),
            "proxy_fill": int(r.get("proxy_fill", 0)),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_prices(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    exclude_proxy: bool = False,
) -> pd.DataFrame:
    """Load daily prices for one ticker.

    Args:
        conn: Open connection.
        ticker: Ticker symbol (e.g. ``"PGR"``).
        start_date: Inclusive lower bound, ISO 8601 (``"YYYY-MM-DD"``).
        end_date: Inclusive upper bound, ISO 8601 (``"YYYY-MM-DD"``).
        exclude_proxy: If True, omit rows where ``proxy_fill = 1``.

    Returns:
        DataFrame indexed by ``date`` (DatetimeIndex), columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``, ``source``,
        ``proxy_fill``. Sorted ascending by date.
    """
    clauses = ["ticker = ?"]
    params: list[Any] = [ticker]
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)
    if exclude_proxy:
        clauses.append("proxy_fill = 0")

    where = " AND ".join(clauses)
    sql = f"""
        SELECT date, open, high, low, close, volume, source, proxy_fill
        FROM daily_prices
        WHERE {where}
        ORDER BY date ASC
    """
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
    if not df.empty:
        df = df.set_index("date")
    return df


# ---------------------------------------------------------------------------
# Dividend helpers
# ---------------------------------------------------------------------------

def upsert_dividends(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Bulk-insert or replace dividend records.

    Args:
        conn: Open connection.
        records: List of dicts with keys ``ticker``, ``ex_date``, ``amount``,
            and optionally ``source``.

    Returns:
        Number of rows written.
    """
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO daily_dividends (ticker, ex_date, amount, source)
        VALUES (:ticker, :ex_date, :amount, :source)
    """
    normalised = [
        {
            "ticker":  r["ticker"],
            "ex_date": r["ex_date"],
            "amount":  r["amount"],
            "source":  r.get("source"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_dividends(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load dividend records for one ticker.

    Returns:
        DataFrame with columns ``ex_date`` (DatetimeIndex), ``amount``,
        ``source``. Sorted ascending by ex_date.
    """
    clauses = ["ticker = ?"]
    params: list[Any] = [ticker]
    if start_date:
        clauses.append("ex_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("ex_date <= ?")
        params.append(end_date)

    where = " AND ".join(clauses)
    sql = f"""
        SELECT ex_date, amount, source
        FROM daily_dividends
        WHERE {where}
        ORDER BY ex_date ASC
    """
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["ex_date"])
    if not df.empty:
        df = df.set_index("ex_date")
    return df


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def upsert_splits(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Bulk-insert or replace split records."""
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO split_history
            (ticker, split_date, split_ratio, numerator, denominator)
        VALUES (:ticker, :split_date, :split_ratio, :numerator, :denominator)
    """
    normalised = [
        {
            "ticker":      r["ticker"],
            "split_date":  r["split_date"],
            "split_ratio": r["split_ratio"],
            "numerator":   r.get("numerator"),
            "denominator": r.get("denominator"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_splits(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Load split history for one ticker, sorted ascending by date."""
    sql = """
        SELECT split_date, split_ratio, numerator, denominator
        FROM split_history
        WHERE ticker = ?
        ORDER BY split_date ASC
    """
    df = pd.read_sql_query(sql, conn, params=[ticker], parse_dates=["split_date"])
    if not df.empty:
        df = df.set_index("split_date")
    return df


# ---------------------------------------------------------------------------
# Fundamentals helpers
# ---------------------------------------------------------------------------

def upsert_pgr_fundamentals(
    conn: sqlite3.Connection, records: list[dict[str, Any]]
) -> int:
    """Bulk-insert or replace PGR quarterly fundamentals from FMP."""
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO pgr_fundamentals_quarterly
            (period_end, pe_ratio, pb_ratio, roe, eps, revenue, net_income, source)
        VALUES
            (:period_end, :pe_ratio, :pb_ratio, :roe, :eps, :revenue, :net_income, :source)
    """
    normalised = [
        {
            "period_end": r["period_end"],
            "pe_ratio":   r.get("pe_ratio"),
            "pb_ratio":   r.get("pb_ratio"),
            "roe":        r.get("roe"),
            "eps":        r.get("eps"),
            "revenue":    r.get("revenue"),
            "net_income": r.get("net_income"),
            "source":     r.get("source"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_pgr_fundamentals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all PGR quarterly fundamentals, sorted ascending by period_end."""
    sql = """
        SELECT period_end, pe_ratio, pb_ratio, roe, eps, revenue, net_income
        FROM pgr_fundamentals_quarterly
        ORDER BY period_end ASC
    """
    df = pd.read_sql_query(sql, conn, parse_dates=["period_end"])
    if not df.empty:
        df = df.set_index("period_end")
    return df


def upsert_pgr_edgar_monthly(
    conn: sqlite3.Connection, records: list[dict[str, Any]]
) -> int:
    """Bulk-insert or replace PGR monthly EDGAR metrics."""
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO pgr_edgar_monthly
            (month_end, combined_ratio, pif_total, pif_growth_yoy, gainshare_estimate)
        VALUES
            (:month_end, :combined_ratio, :pif_total, :pif_growth_yoy, :gainshare_estimate)
    """
    normalised = [
        {
            "month_end":          r["month_end"],
            "combined_ratio":     r.get("combined_ratio"),
            "pif_total":          r.get("pif_total"),
            "pif_growth_yoy":     r.get("pif_growth_yoy"),
            "gainshare_estimate": r.get("gainshare_estimate"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_pgr_edgar_monthly(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all PGR monthly EDGAR metrics, sorted ascending by month_end."""
    sql = """
        SELECT month_end, combined_ratio, pif_total, pif_growth_yoy, gainshare_estimate
        FROM pgr_edgar_monthly
        ORDER BY month_end ASC
    """
    df = pd.read_sql_query(sql, conn, parse_dates=["month_end"])
    if not df.empty:
        df = df.set_index("month_end")
    return df


# ---------------------------------------------------------------------------
# Relative-return helpers
# ---------------------------------------------------------------------------

def upsert_relative_returns(
    conn: sqlite3.Connection, records: list[dict[str, Any]]
) -> int:
    """Bulk-insert or replace pre-computed relative return rows."""
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO monthly_relative_returns
            (date, benchmark, target_horizon,
             pgr_return, benchmark_return, relative_return, proxy_fill)
        VALUES
            (:date, :benchmark, :target_horizon,
             :pgr_return, :benchmark_return, :relative_return, :proxy_fill)
    """
    normalised = [
        {
            "date":             r["date"],
            "benchmark":        r["benchmark"],
            "target_horizon":   int(r["target_horizon"]),
            "pgr_return":       r.get("pgr_return"),
            "benchmark_return": r.get("benchmark_return"),
            "relative_return":  r.get("relative_return"),
            "proxy_fill":       int(r.get("proxy_fill", 0)),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_relative_returns(
    conn: sqlite3.Connection,
    benchmark: str,
    target_horizon: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    """Load the pre-computed relative return series for one benchmark/horizon.

    Returns:
        Series indexed by date (DatetimeIndex), values = relative_return.
        Name = ``f"{benchmark}_{target_horizon}m"``.
    """
    clauses = ["benchmark = ?", "target_horizon = ?"]
    params: list[Any] = [benchmark, target_horizon]
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)

    where = " AND ".join(clauses)
    sql = f"""
        SELECT date, relative_return
        FROM monthly_relative_returns
        WHERE {where}
        ORDER BY date ASC
    """
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
    if df.empty:
        return pd.Series(name=f"{benchmark}_{target_horizon}m", dtype=float)
    series = df.set_index("date")["relative_return"]
    series.name = f"{benchmark}_{target_horizon}m"
    return series


# ---------------------------------------------------------------------------
# FRED macro helpers (v3.0+)
# ---------------------------------------------------------------------------

def upsert_fred_macro(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Bulk-insert or replace FRED macro monthly observations.

    Args:
        conn: Open connection.
        records: List of dicts with keys ``series_id``, ``month_end``
            (ISO date string ``"YYYY-MM-DD"``), and ``value`` (float or None).

    Returns:
        Number of rows written.
    """
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO fred_macro_monthly (series_id, month_end, value)
        VALUES (:series_id, :month_end, :value)
    """
    normalised = [
        {
            "series_id": r["series_id"],
            "month_end": r["month_end"],
            "value":     r.get("value"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_fred_macro(
    conn: sqlite3.Connection,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load FRED macro monthly observations as a wide DataFrame.

    Args:
        conn: Open connection.
        series_ids: If provided, only return these series.  If None,
            returns all series present in the table.

    Returns:
        DataFrame with DatetimeIndex (month_end) and one column per
        ``series_id``.  Missing observations are NaN.  Sorted ascending
        by date.
    """
    if series_ids:
        placeholders = ",".join("?" * len(series_ids))
        sql = f"""
            SELECT series_id, month_end, value
            FROM fred_macro_monthly
            WHERE series_id IN ({placeholders})
            ORDER BY month_end ASC
        """
        df_long = pd.read_sql_query(
            sql, conn, params=series_ids, parse_dates=["month_end"]
        )
    else:
        sql = """
            SELECT series_id, month_end, value
            FROM fred_macro_monthly
            ORDER BY month_end ASC
        """
        df_long = pd.read_sql_query(sql, conn, parse_dates=["month_end"])

    if df_long.empty:
        return pd.DataFrame()

    df_wide = df_long.pivot(index="month_end", columns="series_id", values="value")
    df_wide.index.name = "month_end"
    df_wide.columns.name = None
    return df_wide


# ---------------------------------------------------------------------------
# API rate-limit tracking
# ---------------------------------------------------------------------------

def log_api_request(
    conn: sqlite3.Connection,
    api: str,
    endpoint: str = "",
    utc_date: str | None = None,
) -> None:
    """Increment the daily request counter for the given API.

    Raises:
        RuntimeError: If the daily limit for ``api`` would be exceeded.

    Args:
        conn: Open connection.
        api: ``"av"`` or ``"fmp"``.
        endpoint: Specific endpoint string (for logging detail only).
        utc_date: Override today's UTC date (``"YYYY-MM-DD"``). Defaults
            to the current UTC date.
    """
    today = utc_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    limit = config.AV_DAILY_LIMIT if api == "av" else config.FMP_DAILY_LIMIT

    current = get_api_request_count(conn, api, today)
    if current >= limit:
        raise RuntimeError(
            f"Daily API limit reached for '{api}': {current}/{limit} requests used on {today}."
        )

    conn.execute(
        """
        INSERT INTO api_request_log (api, date, endpoint, count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT (api, date, endpoint)
        DO UPDATE SET count = count + 1
        """,
        (api, today, endpoint),
    )
    conn.commit()


def get_api_request_count(
    conn: sqlite3.Connection,
    api: str,
    utc_date: str | None = None,
) -> int:
    """Return the total number of requests made to ``api`` on ``utc_date``.

    Args:
        conn: Open connection.
        api: ``"av"`` or ``"fmp"``.
        utc_date: Override today's UTC date. Defaults to current UTC date.

    Returns:
        Total request count (sum across all endpoints for this api/date).
    """
    today = utc_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT COALESCE(SUM(count), 0) FROM api_request_log WHERE api = ? AND date = ?",
        (api, today),
    ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Ingestion metadata helpers
# ---------------------------------------------------------------------------

def update_ingestion_metadata(
    conn: sqlite3.Connection,
    ticker: str,
    data_type: str,
    rows_stored: int,
) -> None:
    """Record a successful fetch in the ingestion_metadata table."""
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO ingestion_metadata (ticker, data_type, last_fetched, rows_stored)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (ticker, data_type)
        DO UPDATE SET last_fetched = excluded.last_fetched,
                      rows_stored  = excluded.rows_stored
        """,
        (ticker, data_type, now, rows_stored),
    )
    conn.commit()


def get_ingestion_metadata(
    conn: sqlite3.Connection,
    ticker: str,
    data_type: str,
) -> dict[str, Any] | None:
    """Return the latest ingestion record for a ticker/data_type pair, or None."""
    row = conn.execute(
        """
        SELECT ticker, data_type, last_fetched, rows_stored
        FROM ingestion_metadata
        WHERE ticker = ? AND data_type = ?
        """,
        (ticker, data_type),
    ).fetchone()
    return dict(row) if row else None
