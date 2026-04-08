"""
SQLite database client for the PGR Vesting Decision Support v2 engine.

Responsibilities:
  - Schema initialization and versioned migrations
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
from src.database import migration_runner
from src.ingestion.provider_registry import get_provider_limit


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


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
) -> None:
    """Add a column to an existing table if it is not already present.

    This idempotent migration helper handles the case where a live database
    was created before a new column was added to schema.sql.  SQLite does not
    support ``ALTER TABLE … ADD COLUMN IF NOT EXISTS``, so we check
    ``PRAGMA table_info`` first.

    Args:
        conn:     Open SQLite connection.
        table:    Table name to alter.
        column:   Column name to add.
        col_type: SQLite type string, e.g. ``"REAL"`` or ``"TEXT"``.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def _apply_legacy_column_reconciliation(conn: sqlite3.Connection) -> None:
    """Backfill historically added columns for older live databases.

    v10.1 introduces explicit migration files, but many existing databases in
    the field were created before that policy existed. This reconciliation step
    keeps those databases compatible while future changes should prefer ordered
    migration files under ``src/database/migrations/``.
    """
    # v6.x: book_value_per_share added to pgr_edgar_monthly
    _add_column_if_missing(conn, "pgr_edgar_monthly", "book_value_per_share", "REAL")
    # v6.x: eps_basic added to pgr_edgar_monthly (monthly EPS from 8-K; used for pe_ratio)
    _add_column_if_missing(conn, "pgr_edgar_monthly", "eps_basic", "REAL")
    # v6.2: foundational P&L fields
    _add_column_if_missing(conn, "pgr_edgar_monthly", "net_premiums_written", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "net_premiums_earned", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "net_income", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "eps_diluted", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "loss_lae_ratio", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "expense_ratio", "REAL")
    # v6.2: segment-level channel metrics
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npw_agency", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npw_direct", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npw_commercial", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npw_property", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npe_agency", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npe_direct", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npe_commercial", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npe_property", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "pif_agency_auto", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "pif_direct_auto", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "pif_commercial_lines", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "pif_total_personal_lines", "REAL")
    # v6.2: company-level operating metrics
    _add_column_if_missing(conn, "pgr_edgar_monthly", "investment_income", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_revenues", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_expenses", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "income_before_income_taxes", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "roe_net_income_ttm", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "shareholders_equity", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_assets", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "unearned_premiums", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "shares_repurchased", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "avg_cost_per_share", "REAL")
    # v6.2: investment portfolio metrics
    _add_column_if_missing(conn, "pgr_edgar_monthly", "fte_return_total_portfolio", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "investment_book_yield", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "net_unrealized_gains_fixed", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "fixed_income_duration", "REAL")
    # v6.2: derived fields
    _add_column_if_missing(conn, "pgr_edgar_monthly", "channel_mix_agency_pct", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "npw_growth_yoy", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "underwriting_income", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "unearned_premium_growth_yoy", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "buyback_yield", "REAL")


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Initialize the operational schema and apply ordered migrations safely.

    This method is idempotent and serves two purposes:

    1. apply ordered SQL migrations for fresh or already-migrated databases
    2. reconcile legacy databases that predate the migration framework

    Args:
        conn: An open SQLite connection returned by :func:`get_connection`.
    """
    migration_runner.apply_migrations(conn)
    _apply_legacy_column_reconciliation(conn)
    # v8.9: broaden live EDGAR schema toward the historical CSV layout
    _add_column_if_missing(conn, "pgr_edgar_monthly", "filing_date", "TEXT")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "filing_type", "TEXT")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "accession_number", "TEXT")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "avg_diluted_equivalent_shares", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_net_realized_gains", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "service_revenues", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "fees_and_other_revenues", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "losses_lae", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "policy_acquisition_costs", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "other_underwriting_expenses", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "interest_expense", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "provision_for_income_taxes", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_comprehensive_income", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "comprehensive_eps_diluted", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "avg_shares_basic", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "avg_shares_diluted", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "pif_special_lines", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "pif_property", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "roe_comprehensive_trailing_12m", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_investments", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "loss_lae_reserves", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "debt", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "total_liabilities", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "common_shares_outstanding", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "fte_return_fixed_income", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "fte_return_common_stocks", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "debt_to_total_capital", "REAL")
    _add_column_if_missing(conn, "pgr_edgar_monthly", "weighted_avg_credit_quality", "TEXT")


def get_db_health_report(
    conn: sqlite3.Connection,
    csv_path: str | None = None,
) -> dict[str, Any]:
    """Return a lightweight data/schema parity report for startup checks."""
    required_columns = {
        "month_end",
        "combined_ratio",
        "pif_total",
        "book_value_per_share",
        "eps_basic",
        "net_premiums_written",
        "net_premiums_earned",
        "underwriting_income",
        "investment_book_yield",
        "channel_mix_agency_pct",
        "buyback_yield",
    }

    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(pgr_edgar_monthly)").fetchall()
    }
    missing_columns = sorted(required_columns - existing_columns)

    row = conn.execute(
        """
        SELECT COUNT(*), MIN(month_end), MAX(month_end)
        FROM pgr_edgar_monthly
        """
    ).fetchone()
    row_count = int(row[0]) if row else 0
    min_month = row[1] if row else None
    max_month = row[2] if row else None

    csv_path_resolved = Path(csv_path) if csv_path else Path(config.DATA_PROCESSED_DIR) / "pgr_edgar_cache.csv"
    expected_csv_rows = None
    if csv_path_resolved.exists():
        expected_csv_rows = max(
            sum(1 for _ in csv_path_resolved.open("r", encoding="utf-8")) - 1,
            0,
        )

    warnings: list[str] = []
    if missing_columns:
        warnings.append(
            "pgr_edgar_monthly is missing expanded schema columns: "
            + ", ".join(missing_columns)
        )
    if expected_csv_rows is not None and row_count < expected_csv_rows:
        warnings.append(
            f"pgr_edgar_monthly has {row_count} rows but the committed CSV contains "
            f"{expected_csv_rows}; run scripts/edgar_8k_fetcher.py --load-from-csv."
        )
    if min_month is None:
        warnings.append("pgr_edgar_monthly is empty; monthly features will be incomplete.")
    elif min_month > "2004-08-31":
        warnings.append(
            f"pgr_edgar_monthly starts at {min_month}, later than the committed CSV baseline "
            "starting in 2004-08."
        )

    return {
        "missing_columns": missing_columns,
        "row_count": row_count,
        "min_month_end": min_month,
        "max_month_end": max_month,
        "expected_csv_rows": expected_csv_rows,
        "warnings": warnings,
    }


def warn_if_db_behind(
    conn: sqlite3.Connection,
    context: str,
    csv_path: str | None = None,
) -> list[str]:
    """Print startup warnings when the checked-in DB lags the documented baseline."""
    report = get_db_health_report(conn, csv_path=csv_path)
    for message in report["warnings"]:
        print(f"[db-health] WARNING ({context}): {message}")
    return list(report["warnings"])


def _coerce_iso_date(raw_value: str | None) -> date | None:
    """Return a ``date`` from an ISO-like string, or ``None`` when absent."""
    if raw_value is None:
        return None
    try:
        return date.fromisoformat(str(raw_value)[:10])
    except ValueError:
        return None


def check_data_freshness(
    conn: sqlite3.Connection,
    reference_date: date,
    price_max_age_days: int = config.DATA_FRESHNESS_MAX_PRICE_AGE_DAYS,
    fred_max_age_days: int = config.DATA_FRESHNESS_MAX_FRED_AGE_DAYS,
    edgar_max_age_days: int = config.DATA_FRESHNESS_MAX_EDGAR_AGE_DAYS,
) -> dict[str, Any]:
    """Evaluate whether core feeds are fresh enough for a live monthly run."""
    checks: list[tuple[str, str, str, int]] = [
        ("Daily prices", "daily_prices", "date", price_max_age_days),
        ("FRED macro", "fred_macro_monthly", "month_end", fred_max_age_days),
        ("PGR monthly EDGAR", "pgr_edgar_monthly", "month_end", edgar_max_age_days),
    ]

    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    has_problem = False

    for feed, table, column, max_age_days in checks:
        latest_raw = get_table_max_date(conn, table, column)
        latest_date = _coerce_iso_date(latest_raw)
        if latest_date is None:
            result = {
                "feed": feed,
                "table": table,
                "column": column,
                "max_age_days": max_age_days,
                "latest_date": None,
                "age_days": None,
                "status": "MISSING",
            }
            warnings.append(
                f"{feed} data is missing from {table}."
            )
            has_problem = True
        else:
            age_days = max(0, (reference_date - latest_date).days)
            status = "OK" if age_days <= max_age_days else "STALE"
            result = {
                "feed": feed,
                "table": table,
                "column": column,
                "max_age_days": max_age_days,
                "latest_date": latest_date.isoformat(),
                "age_days": age_days,
                "status": status,
            }
            if status != "OK":
                warnings.append(
                    f"{feed} is stale: latest {latest_date.isoformat()} "
                    f"({age_days} days old, limit {max_age_days})."
                )
                has_problem = True
        results.append(result)

    return {
        "reference_date": reference_date.isoformat(),
        "overall_status": "WARNING" if has_problem else "OK",
        "checks": results,
        "warnings": warnings,
    }


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
    """Bulk-insert or replace PGR monthly EDGAR metrics.

    Accepts any subset of the full v6.2 column set; missing keys default to
    ``None`` (SQLite NULL).  This keeps callers that only supply core fields
    (e.g. the live EDGAR HTML fetcher) compatible with the expanded schema.
    """
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO pgr_edgar_monthly (
            month_end, filing_date, filing_type, accession_number,
            combined_ratio, pif_total, pif_growth_yoy,
            gainshare_estimate, book_value_per_share, eps_basic,
            avg_diluted_equivalent_shares,
            net_premiums_written, net_premiums_earned, net_income,
            eps_diluted, total_net_realized_gains, service_revenues,
            fees_and_other_revenues, losses_lae, policy_acquisition_costs,
            other_underwriting_expenses, interest_expense,
            provision_for_income_taxes, total_comprehensive_income,
            comprehensive_eps_diluted, avg_shares_basic, avg_shares_diluted,
            loss_lae_ratio, expense_ratio,
            npw_agency, npw_direct, npw_commercial, npw_property,
            npe_agency, npe_direct, npe_commercial, npe_property,
            pif_agency_auto, pif_direct_auto, pif_special_lines, pif_property,
            pif_commercial_lines,
            pif_total_personal_lines,
            investment_income, total_revenues, total_expenses,
            income_before_income_taxes, roe_net_income_ttm,
            roe_comprehensive_trailing_12m, shareholders_equity, total_assets,
            total_investments, loss_lae_reserves, unearned_premiums,
            debt, total_liabilities, common_shares_outstanding,
            shares_repurchased, avg_cost_per_share,
            fte_return_fixed_income, fte_return_common_stocks,
            fte_return_total_portfolio, investment_book_yield,
            net_unrealized_gains_fixed, fixed_income_duration,
            debt_to_total_capital, weighted_avg_credit_quality,
            channel_mix_agency_pct, npw_growth_yoy, underwriting_income,
            unearned_premium_growth_yoy, buyback_yield
        ) VALUES (
            :month_end, :filing_date, :filing_type, :accession_number,
            :combined_ratio, :pif_total, :pif_growth_yoy,
            :gainshare_estimate, :book_value_per_share, :eps_basic,
            :avg_diluted_equivalent_shares,
            :net_premiums_written, :net_premiums_earned, :net_income,
            :eps_diluted, :total_net_realized_gains, :service_revenues,
            :fees_and_other_revenues, :losses_lae, :policy_acquisition_costs,
            :other_underwriting_expenses, :interest_expense,
            :provision_for_income_taxes, :total_comprehensive_income,
            :comprehensive_eps_diluted, :avg_shares_basic, :avg_shares_diluted,
            :loss_lae_ratio, :expense_ratio,
            :npw_agency, :npw_direct, :npw_commercial, :npw_property,
            :npe_agency, :npe_direct, :npe_commercial, :npe_property,
            :pif_agency_auto, :pif_direct_auto, :pif_special_lines, :pif_property,
            :pif_commercial_lines,
            :pif_total_personal_lines,
            :investment_income, :total_revenues, :total_expenses,
            :income_before_income_taxes, :roe_net_income_ttm,
            :roe_comprehensive_trailing_12m, :shareholders_equity, :total_assets,
            :total_investments, :loss_lae_reserves, :unearned_premiums,
            :debt, :total_liabilities, :common_shares_outstanding,
            :shares_repurchased, :avg_cost_per_share,
            :fte_return_fixed_income, :fte_return_common_stocks,
            :fte_return_total_portfolio, :investment_book_yield,
            :net_unrealized_gains_fixed, :fixed_income_duration,
            :debt_to_total_capital, :weighted_avg_credit_quality,
            :channel_mix_agency_pct, :npw_growth_yoy, :underwriting_income,
            :unearned_premium_growth_yoy, :buyback_yield
        )
    """
    normalised = [
        {
            "month_end":                   r["month_end"],
            "filing_date":                r.get("filing_date"),
            "filing_type":                r.get("filing_type"),
            "accession_number":           r.get("accession_number"),
            "combined_ratio":              r.get("combined_ratio"),
            "pif_total":                   r.get("pif_total"),
            "pif_growth_yoy":              r.get("pif_growth_yoy"),
            "gainshare_estimate":          r.get("gainshare_estimate"),
            "book_value_per_share":        r.get("book_value_per_share"),
            "eps_basic":                   r.get("eps_basic"),
            "avg_diluted_equivalent_shares": r.get("avg_diluted_equivalent_shares"),
            "net_premiums_written":        r.get("net_premiums_written"),
            "net_premiums_earned":         r.get("net_premiums_earned"),
            "net_income":                  r.get("net_income"),
            "eps_diluted":                 r.get("eps_diluted"),
            "total_net_realized_gains":    r.get("total_net_realized_gains"),
            "service_revenues":            r.get("service_revenues"),
            "fees_and_other_revenues":     r.get("fees_and_other_revenues"),
            "losses_lae":                  r.get("losses_lae"),
            "policy_acquisition_costs":    r.get("policy_acquisition_costs"),
            "other_underwriting_expenses": r.get("other_underwriting_expenses"),
            "interest_expense":            r.get("interest_expense"),
            "provision_for_income_taxes":  r.get("provision_for_income_taxes"),
            "total_comprehensive_income":  r.get("total_comprehensive_income"),
            "comprehensive_eps_diluted":   r.get("comprehensive_eps_diluted"),
            "avg_shares_basic":            r.get("avg_shares_basic"),
            "avg_shares_diluted":          r.get("avg_shares_diluted"),
            "loss_lae_ratio":              r.get("loss_lae_ratio"),
            "expense_ratio":               r.get("expense_ratio"),
            "npw_agency":                  r.get("npw_agency"),
            "npw_direct":                  r.get("npw_direct"),
            "npw_commercial":              r.get("npw_commercial"),
            "npw_property":                r.get("npw_property"),
            "npe_agency":                  r.get("npe_agency"),
            "npe_direct":                  r.get("npe_direct"),
            "npe_commercial":              r.get("npe_commercial"),
            "npe_property":                r.get("npe_property"),
            "pif_agency_auto":             r.get("pif_agency_auto"),
            "pif_direct_auto":             r.get("pif_direct_auto"),
            "pif_special_lines":           r.get("pif_special_lines"),
            "pif_property":                r.get("pif_property"),
            "pif_commercial_lines":        r.get("pif_commercial_lines"),
            "pif_total_personal_lines":    r.get("pif_total_personal_lines"),
            "investment_income":           r.get("investment_income"),
            "total_revenues":              r.get("total_revenues"),
            "total_expenses":              r.get("total_expenses"),
            "income_before_income_taxes":  r.get("income_before_income_taxes"),
            "roe_net_income_ttm":          r.get("roe_net_income_ttm"),
            "roe_comprehensive_trailing_12m": r.get("roe_comprehensive_trailing_12m"),
            "shareholders_equity":         r.get("shareholders_equity"),
            "total_assets":                r.get("total_assets"),
            "total_investments":           r.get("total_investments"),
            "loss_lae_reserves":           r.get("loss_lae_reserves"),
            "unearned_premiums":           r.get("unearned_premiums"),
            "debt":                        r.get("debt"),
            "total_liabilities":           r.get("total_liabilities"),
            "common_shares_outstanding":   r.get("common_shares_outstanding"),
            "shares_repurchased":          r.get("shares_repurchased"),
            "avg_cost_per_share":          r.get("avg_cost_per_share"),
            "fte_return_fixed_income":     r.get("fte_return_fixed_income"),
            "fte_return_common_stocks":    r.get("fte_return_common_stocks"),
            "fte_return_total_portfolio":  r.get("fte_return_total_portfolio"),
            "investment_book_yield":       r.get("investment_book_yield"),
            "net_unrealized_gains_fixed":  r.get("net_unrealized_gains_fixed"),
            "fixed_income_duration":       r.get("fixed_income_duration"),
            "debt_to_total_capital":       r.get("debt_to_total_capital"),
            "weighted_avg_credit_quality": r.get("weighted_avg_credit_quality"),
            "channel_mix_agency_pct":      r.get("channel_mix_agency_pct"),
            "npw_growth_yoy":              r.get("npw_growth_yoy"),
            "underwriting_income":         r.get("underwriting_income"),
            "unearned_premium_growth_yoy": r.get("unearned_premium_growth_yoy"),
            "buyback_yield":               r.get("buyback_yield"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def get_pgr_edgar_monthly(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all PGR monthly EDGAR metrics, sorted ascending by month_end.

    Returns all v6.2 columns.  Pre-v6.2 rows have NULL for the new fields;
    callers should handle NaN accordingly.
    """
    sql = """
        SELECT month_end, filing_date, filing_type, accession_number,
               combined_ratio, pif_total, pif_growth_yoy,
               gainshare_estimate, book_value_per_share, eps_basic,
               avg_diluted_equivalent_shares,
               net_premiums_written, net_premiums_earned, net_income,
               eps_diluted, total_net_realized_gains, service_revenues,
               fees_and_other_revenues, losses_lae, policy_acquisition_costs,
               other_underwriting_expenses, interest_expense,
               provision_for_income_taxes, total_comprehensive_income,
               comprehensive_eps_diluted, avg_shares_basic, avg_shares_diluted,
               loss_lae_ratio, expense_ratio,
               npw_agency, npw_direct, npw_commercial, npw_property,
               npe_agency, npe_direct, npe_commercial, npe_property,
               pif_agency_auto, pif_direct_auto, pif_special_lines, pif_property,
               pif_commercial_lines,
               pif_total_personal_lines,
               investment_income, total_revenues, total_expenses,
               income_before_income_taxes, roe_net_income_ttm,
               roe_comprehensive_trailing_12m, shareholders_equity, total_assets,
               total_investments, loss_lae_reserves, unearned_premiums,
               debt, total_liabilities, common_shares_outstanding,
               shares_repurchased, avg_cost_per_share,
               fte_return_fixed_income, fte_return_common_stocks,
               fte_return_total_portfolio, investment_book_yield,
               net_unrealized_gains_fixed, fixed_income_duration,
               debt_to_total_capital, weighted_avg_credit_quality,
               channel_mix_agency_pct, npw_growth_yoy, underwriting_income,
               unearned_premium_growth_yoy, buyback_yield
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


def upsert_model_performance_log(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
) -> int:
    """Bulk-insert or replace monthly model-performance monitoring rows."""
    if not records:
        return 0

    sql = """
        INSERT OR REPLACE INTO model_performance_log (
            month_end,
            aggregate_oos_r2,
            aggregate_nw_ic,
            aggregate_hit_rate,
            ece,
            ece_ci_lower,
            ece_ci_upper,
            conformal_target_coverage,
            conformal_empirical_coverage,
            conformal_trailing_empirical_coverage,
            conformal_trailing_coverage_gap
        )
        VALUES (
            :month_end,
            :aggregate_oos_r2,
            :aggregate_nw_ic,
            :aggregate_hit_rate,
            :ece,
            :ece_ci_lower,
            :ece_ci_upper,
            :conformal_target_coverage,
            :conformal_empirical_coverage,
            :conformal_trailing_empirical_coverage,
            :conformal_trailing_coverage_gap
        )
    """
    normalized = [
        {
            "month_end": record["month_end"],
            "aggregate_oos_r2": record.get("aggregate_oos_r2"),
            "aggregate_nw_ic": record.get("aggregate_nw_ic"),
            "aggregate_hit_rate": record.get("aggregate_hit_rate"),
            "ece": record.get("ece"),
            "ece_ci_lower": record.get("ece_ci_lower"),
            "ece_ci_upper": record.get("ece_ci_upper"),
            "conformal_target_coverage": record.get("conformal_target_coverage"),
            "conformal_empirical_coverage": record.get("conformal_empirical_coverage"),
            "conformal_trailing_empirical_coverage": record.get(
                "conformal_trailing_empirical_coverage"
            ),
            "conformal_trailing_coverage_gap": record.get(
                "conformal_trailing_coverage_gap"
            ),
        }
        for record in records
    ]
    conn.executemany(sql, normalized)
    conn.commit()
    return len(normalized)


def get_model_performance_log(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return monthly model-performance monitoring rows sorted by month."""
    df = pd.read_sql_query(
        """
        SELECT
            month_end,
            aggregate_oos_r2,
            aggregate_nw_ic,
            aggregate_hit_rate,
            ece,
            ece_ci_lower,
            ece_ci_upper,
            conformal_target_coverage,
            conformal_empirical_coverage,
            conformal_trailing_empirical_coverage,
            conformal_trailing_coverage_gap,
            created_at
        FROM model_performance_log
        ORDER BY month_end ASC
        """,
        conn,
        parse_dates=["month_end"],
    )
    if df.empty:
        return df
    df = df.set_index("month_end")
    return df


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
    limit = get_provider_limit(api)

    if limit is None:
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
        return

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


def get_schema_version(conn: sqlite3.Connection) -> str | None:
    """Return the latest applied schema migration id."""
    return migration_runner.current_schema_version(conn)


def get_table_row_count(conn: sqlite3.Connection, table: str) -> int:
    """Return the row count for a table."""
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def get_table_max_date(conn: sqlite3.Connection, table: str, column: str) -> str | None:
    """Return the maximum ISO date-like value from ``table.column``."""
    row = conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
    return str(row[0]) if row and row[0] is not None else None


def get_operational_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return a compact snapshot for workflow summaries and manifests."""
    snapshot: dict[str, Any] = {
        "schema_version": get_schema_version(conn),
        "row_counts": {},
        "latest_dates": {},
    }
    table_specs = {
        "daily_prices": "date",
        "daily_dividends": "ex_date",
        "pgr_fundamentals_quarterly": "period_end",
        "pgr_edgar_monthly": "month_end",
        "fred_macro_monthly": "month_end",
        "monthly_relative_returns": "date",
        "model_performance_log": "month_end",
    }
    for table, date_col in table_specs.items():
        snapshot["row_counts"][table] = get_table_row_count(conn, table)
        snapshot["latest_dates"][f"{table}.{date_col}"] = get_table_max_date(conn, table, date_col)
    return snapshot


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


# ---------------------------------------------------------------------------
# v35.1 — model_retrain_log helpers
# ---------------------------------------------------------------------------

def record_retrain_event(
    conn: sqlite3.Connection,
    triggered_at: str,
    breach_streak: int,
    triggered: bool,
    cooldown_active: bool,
    last_trigger_date: str | None,
    notes: str,
) -> None:
    """Insert one row into model_retrain_log for audit trail."""
    conn.execute(
        """
        INSERT INTO model_retrain_log
            (triggered_at, breach_streak, triggered, cooldown_active,
             last_trigger_date, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            triggered_at,
            breach_streak,
            int(triggered),
            int(cooldown_active),
            last_trigger_date,
            notes,
        ),
    )
    conn.commit()


def get_last_retrain_trigger_date(conn: sqlite3.Connection) -> str | None:
    """Return the ISO date of the most-recent triggered (not suppressed) retrain, or None."""
    row = conn.execute(
        """
        SELECT triggered_at
        FROM model_retrain_log
        WHERE triggered = 1
        ORDER BY triggered_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    # triggered_at is a full ISO 8601 datetime; return just the date portion
    return str(row["triggered_at"])[:10]
