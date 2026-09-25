"""
SQLite database client for the PGR Vesting Decision Support v2 engine.

Responsibilities:
  - Schema initialization and versioned migrations
  - Bulk upsert operations for prices, dividends, splits, and fundamentals
  - API rate-limit tracking (replaces the v1 JSON counter file)
  - Typed query helpers returning pandas DataFrames

Read-write connections use WAL journal mode for concurrent-write safety
(GitHub Actions commits happen while local reads may be in progress) and enable
foreign key enforcement. Workflows checkpoint the WAL and switch the file back
to ``journal_mode=DELETE`` before committing it (``scripts/finalize_db.py``).
Read-only connections (``get_connection(read_only=True)``) never change the
file.

Usage:
    import config
    from src.database.db_client import get_connection, initialize_schema

    conn = get_connection(config.DB_PATH)
    initialize_schema(conn)
"""

from __future__ import annotations

import logging
import sqlite3
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.database import migration_runner
from src.ingestion.provider_registry import get_provider_limit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_connection(
    db_path: str | None = None,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Return a sqlite3 connection with WAL mode and FK enforcement enabled.

    Args:
        db_path: Path to the SQLite file. Defaults to ``config.DB_PATH``.
            Parent directories are created automatically if missing.
        read_only: When True, open the existing file with ``mode=ro`` so
            that any write raises ``sqlite3.OperationalError``. The journal
            mode is left untouched, because switching it rewrites the file
            header. Used by ``--dry-run`` entry points.

    Returns:
        An open ``sqlite3.Connection`` with row_factory set to
        ``sqlite3.Row`` for dict-like column access.

    Raises:
        FileNotFoundError: If ``read_only`` is True and the file is missing.
    """
    path = db_path or config.DB_PATH
    if read_only:
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Read-only connection requested but database does not exist: {resolved}"
            )
        conn = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def finalize_for_commit(db_path: str | None = None) -> str:
    """Fold the WAL into the main file and switch it to DELETE journal mode.

    Run this before ``git add`` of the database. Afterwards the file on disk
    is self-contained (no ``-wal``/``-shm`` sidecars hold committed pages)
    and its header no longer says WAL, so read-only opens do not need to
    create sidecar files.

    Args:
        db_path: Path to the SQLite file. Defaults to ``config.DB_PATH``.

    Returns:
        The resulting journal mode (always ``"delete"``).

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the checkpoint could not complete or the journal
            mode could not be changed (another connection is open).
    """
    path = Path(db_path or config.DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Database does not exist: {path}")
    conn = sqlite3.connect(str(path))
    try:
        busy, _log_frames, _checkpointed = conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE);"
        ).fetchone()
        if busy:
            raise RuntimeError(f"WAL checkpoint of {path} was blocked by another connection.")
        mode = str(conn.execute("PRAGMA journal_mode=DELETE;").fetchone()[0]).lower()
    finally:
        conn.close()
    if mode != "delete":
        raise RuntimeError(f"Could not switch {path} to journal_mode=DELETE (got {mode!r}).")
    return mode


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


def _month_end(year: int, month: int) -> date:
    """Return the calendar month-end date for a year/month pair."""
    return date(year, month, monthrange(year, month)[1])


def _previous_month_end(reference_date: date) -> date:
    """Return the month-end immediately before the reference month."""
    if reference_date.month == 1:
        return _month_end(reference_date.year - 1, 12)
    return _month_end(reference_date.year, reference_date.month - 1)


def _expected_pgr_edgar_month_end(
    reference_date: date,
    filing_grace_days: int = config.DATA_FRESHNESS_PGR_EDGAR_FILING_GRACE_DAYS,
) -> date:
    """Return the latest PGR 8-K month expected to be available."""
    prior_month_end = _previous_month_end(reference_date)
    due_date = prior_month_end + timedelta(days=filing_grace_days)
    if reference_date >= due_date:
        return prior_month_end
    return _previous_month_end(prior_month_end)


def business_month_end(year: int, month: int) -> date:
    """Return the last weekday of a calendar month (pandas ``BME``)."""
    day = _month_end(year, month)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def fred_month_label(value: str | date | datetime | pd.Timestamp) -> str:
    """Return the ``fred_macro_monthly.month_end`` label for a date's month.

    Every row is labelled with the last business day of its calendar month,
    so a month can only ever have one row per series (review F06/F27).
    """
    ts = pd.Timestamp(value)
    return business_month_end(ts.year, ts.month).isoformat()


def _month_index(year: int, month: int) -> int:
    return year * 12 + month - 1


def _month_from_index(index: int) -> tuple[int, int]:
    return index // 12, index % 12 + 1


def decision_month(reference_date: date) -> tuple[int, int]:
    """Return the (year, month) of the latest feature row on ``reference_date``.

    Feature rows are dated at business month-ends, and a run uses the latest
    row dated on or before its as-of date: the current month once its last
    business day is reached, otherwise the previous month.
    """
    if reference_date >= business_month_end(reference_date.year, reference_date.month):
        return reference_date.year, reference_date.month
    return _month_from_index(_month_index(reference_date.year, reference_date.month) - 1)


def live_feature_fred_series() -> dict[str, list[str]]:
    """Map each FRED series behind a live ensemble feature to those features.

    Live features are the columns each ``config.ENSEMBLE_MODELS`` model is fed
    (``config.MODEL_FEATURE_OVERRIDES``); ``config.FRED_FEATURE_SOURCES`` names
    the series each FRED-derived feature is computed from.
    """
    series_features: dict[str, list[str]] = {}
    for model_type in config.ENSEMBLE_MODELS:
        for feature in config.MODEL_FEATURE_OVERRIDES.get(model_type, []):
            for series_id in config.FRED_FEATURE_SOURCES.get(feature, ()):
                features = series_features.setdefault(series_id, [])
                if feature not in features:
                    features.append(feature)
    return series_features


def _price_freshness_checks(
    conn: sqlite3.Connection,
    reference_date: date,
    tickers: list[str],
    max_age_days: int,
) -> list[dict[str, Any]]:
    """One freshness row per ticker, from its latest non-proxy bar."""
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        found = conn.execute(
            """
            SELECT MAX(date) FROM daily_prices
            WHERE ticker = ? AND COALESCE(proxy_fill, 0) = 0 AND date <= ?
            """,
            (ticker, reference_date.isoformat()),
        ).fetchone()
        latest_date = _coerce_iso_date(found[0] if found else None)
        row: dict[str, Any] = {
            "feed": f"Prices {ticker}",
            "table": "daily_prices",
            "column": "date",
            "ticker": ticker,
            "max_age_days": max_age_days,
            "limit_label": f"{max_age_days} days",
            "latest_date": latest_date.isoformat() if latest_date else None,
            "age_days": None,
            "status": "MISSING",
        }
        if latest_date is not None:
            age_days = (reference_date - latest_date).days
            row["age_days"] = age_days
            row["status"] = "OK" if age_days <= max_age_days else "STALE"
        rows.append(row)
    return rows


def _fred_freshness_checks(
    conn: sqlite3.Connection,
    reference_date: date,
    series_features: dict[str, list[str]],
    grace_months: int,
) -> list[dict[str, Any]]:
    """One freshness row per FRED series, judged by observation month.

    The decision row for month ``D`` uses each series' observation for month
    ``D - lag``. A series is STALE when its latest stored observation month is
    more than ``grace_months`` behind that. Stored labels are month labels,
    not observation dates, so an in-progress month labelled with a future
    month-end counts as that month, never as "0 days old"; months after the
    reference month are ignored.
    """
    row_year, row_month = decision_month(reference_date)
    row_index = _month_index(row_year, row_month)
    reference_month = f"{reference_date.year:04d}-{reference_date.month:02d}"
    rows: list[dict[str, Any]] = []
    for series_id, features in series_features.items():
        lag = int(config.FRED_SERIES_LAGS.get(series_id, config.FRED_DEFAULT_LAG_MONTHS))
        needed_index = row_index - lag
        needed_year, needed_month = _month_from_index(needed_index)
        found = conn.execute(
            """
            SELECT MAX(month_end) FROM fred_macro_monthly
            WHERE series_id = ? AND value IS NOT NULL AND substr(month_end, 1, 7) <= ?
            """,
            (series_id, reference_month),
        ).fetchone()
        latest_date = _coerce_iso_date(found[0] if found else None)
        row: dict[str, Any] = {
            "feed": f"FRED {series_id}",
            "table": "fred_macro_monthly",
            "column": "month_end",
            "series_id": series_id,
            "features": list(features),
            "lag_months": lag,
            "max_age_days": None,
            "limit_label": f"lag {lag} mo; needs {needed_year:04d}-{needed_month:02d}",
            "expected_month_end": business_month_end(needed_year, needed_month).isoformat(),
            "latest_date": latest_date.isoformat() if latest_date else None,
            "age_days": None,
            "months_behind": None,
            "status": "MISSING",
        }
        if latest_date is not None:
            behind = needed_index - _month_index(latest_date.year, latest_date.month)
            row["months_behind"] = max(0, behind)
            row["latest_label"] = f"{latest_date.year:04d}-{latest_date.month:02d}"
            row["age_label"] = f"{max(0, behind)} mo behind"
            row["status"] = "OK" if behind <= grace_months else "STALE"
        rows.append(row)
    return rows


def check_data_freshness(
    conn: sqlite3.Connection,
    reference_date: date,
    price_max_age_days: int = config.DATA_FRESHNESS_MAX_PRICE_AGE_DAYS,
    edgar_max_age_days: int = config.DATA_FRESHNESS_MAX_EDGAR_AGE_DAYS,
    fred_grace_months: int = config.DATA_FRESHNESS_FRED_GRACE_MONTHS,
    price_tickers: list[str] | None = None,
    fred_series: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate whether core feeds are fresh enough for a live monthly run.

    Checks are per feed item (review F07), so one stale input cannot hide
    behind a fresh one in the same table:

    - prices: one row per ticker, PGR and ``config.PRIMARY_FORECAST_UNIVERSE``
      by default, from the latest non-proxy bar on or before the reference
      date;
    - FRED: one row per series behind a live ensemble feature by default
      (:func:`live_feature_fred_series`), judged against the observation month
      the decision row needs (:func:`_fred_freshness_checks`);
    - PGR monthly EDGAR: the latest month against the filing-grace rule.

    Args:
        conn: Open connection.
        reference_date: Run date.
        price_max_age_days: Allowed age of each ticker's latest bar.
        edgar_max_age_days: Kept for the report; EDGAR uses the filing grace.
        fred_grace_months: Months a FRED series may lag the needed month.
        price_tickers: Tickers to check instead of the default set.
        fred_series: FRED series to check instead of the live-feature set.

    Returns:
        Dict with ``reference_date``, ``overall_status`` (``OK`` or
        ``WARNING``), ``checks`` (one dict per item) and ``warnings``.
    """
    tickers = price_tickers or ["PGR", *config.PRIMARY_FORECAST_UNIVERSE]
    series_features = live_feature_fred_series()
    if fred_series is not None:
        series_features = {sid: series_features.get(sid, []) for sid in fred_series}

    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for row in _price_freshness_checks(conn, reference_date, tickers, price_max_age_days):
        results.append(row)
        if row["status"] == "MISSING":
            warnings.append(f"{row['feed']} data is missing from daily_prices.")
        elif row["status"] != "OK":
            warnings.append(
                f"{row['feed']} is stale: latest {row['latest_date']} "
                f"({row['age_days']} days old, limit {price_max_age_days})."
            )

    for row in _fred_freshness_checks(conn, reference_date, series_features, fred_grace_months):
        results.append(row)
        used_by = f" (live features: {', '.join(row['features'])})" if row["features"] else ""
        needed = row["expected_month_end"][:7]
        if row["status"] == "MISSING":
            warnings.append(
                f"{row['feed']} is missing from fred_macro_monthly{used_by}."
            )
        elif row["status"] != "OK":
            warnings.append(
                f"{row['feed']} is stale: latest observation {row['latest_date'][:7]}, "
                f"the decision row needs {needed} (lag {row['lag_months']} mo){used_by}."
            )

    latest_raw = get_table_max_date(conn, "pgr_edgar_monthly", "month_end")
    latest_date = _coerce_iso_date(latest_raw)
    limit_label = (
        f"{config.DATA_FRESHNESS_PGR_EDGAR_FILING_GRACE_DAYS}-day filing grace"
    )
    edgar: dict[str, Any] = {
        "feed": "PGR monthly EDGAR",
        "table": "pgr_edgar_monthly",
        "column": "month_end",
        "max_age_days": edgar_max_age_days,
        "limit_label": limit_label,
        "latest_date": None,
        "age_days": None,
        "status": "MISSING",
    }
    if latest_date is None:
        warnings.append("PGR monthly EDGAR data is missing from pgr_edgar_monthly.")
    else:
        age_days = max(0, (reference_date - latest_date).days)
        expected_month_end = _expected_pgr_edgar_month_end(reference_date)
        edgar.update(
            latest_date=latest_date.isoformat(),
            age_days=age_days,
            expected_month_end=expected_month_end.isoformat(),
            status="OK" if latest_date >= expected_month_end else "STALE",
        )
        if edgar["status"] != "OK":
            warnings.append(
                f"PGR monthly EDGAR is stale: latest {latest_date.isoformat()} "
                f"({age_days} days old); expected at least "
                f"{expected_month_end.isoformat()} after the "
                f"{config.DATA_FRESHNESS_PGR_EDGAR_FILING_GRACE_DAYS}-day "
                "filing grace."
            )
    results.append(edgar)

    has_problem = any(row["status"] != "OK" for row in results)
    return {
        "reference_date": reference_date.isoformat(),
        "overall_status": "WARNING" if has_problem else "OK",
        "checks": results,
        "warnings": warnings,
    }


def check_dividend_freshness(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
    interval_multiple: float = config.DIVIDEND_FRESHNESS_INTERVAL_MULTIPLE,
    history: int = 8,
) -> list[dict[str, Any]]:
    """Per-ticker check that dividends keep up with prices (review F08).

    A ticker is STALE when its latest ex-date is older than its latest
    (non-proxy) price date minus ``interval_multiple`` times its usual payment
    interval, i.e. at least one expected payment is missing. The usual
    interval is the median gap between its last ``history + 1`` ex-dates, so
    monthly, quarterly and annual payers are each judged on their own cadence.

    Args:
        conn: Open connection.
        tickers: Tickers to check. Defaults to PGR, the ETF benchmarks and the
            peer tickers.
        interval_multiple: Allowed lag in units of the payment interval.
        history: Number of recent gaps used for the median interval.

    Returns:
        One dict per ticker with ``ticker``, ``status`` (``OK``, ``STALE`` or
        ``NO_HISTORY`` for tickers with fewer than two dividends or no prices),
        ``last_ex_date``, ``last_price_date``, ``interval_days`` and
        ``due_by`` (the earliest ex-date that would count as fresh).
    """
    if tickers is None:
        tickers = ["PGR", *config.ETF_BENCHMARK_UNIVERSE, *config.PEER_TICKER_UNIVERSE]
    results: list[dict[str, Any]] = []
    for ticker in tickers:
        ex_dates = [
            date.fromisoformat(r[0][:10]) for r in conn.execute(
                "SELECT ex_date FROM daily_dividends WHERE ticker = ? ORDER BY ex_date",
                (ticker,),
            )
        ]
        row = conn.execute(
            "SELECT MAX(date) FROM daily_prices WHERE ticker = ? AND proxy_fill = 0",
            (ticker,),
        ).fetchone()
        last_price = _coerce_iso_date(row[0] if row else None)
        result: dict[str, Any] = {
            "ticker": ticker,
            "status": "NO_HISTORY",
            "last_ex_date": ex_dates[-1].isoformat() if ex_dates else None,
            "last_price_date": last_price.isoformat() if last_price else None,
            "interval_days": None,
            "due_by": None,
        }
        if len(ex_dates) >= 2 and last_price is not None:
            recent = ex_dates[-(history + 1):]
            gaps = sorted((b - a).days for a, b in zip(recent, recent[1:]))
            mid = len(gaps) // 2
            interval = (
                float(gaps[mid]) if len(gaps) % 2
                else (gaps[mid - 1] + gaps[mid]) / 2.0
            )
            due_by = last_price - timedelta(days=interval_multiple * interval)
            result.update(
                status="OK" if ex_dates[-1] >= due_by else "STALE",
                interval_days=interval,
                due_by=due_by.isoformat(),
            )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def _iso_week(date_str: str) -> tuple[int, int]:
    iso = date.fromisoformat(date_str[:10]).isocalendar()
    return iso[0], iso[1]


def upsert_prices(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    one_bar_per_week: bool = False,
) -> int:
    """Bulk-insert or replace price records.

    Args:
        conn: Open connection.
        records: List of dicts with keys matching ``daily_prices`` columns.
            Required keys: ``ticker``, ``date``, ``close``.
            Optional keys: ``open``, ``high``, ``low``, ``volume``,
            ``source``, ``proxy_fill`` (default 0).
        one_bar_per_week: Keep at most one bar per ticker per ISO week, the
            latest-dated one (review F22). Alpha Vantage's weekly series
            labels the in-progress week with its latest trading day, so a
            later fetch returns the completed bar under a different date. With
            this flag the incoming records are collapsed to their latest bar
            per week, an incoming bar older than a stored bar of the same week
            is dropped, and stored bars superseded by a newer bar are deleted.
            The weekly price loaders pass True.

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
    if one_bar_per_week:
        normalised = _latest_bar_per_week(conn, normalised)
    conn.executemany(sql, normalised)
    if one_bar_per_week:
        _delete_superseded_week_bars(conn, sorted({r["ticker"] for r in normalised}))
    conn.commit()
    return len(normalised)


def _latest_bar_per_week(
    conn: sqlite3.Connection, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collapse records to the latest bar per ticker-ISO-week, including stored bars."""
    latest: dict[tuple[str, int, int], dict[str, Any]] = {}
    for rec in records:
        key = (rec["ticker"], *_iso_week(rec["date"]))
        if key not in latest or rec["date"] > latest[key]["date"]:
            latest[key] = rec
    stored_latest: dict[tuple[str, int, int], str] = {}
    for ticker in {k[0] for k in latest}:
        for (stored_date,) in conn.execute(
            "SELECT date FROM daily_prices WHERE ticker = ?", (ticker,)
        ):
            key = (ticker, *_iso_week(stored_date))
            if key in latest and stored_date > stored_latest.get(key, ""):
                stored_latest[key] = stored_date
    return [
        rec for key, rec in latest.items()
        if rec["date"] >= stored_latest.get(key, "")
    ]


def _delete_superseded_week_bars(conn: sqlite3.Connection, tickers: list[str]) -> int:
    """Delete bars that share a ticker-ISO-week with a later-dated bar."""
    doomed: list[tuple[str, str]] = []
    for ticker in tickers:
        latest: dict[tuple[int, int], str] = {}
        dates = [r[0] for r in conn.execute(
            "SELECT date FROM daily_prices WHERE ticker = ?", (ticker,)
        )]
        for d in dates:
            wk = _iso_week(d)
            if d > latest.get(wk, ""):
                latest[wk] = d
        doomed.extend((ticker, d) for d in dates if d != latest[_iso_week(d)])
    conn.executemany(
        "DELETE FROM daily_prices WHERE ticker = ? AND date = ?", doomed
    )
    return len(doomed)


def dedupe_weekly_price_bars(
    conn: sqlite3.Connection, tickers: list[str] | None = None
) -> int:
    """Keep only the latest bar per ticker-ISO-week in ``daily_prices``.

    Removes partial-week bars left behind before ``upsert_prices`` enforced
    one bar per week (31 ticker-weeks in March-May 2026, review F22).

    Returns:
        Number of rows deleted.
    """
    if tickers is None:
        tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM daily_prices")]
    n = _delete_superseded_week_bars(conn, list(tickers))
    conn.commit()
    return n


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
    """Bulk-insert or replace PGR quarterly fundamentals from EDGAR XBRL."""
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO pgr_fundamentals_quarterly
            (period_end, roe, eps, revenue, net_income, filing_date, source)
        VALUES
            (:period_end, :roe, :eps, :revenue, :net_income, :filing_date, :source)
    """
    normalised = [
        {
            "period_end":  r["period_end"],
            "roe":         r.get("roe"),
            "eps":         r.get("eps"),
            "revenue":     r.get("revenue"),
            "net_income":  r.get("net_income"),
            "filing_date": r.get("filing_date"),
            "source":      r.get("source"),
        }
        for r in records
    ]
    conn.executemany(sql, normalised)
    conn.commit()
    return len(normalised)


def replace_pgr_fundamentals(
    conn: sqlite3.Connection, records: list[dict[str, Any]]
) -> int:
    """Replace the whole ``pgr_fundamentals_quarterly`` table with ``records``.

    Used when the definitions change, so rows that the new extraction no
    longer produces (e.g. a full-year row stored as a quarter) do not linger.
    """
    conn.execute("DELETE FROM pgr_fundamentals_quarterly")
    return upsert_pgr_fundamentals(conn, records)


def get_pgr_fundamentals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all PGR quarterly fundamentals, sorted ascending by period_end."""
    sql = """
        SELECT period_end, roe, eps, revenue, net_income, filing_date
        FROM pgr_fundamentals_quarterly
        ORDER BY period_end ASC
    """
    df = pd.read_sql_query(sql, conn, parse_dates=["period_end"])
    if not df.empty:
        df = df.set_index("period_end")
    return df


# Value columns of pgr_edgar_monthly (everything except the key and the
# provenance columns filing_date / filing_type / accession_number).
PGR_EDGAR_MONTHLY_VALUE_COLUMNS: tuple[str, ...] = (
    "combined_ratio", "pif_total", "pif_growth_yoy",
    "gainshare_estimate", "book_value_per_share", "eps_basic",
    "avg_diluted_equivalent_shares",
    "net_premiums_written", "net_premiums_earned", "net_income",
    "eps_diluted", "total_net_realized_gains", "service_revenues",
    "fees_and_other_revenues", "losses_lae", "policy_acquisition_costs",
    "other_underwriting_expenses", "interest_expense",
    "provision_for_income_taxes", "total_comprehensive_income",
    "comprehensive_eps_diluted", "avg_shares_basic", "avg_shares_diluted",
    "loss_lae_ratio", "expense_ratio",
    "npw_agency", "npw_direct", "npw_commercial", "npw_property",
    "npe_agency", "npe_direct", "npe_commercial", "npe_property",
    "pif_agency_auto", "pif_direct_auto", "pif_special_lines", "pif_property",
    "pif_commercial_lines",
    "pif_total_personal_lines",
    "investment_income", "total_revenues", "total_expenses",
    "income_before_income_taxes", "roe_net_income_ttm",
    "roe_comprehensive_trailing_12m", "shareholders_equity", "total_assets",
    "total_investments", "loss_lae_reserves", "unearned_premiums",
    "debt", "total_liabilities", "common_shares_outstanding",
    "shares_repurchased", "avg_cost_per_share",
    "fte_return_fixed_income", "fte_return_common_stocks",
    "fte_return_total_portfolio", "investment_book_yield",
    "net_unrealized_gains_fixed", "fixed_income_duration",
    "debt_to_total_capital", "weighted_avg_credit_quality",
    "channel_mix_agency_pct", "npw_growth_yoy", "underwriting_income",
    "unearned_premium_growth_yoy", "buyback_yield",
)
PGR_EDGAR_MONTHLY_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "filing_date", "filing_type", "accession_number",
)
# Columns computed over the whole time series, never read from one filing.
# They are not recorded in pgr_edgar_monthly_raw.  (pif_total and
# pif_total_personal_lines are also recomputed from their components, but the
# raw table keeps the totals as printed in each filing.)
PGR_EDGAR_MONTHLY_DERIVED_ONLY_COLUMNS: frozenset[str] = frozenset({
    "pif_growth_yoy", "gainshare_estimate", "channel_mix_agency_pct",
    "npw_growth_yoy", "underwriting_income", "unearned_premium_growth_yoy",
    "buyback_yield",
})
PGR_EDGAR_MONTHLY_TEXT_COLUMNS: frozenset[str] = frozenset({
    "filing_date", "filing_type", "accession_number", "weighted_avg_credit_quality",
})


def normalise_accession(accession: Any) -> str | None:
    """Return an EDGAR accession number in dashed ``##########-YY-######`` form.

    The table held both forms (235 dashed, 28 undashed; F33).  Anything that
    is not 18 digits is returned unchanged (as a string).
    """
    if accession is None:
        return None
    text = str(accession).strip()
    if not text or text.lower() == "nan":
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 18:
        return text
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def _normalise_edgar_monthly_record(r: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"month_end": r["month_end"]}
    for col in PGR_EDGAR_MONTHLY_PROVENANCE_COLUMNS + PGR_EDGAR_MONTHLY_VALUE_COLUMNS:
        out[col] = r.get(col)
    # The live 8-K parser emits the CSV-era key ``roe_net_income_trailing_12m``;
    # accept either name.
    if out["roe_net_income_ttm"] is None:
        out["roe_net_income_ttm"] = r.get("roe_net_income_trailing_12m")
    out["accession_number"] = normalise_accession(out["accession_number"])
    for col, value in list(out.items()):
        if isinstance(value, float) and value != value:
            out[col] = None
    return out


def upsert_pgr_edgar_monthly(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    mode: str = "merge",
) -> int:
    """Insert or update PGR monthly EDGAR rows without mixing filings (F33).

    Accepts any subset of the column set; missing keys are NULL.  A row's
    ``accession_number`` and ``filing_date`` say which filing its values came
    from, and an update never makes that false:

    * no row for the month: insert it;
    * same filing (equal accession, or either side has none): the new
      non-NULL values win, others are kept, and the stored provenance
      columns are kept;
    * a different filing: an **earlier-filed** one replaces the whole row
      (first-reported wins); a later one leaves the row unchanged and is
      logged.  Every parse is still recorded in ``pgr_edgar_monthly_raw`` by
      the caller.

    ``mode="insert_missing"`` only inserts months that are absent; existing
    rows are never touched.  ``load_from_csv`` uses it, so re-seeding from
    the CSV cannot overwrite newer rows.

    Derived fields (YoY growth, Gainshare, PIF totals, …) should be refreshed
    afterwards over the whole table with
    ``scripts.edgar_8k_fetcher.recompute_derived_fields``.

    Returns:
        Number of rows inserted or updated.
    """
    if mode not in ("merge", "insert_missing"):
        raise ValueError(f"unknown mode: {mode!r}")
    if not records:
        return 0
    columns = ("month_end",) + PGR_EDGAR_MONTHLY_PROVENANCE_COLUMNS + PGR_EDGAR_MONTHLY_VALUE_COLUMNS
    insert_sql = (
        f"INSERT INTO pgr_edgar_monthly ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)})"
    )
    merge_sql = (
        "UPDATE pgr_edgar_monthly SET "
        + ", ".join(
            [f"{c} = COALESCE({c}, :{c})" for c in PGR_EDGAR_MONTHLY_PROVENANCE_COLUMNS]
            + [f"{c} = COALESCE(:{c}, {c})" for c in PGR_EDGAR_MONTHLY_VALUE_COLUMNS]
        )
        + " WHERE month_end = :month_end"
    )
    replace_sql = (
        "UPDATE pgr_edgar_monthly SET "
        + ", ".join(
            f"{c} = :{c}"
            for c in PGR_EDGAR_MONTHLY_PROVENANCE_COLUMNS + PGR_EDGAR_MONTHLY_VALUE_COLUMNS
        )
        + " WHERE month_end = :month_end"
    )

    written = 0
    for record in records:
        row = _normalise_edgar_monthly_record(record)
        existing = conn.execute(
            "SELECT accession_number, filing_date FROM pgr_edgar_monthly WHERE month_end = ?",
            (row["month_end"],),
        ).fetchone()
        if existing is None:
            conn.execute(insert_sql, row)
            written += 1
            continue
        if mode == "insert_missing":
            continue
        old_accession = normalise_accession(existing[0])
        old_filed = existing[1]
        new_accession = row["accession_number"]
        if old_accession is None or new_accession is None or old_accession == new_accession:
            conn.execute(merge_sql, row)
            written += 1
        elif row["filing_date"] and old_filed and row["filing_date"] < old_filed:
            conn.execute(replace_sql, row)
            written += 1
        else:
            logger.info(
                "pgr_edgar_monthly %s: keeping %s (filed %s); later filing %s "
                "(filed %s) not merged",
                row["month_end"], old_accession, old_filed,
                new_accession, row["filing_date"],
            )
    conn.commit()
    return written


def record_pgr_edgar_raw(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    parser_version: str,
    method: str = "parsed",
    recorded_at: str | None = None,
) -> int:
    """Append parsed 8-K values to the provenance tables (F33).

    Each record becomes one ``pgr_edgar_filing_parses`` row (accession,
    parser version, source URL, fetched-at) and one ``pgr_edgar_monthly_raw``
    row per non-NULL value column.  A (accession, parser_version) pair that is
    already recorded is left as it is: the tables are append-only.  Records
    without an accession number are skipped.  A record's ``derived_fields``
    list marks values computed by the parser (``method='derived'``).

    Returns:
        Number of value rows added.
    """
    if not records:
        return 0
    stamp = recorded_at or datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    added = 0
    for record in records:
        row = _normalise_edgar_monthly_record(record)
        accession = row["accession_number"]
        if accession is None:
            continue
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO pgr_edgar_filing_parses (
                accession_number, parser_version, month_end, filing_date,
                source_url, fetched_at, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                accession, parser_version, row["month_end"], row["filing_date"],
                record.get("document_url"), record.get("fetched_at"), stamp,
            ),
        )
        if cursor.rowcount == 0:
            continue  # this filing was already recorded under this parser
        parse_id = cursor.lastrowid
        derived = set(record.get("derived_fields") or ())
        values = []
        for field in PGR_EDGAR_MONTHLY_VALUE_COLUMNS:
            value = row[field]
            if value is None or field in PGR_EDGAR_MONTHLY_DERIVED_ONLY_COLUMNS:
                continue
            is_text = field in PGR_EDGAR_MONTHLY_TEXT_COLUMNS
            values.append((
                parse_id,
                field,
                None if is_text else float(value),
                str(value) if is_text else None,
                "derived" if field in derived else method,
            ))
        conn.executemany(
            """
            INSERT INTO pgr_edgar_monthly_raw
                (parse_id, field, value_real, value_text, method)
            VALUES (?, ?, ?, ?, ?)
            """,
            values,
        )
        added += len(values)
    conn.commit()
    return added


def get_pgr_edgar_first_reported(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the first-reported value of every (month_end, field) as a long table."""
    return pd.read_sql_query(
        "SELECT * FROM pgr_edgar_monthly_first_reported ORDER BY month_end, field",
        conn,
    )


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


def replace_relative_returns(
    conn: sqlite3.Connection,
    benchmark: str,
    target_horizon: int,
    records: list[dict[str, Any]],
) -> int:
    """Replace every stored row of one benchmark/horizon with ``records``.

    Unlike :func:`upsert_relative_returns`, rows whose date is absent from
    ``records`` are deleted, so the table stays a pure function of the stored
    prices, dividends and splits (e.g. after a window-definition change or a
    newly added split). Runs in one transaction.

    Returns:
        Number of rows written.
    """
    with conn:
        conn.execute(
            "DELETE FROM monthly_relative_returns WHERE benchmark = ? AND target_horizon = ?",
            (benchmark, int(target_horizon)),
        )
    return upsert_relative_returns(conn, records)


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

    Values must be raw (unlagged) observations. ``month_end`` is normalised
    to the month's last business day (:func:`fred_month_label`), so a later
    write for the same series and month replaces the earlier row whatever
    date it was labelled with; migration 005 also enforces one row per
    (series, month) with a unique index.

    Args:
        conn: Open connection.
        records: List of dicts with keys ``series_id``, ``month_end``
            (ISO date string ``"YYYY-MM-DD"``, any day of the month), and
            ``value`` (float or None).

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
            "month_end": fred_month_label(r["month_end"]),
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
