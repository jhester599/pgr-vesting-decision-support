"""Writes the monthly 8-K data to the DB: the EDGAR fetch, the CSV seed and its export.

``fetch_and_upsert`` fetches, parses and upserts recent months;
``load_from_csv`` seeds ``pgr_edgar_monthly`` from the committed
``data/processed/pgr_edgar_cache.csv`` (the one CSV loader); and
``export_edgar_cache_csv`` writes that CSV from the table.

Both writers are idempotent. Every parsed value is appended to
``pgr_edgar_monthly_raw`` (keyed by accession, field and ``PARSER_VERSION``;
an existing key is left alone), ``db_client.upsert_pgr_edgar_monthly`` never
mixes filings within a row, and derived fields are recomputed over the whole
table (``derive.recompute_derived_fields``).
Functions in the other ``edgar_monthly`` modules are called through the module
(``fetch.get(...)``), so a test patches a function once, in the module that
defines it.

"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pgr_vds.ingestion.edgar_monthly import derive, fetch, parse
from src.database import db_client

log = logging.getLogger(__name__)


# Earliest filing date a ``--backfill-years`` run reaches.  (The full history
# back to August 2004 is rebuilt by scripts/repair_edgar_history.py.)
BACKFILL_EARLIEST_DATE: str = "2010-01-01"

# src/pgr_vds/ingestion/edgar_monthly/load.py -> repository root.
REPO_ROOT: Path = Path(__file__).resolve().parents[4]

# The committed seed CSV (``--load-from-csv`` without a path).
DEFAULT_CSV_PATH: str = str(REPO_ROOT / "data" / "processed" / "pgr_edgar_cache.csv")


def check_staleness(conn: sqlite3.Connection) -> None:
    """Log a warning if the most recent 8-K data is more than 45 days old.

    PGR typically files within 20 days of month-end.  If the newest row is
    older than 45 days it almost certainly means a filing was missed (or the
    workflow's primary-pass run failed and the fallback hasn't fired yet).

    Args:
        conn: Open SQLite connection with ``pgr_edgar_monthly`` populated.
    """
    row = conn.execute(
        "SELECT MAX(month_end) FROM pgr_edgar_monthly"
    ).fetchone()

    if row is None or row[0] is None:
        log.warning("WARNING: pgr_edgar_monthly table is empty — no 8-K data present.")
        return

    most_recent = datetime.strptime(row[0], "%Y-%m-%d").date()
    age_days = (date.today() - most_recent).days

    if age_days > 45:
        log.warning(
            "WARNING: Most recent 8-K data is %d days old — "
            "PGR may not have filed yet.",
            age_days,
        )
    else:
        log.info(
            "Most recent 8-K data: %s (%d days old).", row[0], age_days
        )


def fetch_and_upsert(
    conn: sqlite3.Connection,
    backfill_years: int = 2,
    dry_run: bool = False,
) -> int:
    """Fetch PGR 8-K operating metrics and upsert them to the DB.

    Workflow:
      1. Compute cutoff date (today minus backfill_years, floored at
         BACKFILL_EARLIEST_DATE).
      2. Fetch all 8-K (items 7.01/2.02) filings from EDGAR submissions (with
         pagination) back to the cutoff.
      3. For each filing, resolve the primary HTML exhibit URL, parse it for
         combined_ratio and PIF, and collect the result.  Parse failures are
         logged and skipped (never abort the full run).
      4. Compute derived fields (pif_growth_yoy, gainshare_estimate) over the
         full sorted time series.
      5. Deduplicate by month_end (last filing for that period wins).
      6. Upsert all rows via db_client.upsert_pgr_edgar_monthly (INSERT OR REPLACE).

    Args:
        conn: Open SQLite connection.
        backfill_years: How many years back to fetch (default: 2).
        dry_run: If True, parse everything but skip the DB write.

    Returns:
        Number of rows upserted (0 for dry runs).
    """
    today = date.today()
    cutoff_raw = date(today.year - backfill_years, today.month, today.day)
    earliest = date.fromisoformat(BACKFILL_EARLIEST_DATE)
    effective_cutoff = max(cutoff_raw, earliest).isoformat()

    log.info(
        "Backfill window: %s -> %s  (backfill_years=%d)",
        effective_cutoff,
        today.isoformat(),
        backfill_years,
    )

    filings = fetch.fetch_all_8k_filings(cutoff_date=effective_cutoff)
    if not filings:
        log.info("No 8-K (items 7.01/2.02) filings found in the requested date range.")
        return 0

    records: list[dict[str, Any]] = []
    parse_errors = 0

    for filing in filings:
        try:
            parsed = parse.parse_filing(filing)
        except Exception as exc:
            parse_errors += 1
            log.exception(
                "SKIP %s (filed %s) due to parse failure. Error=%r",
                filing["accession_number"],
                filing["filing_date"],
                exc,
            )
            continue
        if parsed is not None:
            records.append(parsed)

    if parse_errors > 0:
        log.warning("%d filing(s) skipped due to parse errors.", parse_errors)

    if not records:
        log.info("No records to upsert.")
        return 0

    deduped = parse.select_monthly_releases(records)

    # Coverage report
    n_total = len(deduped)

    def _cov(field: str) -> str:
        n = sum(1 for r in deduped if r.get(field) is not None)
        return f"{n}/{n_total}"

    log.info(
        "Coverage  combined_ratio=%s  pif_total=%s  npw=%s  npw_agency=%s  "
        "investment_income=%s  bvps=%s  date_range=%s->%s",
        _cov("combined_ratio"),
        _cov("pif_total"),
        _cov("net_premiums_written"),
        _cov("npw_agency"),
        _cov("investment_income"),
        _cov("book_value_per_share"),
        deduped[0]["month_end"],
        deduped[-1]["month_end"],
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows would be upserted).", n_total)
        return 0

    months_before = conn.execute(
        "SELECT COUNT(*) FROM pgr_edgar_monthly"
    ).fetchone()[0]
    # Every parse is kept, append-only, before the monthly table is touched.
    n_raw = db_client.record_pgr_edgar_raw(conn, deduped, parse.PARSER_VERSION)
    n = db_client.upsert_pgr_edgar_monthly(conn, deduped)
    derive.recompute_derived_fields(conn)
    months_after = conn.execute(
        "SELECT COUNT(*) FROM pgr_edgar_monthly"
    ).fetchone()[0]
    log.info(
        "Recorded %d raw values; upserted %d rows to pgr_edgar_monthly "
        "(%d new months).",
        n_raw, n, months_after - months_before,
    )

    # v7.2: warn when no new months were added to alert on format changes.
    if months_after == months_before:
        log.warning(
            "NOTE: No new months added this run (table has %d months). If this "
            "persists, check whether PGR has changed its 8-K filing format.",
            months_after,
        )

    return n


def load_from_csv(
    conn: sqlite3.Connection,
    csv_path: str,
    dry_run: bool = False,
) -> int:
    """Seed ``pgr_edgar_monthly`` from the committed ``pgr_edgar_cache.csv``.

    The CSV (``data/processed/pgr_edgar_cache.csv``) contains 256+ rows of
    monthly PGR data going back to 2004, pre-extracted from SEC EDGAR filings.
    This function converts the CSV's ``report_period`` (``"YYYY-MM"``) to
    ``month_end`` (last calendar day of that month, ``"YYYY-MM-DD"``) and maps
    the CSV columns into the DB schema.

    Only months missing from the table are inserted, so re-running it on a
    live-populated DB changes nothing (F33).  Inserted values are recorded in
    ``pgr_edgar_monthly_raw`` with ``method='csv'``.  Derived fields are then
    recomputed over the whole table.

    No network calls are made.  The regular ``fetch_and_upsert`` EDGAR fetch
    covers recent months not yet in the CSV.

    Args:
        conn: Open SQLite connection.
        csv_path: Path to ``pgr_edgar_cache.csv``.
        dry_run: If True, parse but skip the DB write.

    Returns:
        Number of months inserted (0 for dry runs).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``report_period`` column is missing from the CSV.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info("Loading historical data from %s …", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "report_period" not in df.columns:
        raise ValueError(
            f"Expected 'report_period' column in {csv_path}; "
            f"found: {list(df.columns[:10])}"
        )

    # Convert "YYYY-MM" → last calendar day of that month ("YYYY-MM-DD")
    df["month_end"] = (
        pd.to_datetime(df["report_period"].astype(str), format="%Y-%m")
        + pd.offsets.MonthEnd(0)
    )
    df["month_end"] = df["month_end"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("month_end").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Direct CSV column → DB column mappings (v6.2 expanded schema)
    # CSV column name              DB column name
    # -----------------------------------------------------------------------
    DIRECT_MAP: dict[str, str] = {
        "filing_date":                  "filing_date",
        "filing_type":                  "filing_type",
        "accession_number":             "accession_number",
        "combined_ratio":              "combined_ratio",
        "pif_total":                   "pif_total",
        "net_premiums_written":        "net_premiums_written",
        "net_premiums_earned":         "net_premiums_earned",
        "net_income":                  "net_income",
        "eps_diluted":                 "eps_diluted",
        "eps_basic":                   "eps_basic",
        "avg_diluted_equivalent_shares": "avg_diluted_equivalent_shares",
        "total_net_realized_gains":    "total_net_realized_gains",
        "service_revenues":            "service_revenues",
        "fees_and_other_revenues":     "fees_and_other_revenues",
        "losses_lae":                  "losses_lae",
        "policy_acquisition_costs":    "policy_acquisition_costs",
        "other_underwriting_expenses": "other_underwriting_expenses",
        "interest_expense":            "interest_expense",
        "provision_for_income_taxes":  "provision_for_income_taxes",
        "total_comprehensive_income":  "total_comprehensive_income",
        "comprehensive_eps_diluted":   "comprehensive_eps_diluted",
        "avg_shares_basic":            "avg_shares_basic",
        "avg_shares_diluted":          "avg_shares_diluted",
        "loss_lae_ratio":              "loss_lae_ratio",
        "expense_ratio":               "expense_ratio",
        "book_value_per_share":        "book_value_per_share",
        # Segment-level channel metrics
        "npw_agency":                  "npw_agency",
        "npw_direct":                  "npw_direct",
        "npw_commercial":              "npw_commercial",
        "npw_property":                "npw_property",
        "npe_agency":                  "npe_agency",
        "npe_direct":                  "npe_direct",
        "npe_commercial":              "npe_commercial",
        "npe_property":                "npe_property",
        "pif_agency_auto":             "pif_agency_auto",
        "pif_direct_auto":             "pif_direct_auto",
        "pif_special_lines":           "pif_special_lines",
        "pif_property":                "pif_property",
        "pif_commercial_lines":        "pif_commercial_lines",
        "pif_total_personal_lines":    "pif_total_personal_lines",
        # Company-level operating metrics
        "investment_income":           "investment_income",
        "total_revenues":              "total_revenues",
        "total_expenses":              "total_expenses",
        "income_before_income_taxes":  "income_before_income_taxes",
        "roe_net_income_trailing_12m": "roe_net_income_ttm",  # CSV name differs
        "roe_comprehensive_trailing_12m": "roe_comprehensive_trailing_12m",
        "shareholders_equity":         "shareholders_equity",
        "total_assets":                "total_assets",
        "total_investments":           "total_investments",
        "loss_lae_reserves":           "loss_lae_reserves",
        "unearned_premiums":           "unearned_premiums",
        "debt":                        "debt",
        "total_liabilities":           "total_liabilities",
        "common_shares_outstanding":   "common_shares_outstanding",
        "shares_repurchased":          "shares_repurchased",
        "avg_cost_per_share":          "avg_cost_per_share",
        # Investment portfolio metrics
        "fte_return_fixed_income":     "fte_return_fixed_income",
        "fte_return_common_stocks":    "fte_return_common_stocks",
        "fte_return_total_portfolio":  "fte_return_total_portfolio",
        "investment_book_yield":       "investment_book_yield",
        "net_unrealized_gains_fixed":  "net_unrealized_gains_fixed",
        "fixed_income_duration":       "fixed_income_duration",
        "debt_to_total_capital":       "debt_to_total_capital",
        "weighted_avg_credit_quality": "weighted_avg_credit_quality",
    }

    text_cols = {
        "filing_date",
        "filing_type",
        "accession_number",
        "weighted_avg_credit_quality",
    }
    for csv_col, db_col in DIRECT_MAP.items():
        if csv_col in df.columns and db_col not in text_cols:
            df[db_col] = pd.to_numeric(df[csv_col], errors="coerce")
        elif csv_col in df.columns:
            df[db_col] = df[csv_col].astype(str)
        else:
            df[db_col] = float("nan")

    # -----------------------------------------------------------------------
    # Build records.  Derived fields (YoY growth, Gainshare, PIF totals,
    # channel mix, underwriting income) are not taken from the CSV: they are
    # recomputed over the whole table after the insert, by calendar month (F16).
    # -----------------------------------------------------------------------
    unique_cols = ["month_end"] + list(dict.fromkeys(DIRECT_MAP.values()))
    df_out = df[unique_cols].copy()

    def _nan_to_none(val: Any) -> Any:
        """Convert float NaN to None for SQLite NULL storage."""
        try:
            if val != val:  # NaN check
                return None
        except TypeError:
            pass
        if isinstance(val, str) and val.lower() == "nan":
            return None
        return val

    records_raw: list[dict[str, Any]] = [
        {col: _nan_to_none(row[col]) for col in unique_cols}
        for _, row in df_out.iterrows()
    ]

    log.info(
        "CSV loaded: %d rows  date_range=%s->%s",
        len(records_raw),
        records_raw[0]["month_end"] if records_raw else "n/a",
        records_raw[-1]["month_end"] if records_raw else "n/a",
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows read).", len(records_raw))
        return 0

    # Idempotent against newer rows (F33): only months absent from the table
    # are inserted; rows written by the live parser are never overwritten.
    existing = {
        str(row[0])
        for row in conn.execute("SELECT month_end FROM pgr_edgar_monthly").fetchall()
    }
    new_records = [r for r in records_raw if r["month_end"] not in existing]
    db_client.record_pgr_edgar_raw(
        conn, new_records, parser_version=os.path.basename(csv_path), method="csv",
    )
    n = db_client.upsert_pgr_edgar_monthly(conn, new_records, mode="insert_missing")
    derive.recompute_derived_fields(conn)
    log.info(
        "Inserted %d missing months from CSV into pgr_edgar_monthly "
        "(%d months already present were left unchanged).",
        n, len(records_raw) - len(new_records),
    )
    return n


# Column order of data/processed/pgr_edgar_cache.csv.  ``report_period`` is
# ``YYYY-MM``; ``roe_net_income_trailing_12m`` is the DB's ``roe_net_income_ttm``.
EDGAR_CACHE_CSV_COLUMNS: tuple[str, ...] = (
    "report_period",
    "filing_date",
    "filing_type",
    "accession_number",
    "net_premiums_written",
    "net_premiums_earned",
    "combined_ratio",
    "avg_diluted_equivalent_shares",
    "investment_income",
    "total_net_realized_gains",
    "service_revenues",
    "fees_and_other_revenues",
    "total_revenues",
    "losses_lae",
    "policy_acquisition_costs",
    "other_underwriting_expenses",
    "interest_expense",
    "total_expenses",
    "income_before_income_taxes",
    "provision_for_income_taxes",
    "net_income",
    "total_comprehensive_income",
    "eps_basic",
    "eps_diluted",
    "comprehensive_eps_diluted",
    "avg_shares_basic",
    "avg_shares_diluted",
    "loss_lae_ratio",
    "expense_ratio",
    "pif_agency_auto",
    "pif_direct_auto",
    "pif_special_lines",
    "pif_property",
    "pif_total_personal_lines",
    "pif_commercial_lines",
    "pif_total",
    "npw_agency",
    "npw_direct",
    "npw_property",
    "npw_commercial",
    "npe_agency",
    "npe_direct",
    "npe_property",
    "npe_commercial",
    "total_investments",
    "total_assets",
    "loss_lae_reserves",
    "unearned_premiums",
    "debt",
    "total_liabilities",
    "shareholders_equity",
    "common_shares_outstanding",
    "shares_repurchased",
    "avg_cost_per_share",
    "book_value_per_share",
    "roe_net_income_trailing_12m",
    "roe_comprehensive_trailing_12m",
    "debt_to_total_capital",
    "fixed_income_duration",
    "fte_return_fixed_income",
    "fte_return_common_stocks",
    "fte_return_total_portfolio",
    "investment_book_yield",
    "net_unrealized_gains_fixed",
    "weighted_avg_credit_quality",
)


def export_edgar_cache_csv(conn: sqlite3.Connection, csv_path: str) -> int:
    """Write ``pgr_edgar_monthly`` to ``pgr_edgar_cache.csv`` (the committed seed).

    The CSV is a snapshot of the DB table, so re-seeding an empty DB from it
    with ``load_from_csv`` reproduces the table.  Returns the number of rows.
    """
    df = pd.read_sql_query("SELECT * FROM pgr_edgar_monthly ORDER BY month_end", conn)
    df["report_period"] = df["month_end"].str.slice(0, 7)
    df["roe_net_income_trailing_12m"] = df["roe_net_income_ttm"]
    df[list(EDGAR_CACHE_CSV_COLUMNS)].to_csv(csv_path, index=False, float_format="%.6f")
    return len(df)
