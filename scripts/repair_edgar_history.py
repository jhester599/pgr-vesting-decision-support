#!/usr/bin/env python3
"""Rebuild the PGR EDGAR tables from the filings (review 2026-09-25, step 3b).

``pgr_edgar_monthly``
    Every monthly release since August 2004 is re-fetched from EDGAR and
    re-parsed with the current parser (``edgar_8k_fetcher.PARSER_VERSION``).
    Candidates are 8-Ks with item 7.01 or 2.02, or 9.01-only filings with an
    EX-99 exhibit, read from the primary submissions file and every flat
    pagination file.  One release per month is kept (the earliest filing
    with a combined ratio).  Every parsed value is appended to
    ``pgr_edgar_monthly_raw``; the monthly table is then rebuilt from those
    records and its derived fields are recomputed over the whole table.

``pgr_fundamentals_quarterly``
    Rebuilt from the XBRL companyfacts file: discrete quarters (Q4 = FY −
    9M), earliest-filed values, ROE = TTM net income / average equity.

The script works on the DB given by ``--db`` and refuses the committed DB
unless ``--allow-committed-db`` is passed; run it on a copy and review the
diff first.  EDGAR requests are cached (``--cache-dir``), throttled to 4 per
second, and sent with the ``EDGAR_USER_AGENT`` header, which must be set.

Outputs:
  * a cell-level diff CSV (``--diff-csv``): one row per changed cell, for
    every month and quarter whose values changed;
  * optionally the regenerated ``pgr_edgar_cache.csv`` (``--export-csv``);
  * a validation summary on stdout.

Usage::

    cp data/pgr_financials.db /tmp/repair.db
    EDGAR_USER_AGENT="Name email@example.com" \\
        python scripts/repair_edgar_history.py --db /tmp/repair.db \\
        --diff-csv docs/reviews/2026-09-25_step3b_edgar_cell_diff.csv \\
        --export-csv data/processed/pgr_edgar_cache.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from scripts import edgar_8k_fetcher as fetcher  # noqa: E402
from src.database import db_client  # noqa: E402
from src.ingestion import edgar_client  # noqa: E402
from src.processing import pgr_edgar_validation as validation  # noqa: E402

log = logging.getLogger("repair_edgar_history")

# Filings for August 2004 (the first month) were filed in September 2004.
DEFAULT_SINCE = "2004-09-01"


def _read_table(conn: sqlite3.Connection, table: str, key: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {key}", conn)
    return df.set_index(key)


def _same(a: Any, b: Any) -> bool:
    a_missing = a is None or (isinstance(a, float) and math.isnan(a))
    b_missing = b is None or (isinstance(b, float) and math.isnan(b))
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    return str(a) == str(b)


def cell_diff(before: pd.DataFrame, after: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return one row per changed cell between two versions of a table."""
    rows: list[dict[str, Any]] = []
    columns = list(dict.fromkeys(list(after.columns) + list(before.columns)))
    for key in sorted(set(before.index) | set(after.index)):
        old = before.loc[key] if key in before.index else None
        new = after.loc[key] if key in after.index else None
        for col in columns:
            a = old.get(col) if old is not None else None
            b = new.get(col) if new is not None else None
            if not _same(a, b):
                rows.append({
                    "table": table,
                    "key": key,
                    "column": col,
                    "old": None if a is None or (isinstance(a, float) and math.isnan(a)) else a,
                    "new": None if b is None or (isinstance(b, float) and math.isnan(b)) else b,
                    "row_status": (
                        "added" if old is None else "removed" if new is None else "changed"
                    ),
                })
    return pd.DataFrame(
        rows, columns=["table", "key", "column", "old", "new", "row_status"]
    )


def rebuild_monthly(conn: sqlite3.Connection, since: str) -> list[dict[str, Any]]:
    """Re-fetch, re-parse and rebuild ``pgr_edgar_monthly``; return the records."""
    filings = fetcher.fetch_all_8k_filings(cutoff_date=since)
    parsed: list[dict[str, Any]] = []
    for filing in filings:
        record = fetcher.parse_filing(filing)
        if record is not None:
            parsed.append(record)
    selected = fetcher.select_monthly_releases(parsed)
    log.info(
        "%d candidate filings, %d parsed, %d months selected",
        len(filings), len(parsed), len(selected),
    )
    n_raw = db_client.record_pgr_edgar_raw(conn, selected, fetcher.PARSER_VERSION)
    conn.execute("DELETE FROM pgr_edgar_monthly")
    conn.commit()
    db_client.upsert_pgr_edgar_monthly(conn, selected)
    fetcher.recompute_derived_fields(conn)
    log.info("Recorded %d raw values under parser %s", n_raw, fetcher.PARSER_VERSION)
    return selected


def rebuild_quarterly(conn: sqlite3.Connection) -> int:
    """Rebuild ``pgr_fundamentals_quarterly`` from XBRL companyfacts."""
    facts = edgar_client.fetch_companyfacts()
    records = edgar_client.fundamentals_from_companyfacts(facts)
    return db_client.replace_pgr_fundamentals(conn, records)


def validate(conn: sqlite3.Connection) -> dict[str, int]:
    """Run the row-level validations; return the number of violations per check."""
    monthly = db_client.get_pgr_edgar_monthly(conn)
    quarterly = db_client.get_pgr_fundamentals(conn)
    results = {
        "income identity": len(validation.income_identity_violations(monthly)),
        "CR = LR + ER": len(validation.combined_ratio_violations(monthly)),
        "equity vs BVPS x shares": len(validation.equity_violations(monthly)),
        "monthly NI vs XBRL quarter": len(
            validation.quarterly_net_income_violations(monthly, quarterly)
        ),
        "missing months": len(validation.missing_months(monthly)),
        "PIF jumps > 5%": len(validation.pif_jump_violations(monthly)),
    }
    for name, count in results.items():
        log.info("validation  %-28s %d violation(s)", name, count)
    log.info(
        "validation  quarters compared to XBRL: %d",
        validation.quarters_compared(monthly, quarterly),
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="SQLite DB to repair (use a copy).")
    parser.add_argument(
        "--allow-committed-db",
        action="store_true",
        help="Allow --db to be the committed data/pgr_financials.db.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(config.DATA_RAW_DIR, "edgar_8k_cache"),
        help="EDGAR response cache (default: data/raw/edgar_8k_cache).",
    )
    parser.add_argument("--since", default=DEFAULT_SINCE, help="Earliest filing date.")
    parser.add_argument("--skip-quarterly", action="store_true")
    parser.add_argument("--diff-csv", default=None, help="Write the cell-level diff here.")
    parser.add_argument(
        "--export-csv", default=None, help="Regenerate pgr_edgar_cache.csv here."
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if not os.getenv("EDGAR_USER_AGENT"):
        log.error("Set EDGAR_USER_AGENT (name and e-mail) before calling EDGAR.")
        return 2
    target = Path(args.db).resolve()
    if target == Path(config.DB_PATH).resolve() and not args.allow_committed_db:
        log.error("Refusing to modify the committed DB; run on a copy.")
        return 2

    fetcher.set_http_cache_dir(args.cache_dir)
    conn = db_client.get_connection(str(target))
    try:
        db_client.initialize_schema(conn)
        before_monthly = _read_table(conn, "pgr_edgar_monthly", "month_end")
        before_quarterly = _read_table(conn, "pgr_fundamentals_quarterly", "period_end")

        rebuild_monthly(conn, args.since)
        if not args.skip_quarterly:
            rebuild_quarterly(conn)

        after_monthly = _read_table(conn, "pgr_edgar_monthly", "month_end")
        after_quarterly = _read_table(conn, "pgr_fundamentals_quarterly", "period_end")
        diff = pd.concat(
            [
                cell_diff(before_monthly, after_monthly, "pgr_edgar_monthly"),
                cell_diff(before_quarterly, after_quarterly, "pgr_fundamentals_quarterly"),
            ],
            ignore_index=True,
        )
        for table, group in diff.groupby("table"):
            log.info(
                "%s: %d cells changed in %d rows",
                table, len(group), group["key"].nunique(),
            )
        if args.diff_csv:
            diff.to_csv(args.diff_csv, index=False)
            log.info("Wrote cell-level diff to %s", args.diff_csv)

        failures = validate(conn)
        if args.export_csv:
            n = fetcher.export_edgar_cache_csv(conn, args.export_csv)
            log.info("Wrote %d rows to %s", n, args.export_csv)
    finally:
        conn.close()
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
