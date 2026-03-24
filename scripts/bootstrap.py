"""
Post-initial-fetch bootstrap script.

Runs after the two-day initial data population is complete (prices Day 1,
dividends Day 2) to build the derived data the monthly decision engine needs.
Also generates the first monthly decision report.

Steps
-----
1. Build relative return targets (6M and 12M horizons) for all 20 ETF
   benchmarks → populates ``monthly_relative_returns`` table.
   This is the critical step: ``monthly_decision.py`` reads from this table
   when training WFO models and generating signals.  Without it the decision
   engine has nothing to train on.

2. Run ``monthly_decision.py`` to produce the first recommendation report
   in ``results/monthly_decisions/YYYY-MM/``.

Usage (local or CI):
    python scripts/bootstrap.py [--dry-run] [--skip-decision] [--as-of YYYY-MM-DD]

Options:
    --dry-run        Log actions without making HTTP calls or writing files.
    --skip-decision  Build relative returns only; skip the monthly decision run.
    --as-of          Override the as-of date for the monthly decision (YYYY-MM-DD).
                     Defaults to today.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.processing.multi_total_return import build_relative_return_targets


# ---------------------------------------------------------------------------
# Relative return bootstrap
# ---------------------------------------------------------------------------

def _build_relative_returns(conn, dry_run: bool = False) -> dict[int, int]:
    """Build relative return targets for 6M and 12M horizons.

    Returns:
        Dict mapping horizon (6, 12) to number of rows upserted.
    """
    results: dict[int, int] = {}
    for horizon in (6, 12):
        print(f"\n  [{horizon}M] Computing PGR-vs-ETF relative returns "
              f"for {len(config.ETF_BENCHMARK_UNIVERSE)} benchmarks...")
        if dry_run:
            print(f"  [{horizon}M] [DRY RUN] Skipping — no DB writes.")
            results[horizon] = 0
            continue

        df = build_relative_return_targets(conn, forward_months=horizon, upsert=True)
        n_rows = df.shape[0] * df.shape[1] if not df.empty else 0
        n_benchmarks = df.shape[1] if not df.empty else 0
        print(f"  [{horizon}M] Done — {n_rows} rows across {n_benchmarks} benchmarks "
              f"({df.index.min().date() if not df.empty else 'n/a'} → "
              f"{df.index.max().date() if not df.empty else 'n/a'})")
        results[horizon] = n_rows

    return results


# ---------------------------------------------------------------------------
# Monthly decision
# ---------------------------------------------------------------------------

def _run_monthly_decision(
    as_of: str | None,
    dry_run: bool = False,
    skip_fred: bool = True,
) -> int:
    """Invoke monthly_decision.main() and return its exit code.

    skip_fred defaults to True because FRED data is pre-populated into the DB
    before the bootstrap runs (see scripts/weekly_fetch.py or the local FRED
    fetch done during initial setup on 2026-03-24).  The weekly_data_fetch
    GitHub Action keeps FRED current on an ongoing basis.
    """
    from scripts import monthly_decision  # noqa: PLC0415 (local import)
    try:
        monthly_decision.main(
            as_of_date_str=as_of,
            dry_run=dry_run,
            skip_fred=skip_fred,
        )
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: monthly_decision raised {exc}")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dry_run: bool = False,
    skip_decision: bool = False,
    as_of: str | None = None,
    skip_fred: bool = True,
) -> int:
    today = date.today()
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}PGR Bootstrap — {today}")
    print(f"Database: {config.DB_PATH}")

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    # ------------------------------------------------------------------
    # Step 1: Verify DB has the minimum data needed
    # ------------------------------------------------------------------
    n_prices = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    n_divs   = conn.execute("SELECT COUNT(*) FROM daily_dividends").fetchone()[0]
    print(f"\nDB snapshot: {n_prices:,} price rows, {n_divs:,} dividend rows")

    if n_prices == 0:
        print("ERROR: daily_prices is empty — run initial_fetch.py --prices first.")
        conn.close()
        return 1
    if n_divs == 0:
        print("ERROR: daily_dividends is empty — run initial_fetch.py --dividends first.")
        conn.close()
        return 1

    # ------------------------------------------------------------------
    # Step 2: Build relative return targets (6M and 12M)
    # ------------------------------------------------------------------
    print("\n=== Step 1/2: Building relative return targets ===")
    rr_results = _build_relative_returns(conn, dry_run=dry_run)
    total_rr_rows = sum(rr_results.values())
    print(f"\n  Relative returns: {total_rr_rows:,} total rows upserted "
          f"(6M: {rr_results[6]:,}, 12M: {rr_results[12]:,})")

    conn.close()

    # ------------------------------------------------------------------
    # Step 3: First monthly decision report
    # ------------------------------------------------------------------
    if skip_decision:
        print("\nSkipping monthly decision (--skip-decision).")
        print("\nDone.")
        return 0

    print("\n=== Step 2/2: Generating first monthly decision report ===")
    rc = _run_monthly_decision(as_of, dry_run=dry_run, skip_fred=skip_fred)
    if rc != 0:
        print(f"\nWARNING: monthly_decision exited with code {rc}.")

    print("\nDone.")
    return rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR bootstrap: build relative returns + first monthly decision."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without making HTTP calls or writing files.")
    parser.add_argument("--skip-decision", action="store_true",
                        help="Build relative returns only; skip the monthly decision run.")
    parser.add_argument("--as-of", metavar="YYYY-MM-DD", default=None,
                        help="Override as-of date for the monthly decision.")
    parser.add_argument("--fetch-fred", action="store_true",
                        help="Fetch live FRED data before monthly decision "
                             "(default: use cached DB data; FRED is kept current "
                             "by weekly_data_fetch.yml).")
    args = parser.parse_args()
    sys.exit(main(
        dry_run=args.dry_run,
        skip_decision=args.skip_decision,
        as_of=args.as_of,
        skip_fred=not args.fetch_fred,
    ))
