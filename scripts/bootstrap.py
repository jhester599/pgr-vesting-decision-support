"""
Post-initial-fetch bootstrap script.

Runs after the two-day initial data population is complete to build derived
relative-return targets and optionally generate the first monthly decision
report.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.logging_config import configure_logging
from src.processing.multi_total_return import build_relative_return_targets


logger = logging.getLogger(__name__)


def _build_relative_returns(conn, dry_run: bool = False) -> dict[int, int]:
    """Build relative return targets for 6M and 12M horizons."""
    results: dict[int, int] = {}
    for horizon in (6, 12):
        logger.info(
            "[%sM] Computing PGR-vs-ETF relative returns for %s benchmarks...",
            horizon,
            len(config.ETF_BENCHMARK_UNIVERSE),
        )
        if dry_run:
            logger.info("[%sM] [DRY RUN] Skipping - no DB writes.", horizon)
            results[horizon] = 0
            continue

        df = build_relative_return_targets(conn, forward_months=horizon, upsert=True)
        n_rows = df.shape[0] * df.shape[1] if not df.empty else 0
        n_benchmarks = df.shape[1] if not df.empty else 0
        logger.info(
            "[%sM] Done - %s rows across %s benchmarks (%s -> %s)",
            horizon,
            n_rows,
            n_benchmarks,
            df.index.min().date() if not df.empty else "n/a",
            df.index.max().date() if not df.empty else "n/a",
        )
        results[horizon] = n_rows

    return results


def _run_monthly_decision(
    as_of: str | None,
    dry_run: bool = False,
    skip_fred: bool = True,
) -> int:
    """Invoke monthly_decision.main() and return its exit code."""
    from scripts import monthly_decision

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
        logger.exception("monthly_decision raised an unexpected error. Error=%r", exc)
        return 1


def main(
    dry_run: bool = False,
    skip_decision: bool = False,
    as_of: str | None = None,
    skip_fred: bool = True,
) -> int:
    """Build relative returns and optionally run the first monthly decision."""
    configure_logging()

    today = date.today()
    prefix = "[DRY RUN] " if dry_run else ""
    logger.info("%sPGR Bootstrap - %s", prefix, today)
    logger.info("Database: %s", config.DB_PATH)

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)
    db_client.warn_if_db_behind(conn, context="bootstrap")

    n_prices = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    n_divs = conn.execute("SELECT COUNT(*) FROM daily_dividends").fetchone()[0]
    logger.info("DB snapshot: %s price rows, %s dividend rows", f"{n_prices:,}", f"{n_divs:,}")

    if n_prices == 0:
        logger.error("daily_prices is empty - run initial_fetch.py --prices first.")
        conn.close()
        return 1
    if n_divs == 0:
        logger.error("daily_dividends is empty - run initial_fetch.py --dividends first.")
        conn.close()
        return 1

    logger.info("=== Step 1/2: Building relative return targets ===")
    rr_results = _build_relative_returns(conn, dry_run=dry_run)
    total_rr_rows = sum(rr_results.values())
    logger.info(
        "Relative returns: %s total rows upserted (6M: %s, 12M: %s)",
        f"{total_rr_rows:,}",
        f"{rr_results[6]:,}",
        f"{rr_results[12]:,}",
    )

    conn.close()

    if skip_decision:
        logger.info("Skipping monthly decision (--skip-decision).")
        logger.info("Done.")
        return 0

    logger.info("=== Step 2/2: Generating first monthly decision report ===")
    rc = _run_monthly_decision(as_of, dry_run=dry_run, skip_fred=skip_fred)
    if rc != 0:
        logger.warning("monthly_decision exited with code %s.", rc)

    logger.info("Done.")
    return rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR bootstrap: build relative returns + first monthly decision."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without making HTTP calls or writing files.",
    )
    parser.add_argument(
        "--skip-decision",
        action="store_true",
        help="Build relative returns only; skip the monthly decision run.",
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        default=None,
        help="Override as-of date for the monthly decision.",
    )
    parser.add_argument(
        "--fetch-fred",
        action="store_true",
        help="Fetch live FRED data before monthly decision "
        "(default: use cached DB data; FRED is kept current "
        "by weekly_data_fetch.yml).",
    )
    args = parser.parse_args()
    sys.exit(
        main(
            dry_run=args.dry_run,
            skip_decision=args.skip_decision,
            as_of=args.as_of,
            skip_fred=not args.fetch_fred,
        )
    )
