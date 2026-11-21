"""Fetch PGR monthly 8-K operating metrics from SEC EDGAR and upsert them (entry point).

The logic lives in ``pgr_vds.ingestion.edgar_monthly`` (``fetch``, ``parse``,
``derive``, ``load``); this file only parses the command line. Needs
``pip install -e .``.

Schedule (see .github/workflows/monthly_8k_fetch.yml):
  - Primary run:  20th of each month at 14:00 UTC
  - Fallback run: 25th of each month at 14:00 UTC (covers late filers)

Both runs are idempotent. ``scripts/repair_edgar_history.py`` re-parses the
full history.

Usage:
    python cli/edgar_monthly_fetch.py [--backfill-years N] [--dry-run] [--cache-dir DIR]
    python cli/edgar_monthly_fetch.py [--backfill-years N] --export-csv PATH
    python cli/edgar_monthly_fetch.py --load-from-csv [PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging

import config
from pgr_vds.ingestion.edgar_monthly import fetch, load
from src.database import db_client
from src.logging_config import configure_logging

log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PGR monthly 8-K operating metrics from SEC EDGAR "
            "and upsert them into the local SQLite database."
        )
    )
    parser.add_argument(
        "--backfill-years",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of years back to fetch from EDGAR (default: 2).  "
            "Set to a large value (e.g. 15) for a full historical backfill "
            f"back to {load.BACKFILL_EARLIEST_DATE}.  "
            "Note: if the committed pgr_edgar_cache.csv already covers this "
            "range, use --load-from-csv instead to avoid unnecessary HTTP calls."
        ),
    )
    parser.add_argument(
        "--load-from-csv",
        metavar="PATH",
        nargs="?",
        const=load.DEFAULT_CSV_PATH,
        default=None,
        help=(
            "Seed pgr_edgar_monthly from an existing CSV file instead of "
            "fetching from EDGAR.  Defaults to data/processed/pgr_edgar_cache.csv "
            "when the flag is given without a path.  No network calls are made."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse/read data but do not write to the database. The DB is "
            "opened read-only and migrations are not applied."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        default=None,
        help=(
            "Cache EDGAR responses (filing indexes and exhibits) in DIR. "
            "Filings are immutable; the submissions index is always re-fetched."
        ),
    )
    parser.add_argument(
        "--export-csv",
        metavar="PATH",
        default=None,
        help=(
            "After the upsert, write pgr_edgar_monthly to this CSV "
            "(the committed file is data/processed/pgr_edgar_cache.csv). "
            "Cannot be combined with --dry-run."
        ),
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.export_csv:
        parser.error("--export-csv writes a file, so it cannot be used with --dry-run")
    return args


def main(argv: list[str] | None = None) -> None:
    """Fetch (or seed from the CSV) and upsert; then check staleness."""
    configure_logging()
    args = _parse_args(argv)
    if args.cache_dir:
        fetch.set_http_cache_dir(args.cache_dir)
    if args.dry_run:
        conn = db_client.get_connection(config.DB_PATH, read_only=True)
    else:
        conn = db_client.get_connection(config.DB_PATH)
        db_client.initialize_schema(conn)

    try:
        if args.load_from_csv is not None:
            n = load.load_from_csv(conn, args.load_from_csv, dry_run=args.dry_run)
        else:
            n = load.fetch_and_upsert(
                conn,
                backfill_years=args.backfill_years,
                dry_run=args.dry_run,
            )
        load.check_staleness(conn)
        log.info("Done. %d rows written.", n)
        if args.export_csv:
            exported = load.export_edgar_cache_csv(conn, args.export_csv)
            log.info("Exported %d rows to %s", exported, args.export_csv)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
