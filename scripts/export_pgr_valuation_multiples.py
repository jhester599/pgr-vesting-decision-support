"""
Export PGR's monthly price-to-book and trailing price-to-earnings history.

Reads unadjusted PGR prices, the monthly 8-K EDGAR data, and the split
history from the SQLite database, then writes one row per calendar month to
``data/processed/pgr_valuation_monthly.csv``.  See
``src/processing/valuation_multiples.py`` for the calculation rules.

Usage:
    python scripts/export_pgr_valuation_multiples.py
    python scripts/export_pgr_valuation_multiples.py --output path/to/file.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.processing.valuation_multiples import build_monthly_valuation_multiples

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = os.path.join(config.DATA_PROCESSED_DIR, "pgr_valuation_monthly.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--db-path", default=config.DB_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    conn = db_client.get_connection(args.db_path)
    try:
        prices = db_client.get_prices(conn, "PGR", exclude_proxy=True)
        edgar = db_client.get_pgr_edgar_monthly(conn)
        splits = db_client.get_splits(conn, "PGR")
    finally:
        conn.close()

    table = build_monthly_valuation_multiples(prices, edgar, splits)
    table.to_csv(args.output, index=False, float_format="%.6f")

    logger.info(
        "Wrote %d months (%s to %s) to %s; P/B rows=%d, P/E rows=%d",
        len(table),
        table["month_end"].iloc[0],
        table["month_end"].iloc[-1],
        args.output,
        table["pb_ratio"].notna().sum(),
        table["pe_ratio"].notna().sum(),
    )


if __name__ == "__main__":
    main()
