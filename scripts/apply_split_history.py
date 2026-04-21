"""
One-time migration: populate split_history table and recompute relative returns.

Root cause of the bug
─────────────────────
The split_history SQLite table was left empty after the v2 schema was
initialised.  multi_total_return.build_relative_return_targets() reads
splits from this table; with no rows, the DRIP position calculation for
PGR applied no split adjustments.  Any 6M or 12M forward-return window
that SPANNED a split date produced a large spurious negative return (e.g.
the 4-for-1 split on 2006-05-19 caused windows ending in May 2006 to show
~ −75% returns instead of the actual ~0% price change on split day).

Similarly, KIE (SPDR S&P Insurance ETF, benchmark ticker) had a 3-for-1
split on 2017-12-01.  Windows ending in December 2017 showed ~−66% KIE
benchmark returns, inflating PGR relative returns to ~+90% for those months.

Fix applied
───────────
  1. Upsert known splits for PGR and KIE into split_history.
  2. Re-run build_relative_return_targets() for both horizons (6M, 12M),
     which overwrites the corrupted rows via INSERT OR REPLACE.

Run this script once after a fresh DB initialisation if splits are absent:
    python scripts/apply_split_history.py
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.processing.multi_total_return import build_relative_return_targets

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known splits for all tickers tracked in daily_prices
# ---------------------------------------------------------------------------

KNOWN_SPLITS: list[dict] = [
    # PGR — Progressive Corporation
    {"ticker": "PGR",   "split_date": "1992-12-09", "split_ratio": 3.0, "numerator": 3.0, "denominator": 1.0},
    {"ticker": "PGR",   "split_date": "2002-04-23", "split_ratio": 3.0, "numerator": 3.0, "denominator": 1.0},
    {"ticker": "PGR",   "split_date": "2006-05-19", "split_ratio": 4.0, "numerator": 4.0, "denominator": 1.0},
    # VTI — Vanguard Total Stock Market ETF (benchmark)
    {"ticker": "VTI",   "split_date": "2008-06-20", "split_ratio": 2.0, "numerator": 2.0, "denominator": 1.0},
    # VWO — Vanguard FTSE Emerging Markets ETF (benchmark)
    {"ticker": "VWO",   "split_date": "2008-06-20", "split_ratio": 2.0, "numerator": 2.0, "denominator": 1.0},
    # SCHD — Schwab US Dividend Equity ETF (benchmark)
    {"ticker": "SCHD",  "split_date": "2024-10-11", "split_ratio": 3.0, "numerator": 3.0, "denominator": 1.0},
    # KIE — SPDR S&P Insurance ETF (benchmark)
    {"ticker": "KIE",   "split_date": "2017-12-01", "split_ratio": 3.0, "numerator": 3.0, "denominator": 1.0},
    # CB — Chubb Ltd (peer; in daily_prices but not a model benchmark)
    {"ticker": "CB",    "split_date": "2006-04-21", "split_ratio": 2.0, "numerator": 2.0, "denominator": 1.0},
    # FZROX — Fidelity ZERO Total Market (pre-2018 rows are VTI proxy; split matches VTI)
    {"ticker": "FZROX", "split_date": "2008-06-20", "split_ratio": 2.0, "numerator": 2.0, "denominator": 1.0},
]


def main() -> None:
    conn = db_client.get_connection(config.DB_PATH)

    # ── 1. Check current state ─────────────────────────────────────────────
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM split_history")
    existing = cur.fetchone()[0]
    logger.info("split_history rows before migration: %s", existing)

    # ── 2. Upsert known splits ─────────────────────────────────────────────
    n = db_client.upsert_splits(conn, KNOWN_SPLITS)
    logger.info("Upserted %s split records.", n)

    cur.execute("SELECT ticker, split_date, split_ratio FROM split_history ORDER BY ticker, split_date")
    for row in cur.fetchall():
        logger.info("  split_history: %s  %s  ratio=%.1f", *row)

    # ── 3. Recompute relative returns for both horizons ───────────────────
    for horizon in (6, 12):
        logger.info("Recomputing relative returns — %sM horizon...", horizon)
        df = build_relative_return_targets(conn, forward_months=horizon, upsert=True)
        logger.info("  %sM: %s month-end dates × %s benchmarks", horizon, df.shape[0], df.shape[1])

    # ── 4. Spot-check: PGR returns around split months should be near zero ─
    logger.info("Spot-check: PGR returns at split months (expect ~0%, not −66%/−75%):")
    for check_date, label in [
        ("2002-04-30", "3-for-1 2002-04-23"),
        ("2002-05-31", "month after 2002 split"),
        ("2006-05-31", "4-for-1 2006-05-19"),
        ("2006-06-30", "month after 2006 split"),
    ]:
        cur.execute(
            "SELECT pgr_return FROM monthly_relative_returns "
            "WHERE date=? AND benchmark='VTI' AND target_horizon=6",
            (check_date,),
        )
        row = cur.fetchone()
        val = f"{row[0]:+.1%}" if row else "N/A"
        logger.info("  %s (%s): pgr_return(6M) = %s", check_date, label, val)

    logger.info("Spot-check: KIE returns around 2017-12 split:")
    for check_date in ["2017-08-31", "2017-11-30", "2017-12-29", "2018-01-31"]:
        cur.execute(
            "SELECT benchmark_return, relative_return FROM monthly_relative_returns "
            "WHERE date=? AND benchmark='KIE' AND target_horizon=6",
            (check_date,),
        )
        row = cur.fetchone()
        if row:
            logger.info("  %s: KIE 6M return=%.1f%%, relative=%.1f%%", check_date, row[0] * 100, row[1] * 100)
        else:
            logger.info("  %s: N/A", check_date)

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
