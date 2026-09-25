"""
Read-only integrity report for prices, splits and dividends (review F03/F05/F08/F22).

Checks:
  * weekly close ratios outside [0.6, 1.7] with no ``split_history`` row
    within 7 days (a split missing from ``config/splits.py``);
  * more than one price bar per ticker per ISO week;
  * per-ticker dividend freshness (``db_client.check_dividend_freshness``).

Prints a Markdown summary (suitable for ``$GITHUB_STEP_SUMMARY``) and exits
1 when a price check fails, or when dividends are stale and
``--fail-on-stale-dividends`` is given. Opens the DB read-only.

Usage:
    python scripts/check_data_integrity.py [--fail-on-stale-dividends]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.processing.price_integrity import (
    find_duplicate_week_bars,
    find_unexplained_price_jumps,
)


def build_report(conn) -> tuple[list[str], bool, bool]:
    """Return (markdown lines, price_problem, stale_dividends)."""
    lines = ["## Data integrity", ""]
    jumps = find_unexplained_price_jumps(conn)
    dupes = find_duplicate_week_bars(conn)
    freshness = db_client.check_dividend_freshness(conn)
    stale = [r for r in freshness if r["status"] == "STALE"]

    if jumps.empty:
        lines.append("- Price jumps: OK (every jump outside [0.6, 1.7] has a split row)")
    else:
        lines.append(
            f"- **Price jumps: {len(jumps)} unexplained** "
            "(add the split to `config/splits.py`, then run "
            "`python scripts/rebuild_relative_returns.py`)"
        )
        for row in jumps.itertuples(index=False):
            lines.append(
                f"  - {row.ticker} {row.prev_date} -> {row.date}: "
                f"{row.prev_close:g} -> {row.close:g} ({row.ratio:.3f}x)"
            )
    if dupes.empty:
        lines.append("- One bar per ticker-week: OK")
    else:
        lines.append(f"- **Duplicate ticker-week bars: {len(dupes)}**")
    if stale:
        lines.append(f"- **Dividends stale for {len(stale)} tickers**")
        for r in stale:
            lines.append(
                f"  - {r['ticker']}: last ex-date {r['last_ex_date']}, "
                f"expected one on/after {r['due_by']} "
                f"(interval {r['interval_days']:.0f} d, last price {r['last_price_date']})"
            )
    else:
        lines.append("- Dividend freshness: OK")
    return lines, not (jumps.empty and dupes.empty), bool(stale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--fail-on-stale-dividends",
        action="store_true",
        help="Exit 1 when any ticker's dividends are stale.",
    )
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite DB path.")
    args = parser.parse_args(argv)

    conn = db_client.get_connection(args.db, read_only=True)
    try:
        lines, price_problem, stale = build_report(conn)
    finally:
        conn.close()
    print("\n".join(lines))
    if price_problem:
        return 1
    if stale and args.fail_on_stale_dividends:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
