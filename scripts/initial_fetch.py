"""
One-time (or quarterly) full data population script.

Fetches complete price AND dividend history for all 23 tickers
(PGR + 22 ETF benchmarks) from Alpha Vantage.  Because the free tier
allows only 25 calls/day, the two fetches must be run on separate days:

    Day 1:  python scripts/initial_fetch.py --prices
    Day 2:  python scripts/initial_fetch.py --dividends

After both days the database will have complete historical data back to
~2000 for all tickers that AV covers (proxy backfill via fill_proxy_history
is a separate post-processing step).

Usage:
    python scripts/initial_fetch.py --prices     [--dry-run]
    python scripts/initial_fetch.py --dividends  [--dry-run]
    python scripts/initial_fetch.py --prices --dividends   # NOT recommended: exceeds 25 call/day

Options:
    --prices     Fetch full weekly price history for all 23 tickers (23 AV calls).
    --dividends  Fetch full dividend history for all 23 tickers (23 AV calls).
    --dry-run    Log which tickers would be fetched but make no HTTP calls.
    --force      Re-fetch even if data was already fetched today.

Typical re-use:
    Run --dividends quarterly to capture new ETF dividend events.
    Re-run --prices at any time; the upsert is idempotent.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import get_all_dividend_tickers, get_all_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader


def main(
    do_prices: bool = False,
    do_dividends: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    if not do_prices and not do_dividends:
        print("Error: specify --prices and/or --dividends.")
        sys.exit(1)

    today = date.today()
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}PGR v2 Initial Fetch — {today}")
    print(f"Database: {config.DB_PATH}")

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    all_tickers = get_all_price_tickers()   # ["PGR"] + 22 ETFs

    # --- Price fetch (23 AV calls) ---
    if do_prices:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Fetching prices for {len(all_tickers)} tickers...")
        loader = MultiTickerLoader(conn)
        price_results = loader.fetch_all_prices(
            all_tickers,
            dry_run=dry_run,
        )
        total_price_rows = sum(price_results.values())
        print(f"Prices — {total_price_rows} total rows upserted")
        for ticker, n in price_results.items():
            if n:
                print(f"  {ticker}: {n} rows")
            else:
                print(f"  {ticker}: skipped (already fresh; use --force to override)")

    # --- Dividend fetch (23 AV calls) ---
    if do_dividends:
        div_tickers = get_all_dividend_tickers()  # same 23 tickers
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Fetching dividends for {len(div_tickers)} tickers...")
        div_loader = MultiDividendLoader(conn)
        div_results = div_loader.fetch_for_tickers(
            div_tickers,
            dry_run=dry_run,
        )
        total_div_rows = sum(div_results.values())
        print(f"Dividends — {total_div_rows} total rows upserted")
        for ticker, n in div_results.items():
            if n:
                print(f"  {ticker}: {n} rows")
            else:
                print(f"  {ticker}: skipped (already fresh; use --force to override)")

    # --- Budget summary ---
    today_str = today.isoformat()
    if dry_run:
        av_projected = (len(all_tickers) if do_prices else 0) + \
                       (len(get_all_dividend_tickers()) if do_dividends else 0)
        print(f"\n[DRY RUN] Projected AV calls: {av_projected}/{config.AV_DAILY_LIMIT}")
        if av_projected > config.AV_DAILY_LIMIT:
            print("  WARNING: projected calls exceed daily limit — run --prices and "
                  "--dividends on separate days.")
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        print(f"\nAV budget used today: {av_used}/{config.AV_DAILY_LIMIT}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR v2 one-time full data population (run --prices and "
                    "--dividends on separate days to stay within AV limits)."
    )
    parser.add_argument("--prices", action="store_true",
                        help="Fetch full price history for all 23 tickers.")
    parser.add_argument("--dividends", action="store_true",
                        help="Fetch full dividend history for all 23 tickers.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without making HTTP calls.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if data was already fetched today.")
    args = parser.parse_args()
    main(
        do_prices=args.prices,
        do_dividends=args.dividends,
        dry_run=args.dry_run,
        force=args.force,
    )
