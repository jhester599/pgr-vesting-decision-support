"""
Peer ticker data fetch for v6.0 cross-asset signals.

Fetches weekly price and dividend history for the four PGR insurance-peer
tickers (ALL, TRV, CB, HIG) defined in ``config.PEER_TICKER_UNIVERSE``.

This script runs on a dedicated Sunday 04:00 UTC cron — exactly 30 hours
after the main ``weekly_fetch.py`` cron fires at Friday 22:00 UTC.  Running
on a separate calendar day guarantees both fetches stay well within the
25 calls/day Alpha Vantage free-tier limit regardless of GitHub Actions
scheduler lag.

Budget per run:
  4 AV calls — peer prices (TIME_SERIES_WEEKLY, full history)
  4 AV calls — peer dividends (DIVIDENDS)
  ─────────────
  8 AV total  (free-tier limit: 25/day; 17 calls of margin)

No EDGAR, no FRED, no relative-return computation — those are handled by
the main ``weekly_fetch.py`` run.  The peer price and dividend data will
be consumed by v6.0 feature engineering:

  - ``pgr_vs_peers_6m``:  PGR 6M DRIP total return minus equal-weight
    peer composite 6M DRIP total return.
  - Residual momentum baseline (Blitz et al. 2011): Fama-French 3-factor
    neutral returns use peer betas estimated from this price history.

Why NOT yfinance:
  yfinance scrapes Yahoo Finance's undocumented internal endpoints and
  carries no API contract, no SLA, and a history of silent breaks.  Using
  Alpha Vantage keeps the entire price/dividend data stack on a single
  source with consistent split/dividend handling and a known call budget.

Usage (local or CI):
    python scripts/peer_fetch.py [--dry-run]

Options:
    --dry-run    Log which tickers would be fetched but make no HTTP calls.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import get_peer_dividend_tickers, get_peer_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader


def main(dry_run: bool = False) -> None:
    today = date.today()
    print(f"{'[DRY RUN] ' if dry_run else ''}PGR v6.0 Peer Data Fetch — {today}")
    print(f"Database: {config.DB_PATH}")

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    peer_tickers = get_peer_price_tickers()
    peer_div_tickers = get_peer_dividend_tickers()

    av_projected = len(peer_tickers) + len(peer_div_tickers)
    print(f"\nPeer price tickers  ({len(peer_tickers)}): {peer_tickers}")
    print(f"Peer dividend tickers ({len(peer_div_tickers)}): {peer_div_tickers}")
    print(f"Projected AV calls: {av_projected}/{config.AV_DAILY_LIMIT}")

    # --- Price fetch (4 AV calls) ---
    print("\nFetching peer prices...")
    loader = MultiTickerLoader(conn)
    price_results = loader.fetch_all_prices(peer_tickers, dry_run=dry_run)
    total_price_rows = sum(v for v in price_results.values() if v is not None)
    price_deferred = [t for t, v in price_results.items() if v is None]
    print(f"  Prices — {total_price_rows} total rows upserted")
    for ticker, n in price_results.items():
        if n:
            print(f"    {ticker}: {n} rows")
    if price_deferred:
        print(f"  WARNING: {len(price_deferred)} price tickers deferred by AV rate limit: "
              f"{price_deferred}")

    # --- Dividend fetch (4 AV calls) ---
    print("\nFetching peer dividends...")
    div_loader = MultiDividendLoader(conn)
    div_results = div_loader.fetch_for_tickers(peer_div_tickers, dry_run=dry_run)
    total_div_rows = sum(v for v in div_results.values() if v is not None)
    div_deferred = [t for t, v in div_results.items() if v is None]
    print(f"  Dividends — {total_div_rows} total rows upserted")
    for ticker, n in div_results.items():
        if n:
            print(f"    {ticker}: {n} rows")
    if div_deferred:
        print(f"  WARNING: {len(div_deferred)} dividend tickers deferred by AV rate limit: "
              f"{div_deferred}")

    # --- Budget summary ---
    today_str = today.isoformat()
    if dry_run:
        print(f"\n[DRY RUN] Projected API calls: AV {av_projected}/{config.AV_DAILY_LIMIT}")
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        print(f"\nAPI budget used today: AV {av_used}/{config.AV_DAILY_LIMIT}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR v6.0 peer ticker weekly data fetch (ALL, TRV, CB, HIG)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without making HTTP calls or writing to the database.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
