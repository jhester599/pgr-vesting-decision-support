"""
Peer ticker data fetch for v6.0 cross-asset signals.

Fetches weekly price and dividend history for the four PGR insurance-peer
tickers (ALL, TRV, CB, HIG) defined in ``config.PEER_TICKER_UNIVERSE``.

This script runs on a dedicated Sunday 04:00 UTC cron, 30 hours after the
main ``weekly_fetch.py`` cron at Friday 22:00 UTC. Running on a separate
calendar day keeps both workflows within the 25 calls/day Alpha Vantage
free-tier limit regardless of scheduler lag.

Budget per run:
  4 AV calls - peer prices (TIME_SERIES_WEEKLY, full history)
  4 AV calls - peer dividends (DIVIDENDS)
  8 AV total (free-tier limit: 25/day; 17 calls of margin)

No EDGAR, no FRED, no relative-return computation. Those are handled by the
main ``weekly_fetch.py`` run.
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
from src.ingestion.fetch_scheduler import get_peer_dividend_tickers, get_peer_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


def main(dry_run: bool = False) -> None:
    """Run the peer-price and dividend fetch workflow."""
    configure_logging()

    today = date.today()
    logger.info("%sPGR v6.0 Peer Data Fetch - %s", "[DRY RUN] " if dry_run else "", today)
    logger.info("Database: %s", config.DB_PATH)

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    peer_tickers = get_peer_price_tickers()
    peer_div_tickers = get_peer_dividend_tickers()

    av_projected = len(peer_tickers) + len(peer_div_tickers)
    logger.info("Peer price tickers (%s): %s", len(peer_tickers), peer_tickers)
    logger.info(
        "Peer dividend tickers (%s): %s",
        len(peer_div_tickers),
        peer_div_tickers,
    )
    logger.info("Projected AV calls: %s/%s", av_projected, config.AV_DAILY_LIMIT)

    logger.info("Fetching peer prices...")
    loader = MultiTickerLoader(conn)
    price_results = loader.fetch_all_prices(peer_tickers, dry_run=dry_run)
    total_price_rows = sum(v for v in price_results.values() if v is not None)
    price_deferred = [t for t, v in price_results.items() if v is None]
    logger.info("Prices - %s total rows upserted", total_price_rows)
    for ticker, n in price_results.items():
        if n:
            logger.info("%s: %s rows", ticker, n)
    if price_deferred:
        logger.warning(
            "%s price tickers deferred by AV rate limit: %s",
            len(price_deferred),
            price_deferred,
        )

    logger.info("Fetching peer dividends...")
    div_loader = MultiDividendLoader(conn)
    div_results = div_loader.fetch_for_tickers(peer_div_tickers, dry_run=dry_run)
    total_div_rows = sum(v for v in div_results.values() if v is not None)
    div_deferred = [t for t, v in div_results.items() if v is None]
    logger.info("Dividends - %s total rows upserted", total_div_rows)
    for ticker, n in div_results.items():
        if n:
            logger.info("%s: %s rows", ticker, n)
    if div_deferred:
        logger.warning(
            "%s dividend tickers deferred by AV rate limit: %s",
            len(div_deferred),
            div_deferred,
        )

    today_str = today.isoformat()
    if dry_run:
        logger.info("[DRY RUN] Projected API calls: AV %s/%s", av_projected, config.AV_DAILY_LIMIT)
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        logger.info("API budget used today: AV %s/%s", av_used, config.AV_DAILY_LIMIT)

    conn.close()
    logger.info("Done.")


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
