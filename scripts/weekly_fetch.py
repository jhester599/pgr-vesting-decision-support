"""
Weekly data accumulation entrypoint for GitHub Actions.

Fetches prices for all 23 tickers (PGR + 22 ETF benchmarks) and PGR dividends
via Alpha Vantage TIME_SERIES_WEEKLY, then refreshes PGR quarterly fundamentals
from SEC EDGAR XBRL and FRED macro series (v3.0+). All results are upserted
into the v2 SQLite database.

Budget per run:
  23 AV calls - all ticker prices (TIME_SERIES_WEEKLY, full history)
   1 AV call  - PGR dividends
  24 AV total (free-tier limit: 25/day)
   1 EDGAR call - PGR companyfacts JSON (7-day cache; most runs cost 0 calls)
  N FRED calls - one per series in FRED_SERIES_MACRO (no daily limit)

ETF dividends are NOT fetched here; use scripts/initial_fetch.py for the
one-time bootstrap and for quarterly ETF dividend refreshes.

Usage (local or CI):
    python scripts/weekly_fetch.py [--dry-run] [--skip-fred]

Options:
    --dry-run    Log which tickers would be fetched but make no HTTP calls.
                 Useful for verifying budget projection before a real run.
    --skip-fred  Skip the FRED macro fetch step. Useful if FRED_API_KEY
                 is not set or during budget-constrained testing.
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
from src.ingestion.fetch_scheduler import get_all_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader
from src.logging_config import configure_logging
from src.processing.multi_total_return import build_relative_return_targets


logger = logging.getLogger(__name__)

# Known stock splits for every ticker stored in daily_prices.
# build_relative_return_targets() reads split_history from the DB; an empty
# table causes split-date price discontinuities to appear as real returns.
# These records are idempotent (INSERT OR REPLACE) so re-seeding on every
# run is safe and cheap.
_KNOWN_SPLITS: list[dict] = [
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


def _seed_known_splits(conn) -> None:
    """Idempotently upsert all known stock splits into split_history."""
    n = db_client.upsert_splits(conn, _KNOWN_SPLITS)
    if n:
        logger.info("Split history seeded: %s records upserted.", n)


def _refresh_pgr_fundamentals(conn, dry_run: bool = False) -> int:
    """Fetch PGR quarterly fundamentals from SEC EDGAR XBRL and upsert into DB."""
    from src.ingestion import edgar_client

    db_client.log_api_request(conn, "edgar", endpoint="companyfacts")

    if dry_run:
        return 0

    records = edgar_client.fetch_pgr_fundamentals_quarterly()
    if not records:
        return 0

    n = db_client.upsert_pgr_fundamentals(conn, records)
    if n:
        db_client.update_ingestion_metadata(conn, "PGR", "fundamentals", n)
    return n


def _fetch_fred_step(conn, dry_run: bool = False) -> None:
    """Fetch FRED macro series and upsert into fred_macro_monthly."""
    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    series_list = config.FRED_SERIES_MACRO
    logger.info("Fetching %s FRED macro series...", len(series_list))
    if dry_run:
        logger.info("[DRY RUN] Would fetch: %s", series_list)
        return

    if config.FRED_API_KEY is None:
        logger.warning("FRED_API_KEY not set. Skipping FRED fetch.")
        return

    try:
        df = fetch_all_fred_macro(series_list)
        n = upsert_fred_to_db(conn, df)
        logger.info("FRED macro: %s rows upserted (%s series)", n, len(series_list))
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "FRED fetch failed. Continuing with cached data. Error=%r",
            exc,
        )


def main(dry_run: bool = False, skip_fred: bool = False) -> None:
    """Run the weekly production ingestion workflow."""
    configure_logging()

    today = date.today()
    logger.info("%sPGR v2 Weekly Fetch - %s", "[DRY RUN] " if dry_run else "", today)
    logger.info("Database: %s", config.DB_PATH)

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    all_tickers = get_all_price_tickers()
    pgr_only = ["PGR"]

    logger.info("Price tickers (%s): %s", len(all_tickers), all_tickers)
    logger.info(
        "Dividend tickers: %s (ETF dividends via initial_fetch.py)",
        pgr_only,
    )
    logger.info("PGR fundamentals refresh: True (SEC EDGAR XBRL, always on weekly run)")

    loader = MultiTickerLoader(conn)
    price_results = loader.fetch_all_prices(all_tickers, dry_run=dry_run)
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

    div_loader = MultiDividendLoader(conn)
    div_results = div_loader.fetch_for_tickers(pgr_only, dry_run=dry_run)
    total_div_rows = sum(v for v in div_results.values() if v is not None)
    div_deferred = [t for t, v in div_results.items() if v is None]
    logger.info("Dividends - %s rows upserted", total_div_rows)
    for ticker, n in div_results.items():
        if n:
            logger.info("%s: %s rows", ticker, n)
    if div_deferred:
        logger.warning(
            "%s dividend tickers deferred by AV rate limit: %s",
            len(div_deferred),
            div_deferred,
        )

    logger.info("Refreshing PGR quarterly fundamentals from SEC EDGAR XBRL...")
    try:
        n = _refresh_pgr_fundamentals(conn, dry_run=dry_run)
        logger.info("PGR fundamentals: %s rows upserted", n)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "EDGAR fundamentals fetch failed. Continuing with previously cached data. Error=%r",
            exc,
        )

    if not skip_fred:
        _fetch_fred_step(conn, dry_run=dry_run)
    else:
        logger.info("Skipping FRED fetch (--skip-fred).")

    # Ensure split_history is populated before computing returns; safe to run
    # every time because upsert_splits uses INSERT OR REPLACE.
    _seed_known_splits(conn)

    logger.info("Refreshing relative return targets (6M and 12M)...")
    if dry_run:
        logger.info("[DRY RUN] Skipping relative return computation.")
    else:
        for horizon in (6, 12):
            df = build_relative_return_targets(conn, forward_months=horizon, upsert=True)
            n_rows = df.shape[0] * df.shape[1] if not df.empty else 0
            logger.info(
                "%sM: %s rows upserted across %s benchmarks",
                horizon,
                n_rows,
                df.shape[1] if not df.empty else 0,
            )

    today_str = today.isoformat()
    if dry_run:
        av_projected = len(all_tickers) + len(pgr_only)
        logger.info(
            "[DRY RUN] Projected API calls: AV %s/%s EDGAR: 0-1 (7-day cache, no hard limit)",
            av_projected,
            config.AV_DAILY_LIMIT,
        )
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        logger.info(
            "API budget used today: AV %s/%s EDGAR: free (no daily limit)",
            av_used,
            config.AV_DAILY_LIMIT,
        )

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGR v3.0 weekly data accumulation.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without making HTTP calls.",
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip FRED macro fetch step.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, skip_fred=args.skip_fred)
