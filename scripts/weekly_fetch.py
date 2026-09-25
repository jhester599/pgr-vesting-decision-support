"""
Weekly data accumulation entrypoint for GitHub Actions.

Fetches prices for all 23 tickers (PGR + 22 ETF benchmarks) and PGR dividends
via Alpha Vantage TIME_SERIES_WEEKLY, then refreshes PGR quarterly fundamentals
from SEC EDGAR XBRL and FRED macro series (v3.0+). All results are upserted
into the v2 SQLite database.

Budget per run (Friday price run):
  22 AV calls - all ticker prices (TIME_SERIES_WEEKLY, full history)
   1 AV call  - PGR dividends (+ up to 2 advisory retries)
  23 AV total (free-tier limit: 25/day)
   1 EDGAR call - PGR companyfacts JSON (7-day cache; most runs cost 0 calls)
  N FRED calls - one per series in FRED_SERIES_MACRO + FRED_SERIES_PGR
                 (no daily limit)

Dividend refresh run (``--dividend-refresh``, Wednesday cron; review F08):
  Re-fetches DIVIDENDS for PGR and every ETF benchmark whose last dividend
  fetch is older than ``config.DIVIDEND_REFRESH_MIN_AGE_DAYS`` (weekly for
  monthly payers), oldest first, capped at the AV calls left today minus
  ``config.DIVIDEND_REFRESH_AV_RESERVE``. No prices, EDGAR or FRED calls.
  With 21 ETFs plus PGR this refreshes every ETF about once a month.

Both modes seed ``split_history`` from ``config.KNOWN_SPLITS`` and then
rebuild ``monthly_relative_returns`` -- unless a benchmark or PGR shows a
weekly close ratio outside [0.6, 1.7] with no split row nearby (a split
missing from the registry), in which case the rebuild is skipped and an error
is logged so corrupted targets are never written (review F03/F05).

Usage (local or CI):
    python scripts/weekly_fetch.py [--dry-run] [--skip-fred]
    python scripts/weekly_fetch.py --dividend-refresh [--dry-run]

Options:
    --dry-run    Log which tickers would be fetched but make no HTTP calls.
                 The database is opened read-only: no schema migrations,
                 API-log rows, split seeding or target rebuilds are written.
                 Useful for verifying budget projection before a real run.
    --skip-fred  Skip the FRED macro fetch step. Useful if FRED_API_KEY
                 is not set or during budget-constrained testing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import get_all_dividend_tickers, get_all_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader
from src.logging_config import configure_logging
from src.processing.multi_total_return import build_relative_return_targets
from src.processing.price_integrity import find_unexplained_price_jumps


logger = logging.getLogger(__name__)


def _seed_known_splits(conn) -> None:
    """Idempotently upsert the canonical split registry into split_history.

    ``build_relative_return_targets()`` reads split_history from the DB; a
    missing row turns a split-day price discontinuity into a fake return.
    ``config.KNOWN_SPLITS`` (config/splits.py) is the only split list.
    """
    n = db_client.upsert_splits(conn, config.KNOWN_SPLITS)
    if n:
        logger.info("Split history seeded: %s records upserted.", n)


def _refresh_pgr_fundamentals(conn, dry_run: bool = False) -> int:
    """Fetch PGR quarterly fundamentals from SEC EDGAR XBRL and upsert into DB."""
    from src.ingestion import edgar_client

    if dry_run:
        return 0

    db_client.log_api_request(conn, "edgar", endpoint="companyfacts")

    records = edgar_client.fetch_pgr_fundamentals_quarterly()
    if not records:
        return 0

    n = db_client.upsert_pgr_fundamentals(conn, records)
    if n:
        db_client.update_ingestion_metadata(conn, "PGR", "fundamentals", n)
    return n


def _fetch_fred_step(conn, dry_run: bool = False) -> None:
    """Fetch FRED macro and PGR-specific series into fred_macro_monthly."""
    from src.ingestion.fred_loader import (
        fetch_all_fred_macro,
        production_fred_series,
        upsert_fred_to_db,
    )

    series_list = production_fred_series()
    logger.info("Fetching %s FRED series...", len(series_list))
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


def _days_since_fetch(conn, ticker: str, today: date) -> float:
    meta = db_client.get_ingestion_metadata(conn, ticker, "dividends")
    last = (meta or {}).get("last_fetched")
    if not last:
        return float("inf")
    return float((today - date.fromisoformat(last[:10])).days)


def select_dividend_refresh_tickers(
    conn,
    tickers: list[str],
    today: date,
    max_calls: int,
    min_age_days: int = config.DIVIDEND_REFRESH_MIN_AGE_DAYS,
    min_age_days_monthly: int = config.DIVIDEND_REFRESH_MIN_AGE_DAYS_MONTHLY_PAYER,
) -> list[str]:
    """Pick the tickers whose dividends are due for a refresh, within budget.

    A ticker is due when its last dividend fetch (``ingestion_metadata``) is at
    least ``min_age_days`` old, or ``min_age_days_monthly`` for monthly payers
    (usual payment interval <= 45 days). Never-fetched tickers are always due.
    Due tickers are returned oldest-fetch first and capped at ``max_calls``.
    """
    if max_calls <= 0:
        return []
    intervals = {
        row["ticker"]: row["interval_days"]
        for row in db_client.check_dividend_freshness(conn, tickers)
    }
    due: list[tuple[float, str]] = []
    for ticker in tickers:
        age = _days_since_fetch(conn, ticker, today)
        interval = intervals.get(ticker)
        threshold = (
            min_age_days_monthly
            if interval is not None and interval <= 45
            else min_age_days
        )
        if age >= threshold:
            due.append((age, ticker))
    due.sort(key=lambda item: -item[0])
    return [ticker for _, ticker in due[:max_calls]]


def _log_dividend_freshness(conn) -> list[str]:
    """Log per-ticker dividend freshness; return the STALE tickers."""
    stale = []
    for row in db_client.check_dividend_freshness(conn):
        if row["status"] == "STALE":
            stale.append(row["ticker"])
            logger.warning(
                "Dividends STALE for %s: last ex-date %s, last price %s, "
                "usual interval %.0f days (expected an ex-date on/after %s).",
                row["ticker"], row["last_ex_date"], row["last_price_date"],
                row["interval_days"], row["due_by"],
            )
    if not stale:
        logger.info("Dividend freshness: all tickers OK.")
    return stale


def _rebuild_targets(conn, dry_run: bool) -> bool:
    """Seed splits, run the jump guard and rebuild relative-return targets.

    Returns:
        False when the rebuild was skipped because of an unexplained price jump.
    """
    # Ensure split_history is populated before computing returns; safe to run
    # every time because upsert_splits uses INSERT OR REPLACE.
    if dry_run:
        logger.info("[DRY RUN] Skipping split-history seeding.")
    else:
        _seed_known_splits(conn)

    target_tickers = ["PGR", *config.ETF_BENCHMARK_UNIVERSE]
    jumps = find_unexplained_price_jumps(conn, tickers=target_tickers)
    if not jumps.empty:
        logger.error(
            "Unexplained weekly price jumps (a split missing from "
            "config/splits.py?). Skipping the relative-return rebuild so no "
            "corrupted targets are written:\n%s",
            jumps.to_string(index=False),
        )
        return False

    logger.info("Refreshing relative return targets (6M and 12M)...")
    if dry_run:
        logger.info("[DRY RUN] Skipping relative return computation.")
        return True
    for horizon in (6, 12):
        df = build_relative_return_targets(conn, forward_months=horizon, upsert=True)
        n_rows = int(df.notna().sum().sum()) if not df.empty else 0
        logger.info(
            "%sM: %s rows written across %s benchmarks",
            horizon,
            n_rows,
            df.shape[1] if not df.empty else 0,
        )
    return True


def _log_dividend_results(div_results: dict[str, int | None]) -> None:
    total_div_rows = sum(v for v in div_results.values() if v is not None)
    div_deferred = [t for t, v in div_results.items() if v is None]
    logger.info("Dividends - %s rows upserted", total_div_rows)
    for ticker, n in div_results.items():
        if n:
            logger.info("%s: %s rows", ticker, n)
    if div_deferred:
        logger.warning(
            "%s dividend tickers not fetched (AV advisory/limit): %s",
            len(div_deferred),
            div_deferred,
        )


def run_dividend_refresh(dry_run: bool = False) -> None:
    """Budget-aware dividend refresh for PGR and the ETF benchmarks (review F08)."""
    today = date.today()
    logger.info("%sPGR dividend refresh - %s", "[DRY RUN] " if dry_run else "", today)
    if dry_run:
        conn = db_client.get_connection(config.DB_PATH, read_only=True)
    else:
        conn = db_client.get_connection(config.DB_PATH)
        db_client.initialize_schema(conn)

    utc_today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")  # api_request_log key
    av_used = db_client.get_api_request_count(conn, "av", utc_today)
    max_calls = config.AV_DAILY_LIMIT - av_used - config.DIVIDEND_REFRESH_AV_RESERVE
    due = select_dividend_refresh_tickers(
        conn, get_all_dividend_tickers(), today=today, max_calls=max_calls,
    )
    logger.info(
        "AV used today %s/%s; refreshing %s due dividend tickers: %s",
        av_used, config.AV_DAILY_LIMIT, len(due), due,
    )
    if due:
        div_results = MultiDividendLoader(conn).fetch_for_tickers(due, dry_run=dry_run)
        _log_dividend_results(div_results)
        _rebuild_targets(conn, dry_run=dry_run)
    _log_dividend_freshness(conn)
    conn.close()
    logger.info("Done.")


def main(dry_run: bool = False, skip_fred: bool = False) -> None:
    """Run the weekly production ingestion workflow."""
    configure_logging()

    today = date.today()
    logger.info("%sPGR v2 Weekly Fetch - %s", "[DRY RUN] " if dry_run else "", today)
    logger.info("Database: %s", config.DB_PATH)

    if dry_run:
        # Read-only: any accidental write raises instead of mutating the DB.
        conn = db_client.get_connection(config.DB_PATH, read_only=True)
    else:
        conn = db_client.get_connection(config.DB_PATH)
        db_client.initialize_schema(conn)

    all_tickers = get_all_price_tickers()
    pgr_only = ["PGR"]

    logger.info("Price tickers (%s): %s", len(all_tickers), all_tickers)
    logger.info(
        "Dividend tickers: %s (ETF dividends via --dividend-refresh)",
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
    _log_dividend_results(div_results)

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

    _rebuild_targets(conn, dry_run=dry_run)
    _log_dividend_freshness(conn)

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
        help="Log actions without making HTTP calls or writing to the DB.",
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip FRED macro fetch step.",
    )
    parser.add_argument(
        "--dividend-refresh",
        action="store_true",
        help=(
            "Budget-aware dividend refresh for PGR and the ETF benchmarks "
            "(no prices, EDGAR or FRED), then rebuild targets."
        ),
    )
    args = parser.parse_args()
    if args.dividend_refresh:
        configure_logging()
        run_dividend_refresh(dry_run=args.dry_run)
    else:
        main(dry_run=args.dry_run, skip_fred=args.skip_fred)
