"""
Weekly data accumulation entrypoint for GitHub Actions.

Fetches prices for all 23 tickers (PGR + 22 ETF benchmarks) and PGR dividends
via Alpha Vantage TIME_SERIES_WEEKLY, then refreshes PGR quarterly fundamentals
from SEC EDGAR XBRL and FRED macro series (v3.0+).  All results are upserted
into the v2 SQLite database.

Budget per run:
  23 AV calls  — all ticker prices (TIME_SERIES_WEEKLY, full history)
   1 AV call   — PGR dividends
  ─────────────
  24 AV total  (free-tier limit: 25/day)
   1 EDGAR call — PGR companyfacts JSON (7-day cache; most runs cost 0 calls)
  ─────────────
  EDGAR is a free public API; no daily limit, no API key required.
  N FRED calls — one per series in FRED_SERIES_MACRO (no daily limit)
  ─────────────
  FRED is a free public API; calls do not count against AV budget.

ETF dividends are NOT fetched here; use scripts/initial_fetch.py for the
one-time bootstrap and for quarterly ETF dividend refreshes.

Usage (local or CI):
    python scripts/weekly_fetch.py [--dry-run] [--skip-fred]

Options:
    --dry-run    Log which tickers would be fetched but make no HTTP calls.
                 Useful for verifying budget projection before a real run.
    --skip-fred  Skip the FRED macro fetch step.  Useful if FRED_API_KEY
                 is not set or during budget-constrained testing.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import get_all_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader
from src.processing.multi_total_return import build_relative_return_targets


# ---------------------------------------------------------------------------
# EDGAR PGR fundamentals refresh
# ---------------------------------------------------------------------------

def _refresh_pgr_fundamentals(conn, dry_run: bool = False) -> int:
    """Fetch PGR quarterly fundamentals from SEC EDGAR XBRL and upsert into DB.

    Uses the companyfacts API (single request, 7-day cache).  No API key
    required.  Rate limit: 10 req/sec; the 7-day cache means most weekly
    runs make zero HTTP calls.

    Returns:
        Number of rows upserted.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fetch_fred_step(conn, dry_run: bool = False) -> None:
    """Fetch FRED macro series and upsert into fred_macro_monthly (v3.0+)."""
    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    series_list = config.FRED_SERIES_MACRO
    print(f"\nFetching {len(series_list)} FRED macro series...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch: {series_list}")
        return

    if config.FRED_API_KEY is None:
        print("  WARNING: FRED_API_KEY not set. Skipping FRED fetch.")
        return

    try:
        df = fetch_all_fred_macro(series_list)
        n = upsert_fred_to_db(conn, df)
        print(f"  FRED macro: {n} rows upserted ({len(series_list)} series)")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: FRED fetch failed: {exc}. Continuing with cached data.")


def main(dry_run: bool = False, skip_fred: bool = False) -> None:
    today = date.today()
    print(f"{'[DRY RUN] ' if dry_run else ''}PGR v2 Weekly Fetch — {today}")
    print(f"Database: {config.DB_PATH}")

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    all_tickers = get_all_price_tickers()   # PGR + 22 ETFs = 23 tickers
    pgr_only = ["PGR"]

    print(f"\nPrice tickers ({len(all_tickers)}): {all_tickers}")
    print(f"Dividend tickers: {pgr_only} (ETF dividends via initial_fetch.py)")
    print("PGR fundamentals refresh: True (SEC EDGAR XBRL, always on weekly run)")

    # --- Price fetch (23 AV calls) ---
    loader = MultiTickerLoader(conn)
    price_results = loader.fetch_all_prices(all_tickers, dry_run=dry_run)
    total_price_rows = sum(v for v in price_results.values() if v is not None)
    price_deferred = [t for t, v in price_results.items() if v is None]
    print(f"\nPrices — {total_price_rows} total rows upserted")
    for ticker, n in price_results.items():
        if n:
            print(f"  {ticker}: {n} rows")
    if price_deferred:
        print(f"  WARNING: {len(price_deferred)} price tickers deferred by AV rate limit: "
              f"{price_deferred}")

    # --- PGR dividend fetch (1 AV call) ---
    div_loader = MultiDividendLoader(conn)
    div_results = div_loader.fetch_for_tickers(pgr_only, dry_run=dry_run)
    total_div_rows = sum(v for v in div_results.values() if v is not None)
    div_deferred = [t for t, v in div_results.items() if v is None]
    print(f"\nDividends — {total_div_rows} rows upserted")
    for ticker, n in div_results.items():
        if n:
            print(f"  {ticker}: {n} rows")
    if div_deferred:
        print(f"  WARNING: {len(div_deferred)} dividend tickers deferred by AV rate limit: "
              f"{div_deferred}")

    # --- EDGAR fundamentals (1 EDGAR call, 7-day cached) ---
    print("\nRefreshing PGR quarterly fundamentals from SEC EDGAR XBRL...")
    try:
        n = _refresh_pgr_fundamentals(conn, dry_run=dry_run)
        print(f"  PGR fundamentals: {n} rows upserted")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: EDGAR fundamentals fetch failed: {exc}")
        print("  Continuing with previously cached data in the database.")

    # --- FRED macro series (v3.0+, no AV budget impact) ---
    if not skip_fred:
        _fetch_fred_step(conn, dry_run=dry_run)
    else:
        print("\nSkipping FRED fetch (--skip-fred).")

    # --- Relative return targets (derived; no API calls) ---
    # Refresh monthly_relative_returns after new prices/dividends are upserted
    # so the WFO models always train on the latest targets.
    print("\nRefreshing relative return targets (6M and 12M)...")
    if dry_run:
        print("  [DRY RUN] Skipping relative return computation.")
    else:
        for horizon in (6, 12):
            df = build_relative_return_targets(conn, forward_months=horizon, upsert=True)
            n_rows = df.shape[0] * df.shape[1] if not df.empty else 0
            print(f"  {horizon}M: {n_rows} rows upserted across "
                  f"{df.shape[1] if not df.empty else 0} benchmarks")

    # --- Budget summary ---
    today_str = today.isoformat()
    if dry_run:
        av_projected = len(all_tickers) + len(pgr_only)   # prices + PGR dividends
        print(f"\n[DRY RUN] Projected API calls: AV {av_projected}/{config.AV_DAILY_LIMIT}  "
              "EDGAR: 0–1 (7-day cache, no hard limit)")
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        print(f"\nAPI budget used today: AV {av_used}/{config.AV_DAILY_LIMIT}  "
              "EDGAR: free (no daily limit)")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGR v3.0 weekly data accumulation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without making HTTP calls.")
    parser.add_argument("--skip-fred", action="store_true",
                        help="Skip FRED macro fetch step.")
    args = parser.parse_args()
    main(dry_run=args.dry_run, skip_fred=args.skip_fred)
