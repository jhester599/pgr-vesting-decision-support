"""
Weekly data accumulation entrypoint for GitHub Actions.

Fetches prices for all 23 tickers (PGR + 22 ETF benchmarks) and PGR dividends
via Alpha Vantage TIME_SERIES_WEEKLY, then refreshes PGR quarterly fundamentals
from FMP and FRED macro series (v3.0+).  All results are upserted into the
v2 SQLite database.

Budget per run:
  23 AV calls  — all ticker prices (TIME_SERIES_WEEKLY, full history)
   1 AV call   — PGR dividends
  ─────────────
  24 AV total  (free-tier limit: 25/day)
   2 FMP calls — PGR key-metrics + income-statement
  ─────────────
   2 FMP total (free-tier limit: 250/day)
  N FRED calls — one per series in FRED_SERIES_MACRO (no daily limit)
  ─────────────
  FRED is a free public API; calls do not count against AV or FMP budgets.

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
# FMP PGR fundamentals refresh
# ---------------------------------------------------------------------------

def _refresh_pgr_fundamentals(conn, dry_run: bool = False) -> int:
    """Fetch PGR quarterly fundamentals from FMP and upsert into the DB.

    Returns:
        Number of rows upserted.
    """
    import requests
    import pandas as pd

    endpoints = [
        (
            f"/v3/key-metrics/{config.TICKER}",
            {"period": "quarter", "limit": 40},
            "fmp_key_metrics",
        ),
        (
            f"/v3/income-statement/{config.TICKER}",
            {"period": "quarter", "limit": 40},
            "fmp_income_statement",
        ),
    ]

    raw_results: dict[str, list] = {}
    for endpoint, params, label in endpoints:
        db_client.log_api_request(conn, "fmp", endpoint=label)
        if dry_run:
            raw_results[label] = []
            continue

        if config.FMP_API_KEY is None:
            raise RuntimeError("FMP_API_KEY is not set. Add it to your .env file.")

        url = f"{config.FMP_BASE_URL}{endpoint}"
        full_params = {**params, "apikey": config.FMP_API_KEY}
        resp = requests.get(url, params=full_params, timeout=30)
        resp.raise_for_status()
        raw_results[label] = resp.json()

    if dry_run:
        return 0

    def _parse_fmp(records: list, cols: dict[str, str]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty or "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        rename = {src: dst for src, dst in cols.items() if src in df.columns}
        df = df.rename(columns=rename)
        wanted = ["date"] + list(rename.values())
        df = df[[c for c in wanted if c in df.columns]]
        return df.sort_values("date").set_index("date")

    km_df = _parse_fmp(
        raw_results["fmp_key_metrics"],
        {"peRatio": "pe_ratio", "pbRatio": "pb_ratio", "roe": "roe"},
    )
    is_df = _parse_fmp(
        raw_results["fmp_income_statement"],
        {"eps": "eps", "revenue": "revenue", "netIncome": "net_income"},
    )

    if km_df.empty and is_df.empty:
        return 0

    combined = km_df.join(is_df, how="outer") if not km_df.empty else is_df
    combined = combined.reset_index()
    combined = combined.rename(columns={"date": "period_end"})
    combined["period_end"] = combined["period_end"].dt.strftime("%Y-%m-%d")
    combined["source"] = "fmp"

    records_out = combined.to_dict("records")
    n = db_client.upsert_pgr_fundamentals(conn, records_out)
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
    print("PGR fundamentals refresh: True (always on weekly run)")

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

    # --- FMP fundamentals (2 FMP calls) ---
    print("\nRefreshing PGR quarterly fundamentals from FMP...")
    n = _refresh_pgr_fundamentals(conn, dry_run=dry_run)
    print(f"  PGR fundamentals: {n} rows upserted")

    # --- FRED macro series (v3.0+, no AV/FMP budget impact) ---
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
        fmp_projected = 2
        print(f"\n[DRY RUN] Projected API calls: AV {av_projected}/{config.AV_DAILY_LIMIT}  "
              f"FMP {fmp_projected}/{config.FMP_DAILY_LIMIT}")
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        fmp_used = db_client.get_api_request_count(conn, "fmp", today_str)
        print(f"\nAPI budget used today: AV {av_used}/{config.AV_DAILY_LIMIT}  "
              f"FMP {fmp_used}/{config.FMP_DAILY_LIMIT}")

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
