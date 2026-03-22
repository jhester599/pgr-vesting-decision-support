"""
Daily data accumulation entrypoint for GitHub Actions.

Fetches prices, dividends, and (weekly) PGR fundamentals into the
v2 SQLite database.  Respects daily API budgets via db_client.

Usage (local or CI):
    python scripts/daily_fetch.py [--dry-run] [--date YYYY-MM-DD]

Options:
    --dry-run          Log which tickers would be fetched but make no
                       HTTP calls.  Useful for verifying scheduler logic.
    --date YYYY-MM-DD  Override today's date for scheduler parity logic.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

# Ensure the repository root is on sys.path regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import (
    get_dividend_tickers_for_today,
    get_price_tickers_for_today,
    should_refresh_pgr_fundamentals,
)
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader


# ---------------------------------------------------------------------------
# FMP PGR fundamentals refresh (uses FMP, 2 calls/week)
# ---------------------------------------------------------------------------

def _refresh_pgr_fundamentals(conn, dry_run: bool = False) -> int:
    """Fetch PGR quarterly fundamentals from FMP and upsert into the DB.

    Uses the same FMP endpoints as v1's fundamentals_loader but writes
    directly to the DB (not to a Parquet file) and uses DB-backed rate
    limiting.

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

def main(today: date | None = None, dry_run: bool = False) -> None:
    print(f"{'[DRY RUN] ' if dry_run else ''}PGR v2 Daily Fetch — {today or date.today()}")
    print(f"Database: {config.DB_PATH}")

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    price_tickers = get_price_tickers_for_today(today)
    div_tickers = get_dividend_tickers_for_today(today)
    do_fundamentals = should_refresh_pgr_fundamentals(today)

    print(f"\nPrice tickers ({len(price_tickers)}): {price_tickers}")
    print(f"Dividend tickers ({len(div_tickers)}): {div_tickers}")
    print(f"PGR fundamentals refresh: {do_fundamentals}")

    # --- Price fetch ---
    loader = MultiTickerLoader(conn)
    price_results = loader.fetch_all_prices(price_tickers, dry_run=dry_run)
    total_price_rows = sum(price_results.values())
    print(f"\nPrices — {total_price_rows} total rows upserted")
    for ticker, n in price_results.items():
        if n:
            print(f"  {ticker}: {n} rows")

    # --- Dividend fetch ---
    div_loader = MultiDividendLoader(conn)
    div_results = div_loader.fetch_for_tickers(div_tickers, dry_run=dry_run)
    total_div_rows = sum(div_results.values())
    print(f"\nDividends — {total_div_rows} total rows upserted")
    for ticker, n in div_results.items():
        if n:
            print(f"  {ticker}: {n} rows")

    # --- FMP fundamentals (Tuesdays only) ---
    if do_fundamentals:
        print("\nRefreshing PGR quarterly fundamentals from FMP...")
        n = _refresh_pgr_fundamentals(conn, dry_run=dry_run)
        print(f"  PGR fundamentals: {n} rows upserted")
    else:
        print("\nSkipping PGR fundamentals (not Tuesday).")

    # --- Budget summary ---
    today_str = (today or date.today()).isoformat()
    if dry_run:
        # Compute projected usage without touching the DB
        av_projected = len(price_tickers) + len(div_tickers)
        fmp_projected = 2 if do_fundamentals else 0
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
    parser = argparse.ArgumentParser(description="PGR v2 daily data accumulation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without making HTTP calls.")
    parser.add_argument("--date", metavar="YYYY-MM-DD",
                        help="Override today's date for scheduler logic.")
    args = parser.parse_args()

    override_date: date | None = None
    if args.date:
        override_date = date.fromisoformat(args.date)

    main(today=override_date, dry_run=args.dry_run)
