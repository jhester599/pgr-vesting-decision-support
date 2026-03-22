"""
One-time migration script: v1 JSON/Parquet cache → v2 SQLite database.

Reads all v1 data artifacts (Parquet files + pgr_edgar_cache.csv) and
inserts them into ``pgr_financials.db`` using the v2 schema.  Does NOT
delete v1 files — they are retained for rollback safety.

Run once from the repository root:
    python scripts/migrate_v1_to_v2.py

The script is idempotent: re-running will UPSERT (replace) existing rows
rather than duplicating them.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Ensure the repo root is on the Python path so ``import config`` works when
# the script is launched from the repo root directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client


def _migrate_prices(conn: db_client.sqlite3.Connection) -> int:
    """Migrate PGR weekly price history from price_history.parquet."""
    path = os.path.join(config.DATA_PROCESSED_DIR, "price_history.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found.")
        return 0

    df = pd.read_parquet(path)
    # v1 parquet has a DatetimeIndex and columns: open, high, low, close, volume
    df = df.reset_index()
    date_col = [c for c in df.columns if "date" in c.lower() or c.lower() == "index"]
    if not date_col:
        print(f"  [WARN] Cannot detect date column in price_history.parquet; skipping.")
        return 0

    df = df.rename(columns={date_col[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = "PGR"
    df["source"] = "av"
    df["proxy_fill"] = 0

    records = df[["ticker", "date", "open", "high", "low", "close", "volume",
                  "source", "proxy_fill"]].to_dict("records")
    n = db_client.upsert_prices(conn, records)
    print(f"  [OK]   Migrated {n} PGR price rows.")
    return n


def _migrate_dividends(conn: db_client.sqlite3.Connection) -> int:
    """Migrate PGR dividend history from dividend_history.parquet."""
    path = os.path.join(config.DATA_PROCESSED_DIR, "dividend_history.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found.")
        return 0

    df = pd.read_parquet(path)
    df = df.reset_index()
    date_col = [c for c in df.columns if "date" in c.lower() or c.lower() == "index"]
    if not date_col:
        print(f"  [WARN] Cannot detect date column in dividend_history.parquet; skipping.")
        return 0

    df = df.rename(columns={date_col[0]: "ex_date"})
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = "PGR"
    df["source"] = "av"

    # v1 has a column named 'dividend'; normalise to 'amount'
    amount_col = next((c for c in df.columns if "div" in c.lower() or "amount" in c.lower()), None)
    if amount_col is None:
        print("  [WARN] Cannot find dividend amount column; skipping.")
        return 0
    df = df.rename(columns={amount_col: "amount"})

    records = df[["ticker", "ex_date", "amount", "source"]].to_dict("records")
    n = db_client.upsert_dividends(conn, records)
    print(f"  [OK]   Migrated {n} PGR dividend rows.")
    return n


def _migrate_splits(conn: db_client.sqlite3.Connection) -> int:
    """Migrate PGR split history from config (and optionally the parquet)."""
    # First try the parquet (may have richer data)
    path = os.path.join(config.DATA_PROCESSED_DIR, "split_history.parquet")
    records: list[dict] = []

    if os.path.exists(path):
        df = pd.read_parquet(path)
        df = df.reset_index()
        date_col = [c for c in df.columns if "date" in c.lower() or c.lower() == "index"]
        if date_col:
            df = df.rename(columns={date_col[0]: "split_date"})
            df["split_date"] = pd.to_datetime(df["split_date"]).dt.strftime("%Y-%m-%d")
            df["ticker"] = "PGR"
            for _, row in df.iterrows():
                records.append({
                    "ticker":      "PGR",
                    "split_date":  row["split_date"],
                    "split_ratio": float(row.get("split_ratio", row.get("ratio", np.nan))),
                    "numerator":   float(row.get("numerator", np.nan)),
                    "denominator": float(row.get("denominator", np.nan)),
                })

    if not records:
        # Fall back to the hardcoded config values
        for s in config.PGR_KNOWN_SPLITS:
            records.append({
                "ticker":      "PGR",
                "split_date":  s["date"],
                "split_ratio": s["ratio"],
                "numerator":   None,
                "denominator": None,
            })

    n = db_client.upsert_splits(conn, records)
    print(f"  [OK]   Migrated {n} PGR split rows.")
    return n


def _migrate_fundamentals(conn: db_client.sqlite3.Connection) -> int:
    """Migrate PGR quarterly fundamentals from the v1 parquet (if it exists)."""
    # v1 saves fundamentals via fundamentals_loader.py to a parquet; check for it.
    path = os.path.join(config.DATA_PROCESSED_DIR, "fundamentals.parquet")
    if not os.path.exists(path):
        # Try alternate name used by some v1 runs
        alt = os.path.join(config.DATA_PROCESSED_DIR, "pgr_fundamentals.parquet")
        if not os.path.exists(alt):
            print("  [SKIP] No quarterly fundamentals parquet found.")
            return 0
        path = alt

    df = pd.read_parquet(path)
    df = df.reset_index()
    date_col = [c for c in df.columns if "date" in c.lower() or "period" in c.lower()
                or c.lower() == "index"]
    if not date_col:
        print("  [WARN] Cannot detect date column in fundamentals parquet; skipping.")
        return 0

    df = df.rename(columns={date_col[0]: "period_end"})
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.strftime("%Y-%m-%d")
    df["source"] = "fmp"

    col_map = {
        "pe_ratio":   ["pe_ratio", "pe"],
        "pb_ratio":   ["pb_ratio", "pb"],
        "roe":        ["roe", "return_on_equity"],
        "eps":        ["eps", "eps_diluted", "epsdiluted"],
        "revenue":    ["revenue", "total_revenue"],
        "net_income": ["net_income", "netincome"],
    }
    for canon, aliases in col_map.items():
        matched = next((c for c in aliases if c in df.columns), None)
        if matched and matched != canon:
            df = df.rename(columns={matched: canon})
        if canon not in df.columns:
            df[canon] = np.nan

    records = df[["period_end", "pe_ratio", "pb_ratio", "roe", "eps",
                  "revenue", "net_income", "source"]].to_dict("records")
    n = db_client.upsert_pgr_fundamentals(conn, records)
    print(f"  [OK]   Migrated {n} PGR quarterly fundamental rows.")
    return n


def _migrate_edgar_monthly(conn: db_client.sqlite3.Connection) -> int:
    """Migrate PGR monthly EDGAR data via the v1 pgr_monthly_loader."""
    # Import v1 loader to reuse its parsing logic
    try:
        from src.ingestion.pgr_monthly_loader import load as load_edgar
        df = load_edgar(force_refresh=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Could not load EDGAR cache via v1 loader: {exc}; skipping.")
        return 0

    if df.empty:
        print("  [SKIP] EDGAR cache is empty.")
        return 0

    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "month_end"})
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.strftime("%Y-%m-%d")

    records = df[["month_end", "combined_ratio", "pif_total",
                  "pif_growth_yoy", "gainshare_estimate"]].to_dict("records")
    n = db_client.upsert_pgr_edgar_monthly(conn, records)
    print(f"  [OK]   Migrated {n} PGR EDGAR monthly rows.")
    return n


def main() -> None:
    print(f"PGR v1 → v2 Migration")
    print(f"Target database: {config.DB_PATH}")
    print()

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)
    print("Schema initialised (or already exists).")
    print()

    total = 0
    print("Migrating PGR price history...")
    total += _migrate_prices(conn)

    print("Migrating PGR dividend history...")
    total += _migrate_dividends(conn)

    print("Migrating PGR split history...")
    total += _migrate_splits(conn)

    print("Migrating PGR quarterly fundamentals...")
    total += _migrate_fundamentals(conn)

    print("Migrating PGR EDGAR monthly metrics...")
    total += _migrate_edgar_monthly(conn)

    conn.close()
    print()
    print(f"Migration complete. Total rows written: {total}")
    print("v1 source files have NOT been deleted. Verify the DB before removing them.")


if __name__ == "__main__":
    main()
