"""
Load historical dividend payments for PGR.

Primary source: Alpha Vantage DIVIDENDS endpoint (free tier).
Returns ex-dividend dates and per-share amounts (unadjusted) for accurate
DRIP fractional share calculations in total_return.py.

Note: FMP /v3/historical-price-full/stock_dividend requires a paid plan.
"""

import os

import pandas as pd

from src.ingestion import av_client
import config


_PROCESSED_PATH = os.path.join(config.DATA_PROCESSED_DIR, "dividend_history.parquet")


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return PGR historical dividend payments.

    Args:
        force_refresh: If True, bypass cache and re-fetch from Alpha Vantage.

    Returns:
        DataFrame with columns: date (DatetimeIndex, ascending), dividend (float64).
        ``date`` is the ex-dividend date. ``dividend`` is the raw per-share amount.
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    raw = av_client.get(
        "DIVIDENDS",
        cache_hours=168,  # dividends change infrequently; 1-week TTL
    )

    records = raw.get("data", [])
    if not records:
        raise ValueError("Alpha Vantage returned no dividend history for PGR.")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["ex_dividend_date"])
    df["dividend"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[["date", "dividend"]].copy()
    df = df[df["dividend"] > 0].dropna(subset=["dividend"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
