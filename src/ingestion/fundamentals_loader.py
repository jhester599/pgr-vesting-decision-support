"""
Load quarterly fundamental data for PGR from FMP.

Retrieves income statement and key financial metrics. On the FMP free tier
this data is available for approximately the most recent 5 years. Older
observations will appear as NaN in the feature matrix and are handled
gracefully by feature_engineering.py.

Point-in-time integrity: FMP serves data keyed by the fiscal period end
date, which is used as the feature observation date. No future information
is introduced.
"""

import os

import pandas as pd

from src.ingestion import fmp_client
import config


_PROCESSED_PATH = os.path.join(
    config.DATA_PROCESSED_DIR, "fundamentals_quarterly.parquet"
)
_FUNDAMENTALS_CACHE_HOURS = 168  # 7 days


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return quarterly fundamentals for PGR (P/E, P/B, ROE, EPS).

    Args:
        force_refresh: If True, bypass cache and re-fetch from FMP.

    Returns:
        DataFrame indexed by period end date (DatetimeIndex, ascending) with
        columns: pe_ratio, pb_ratio, roe, eps, revenue, net_income.
        All values are float64; missing data appears as NaN.
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    # Fetch key metrics (P/E, P/B, ROE)
    km_raw = fmp_client.get(
        f"/v3/key-metrics/{config.TICKER}",
        params={"period": "quarter", "limit": 40},
        cache_hours=_FUNDAMENTALS_CACHE_HOURS,
    )
    # Fetch income statement (revenue, net income, EPS)
    is_raw = fmp_client.get(
        f"/v3/income-statement/{config.TICKER}",
        params={"period": "quarter", "limit": 40},
        cache_hours=_FUNDAMENTALS_CACHE_HOURS,
    )

    def _parse(records: list, cols: list[str]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        available = [c for c in cols if c in df.columns]
        df = df[["date"] + available].copy()
        df = df.sort_values("date").set_index("date")
        return df

    km_df = _parse(
        km_raw,
        ["peRatio", "pbRatio", "roe"],
    )
    is_df = _parse(
        is_raw,
        ["eps", "revenue", "netIncome"],
    )

    df = km_df.join(is_df, how="outer")
    df = df.rename(
        columns={
            "peRatio": "pe_ratio",
            "pbRatio": "pb_ratio",
            "roe": "roe",
            "eps": "eps",
            "revenue": "revenue",
            "netIncome": "net_income",
        }
    )
    df = df.astype("float64")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
