"""
Load unadjusted daily OHLCV history for PGR.

Primary source: Alpha Vantage TIME_SERIES_WEEKLY (free tier, 1,000+ weeks)
for grant-date price lookups and return calculations.

Note: FMP's /v3/historical-price-full endpoint requires a paid plan.
The weekly AV series gives sufficient resolution for monthly feature
engineering and grant-date cost basis lookups.
"""

import json
import os
import time

import pandas as pd
import requests

import config


_PROCESSED_PATH = os.path.join(config.DATA_PROCESSED_DIR, "price_history.parquet")
_AV_CACHE_PATH  = os.path.join(config.DATA_RAW_DIR, "av_TIME_SERIES_WEEKLY_PGR.json")
_CACHE_HOURS    = 24


def _is_cache_valid(path: str, hours: int) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) < hours * 3600


def _fetch_av_weekly() -> dict:
    """Fetch or load-from-cache the AV weekly price series."""
    if _is_cache_valid(_AV_CACHE_PATH, _CACHE_HOURS):
        with open(_AV_CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)

    if config.AV_API_KEY is None:
        raise RuntimeError("AV_API_KEY is not set. Add it to your .env file.")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_WEEKLY&symbol={config.TICKER}"
        f"&apikey={config.AV_API_KEY}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "Weekly Time Series" not in data:
        raise ValueError(f"Alpha Vantage error: {data}")

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    with open(_AV_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return PGR weekly price data as a date-indexed DataFrame.

    Columns: open, high, low, close, volume.
    Index: DatetimeIndex of week-ending dates (Fridays), ascending.

    The weekly frequency is sufficient for:
      - Grant-date cost basis lookups (via .asof())
      - Monthly feature engineering (resampled to month-end)
      - 6-month forward return target construction

    Args:
        force_refresh: Bypass the Parquet cache and re-fetch from AV.
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    data = _fetch_av_weekly()
    ts   = data["Weekly Time Series"]

    records = []
    for date_str, ohlcv in ts.items():
        records.append({
            "date":   pd.Timestamp(date_str),
            "open":   float(ohlcv["1. open"]),
            "high":   float(ohlcv["2. high"]),
            "low":    float(ohlcv["3. low"]),
            "close":  float(ohlcv["4. close"]),
            "volume": int(ohlcv["5. volume"]),
        })

    df = (
        pd.DataFrame(records)
        .sort_values("date")
        .set_index("date")
    )

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
