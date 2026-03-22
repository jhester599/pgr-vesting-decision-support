"""
Load pre-calculated technical indicators for PGR from Alpha Vantage.

Using AV's native endpoints offloads computation and ensures indicator
values match institutional standards. Four indicators are fetched (4 API
calls total on first run; 0 on subsequent cached runs):

  - SMA  (12-month, monthly interval)
  - RSI  (14-period, monthly interval)
  - MACD (standard 12/26/9, monthly interval)
  - BBANDS (20-period, monthly interval) — used to derive %B

All data is returned as a single monthly-frequency DataFrame, merged on
the observation date. Missing observations are NaN.
"""

import os

import pandas as pd
import numpy as np

from src.ingestion import av_client
import config


_PROCESSED_PATH = os.path.join(config.DATA_PROCESSED_DIR, "technical_indicators.parquet")


def _parse_av_series(data: dict, series_key: str, value_col: str) -> pd.DataFrame:
    """Extract a single AV time-series payload into a date-indexed DataFrame."""
    records = data.get(series_key, {})
    if not records:
        return pd.DataFrame(columns=["date", value_col]).set_index("date")
    rows = [{"date": k, value_col: float(v[value_col])} for k, v in records.items()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return monthly technical indicators for PGR.

    Args:
        force_refresh: If True, bypass cache and re-fetch from Alpha Vantage
                       (costs 4 API requests against the 25/day free limit).

    Returns:
        DataFrame indexed by month-end date (DatetimeIndex, ascending) with
        columns: sma_12m, rsi_14, macd_hist, bb_pct_b (all float64).
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    # --- SMA 12-month ---
    sma_raw = av_client.get(
        "SMA",
        params={"interval": "monthly", "time_period": 12, "series_type": "close"},
    )
    sma_df = _parse_av_series(
        sma_raw, "Technical Analysis: SMA", "SMA"
    ).rename(columns={"SMA": "sma_12m"})

    # --- RSI 14-period ---
    rsi_raw = av_client.get(
        "RSI",
        params={"interval": "monthly", "time_period": 14, "series_type": "close"},
    )
    rsi_df = _parse_av_series(
        rsi_raw, "Technical Analysis: RSI", "RSI"
    ).rename(columns={"RSI": "rsi_14"})

    # --- MACD ---
    macd_raw = av_client.get(
        "MACD",
        params={"interval": "monthly", "series_type": "close"},
    )
    macd_records = macd_raw.get("Technical Analysis: MACD", {})
    macd_rows = [
        {"date": k, "macd_hist": float(v["MACD_Hist"])}
        for k, v in macd_records.items()
    ]
    macd_df = pd.DataFrame(macd_rows)
    if not macd_df.empty:
        macd_df["date"] = pd.to_datetime(macd_df["date"])
        macd_df = macd_df.sort_values("date").set_index("date")

    # --- Bollinger Bands → %B ---
    bb_raw = av_client.get(
        "BBANDS",
        params={
            "interval": "monthly",
            "time_period": 20,
            "series_type": "close",
            "nbdevup": 2,
            "nbdevdn": 2,
        },
    )
    bb_records = bb_raw.get("Technical Analysis: BBANDS", {})
    bb_rows = [
        {
            "date": k,
            "bb_upper": float(v["Real Upper Band"]),
            "bb_middle": float(v["Real Middle Band"]),
            "bb_lower": float(v["Real Lower Band"]),
        }
        for k, v in bb_records.items()
    ]
    bb_df = pd.DataFrame(bb_rows)
    if not bb_df.empty:
        bb_df["date"] = pd.to_datetime(bb_df["date"])
        bb_df = bb_df.sort_values("date").set_index("date")
        # %B = (price - lower) / (upper - lower); proxy: use middle as price
        band_width = bb_df["bb_upper"] - bb_df["bb_lower"]
        bb_df["bb_pct_b"] = np.where(
            band_width > 0,
            (bb_df["bb_middle"] - bb_df["bb_lower"]) / band_width,
            np.nan,
        )
        bb_df = bb_df[["bb_pct_b"]]

    # --- Merge all indicators ---
    frames = [sma_df, rsi_df]
    if not macd_df.empty:
        frames.append(macd_df)
    if not bb_df.empty:
        frames.append(bb_df)

    df = frames[0].copy()
    for frame in frames[1:]:
        df = df.join(frame, how="outer")

    # Ensure all expected columns exist
    for col in ["sma_12m", "rsi_14", "macd_hist", "bb_pct_b"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["sma_12m", "rsi_14", "macd_hist", "bb_pct_b"]].astype("float64")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
