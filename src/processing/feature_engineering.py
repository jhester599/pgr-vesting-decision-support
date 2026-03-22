"""
Rolling monthly feature matrix construction.

Computes all predictive features on a month-end frequency, expanding the
effective dataset from ~30 semi-annual observations to 180+ monthly samples.
This directly addresses the "Large P, Small N" constraint identified in
the Gemini peer review.

No feature uses data from t+1 or later (strict no-leakage guarantee).
The target variable (6-month forward DRIP total return) is appended as the
final column, with NaN for the last 6 months of history.

Feature groups:
  Price-derived (always available):
    - mom_3m, mom_6m, mom_12m   (price momentum)
    - vol_21d, vol_63d           (realized volatility)

  Technical indicators (from Alpha Vantage cache):
    - sma_12m, rsi_14, macd_hist, bb_pct_b

  Fundamental (from FMP; ~5-year depth on free tier):
    - pe_ratio, pb_ratio, roe

  Optional — Gainshare / PGR-specific (from EDGAR cache):
    - combined_ratio_ttm, pif_growth_yoy, gainshare_est
    Dropped entirely if fewer than WFO_MIN_GAINSHARE_OBS non-NaN rows.

"""

import os

import numpy as np
import pandas as pd

import config


_PROCESSED_PATH = os.path.join(config.DATA_PROCESSED_DIR, "feature_matrix.parquet")

# Nominal trading days for each momentum lookback
_MOMENTUM_WINDOWS: dict[str, int] = {
    "mom_3m": 63,
    "mom_6m": 126,
    "mom_12m": 252,
}

_VOL_WINDOWS: dict[str, int] = {
    "vol_21d": 21,
    "vol_63d": 63,
}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    price_history: pd.DataFrame,
    dividend_history: pd.DataFrame,
    split_history: pd.DataFrame,
    technical_indicators: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    pgr_monthly: pd.DataFrame | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build the full monthly feature matrix for ML training.

    Args:
        price_history:       DataFrame from price_loader.load().
        dividend_history:    DataFrame from dividend_loader.load().
        split_history:       DataFrame from split_loader.load().
        technical_indicators: DataFrame from technical_loader.load() (optional).
        fundamentals:        DataFrame from fundamentals_loader.load() (optional).
        pgr_monthly:         DataFrame from pgr_monthly_loader.load() (optional).
        force_refresh:       If True, recompute even if cached Parquet exists.

    Returns:
        DataFrame indexed by month-end date with feature columns and
        ``target_6m_return`` as the final column.
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    # Resample daily prices to month-end (last business day)
    monthly_close = price_history["close"].resample("BME").last()
    monthly_dates = monthly_close.index

    df = pd.DataFrame(index=monthly_dates)
    df.index.name = "date"

    # ------------------------------------------------------------------
    # Price momentum features (no-leakage: only data up to t used at t)
    # ------------------------------------------------------------------
    # shift(n) on the daily series moves each value n trading days into the
    # future, so daily_close.shift(n)[t] == close[t - n trading days].
    # Reindexing to month-end dates with ffill aligns the lagged price to
    # the last available trading day on or before each month-end.
    daily_close = price_history["close"]

    for col, n_days in _MOMENTUM_WINDOWS.items():
        lagged_daily = daily_close.shift(n_days)
        lagged_monthly = lagged_daily.reindex(monthly_dates, method="ffill")
        df[col] = (monthly_close / lagged_monthly) - 1.0

    # ------------------------------------------------------------------
    # Realized volatility
    # ------------------------------------------------------------------
    daily_log_ret = np.log(daily_close / daily_close.shift(1))

    for col, n_days in _VOL_WINDOWS.items():
        rolling_vol = (
            daily_log_ret.rolling(window=n_days, min_periods=n_days // 2)
            .std()
            * np.sqrt(252)
        )
        df[col] = rolling_vol.reindex(monthly_dates, method="ffill")

    # ------------------------------------------------------------------
    # Technical indicators (Alpha Vantage)
    # ------------------------------------------------------------------
    if technical_indicators is not None and not technical_indicators.empty:
        for col in ["sma_12m", "rsi_14", "macd_hist", "bb_pct_b"]:
            if col in technical_indicators.columns:
                df[col] = technical_indicators[col].reindex(
                    monthly_dates, method="ffill"
                )

    # ------------------------------------------------------------------
    # Fundamental features (FMP, ~5-year depth on free tier)
    # ------------------------------------------------------------------
    if fundamentals is not None and not fundamentals.empty:
        for col in ["pe_ratio", "pb_ratio", "roe"]:
            if col in fundamentals.columns:
                # Forward-fill quarterly data to monthly frequency
                df[col] = fundamentals[col].reindex(monthly_dates, method="ffill")

    # ------------------------------------------------------------------
    # PGR-specific Gainshare features (optional, from EDGAR cache)
    # ------------------------------------------------------------------
    if pgr_monthly is not None and not pgr_monthly.empty:
        # Trailing 12-month combined ratio (rolling mean of monthly values)
        if "combined_ratio" in pgr_monthly.columns:
            cr_monthly = pgr_monthly["combined_ratio"].reindex(
                monthly_dates, method="ffill"
            )
            df["combined_ratio_ttm"] = cr_monthly.rolling(12, min_periods=6).mean()

        if "pif_growth_yoy" in pgr_monthly.columns:
            df["pif_growth_yoy"] = pgr_monthly["pif_growth_yoy"].reindex(
                monthly_dates, method="ffill"
            )

        if "gainshare_estimate" in pgr_monthly.columns:
            df["gainshare_est"] = pgr_monthly["gainshare_estimate"].reindex(
                monthly_dates, method="ffill"
            )

    # Drop optional Gainshare columns if insufficient observations
    for col in ["combined_ratio_ttm", "pif_growth_yoy", "gainshare_est"]:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            if n_valid < config.WFO_MIN_GAINSHARE_OBS:
                df = df.drop(columns=[col])

    # ------------------------------------------------------------------
    # Target variable: 6-month forward DRIP total return
    # ------------------------------------------------------------------
    from src.processing.total_return import build_monthly_returns
    target = build_monthly_returns(
        price_history, dividend_history, split_history, forward_months=6
    )
    df["target_6m_return"] = target.reindex(monthly_dates)

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------
    # Drop rows where ALL price-derived features are NaN (burn-in period)
    price_feature_cols = list(_MOMENTUM_WINDOWS.keys()) + list(_VOL_WINDOWS.keys())
    df = df.dropna(subset=price_feature_cols, how="all")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excludes the target variable)."""
    return [c for c in df.columns if c != "target_6m_return"]


def get_X_y(
    df: pd.DataFrame,
    drop_na_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature matrix into X (features) and y (target).

    Args:
        df:             DataFrame from build_feature_matrix().
        drop_na_target: If True, drop rows where target_6m_return is NaN
                        (i.e., the final 6 months of history).

    Returns:
        (X, y) tuple where X is the feature DataFrame and y is the target Series.
    """
    if drop_na_target:
        df = df.dropna(subset=["target_6m_return"])
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["target_6m_return"].copy()
    return X, y
