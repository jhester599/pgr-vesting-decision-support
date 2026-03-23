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

  FRED macro features (v3.0+, from fred_macro_monthly table):
    - yield_slope      (T10Y2Y — 10Y-2Y spread)
    - yield_curvature  (2×GS5 − GS2 − GS10)
    - real_rate_10y    (GS10 − T10YIE)
    - credit_spread_ig (BAA10Y — investment-grade credit spread)
    - credit_spread_hy (BAMLH0A0HYM2 — high-yield OAS)
    - nfci             (Chicago Fed NFCI composite)
    Dropped silently if FRED data is absent (backward-compatible with pre-v3.0 runs).

"""

import os
import sqlite3

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
    fred_macro: pd.DataFrame | None = None,
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
        fred_macro:          Wide DataFrame from db_client.get_fred_macro() with
                             DatetimeIndex (month-end) and one column per FRED
                             series_id.  Optional; omitted in pre-v3.0 runs.
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
    # FRED macro features (v3.0+) — derived from the fred_macro DataFrame.
    # Each feature is forward-filled from the FRED month-end observations
    # to align with the feature matrix dates.  If a required FRED series
    # is absent, the corresponding feature column is skipped silently so
    # that pre-v3.0 runs remain unaffected.
    # ------------------------------------------------------------------
    if fred_macro is not None and not fred_macro.empty:
        # Reindex to the feature matrix dates; forward-fill end-of-month gaps
        fred_aligned = fred_macro.reindex(monthly_dates, method="ffill")

        # yield_slope: 10Y-2Y spread (direct series)
        if "T10Y2Y" in fred_aligned.columns:
            df["yield_slope"] = fred_aligned["T10Y2Y"]

        # yield_curvature: butterfly = 2×GS5 − GS2 − GS10
        if all(s in fred_aligned.columns for s in ("GS5", "GS2", "GS10")):
            df["yield_curvature"] = (
                2.0 * fred_aligned["GS5"]
                - fred_aligned["GS2"]
                - fred_aligned["GS10"]
            )

        # real_rate_10y: nominal 10Y minus breakeven inflation
        if all(s in fred_aligned.columns for s in ("GS10", "T10YIE")):
            df["real_rate_10y"] = fred_aligned["GS10"] - fred_aligned["T10YIE"]

        # credit_spread_ig: Moody's Baa minus 10Y Treasury
        if "BAA10Y" in fred_aligned.columns:
            df["credit_spread_ig"] = fred_aligned["BAA10Y"]

        # credit_spread_hy: ICE BofA HY OAS
        if "BAMLH0A0HYM2" in fred_aligned.columns:
            df["credit_spread_hy"] = fred_aligned["BAMLH0A0HYM2"]

        # nfci: Chicago Fed National Financial Conditions Index
        if "NFCI" in fred_aligned.columns:
            df["nfci"] = fred_aligned["NFCI"]

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


def build_feature_matrix_from_db(
    conn: sqlite3.Connection,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """DB-backed entry point for the PGR feature matrix.

    Loads PGR price, dividend, split, fundamental, and EDGAR data from the
    v2 SQLite database and delegates to the existing build_feature_matrix().
    The Parquet cache is always bypassed because the DB is the authoritative
    source for v2 runs.

    Args:
        conn:          Open SQLite connection with v2 schema.
        force_refresh: When True, forces recomputation even if a Parquet cache
                       file already exists (always True internally; parameter
                       preserved for API symmetry with build_feature_matrix).

    Returns:
        DataFrame identical in structure to build_feature_matrix() output:
        monthly DatetimeIndex, feature columns, and ``target_6m_return``.

    Raises:
        ValueError: If PGR has no price data in the database.
    """
    from src.database import db_client

    prices = db_client.get_prices(conn, "PGR")

    dividends_raw = db_client.get_dividends(conn, "PGR")
    if dividends_raw.empty:
        dividends = pd.DataFrame(
            columns=["dividend", "source"],
            index=pd.DatetimeIndex([], name="ex_date"),
        )
    else:
        dividends = dividends_raw.rename(columns={"amount": "dividend"})

    splits_raw = db_client.get_splits(conn, "PGR")
    if splits_raw.empty:
        splits = pd.DataFrame(
            columns=["split_ratio", "numerator", "denominator"],
            index=pd.DatetimeIndex([], name="split_date"),
        )
    else:
        splits = splits_raw

    fundamentals_raw = db_client.get_pgr_fundamentals(conn)
    fundamentals = fundamentals_raw if not fundamentals_raw.empty else None

    edgar_raw = db_client.get_pgr_edgar_monthly(conn)
    pgr_monthly = edgar_raw if not edgar_raw.empty else None

    # v3.0: load FRED macro series if the table is populated
    fred_raw = db_client.get_fred_macro(conn, config.FRED_SERIES_MACRO)
    fred_macro = fred_raw if not fred_raw.empty else None

    return build_feature_matrix(
        price_history=prices,
        dividend_history=dividends,
        split_history=splits,          # may be empty DataFrame with DatetimeIndex
        fundamentals=fundamentals,
        pgr_monthly=pgr_monthly,
        fred_macro=fred_macro,
        force_refresh=True,            # always recompute from DB; never use stale Parquet
    )


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


def get_X_y_relative(
    df: pd.DataFrame,
    relative_returns: pd.Series,
    drop_na_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align the feature matrix to a relative return target series.

    Used by the v2 WFO engine to build (X, y) pairs where y is the
    PGR-minus-ETF relative return rather than the absolute PGR return.

    Args:
        df:               Feature matrix from build_feature_matrix() or
                          build_feature_matrix_from_db().  Must have a
                          DatetimeIndex of month-end dates.
        relative_returns: Pre-computed relative return Series (e.g. from
                          load_relative_return_matrix() or a column from
                          build_relative_return_targets()).  DatetimeIndex
                          must overlap with ``df``.
        drop_na_target:   If True, drop rows where ``relative_returns`` is NaN.

    Returns:
        ``(X, y)`` where X is the aligned feature DataFrame (same columns as
        ``get_feature_columns(df)``) and y is the relative return Series.

    Raises:
        ValueError: If the aligned index is empty (no date overlap between
                    ``df`` and ``relative_returns``, or all NaN after dropping).
    """
    feature_cols = get_feature_columns(df)
    X_raw = df[feature_cols]

    # Inner join: retain only dates present in both the feature matrix and
    # the relative return series.
    aligned = X_raw.join(relative_returns, how="inner")

    if drop_na_target:
        aligned = aligned.dropna(subset=[relative_returns.name])

    if aligned.empty:
        raise ValueError(
            f"No overlapping non-NaN dates between feature matrix and relative "
            f"return series '{relative_returns.name}'. Verify that both cover "
            f"the same date range and that the series is not all-NaN."
        )

    X = aligned[feature_cols].copy()
    y = aligned[relative_returns.name].copy()
    return X, y
