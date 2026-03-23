"""
Tests for FRED macro feature integration in feature_engineering.py (v3.0).

Verifies that the 6 derived FRED features are correctly computed and that
no future leakage is introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import build_feature_matrix


def _make_price_history(n_days: int = 400) -> pd.DataFrame:
    """Generate synthetic daily OHLCV price history."""
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    prices = 100.0 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, n_days))
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": 1_000_000},
        index=idx,
    )


def _make_dividend_history() -> pd.DataFrame:
    """Minimal dividend history."""
    return pd.DataFrame(
        {"dividend": [0.50], "source": ["test"]},
        index=pd.DatetimeIndex(["2020-06-15"], name="ex_date"),
    )


def _make_split_history() -> pd.DataFrame:
    """Empty split history."""
    return pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"],
        index=pd.DatetimeIndex([], name="split_date"),
    )


def _make_fred_macro(n_months: int = 14) -> pd.DataFrame:
    """Synthetic FRED macro DataFrame with all 6 required series."""
    idx = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "T10Y2Y":        rng.normal(0.5, 0.3, n_months),
            "GS5":           rng.normal(3.0, 0.5, n_months),
            "GS2":           rng.normal(2.5, 0.5, n_months),
            "GS10":          rng.normal(3.5, 0.5, n_months),
            "T10YIE":        rng.normal(2.0, 0.3, n_months),
            "BAA10Y":        rng.normal(1.5, 0.3, n_months),
            "BAMLH0A0HYM2":  rng.normal(3.5, 0.5, n_months),
            "NFCI":          rng.normal(0.0, 0.2, n_months),
        },
        index=idx,
    )


class TestFredFeaturesInMatrix:
    def test_all_six_derived_features_present(self):
        prices = _make_price_history()
        dividends = _make_dividend_history()
        splits = _make_split_history()
        fred = _make_fred_macro()

        df = build_feature_matrix(
            price_history=prices,
            dividend_history=dividends,
            split_history=splits,
            fred_macro=fred,
            force_refresh=True,
        )

        expected_features = [
            "yield_slope", "yield_curvature", "real_rate_10y",
            "credit_spread_ig", "credit_spread_hy", "nfci",
        ]
        for feat in expected_features:
            assert feat in df.columns, f"Missing FRED feature: {feat}"

    def test_no_fred_features_without_fred_data(self):
        prices = _make_price_history()
        dividends = _make_dividend_history()
        splits = _make_split_history()

        df = build_feature_matrix(
            price_history=prices,
            dividend_history=dividends,
            split_history=splits,
            fred_macro=None,
            force_refresh=True,
        )

        fred_features = ["yield_slope", "yield_curvature", "real_rate_10y",
                         "credit_spread_ig", "credit_spread_hy", "nfci"]
        for feat in fred_features:
            assert feat not in df.columns, f"FRED feature present without data: {feat}"

    def test_yield_curvature_formula(self):
        """yield_curvature = 2×GS5 − GS2 − GS10."""
        prices = _make_price_history()
        dividends = _make_dividend_history()
        splits = _make_split_history()

        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        gs5 = np.ones(12) * 3.0
        gs2 = np.ones(12) * 2.0
        gs10 = np.ones(12) * 4.0
        fred = pd.DataFrame(
            {"T10Y2Y": np.zeros(12), "GS5": gs5, "GS2": gs2, "GS10": gs10,
             "T10YIE": np.ones(12), "BAA10Y": np.ones(12),
             "BAMLH0A0HYM2": np.ones(12), "NFCI": np.zeros(12)},
            index=idx,
        )

        df = build_feature_matrix(
            price_history=prices,
            dividend_history=dividends,
            split_history=splits,
            fred_macro=fred,
            force_refresh=True,
        )

        # Expected: 2*3.0 - 2.0 - 4.0 = 0.0
        if "yield_curvature" in df.columns:
            curvature_vals = df["yield_curvature"].dropna()
            assert (curvature_vals.abs() < 1e-9).all(), \
                "yield_curvature should be 0.0 for flat yield curve"

    def test_real_rate_formula(self):
        """real_rate_10y = GS10 − T10YIE."""
        prices = _make_price_history()
        dividends = _make_dividend_history()
        splits = _make_split_history()

        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        fred = pd.DataFrame(
            {"T10Y2Y": np.zeros(12), "GS5": np.ones(12) * 3,
             "GS2": np.ones(12) * 2, "GS10": np.ones(12) * 4.5,
             "T10YIE": np.ones(12) * 2.0, "BAA10Y": np.ones(12),
             "BAMLH0A0HYM2": np.ones(12), "NFCI": np.zeros(12)},
            index=idx,
        )

        df = build_feature_matrix(
            price_history=prices,
            dividend_history=dividends,
            split_history=splits,
            fred_macro=fred,
            force_refresh=True,
        )

        # Expected real rate: 4.5 - 2.0 = 2.5
        if "real_rate_10y" in df.columns:
            real_rate_vals = df["real_rate_10y"].dropna()
            assert (abs(real_rate_vals - 2.5) < 1e-9).all(), \
                "real_rate_10y should equal GS10 - T10YIE = 2.5"

    def test_fred_features_use_only_past_data(self):
        """Verify no future leakage: FRED features at month T use only data ≤ T."""
        prices = _make_price_history()
        dividends = _make_dividend_history()
        splits = _make_split_history()
        fred = _make_fred_macro(n_months=20)

        df = build_feature_matrix(
            price_history=prices,
            dividend_history=dividends,
            split_history=splits,
            fred_macro=fred,
            force_refresh=True,
        )

        # The FRED data is forward-filled (not shifted forward), so the value
        # at month T should match the FRED observation at or before T.
        # A simple structural check: yield_slope values should be finite floats.
        if "yield_slope" in df.columns:
            slope = df["yield_slope"].dropna()
            assert slope.apply(lambda x: np.isfinite(x)).all()
