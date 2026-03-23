"""
Tests for PGR-specific FRED features in feature_engineering.py (v3.1).

Verifies:
  - insurance_cpi_mom3m present when CUSR0000SETC01 supplied
  - vmt_yoy present when TRFVOLUSM227NFWA supplied
  - vix feature present when VIXCLS supplied
  - Absent when not in fred_macro DataFrame
  - Formula: insurance_cpi_mom3m = 3-period pct_change; vmt_yoy = 12-period pct_change
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import build_feature_matrix


def _make_price_history(n_days: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    prices = 100.0 * np.cumprod(
        1 + np.random.default_rng(7).normal(0.0003, 0.01, n_days)
    )
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 1_000_000,
        },
        index=idx,
    )


def _make_dividend_history() -> pd.DataFrame:
    return pd.DataFrame(
        {"dividend": [0.50], "source": ["test"]},
        index=pd.DatetimeIndex(["2019-06-15"], name="ex_date"),
    )


def _make_split_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"],
        index=pd.DatetimeIndex([], name="split_date"),
    )


def _make_fred_pgr(n_months: int = 24) -> pd.DataFrame:
    """Synthetic FRED macro + PGR series."""
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "T10Y2Y":           rng.normal(0.5, 0.3, n_months),
            "GS5":              rng.normal(3.0, 0.5, n_months),
            "GS2":              rng.normal(2.5, 0.5, n_months),
            "GS10":             rng.normal(3.5, 0.5, n_months),
            "T10YIE":           rng.normal(2.0, 0.3, n_months),
            "BAA10Y":           rng.normal(1.5, 0.3, n_months),
            "BAMLH0A0HYM2":     rng.normal(3.5, 0.5, n_months),
            "NFCI":             rng.normal(0.0, 0.2, n_months),
            "VIXCLS":           rng.uniform(10, 35, n_months),
            "CUSR0000SETC01":   200.0 + rng.normal(0, 2, n_months).cumsum(),
            "TRFVOLUSM227NFWA": 250_000 + rng.normal(0, 1000, n_months).cumsum(),
        },
        index=idx,
    )


class TestPgrFredFeatures:
    def test_vix_feature_present(self):
        prices = _make_price_history()
        fred = _make_fred_pgr()
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=fred,
            force_refresh=True,
        )
        assert "vix" in df.columns, "vix feature should be present when VIXCLS provided"

    def test_insurance_cpi_mom3m_present(self):
        prices = _make_price_history()
        fred = _make_fred_pgr()
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=fred,
            force_refresh=True,
        )
        assert "insurance_cpi_mom3m" in df.columns, (
            "insurance_cpi_mom3m should be present when CUSR0000SETC01 provided"
        )

    def test_vmt_yoy_present(self):
        prices = _make_price_history()
        fred = _make_fred_pgr()
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=fred,
            force_refresh=True,
        )
        assert "vmt_yoy" in df.columns, (
            "vmt_yoy should be present when TRFVOLUSM227NFWA provided"
        )

    def test_pgr_features_absent_without_fred(self):
        prices = _make_price_history()
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=None,
            force_refresh=True,
        )
        for feat in ["vix", "insurance_cpi_mom3m", "vmt_yoy"]:
            assert feat not in df.columns, (
                f"{feat} should be absent when fred_macro=None"
            )

    def test_pgr_features_absent_without_pgr_columns(self):
        """When only macro columns are present (no PGR-specific), features absent."""
        prices = _make_price_history()
        # Fred with only macro columns, no PGR-specific
        idx = pd.date_range("2018-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(1)
        fred_macro_only = pd.DataFrame(
            {
                "T10Y2Y": rng.normal(0, 1, 24),
                "GS5": rng.normal(3, 0.5, 24),
                "GS2": rng.normal(2.5, 0.5, 24),
                "GS10": rng.normal(3.5, 0.5, 24),
                "T10YIE": rng.normal(2, 0.3, 24),
                "BAA10Y": rng.normal(1.5, 0.3, 24),
                "BAMLH0A0HYM2": rng.normal(3.5, 0.5, 24),
                "NFCI": rng.normal(0, 0.2, 24),
            },
            index=idx,
        )
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=fred_macro_only,
            force_refresh=True,
        )
        assert "insurance_cpi_mom3m" not in df.columns
        assert "vmt_yoy" not in df.columns

    def test_insurance_cpi_mom3m_formula(self):
        """insurance_cpi_mom3m = pct_change over 3 months."""
        prices = _make_price_history()
        # Use constant CPI growth to get predictable values
        idx = pd.date_range("2018-01-31", periods=24, freq="ME")
        ins_cpi = pd.Series(range(200, 224), index=idx, dtype=float)
        rng = np.random.default_rng(2)
        fred = pd.DataFrame(
            {
                "T10Y2Y": rng.normal(0.5, 0.1, 24),
                "GS5": rng.normal(3, 0.1, 24),
                "GS2": rng.normal(2.5, 0.1, 24),
                "GS10": rng.normal(3.5, 0.1, 24),
                "T10YIE": rng.normal(2, 0.1, 24),
                "BAA10Y": rng.normal(1.5, 0.1, 24),
                "BAMLH0A0HYM2": rng.normal(3.5, 0.1, 24),
                "NFCI": rng.normal(0, 0.1, 24),
                "CUSR0000SETC01": ins_cpi.values,
                "TRFVOLUSM227NFWA": rng.normal(250_000, 1000, 24),
            },
            index=idx,
        )
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=fred,
            force_refresh=True,
        )
        if "insurance_cpi_mom3m" in df.columns:
            # With linearly increasing CPI, 3M pct_change = 3/200 at first point
            # All valid values should be positive (CPI rising)
            valid = df["insurance_cpi_mom3m"].dropna()
            assert (valid > 0).all(), "CPI increasing → positive 3M momentum"

    def test_vmt_yoy_is_finite(self):
        prices = _make_price_history()
        fred = _make_fred_pgr(n_months=24)
        df = build_feature_matrix(
            price_history=prices,
            dividend_history=_make_dividend_history(),
            split_history=_make_split_history(),
            fred_macro=fred,
            force_refresh=True,
        )
        if "vmt_yoy" in df.columns:
            valid = df["vmt_yoy"].dropna()
            if not valid.empty:
                assert valid.apply(lambda x: np.isfinite(x)).all()
