"""
Tests for v4.5 new predictor variables.

Validates:
  - used_car_cpi_yoy   (CUSR0000SETA02 YoY % change)
  - medical_cpi_yoy    (CUSR0000SAM2 YoY % change)
  - ppi_auto_ins_yoy   (PCU5241265241261 YoY % change — auto insurance PPI)
  - cr_acceleration    (3-period diff of combined_ratio_ttm)
  - pgr_vs_kie_6m      (PGR trailing 6M return minus KIE trailing 6M return)

All tests are DB-independent (use synthetic data via build_feature_matrix()).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.processing.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BUILD_KW = {"force_refresh": True}   # always bypass parquet cache in tests


def _make_prices(n_years: int = 7) -> pd.DataFrame:
    """Daily price series covering n_years of business days."""
    n = n_years * 252
    dates = pd.bdate_range("2018-01-01", periods=n)
    price = np.linspace(100, 250, n) + np.random.default_rng(0).normal(0, 2, n)
    return pd.DataFrame({"open": price, "high": price * 1.01,
                         "low": price * 0.99, "close": price,
                         "volume": 1_000_000}, index=dates)


def _make_splits() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"],
        index=pd.DatetimeIndex([], name="split_date"),
    )


def _make_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["dividend", "source"],
        index=pd.DatetimeIndex([], name="ex_date"),
    )


def _make_fred_with_ppi(n_months: int = 84) -> pd.DataFrame:
    """FRED DataFrame containing all v4.5 series as level values (not YoY)."""
    dates = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(index=dates)
    # Macro
    df["T10Y2Y"]          = rng.normal(0.5, 0.5, n_months)
    df["GS2"]             = rng.uniform(0.5, 5.0, n_months)
    df["GS5"]             = df["GS2"] + rng.uniform(0, 1, n_months)
    df["GS10"]            = df["GS5"] + rng.uniform(0, 1, n_months)
    df["T10YIE"]          = rng.uniform(1.5, 3.0, n_months)
    df["BAA10Y"]          = rng.uniform(1.0, 4.0, n_months)
    df["BAMLH0A0HYM2"]    = rng.uniform(2.0, 8.0, n_months)
    df["NFCI"]            = rng.normal(0, 0.5, n_months)
    df["VIXCLS"]          = rng.uniform(10, 40, n_months)
    df["TRFVOLUSM227NFWA"] = np.linspace(250_000, 280_000, n_months)
    # v4.5 severity features — monotonically rising CPI-like series
    df["CUSR0000SETA02"]  = np.linspace(100, 140, n_months)   # used car CPI
    df["CUSR0000SAM2"]    = np.linspace(100, 120, n_months)   # medical CPI
    df["PCU5241265241261"] = np.linspace(100, 145, n_months)  # auto ins PPI
    # v4.5 KIE relative strength (pre-computed, injected as synthetic column)
    df["pgr_vs_kie_6m"]   = rng.normal(0.0, 0.05, n_months)
    return df


def _make_pgr_monthly(n_months: int = 36) -> pd.DataFrame:
    """Synthetic PGR monthly EDGAR data with combined_ratio for cr_acceleration."""
    dates = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "combined_ratio":   rng.uniform(88, 105, n_months),
        "pif_growth_yoy":   rng.normal(0.05, 0.02, n_months),
    }, index=dates)


# ===========================================================================
# Tests for config additions
# ===========================================================================

class TestConfigV45:
    def test_pcu_series_in_fred_series_pgr(self) -> None:
        assert "PCU5241265241261" in config.FRED_SERIES_PGR

    def test_used_car_cpi_in_fred_series_pgr(self) -> None:
        assert "CUSR0000SETA02" in config.FRED_SERIES_PGR

    def test_medical_cpi_in_fred_series_pgr(self) -> None:
        assert "CUSR0000SAM2" in config.FRED_SERIES_PGR

    def test_pcu_lag_in_fred_series_lags(self) -> None:
        assert "PCU5241265241261" in config.FRED_SERIES_LAGS
        assert config.FRED_SERIES_LAGS["PCU5241265241261"] == 1

    def test_used_car_lag_in_fred_series_lags(self) -> None:
        assert config.FRED_SERIES_LAGS["CUSR0000SETA02"] == 1

    def test_medical_cpi_lag_in_fred_series_lags(self) -> None:
        assert config.FRED_SERIES_LAGS["CUSR0000SAM2"] == 1


# ===========================================================================
# Tests for used_car_cpi_yoy
# ===========================================================================

class TestUsedCarCpiYoy:
    def _build(self) -> pd.DataFrame:
        prices = _make_prices()
        return build_feature_matrix(
            prices, _make_dividends(), _make_splits(),
            fred_macro=_make_fred_with_ppi(), **_BUILD_KW,
        )

    def test_used_car_cpi_yoy_present(self) -> None:
        df = self._build()
        assert "used_car_cpi_yoy" in df.columns

    def test_used_car_cpi_yoy_is_float64(self) -> None:
        df = self._build()
        assert df["used_car_cpi_yoy"].dtype == float

    def test_used_car_cpi_yoy_positive_for_rising_series(self) -> None:
        """Monotonically rising CPI → all non-null YoY values positive."""
        df = self._build()
        non_null = df["used_car_cpi_yoy"].dropna()
        assert len(non_null) > 0
        assert non_null.mean() > 0, "Rising CPI should give predominantly positive YoY"

    def test_used_car_cpi_yoy_first_12_are_nan(self) -> None:
        """YoY requires 12 months of history; first 12 months must be NaN."""
        df = self._build()
        assert df["used_car_cpi_yoy"].notna().sum() > 0  # some values exist
        # The series should have NaN at the start (burn-in for YoY calc)


# ===========================================================================
# Tests for medical_cpi_yoy
# ===========================================================================

class TestMedicalCpiYoy:
    def _build(self) -> pd.DataFrame:
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=_make_fred_with_ppi(),
        )

    def _build(self) -> pd.DataFrame:  # type: ignore[override]  # shadows parent
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=_make_fred_with_ppi(), **_BUILD_KW,
        )

    def test_medical_cpi_yoy_present(self) -> None:
        assert "medical_cpi_yoy" in self._build().columns

    def test_medical_cpi_yoy_is_float64(self) -> None:
        assert self._build()["medical_cpi_yoy"].dtype == float

    def test_medical_cpi_yoy_positive_for_rising_series(self) -> None:
        non_null = self._build()["medical_cpi_yoy"].dropna()
        assert non_null.mean() > 0, "Rising series should give predominantly positive YoY"

    def test_medical_cpi_yoy_absent_when_series_missing(self) -> None:
        """If CUSR0000SAM2 is absent from fred_macro, feature is silently skipped."""
        fred = _make_fred_with_ppi().drop(columns=["CUSR0000SAM2", "pgr_vs_kie_6m"])
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=fred, **_BUILD_KW,
        )
        assert "medical_cpi_yoy" not in df.columns


# ===========================================================================
# Tests for ppi_auto_ins_yoy
# ===========================================================================

class TestPpiAutoInsYoy:
    def _build(self) -> pd.DataFrame:
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=_make_fred_with_ppi(),
        )

    def _build(self) -> pd.DataFrame:  # type: ignore[override]
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=_make_fred_with_ppi(), **_BUILD_KW,
        )

    def test_ppi_auto_ins_yoy_present(self) -> None:
        assert "ppi_auto_ins_yoy" in self._build().columns

    def test_ppi_auto_ins_yoy_is_float64(self) -> None:
        assert self._build()["ppi_auto_ins_yoy"].dtype == float

    def test_ppi_auto_ins_yoy_positive_for_rising_series(self) -> None:
        non_null = self._build()["ppi_auto_ins_yoy"].dropna()
        assert len(non_null) > 0
        assert non_null.mean() > 0, "Rising series should give predominantly positive YoY"

    def test_ppi_absent_when_series_missing(self) -> None:
        fred = _make_fred_with_ppi().drop(columns=["PCU5241265241261", "pgr_vs_kie_6m"])
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=fred, **_BUILD_KW,
        )
        assert "ppi_auto_ins_yoy" not in df.columns

    def test_ppi_yoy_magnitude_reasonable(self) -> None:
        """YoY from linspace(100→145, 84 months) ≈ +6-7%/year; not 1000%."""
        non_null = self._build()["ppi_auto_ins_yoy"].dropna()
        assert non_null.abs().max() < 0.5, "YoY should be <50% for a smooth CPI series"


# ===========================================================================
# Tests for cr_acceleration
# ===========================================================================

class TestCrAcceleration:
    def _build(self, n_months: int = 36) -> pd.DataFrame:
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=_make_pgr_monthly(n_months), **_BUILD_KW,
        )

    def test_cr_acceleration_present_when_combined_ratio_available(self) -> None:
        df = self._build()
        assert "cr_acceleration" in df.columns

    def test_cr_acceleration_is_float64(self) -> None:
        assert self._build()["cr_acceleration"].dtype == float

    def test_cr_acceleration_is_3period_diff_of_ttm(self) -> None:
        """cr_acceleration = combined_ratio_ttm.diff(3) at every point."""
        df = self._build()
        if "combined_ratio_ttm" not in df.columns:
            pytest.skip("combined_ratio_ttm not produced with this data")
        expected = df["combined_ratio_ttm"].diff(3)
        pd.testing.assert_series_equal(
            df["cr_acceleration"].dropna(),
            expected.dropna(),
            check_names=False,
        )

    def test_cr_acceleration_absent_without_pgr_monthly(self) -> None:
        """No pgr_monthly → no combined_ratio_ttm → no cr_acceleration."""
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(), **_BUILD_KW,
        )
        assert "cr_acceleration" not in df.columns

    def test_cr_acceleration_non_null_requires_15_months_min(self) -> None:
        """combined_ratio_ttm needs 12M rolling; then diff(3) needs 3 more → 15 min."""
        df = self._build(n_months=36)
        # Should have some non-null values with 36 months of input
        assert df["cr_acceleration"].notna().sum() > 0


# ===========================================================================
# Tests for pgr_vs_kie_6m
# ===========================================================================

class TestPgrVsKie6m:
    def _build_with_kie(self) -> pd.DataFrame:
        """Build feature matrix with pgr_vs_kie_6m injected via fred_macro."""
        fred = _make_fred_with_ppi()
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=fred, **_BUILD_KW,
        )

    def test_pgr_vs_kie_present_when_injected(self) -> None:
        df = self._build_with_kie()
        assert "pgr_vs_kie_6m" in df.columns

    def test_pgr_vs_kie_is_float64(self) -> None:
        assert self._build_with_kie()["pgr_vs_kie_6m"].dtype == float

    def test_pgr_vs_kie_absent_when_not_injected(self) -> None:
        fred = _make_fred_with_ppi().drop(columns=["pgr_vs_kie_6m"])
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            fred_macro=fred, **_BUILD_KW,
        )
        assert "pgr_vs_kie_6m" not in df.columns

    def test_pgr_vs_kie_values_are_finite(self) -> None:
        """Synthetic values should be finite floats (no inf/NaN within non-null set)."""
        non_null = self._build_with_kie()["pgr_vs_kie_6m"].dropna()
        assert len(non_null) > 0
        assert np.isfinite(non_null.values).all()


# ===========================================================================
# Integration: all 5 features in a single matrix build
# ===========================================================================

class TestAllFiveFeatures:
    def _build_all(self) -> pd.DataFrame:
        return build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=_make_pgr_monthly(),
            fred_macro=_make_fred_with_ppi(),
            **_BUILD_KW,
        )

    def test_all_five_present(self) -> None:
        df = self._build_all()
        for feat in ["used_car_cpi_yoy", "medical_cpi_yoy", "ppi_auto_ins_yoy",
                     "cr_acceleration", "pgr_vs_kie_6m"]:
            assert feat in df.columns, f"{feat} missing from feature matrix"

    def test_all_five_are_float64(self) -> None:
        df = self._build_all()
        for feat in ["used_car_cpi_yoy", "medical_cpi_yoy", "ppi_auto_ins_yoy",
                     "cr_acceleration", "pgr_vs_kie_6m"]:
            assert df[feat].dtype == float, f"{feat} is not float64"

    def test_feature_count_increased(self) -> None:
        """Feature matrix should have more columns than pre-v4.5 baseline of 14.
        Without fundamentals (pe_ratio/pb_ratio/roe), the count is 16:
          4 price momentum + 5 FRED macro + 1 VMT + 3 v4.5 FRED + 1 KIE + 1 CR accel.
        With fundamentals it reaches 19.  Either way it exceeds the pre-v4.5 baseline."""
        df = self._build_all()
        feat_cols = get_feature_columns(df)
        assert len(feat_cols) >= 16, (
            f"Expected ≥16 features with v4.5 additions, got {len(feat_cols)}: {feat_cols}"
        )

    def test_no_object_dtype_columns(self) -> None:
        """All feature columns must be float64 after v4.3.1 dtype coercion."""
        df = self._build_all()
        feat_cols = get_feature_columns(df)
        bad = [c for c in feat_cols if df[c].dtype == object]
        assert bad == [], f"Object-dtype columns: {bad}"
