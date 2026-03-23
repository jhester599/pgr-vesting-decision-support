"""
Tests for v3.0 statistical metrics in src/reporting/backtest_report.py.

Covers:
  - compute_oos_r_squared (Campbell-Thompson OOS R²)
  - apply_bhy_correction (BHY multiple testing)
  - compute_newey_west_ic (HAC-corrected IC)
  - generate_rolling_ic_series
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reporting.backtest_report import (
    apply_bhy_correction,
    compute_newey_west_ic,
    compute_oos_r_squared,
    generate_rolling_ic_series,
)


# ---------------------------------------------------------------------------
# compute_oos_r_squared
# ---------------------------------------------------------------------------

class TestComputeOosRSquared:
    def test_perfect_prediction_gives_positive_r2(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = compute_oos_r_squared(y, y)
        assert r2 > 0.0

    def test_historical_mean_prediction_gives_zero(self):
        """When predicted = expanding historical mean, OOS R² should be 0."""
        realized = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # Expanding mean as the naive prediction
        predicted = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0])
        r2 = compute_oos_r_squared(predicted, realized)
        # The model IS the historical mean benchmark → R² ≈ 0
        assert abs(r2) < 0.1

    def test_random_prediction_gives_negative_or_low_r2(self):
        rng = np.random.default_rng(99)
        realized = pd.Series(rng.normal(size=50))
        predicted = pd.Series(rng.normal(size=50))  # uncorrelated noise
        r2 = compute_oos_r_squared(predicted, realized)
        # Noise predictions should not systematically beat naive mean
        assert r2 < 0.5

    def test_returns_nan_for_insufficient_data(self):
        r2 = compute_oos_r_squared(pd.Series([1.0]), pd.Series([1.0]))
        assert np.isnan(r2)

    def test_handles_nan_values_in_input(self):
        predicted = pd.Series([1.0, np.nan, 3.0, 4.0])
        realized = pd.Series([1.1, 2.0, 2.9, 3.8])
        r2 = compute_oos_r_squared(predicted, realized)
        assert np.isfinite(r2)

    def test_returns_float(self):
        y = pd.Series(np.random.default_rng(0).normal(size=20))
        r2 = compute_oos_r_squared(y * 1.1, y)
        assert isinstance(r2, float)


# ---------------------------------------------------------------------------
# apply_bhy_correction
# ---------------------------------------------------------------------------

class TestApplyBhyCorrection:
    def test_empty_input_returns_empty(self):
        result = apply_bhy_correction({})
        assert result == {}

    def test_significant_p_value_rejected(self):
        """Very small p-values should be rejected after BHY correction."""
        p_values = {"VTI": 0.0001, "VOO": 0.0001, "GLD": 0.0001}
        result = apply_bhy_correction(p_values, alpha=0.05)
        assert all(result.values()), "Very small p-values should all be rejected"

    def test_large_p_values_not_rejected(self):
        """Large p-values should not survive BHY correction."""
        p_values = {"VTI": 0.9, "VOO": 0.8, "GLD": 0.95}
        result = apply_bhy_correction(p_values, alpha=0.05)
        assert not any(result.values()), "Large p-values should not be rejected"

    def test_returns_bool_dict(self):
        p_values = {"VTI": 0.05, "VOO": 0.1}
        result = apply_bhy_correction(p_values)
        for k, v in result.items():
            assert isinstance(v, bool), f"Expected bool for {k}, got {type(v)}"

    def test_keys_preserved(self):
        p_values = {"VTI": 0.01, "GLD": 0.99, "BND": 0.05}
        result = apply_bhy_correction(p_values)
        assert set(result.keys()) == set(p_values.keys())

    def test_fdr_level_respected(self):
        """More conservative alpha should reject fewer hypotheses."""
        p_values = {f"B{i}": 0.05 for i in range(10)}
        result_loose = apply_bhy_correction(p_values, alpha=0.20)
        result_strict = apply_bhy_correction(p_values, alpha=0.01)
        # Strict should reject ≤ loose
        assert sum(result_strict.values()) <= sum(result_loose.values())


# ---------------------------------------------------------------------------
# compute_newey_west_ic
# ---------------------------------------------------------------------------

class TestComputeNeweyWestIc:
    def test_returns_tuple_of_two_floats(self):
        rng = np.random.default_rng(5)
        y = pd.Series(rng.normal(size=40))
        x = y + rng.normal(scale=0.5, size=40)  # correlated
        ic, pval = compute_newey_west_ic(x, y, lags=5)
        assert isinstance(ic, float)
        assert isinstance(pval, float)

    def test_finite_values_for_valid_input(self):
        rng = np.random.default_rng(6)
        y = pd.Series(rng.normal(size=40))
        x = y + rng.normal(scale=0.5, size=40)
        ic, pval = compute_newey_west_ic(x, y, lags=5)
        assert np.isfinite(ic)
        assert np.isfinite(pval)

    def test_p_value_between_0_and_1(self):
        rng = np.random.default_rng(7)
        y = pd.Series(rng.normal(size=40))
        x = pd.Series(rng.normal(size=40))
        _, pval = compute_newey_west_ic(x, y, lags=5)
        assert 0.0 <= pval <= 1.0

    def test_correlated_series_has_low_p_value(self):
        """Strongly correlated series should produce low p-value."""
        rng = np.random.default_rng(8)
        y = pd.Series(rng.normal(size=60))
        x = y + rng.normal(scale=0.1, size=60)  # nearly identical
        ic, pval = compute_newey_west_ic(x, y, lags=5)
        assert pval < 0.05, f"Expected significant p-value, got {pval:.4f}"

    def test_insufficient_data_returns_nan(self):
        ic, pval = compute_newey_west_ic(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            lags=1,
        )
        assert np.isnan(ic) and np.isnan(pval)


# ---------------------------------------------------------------------------
# generate_rolling_ic_series
# ---------------------------------------------------------------------------

class TestGenerateRollingIcSeries:
    def _make_results(self, n: int = 36):
        """Create synthetic BacktestEventResult-like objects."""
        from dataclasses import dataclass
        from datetime import date, timedelta
        from src.backtest.vesting_events import VestingEvent

        @dataclass
        class MockResult:
            event: VestingEvent
            benchmark: str
            target_horizon: int
            predicted_relative_return: float
            realized_relative_return: float
            signal_direction: str
            correct_direction: bool
            predicted_sell_pct: float
            ic_at_event: float
            hit_rate_at_event: float
            n_train_observations: int
            proxy_fill_fraction: float

        results = []
        rng = np.random.default_rng(42)
        base = date(2020, 1, 31)
        for i in range(n):
            d = base + timedelta(days=30 * i)
            ev = VestingEvent(
                event_date=d, rsu_type="time",
                horizon_6m_end=d + timedelta(days=180),
                horizon_12m_end=d + timedelta(days=365),
            )
            results.append(MockResult(
                event=ev, benchmark="VTI", target_horizon=6,
                predicted_relative_return=float(rng.normal()),
                realized_relative_return=float(rng.normal()),
                signal_direction="OUTPERFORM",
                correct_direction=bool(rng.integers(2)),
                predicted_sell_pct=0.5,
                ic_at_event=float(rng.uniform(0, 0.2)),
                hit_rate_at_event=float(rng.uniform(0.4, 0.7)),
                n_train_observations=80,
                proxy_fill_fraction=0.0,
            ))
        return results

    def test_returns_dataframe(self):
        results = self._make_results()
        df = generate_rolling_ic_series(results)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        results = self._make_results()
        df = generate_rolling_ic_series(results, window_months=6)
        assert "ic_rolling" in df.columns
        assert "hit_rate_rolling" in df.columns
        assert "n_obs" in df.columns

    def test_empty_results_returns_empty(self):
        df = generate_rolling_ic_series([])
        assert df.empty

    def test_filter_by_benchmark(self):
        results = self._make_results()
        # All results have benchmark="VTI"
        df_vti = generate_rolling_ic_series(results, benchmark="VTI")
        df_gld = generate_rolling_ic_series(results, benchmark="GLD")
        assert not df_vti.empty
        assert df_gld.empty

    def test_rolling_window_applied(self):
        results = self._make_results(n=36)
        df = generate_rolling_ic_series(results, window_months=24)
        # With n=36 and window=24, we should have some valid rows
        assert len(df.dropna()) >= 1
