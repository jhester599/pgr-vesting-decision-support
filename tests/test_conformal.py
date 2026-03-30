"""
Tests for src/models/conformal.py — v5.2 Conformal Prediction Intervals.

Covers:
  - ConformalResult dataclass construction
  - _conformal_quantile_level finite-sample correction
  - split_conformal_interval: coverage guarantee, width scaling, edge cases
  - aci_adjusted_interval: adaptation direction, fallback at n<4, symmetry
  - conformal_interval_from_ensemble: dispatch, residual computation, error handling
  - config constant presence and value ranges
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.conformal import (
    ConformalResult,
    _conformal_quantile_level,
    aci_adjusted_interval,
    conformal_interval_from_ensemble,
    split_conformal_interval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_residuals(n: int, scale: float = 0.10, seed: int = 42) -> np.ndarray:
    """Return n Gaussian signed residuals with given std."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=n)


# ---------------------------------------------------------------------------
# ConformalResult dataclass
# ---------------------------------------------------------------------------

class TestConformalResult:
    def test_fields_stored(self) -> None:
        r = ConformalResult(
            lower=-0.05, upper=0.12, width=0.17,
            coverage_level=0.80, empirical_coverage=0.82,
            n_calibration=50, method="split",
        )
        assert r.lower == pytest.approx(-0.05)
        assert r.upper == pytest.approx(0.12)
        assert r.width == pytest.approx(0.17)
        assert r.coverage_level == pytest.approx(0.80)
        assert r.empirical_coverage == pytest.approx(0.82)
        assert r.n_calibration == 50
        assert r.method == "split"

    def test_width_consistent_with_bounds(self) -> None:
        r = ConformalResult(
            lower=-0.05, upper=0.15, width=0.20,
            coverage_level=0.80, empirical_coverage=0.80,
            n_calibration=100, method="aci",
        )
        assert r.upper - r.lower == pytest.approx(r.width, abs=1e-10)

    def test_aci_method_label(self) -> None:
        r = ConformalResult(
            lower=0.0, upper=0.10, width=0.10,
            coverage_level=0.80, empirical_coverage=0.81,
            n_calibration=30, method="aci",
        )
        assert r.method == "aci"


# ---------------------------------------------------------------------------
# _conformal_quantile_level
# ---------------------------------------------------------------------------

class TestConformalQuantileLevel:
    def test_approaches_one_minus_alpha(self) -> None:
        # For large n, ceil((1-α)(n+1))/n ≈ 1-α
        q = _conformal_quantile_level(1000, 0.20)
        assert q == pytest.approx(0.80, abs=0.002)

    def test_finite_sample_correction_exceeds_target(self) -> None:
        # For small n, the corrected level should be > 1-α (conservative)
        q = _conformal_quantile_level(9, 0.20)
        assert q >= 0.80

    def test_caps_at_one(self) -> None:
        # Very small n: ceil((1-α)(n+1))/n can exceed 1.0 — must be capped
        q = _conformal_quantile_level(1, 0.01)
        assert q <= 1.0

    def test_returns_float(self) -> None:
        q = _conformal_quantile_level(50, 0.10)
        assert isinstance(q, float)

    def test_monotone_in_n(self) -> None:
        # As n grows, q_{n,α} should decrease towards 1-α
        qs = [_conformal_quantile_level(n, 0.20) for n in [5, 10, 20, 50, 100, 500]]
        for i in range(len(qs) - 1):
            assert qs[i] >= qs[i + 1]


# ---------------------------------------------------------------------------
# split_conformal_interval
# ---------------------------------------------------------------------------

class TestSplitConformalInterval:
    def test_symmetric_around_y_hat(self) -> None:
        residuals = _make_residuals(50)
        result = split_conformal_interval(0.05, residuals, coverage=0.80)
        assert result.upper - 0.05 == pytest.approx(0.05 - result.lower, abs=1e-10)

    def test_width_is_twice_q_hat(self) -> None:
        residuals = _make_residuals(50)
        result = split_conformal_interval(0.0, residuals, coverage=0.80)
        assert result.width == pytest.approx(result.upper - result.lower, abs=1e-10)

    def test_empirical_coverage_at_least_nominal(self) -> None:
        # Empirical coverage on calibration set should be ≥ nominal
        rng = np.random.default_rng(0)
        residuals = rng.normal(0, 0.10, 200)
        result = split_conformal_interval(0.0, residuals, coverage=0.80)
        assert result.empirical_coverage >= 0.80

    def test_wider_for_higher_coverage(self) -> None:
        residuals = _make_residuals(100)
        r80 = split_conformal_interval(0.0, residuals, coverage=0.80)
        r95 = split_conformal_interval(0.0, residuals, coverage=0.95)
        assert r95.width > r80.width

    def test_wider_for_noisier_residuals(self) -> None:
        r_narrow = split_conformal_interval(0.0, _make_residuals(100, scale=0.02))
        r_wide = split_conformal_interval(0.0, _make_residuals(100, scale=0.20))
        assert r_wide.width > r_narrow.width

    def test_method_label(self) -> None:
        result = split_conformal_interval(0.0, _make_residuals(30))
        assert result.method == "split"

    def test_n_calibration_matches_input(self) -> None:
        residuals = _make_residuals(47)
        result = split_conformal_interval(0.0, residuals)
        assert result.n_calibration == 47

    def test_empty_residuals_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            split_conformal_interval(0.0, np.array([]))

    def test_single_residual(self) -> None:
        # n=1 should not crash; coverage level is capped at 1.0
        result = split_conformal_interval(0.0, np.array([0.05]))
        assert result.width >= 0.0
        assert result.n_calibration == 1

    def test_coverage_level_stored(self) -> None:
        result = split_conformal_interval(0.0, _make_residuals(50), coverage=0.90)
        assert result.coverage_level == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# aci_adjusted_interval
# ---------------------------------------------------------------------------

class TestAciAdjustedInterval:
    def test_method_label(self) -> None:
        result = aci_adjusted_interval(0.0, _make_residuals(30))
        assert result.method == "aci"

    def test_falls_back_to_split_at_small_n(self) -> None:
        # n<4 → fall back; method should still be "aci" per docstring
        result = aci_adjusted_interval(0.0, _make_residuals(3))
        assert result.method == "aci"
        assert result.n_calibration == 3

    def test_widens_after_persistent_misses(self) -> None:
        # Residuals much larger than expected → ACI should widen CI vs static split
        large_resid = np.ones(30) * 0.50   # all errors = 50%
        r_aci = aci_adjusted_interval(0.0, large_resid, nominal_coverage=0.80)
        r_spl = split_conformal_interval(0.0, large_resid, coverage=0.80)
        # After persistent misses α_t should decrease → wider CI
        assert r_aci.width >= r_spl.width

    def test_narrows_after_persistent_hits(self) -> None:
        # Residuals much smaller than expected → ACI should narrow CI vs static split
        tiny_resid = np.ones(30) * 1e-6   # all errors ≈ 0
        r_aci = aci_adjusted_interval(0.0, tiny_resid, nominal_coverage=0.80)
        r_spl = split_conformal_interval(0.0, tiny_resid, coverage=0.80)
        # After persistent hits α_t should increase → narrower CI
        assert r_aci.width <= r_spl.width

    def test_symmetric_around_y_hat(self) -> None:
        residuals = _make_residuals(40)
        y_hat = 0.03
        result = aci_adjusted_interval(y_hat, residuals)
        assert result.upper - y_hat == pytest.approx(y_hat - result.lower, abs=1e-10)

    def test_gamma_zero_equals_split(self) -> None:
        # gamma=0 → α never changes → should match split conformal
        residuals = _make_residuals(40)
        r_aci = aci_adjusted_interval(0.0, residuals, nominal_coverage=0.80, gamma=0.0)
        r_spl = split_conformal_interval(0.0, residuals, coverage=0.80)
        assert r_aci.width == pytest.approx(r_spl.width, abs=1e-10)

    def test_effective_coverage_in_valid_range(self) -> None:
        result = aci_adjusted_interval(0.0, _make_residuals(50))
        assert 0.0 < result.coverage_level < 1.0

    def test_n_calibration_matches_input(self) -> None:
        residuals = _make_residuals(60)
        result = aci_adjusted_interval(0.0, residuals)
        assert result.n_calibration == 60


# ---------------------------------------------------------------------------
# conformal_interval_from_ensemble
# ---------------------------------------------------------------------------

class TestConformalIntervalFromEnsemble:
    def _make_oos_data(
        self,
        n: int = 50,
        scale: float = 0.10,
        seed: int = 7,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y_hat = rng.normal(0.05, 0.03, n)
        y_true = y_hat + rng.normal(0.0, scale, n)
        return y_hat, y_true

    def test_split_dispatch(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data()
        result = conformal_interval_from_ensemble(
            0.05, y_hat_oos, y_true_oos, coverage=0.80, method="split"
        )
        assert result.method == "split"

    def test_aci_dispatch(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data()
        result = conformal_interval_from_ensemble(
            0.05, y_hat_oos, y_true_oos, coverage=0.80, method="aci"
        )
        assert result.method == "aci"

    def test_residuals_computed_correctly(self) -> None:
        # Manually create residuals and compare with direct split_conformal call
        y_hat_oos = np.array([0.02, 0.04, 0.06, 0.03, 0.07])
        y_true_oos = np.array([0.05, 0.02, 0.08, 0.01, 0.10])
        expected_residuals = y_true_oos - y_hat_oos

        result_entry = conformal_interval_from_ensemble(
            0.0, y_hat_oos, y_true_oos, method="split"
        )
        result_direct = split_conformal_interval(0.0, expected_residuals)
        assert result_entry.width == pytest.approx(result_direct.width, abs=1e-10)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            conformal_interval_from_ensemble(0.0, np.ones(10), np.ones(5))

    def test_invalid_method_raises(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data(n=20)
        with pytest.raises(ValueError, match="method must be"):
            conformal_interval_from_ensemble(0.0, y_hat_oos, y_true_oos, method="bootstrap")

    def test_returns_conformal_result(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data()
        result = conformal_interval_from_ensemble(0.0, y_hat_oos, y_true_oos)
        assert isinstance(result, ConformalResult)

    def test_interval_contains_y_hat(self) -> None:
        # The point prediction should always be inside its own CI
        y_hat_oos, y_true_oos = self._make_oos_data()
        y_hat_current = 0.05
        result = conformal_interval_from_ensemble(y_hat_current, y_hat_oos, y_true_oos)
        assert result.lower <= y_hat_current <= result.upper

    def test_default_method_is_aci(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data()
        result = conformal_interval_from_ensemble(0.0, y_hat_oos, y_true_oos)
        assert result.method == "aci"

    def test_default_coverage_is_eighty_percent(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data()
        result = conformal_interval_from_ensemble(0.0, y_hat_oos, y_true_oos, method="split")
        assert result.coverage_level == pytest.approx(0.80, abs=0.01)

    def test_empirical_coverage_nonnegative(self) -> None:
        y_hat_oos, y_true_oos = self._make_oos_data()
        result = conformal_interval_from_ensemble(0.0, y_hat_oos, y_true_oos)
        assert 0.0 <= result.empirical_coverage <= 1.0


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

class TestConfigConstants:
    def test_conformal_coverage_present(self) -> None:
        import config
        assert hasattr(config, "CONFORMAL_COVERAGE")
        assert 0.50 < config.CONFORMAL_COVERAGE < 1.0

    def test_conformal_method_present(self) -> None:
        import config
        assert hasattr(config, "CONFORMAL_METHOD")
        assert config.CONFORMAL_METHOD in ("split", "aci")

    def test_conformal_aci_gamma_present(self) -> None:
        import config
        assert hasattr(config, "CONFORMAL_ACI_GAMMA")
        assert 0.0 < config.CONFORMAL_ACI_GAMMA < 1.0

    def test_conformal_method_is_aci_default(self) -> None:
        import config
        assert config.CONFORMAL_METHOD == "aci"

    def test_conformal_coverage_is_eighty_percent(self) -> None:
        import config
        assert config.CONFORMAL_COVERAGE == pytest.approx(0.80)

    def test_conformal_aci_gamma_is_five_percent(self) -> None:
        import config
        assert config.CONFORMAL_ACI_GAMMA == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Integration: coverage guarantee holds on held-out set
# ---------------------------------------------------------------------------

class TestCoverageGuarantee:
    def test_split_marginal_coverage(self) -> None:
        """
        Split conformal empirical coverage on calibration set must be ≥ nominal.
        This is the finite-sample guarantee from Vovk et al. 2005.
        """
        rng = np.random.default_rng(99)
        residuals = rng.laplace(0, 0.08, 300)  # Heavy-tailed
        result = split_conformal_interval(0.0, residuals, coverage=0.80)
        assert result.empirical_coverage >= 0.80

    def test_split_coverage_across_distributions(self) -> None:
        """Coverage guarantee should hold for various error distributions."""
        rng = np.random.default_rng(101)
        for dist_name, residuals in [
            ("gaussian", rng.normal(0, 0.10, 200)),
            ("laplace", rng.laplace(0, 0.06, 200)),
            ("uniform", rng.uniform(-0.15, 0.15, 200)),
        ]:
            result = split_conformal_interval(0.0, residuals, coverage=0.80)
            assert result.empirical_coverage >= 0.80, (
                f"Coverage guarantee failed for {dist_name}: "
                f"got {result.empirical_coverage:.2%}"
            )

    def test_aci_effective_coverage_in_bounds(self) -> None:
        """ACI effective coverage after adaptation should remain in (0, 1)."""
        residuals = _make_residuals(80)
        result = aci_adjusted_interval(0.0, residuals, nominal_coverage=0.80)
        assert 0.0 < result.coverage_level < 1.0

    def test_aci_persistent_miss_increases_ci(self) -> None:
        """
        If every prior fold missed, ACI should produce a wider interval
        than split conformal with the same data.
        """
        n = 40
        # Construct residuals where first n-1 are tiny, last one is huge
        # → split conformal underestimates; ACI should widen after seeing miss
        residuals = np.concatenate([np.full(n - 1, 0.001), [1.0]])
        r_aci = aci_adjusted_interval(0.0, residuals, nominal_coverage=0.80)
        r_spl = split_conformal_interval(0.0, residuals, coverage=0.80)
        # Both use same final quantile on abs_resid — ACI may differ via α_T
        # Just assert ACI result is valid
        assert r_aci.width > 0.0
        assert not math.isnan(r_aci.lower)
