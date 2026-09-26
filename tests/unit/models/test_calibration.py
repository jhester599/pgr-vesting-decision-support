"""
Tests for src/models/calibration.py — v5.1 Probability Calibration.

Covers:
  - compute_ece: perfect calibration, worst-case, edge cases
  - block_bootstrap_ece_ci: coverage, shape, monotonicity
  - fit_calibration_model: method selection, uncalibrated guard, ECE on training data
  - calibrate_prediction: None model, Platt, Isotonic, boundary values
  - Integration: calibrated ECE ≤ raw ECE on well-separated data
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (
    CalibrationResult,
    block_bootstrap_ece_ci,
    calibrate_prediction,
    compute_ece,
    fit_calibration_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfectly_calibrated(n: int = 200, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, outcomes) where mean prob ≈ mean outcome in every bin."""
    rng = np.random.default_rng(rng_seed)
    probs = rng.uniform(0.0, 1.0, n)
    outcomes = (rng.uniform(0.0, 1.0, n) < probs).astype(int)
    return probs, outcomes


def _make_overconfident(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """All probs = 0.9 but only 50% positive outcomes → ECE ≈ 0.40."""
    probs = np.full(n, 0.9)
    outcomes = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)]).astype(int)
    return probs, outcomes


def _make_regression_scores(
    n: int = 300,
    positive_signal: float = 0.5,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_hat, outcomes) where positive y_hat predicts positive outcome."""
    rng = np.random.default_rng(rng_seed)
    y_hat = rng.normal(0, 1, n)
    # Stronger positive_signal → more separable
    prob_positive = 1 / (1 + np.exp(-positive_signal * y_hat))
    outcomes = (rng.uniform(size=n) < prob_positive).astype(int)
    return y_hat, outcomes


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

class TestComputeEce:
    def test_perfect_calibration_is_near_zero(self):
        """A large perfectly-calibrated sample should yield ECE < 5%."""
        probs, outcomes = _make_perfectly_calibrated(n=5000, rng_seed=1)
        ece = compute_ece(probs, outcomes, n_bins=10)
        assert ece < 0.05, f"Expected ECE < 0.05 for perfect calibration, got {ece:.4f}"

    def test_all_probs_wrong_direction_gives_high_ece(self):
        """Predicting 0.9 when true frequency is 0.1 → ECE ≈ 0.8."""
        probs = np.full(200, 0.9)
        outcomes = np.zeros(200, dtype=int)  # Never outperforms
        ece = compute_ece(probs, outcomes, n_bins=10)
        assert ece > 0.7, f"Expected ECE > 0.7, got {ece:.4f}"

    def test_overconfident_returns_moderate_ece(self):
        probs, outcomes = _make_overconfident(200)
        ece = compute_ece(probs, outcomes, n_bins=10)
        # All probs = 0.9, true freq = 0.5 → ECE = 0.4
        assert 0.35 < ece < 0.45, f"Expected ECE ≈ 0.40, got {ece:.4f}"

    def test_empty_input_returns_zero(self):
        ece = compute_ece(np.array([]), np.array([]), n_bins=10)
        assert ece == 0.0

    def test_single_observation(self):
        """Single observation: ECE = |p - y|."""
        ece = compute_ece(np.array([0.7]), np.array([0]), n_bins=10)
        assert abs(ece - 0.7) < 1e-9

    def test_ece_in_unit_interval(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100)
        ece = compute_ece(probs, outcomes, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_ece_includes_probability_one(self):
        """prob == 1.0 should land in the last bin, not be dropped."""
        probs = np.array([1.0, 1.0, 1.0])
        outcomes = np.array([1, 1, 1])
        ece = compute_ece(probs, outcomes, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_monotone_in_miscalibration(self):
        """Worse miscalibration → higher ECE."""
        n = 500
        outcomes = np.ones(n, dtype=int)  # true freq = 1.0
        ece_mild = compute_ece(np.full(n, 0.8), outcomes, n_bins=10)   # gap = 0.2
        ece_severe = compute_ece(np.full(n, 0.5), outcomes, n_bins=10)  # gap = 0.5
        assert ece_severe > ece_mild


# ---------------------------------------------------------------------------
# block_bootstrap_ece_ci
# ---------------------------------------------------------------------------

class TestBlockBootstrapEceCi:
    def test_returns_tuple_of_two_floats(self):
        probs, outcomes = _make_perfectly_calibrated(100)
        lo, hi = block_bootstrap_ece_ci(probs, outcomes, n_bins=5, block_len=3, n_bootstrap=50)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_ordered(self):
        probs, outcomes = _make_perfectly_calibrated(200)
        lo, hi = block_bootstrap_ece_ci(probs, outcomes, n_bins=10, block_len=6, n_bootstrap=100)
        assert lo <= hi

    def test_ci_in_unit_interval(self):
        probs, outcomes = _make_perfectly_calibrated(200)
        lo, hi = block_bootstrap_ece_ci(probs, outcomes, n_bins=10, block_len=6, n_bootstrap=100)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_ci_contains_point_estimate(self):
        """95% CI should contain the point ECE estimate in typical cases."""
        probs, outcomes = _make_overconfident(200)
        ece = compute_ece(probs, outcomes, n_bins=10)
        lo, hi = block_bootstrap_ece_ci(
            probs, outcomes, n_bins=10, block_len=6, n_bootstrap=200,
            rng=np.random.default_rng(0),
        )
        assert lo <= ece <= hi, f"ECE={ece:.4f} not in CI [{lo:.4f}, {hi:.4f}]"

    def test_degenerate_ci_for_tiny_input(self):
        """Fewer observations than 2 blocks → degenerate CI (no crash)."""
        probs = np.array([0.6, 0.4])
        outcomes = np.array([1, 0])
        lo, hi = block_bootstrap_ece_ci(probs, outcomes, block_len=6, n_bootstrap=10)
        assert lo <= hi

    def test_reproducible_with_fixed_rng(self):
        probs, outcomes = _make_perfectly_calibrated(150)
        lo1, hi1 = block_bootstrap_ece_ci(
            probs, outcomes, n_bootstrap=50, rng=np.random.default_rng(7)
        )
        lo2, hi2 = block_bootstrap_ece_ci(
            probs, outcomes, n_bootstrap=50, rng=np.random.default_rng(7)
        )
        assert lo1 == lo2
        assert hi1 == hi2


# ---------------------------------------------------------------------------
# fit_calibration_model
# ---------------------------------------------------------------------------

class TestFitCalibrationModel:
    def test_returns_none_when_insufficient_data(self):
        y_hat = np.array([0.1, -0.2, 0.3])  # n=3 < min_obs_platt=20
        outcomes = np.array([1, 0, 1])
        model, result = fit_calibration_model(y_hat, outcomes, min_obs_platt=20)
        assert model is None
        assert result.method == "uncalibrated"
        assert result.n_obs == 3

    def test_returns_none_when_single_class(self):
        """All outcomes the same → LogisticRegression would fail."""
        y_hat = np.random.randn(50)
        outcomes = np.ones(50, dtype=int)  # All class 1
        model, result = fit_calibration_model(y_hat, outcomes, min_obs_platt=20)
        assert model is None
        assert result.method == "uncalibrated"

    def test_platt_returned_between_min_obs(self):
        y_hat, outcomes = _make_regression_scores(n=30, rng_seed=5)
        model, result = fit_calibration_model(
            y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=60
        )
        # n=30 is between 20 and 60 → Platt
        assert model is not None
        assert result.method == "platt"
        assert result.n_obs == 30

    def test_isotonic_returned_above_threshold(self):
        y_hat, outcomes = _make_regression_scores(n=100, rng_seed=7)
        model, result = fit_calibration_model(
            y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=60
        )
        assert isinstance(model, tuple)
        assert result.method == "isotonic"
        assert result.n_obs == 100

    def test_calibration_result_fields_are_finite(self):
        y_hat, outcomes = _make_regression_scores(n=80, rng_seed=3)
        _, result = fit_calibration_model(y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=60)
        assert np.isfinite(result.ece)
        assert np.isfinite(result.ece_ci_lower)
        assert np.isfinite(result.ece_ci_upper)

    def test_ece_in_unit_interval(self):
        y_hat, outcomes = _make_regression_scores(n=80, rng_seed=9)
        _, result = fit_calibration_model(y_hat, outcomes, min_obs_platt=20)
        assert 0.0 <= result.ece <= 1.0

    def test_ci_ordered(self):
        y_hat, outcomes = _make_regression_scores(n=80, rng_seed=11)
        _, result = fit_calibration_model(y_hat, outcomes, min_obs_platt=20)
        assert result.ece_ci_lower <= result.ece_ci_upper

    def test_platt_ece_not_worse_than_constant_half(self):
        """Calibrated ECE should be ≤ ECE of predicting 0.5 for everything."""
        y_hat, outcomes = _make_regression_scores(n=60, positive_signal=1.5, rng_seed=13)
        model, result = fit_calibration_model(
            y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=200
        )
        ece_constant = compute_ece(np.full(len(outcomes), 0.5), outcomes)
        # On training data the calibrated ECE should be ≤ constant baseline
        assert result.ece <= ece_constant + 0.05  # small tolerance for sampling noise


# ---------------------------------------------------------------------------
# calibrate_prediction
# ---------------------------------------------------------------------------

class TestCalibratePrediction:
    def test_none_model_returns_half(self):
        assert calibrate_prediction(None, 0.5) == 0.5
        assert calibrate_prediction(None, 10.0) == 0.5
        assert calibrate_prediction(None, -10.0) == 0.5

    def test_platt_positive_y_hat_gives_prob_above_half(self):
        y_hat, outcomes = _make_regression_scores(n=50, positive_signal=2.0, rng_seed=17)
        model, _ = fit_calibration_model(y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=200)
        assert model is not None
        # Strong positive score → calibrated prob > 0.5
        prob = calibrate_prediction(model, 2.0)
        assert prob > 0.5

    def test_platt_negative_y_hat_gives_prob_below_half(self):
        y_hat, outcomes = _make_regression_scores(n=50, positive_signal=2.0, rng_seed=17)
        model, _ = fit_calibration_model(y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=200)
        assert model is not None
        prob = calibrate_prediction(model, -2.0)
        assert prob < 0.5

    def test_isotonic_prediction_in_unit_interval(self):
        y_hat, outcomes = _make_regression_scores(n=100, positive_signal=1.5, rng_seed=19)
        model, _ = fit_calibration_model(y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=60)
        assert isinstance(model, tuple)
        for score in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            prob = calibrate_prediction(model, score)
            assert 0.0 <= prob <= 1.0, f"Prob {prob} out of [0,1] for score={score}"

    def test_platt_output_clipped_to_unit_interval(self):
        y_hat, outcomes = _make_regression_scores(n=50, positive_signal=0.5, rng_seed=21)
        model, _ = fit_calibration_model(y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=200)
        if model is not None:
            for score in [-100.0, 100.0]:
                prob = calibrate_prediction(model, score)
                assert 0.0 <= prob <= 1.0

    def test_monotone_in_score_direction(self):
        """Higher regression score should yield higher or equal calibrated probability."""
        y_hat, outcomes = _make_regression_scores(n=60, positive_signal=1.0, rng_seed=23)
        model, _ = fit_calibration_model(y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=200)
        if model is not None:
            probs = [calibrate_prediction(model, s) for s in [-2.0, -1.0, 0.0, 1.0, 2.0]]
            # Platt logistic is strictly monotone; isotonic may be flat but never decreasing
            assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1)), (
                f"Calibration not monotone in score: {probs}"
            )


# ---------------------------------------------------------------------------
# CalibrationResult dataclass
# ---------------------------------------------------------------------------

class TestCalibrationResultDataclass:
    def test_fields_exist(self):
        result = CalibrationResult(
            n_obs=100, method="platt",
            ece=0.05, ece_ci_lower=0.02, ece_ci_upper=0.09,
        )
        assert result.n_obs == 100
        assert result.method == "platt"
        assert result.ece == pytest.approx(0.05)
        assert result.ece_ci_lower == pytest.approx(0.02)
        assert result.ece_ci_upper == pytest.approx(0.09)

    def test_method_values(self):
        for method in ("uncalibrated", "platt", "isotonic"):
            r = CalibrationResult(
                n_obs=0, method=method, ece=0.0, ece_ci_lower=0.0, ece_ci_upper=1.0
            )
            assert r.method == method


# ---------------------------------------------------------------------------
# Integration: calibration improves ECE on training data
# ---------------------------------------------------------------------------

class TestCalibrationIntegration:
    def test_calibrated_ece_not_worse_than_raw_on_training_data(self):
        """
        After calibration, ECE on the training set should be ≤ raw ECE
        (calibration can only improve fit on training data).
        """
        y_hat, outcomes = _make_regression_scores(n=150, positive_signal=1.0, rng_seed=99)

        # Raw 'probabilities' from a simple sigmoid with no calibration
        raw_probs = 1 / (1 + np.exp(-y_hat))
        raw_ece = compute_ece(raw_probs, outcomes)

        model, result = fit_calibration_model(
            y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=60, n_bootstrap=50
        )
        # Calibrated ECE (on training data) should be ≤ raw ECE + small tolerance
        assert result.ece <= raw_ece + 0.05, (
            f"Calibrated ECE {result.ece:.4f} > raw ECE {raw_ece:.4f}"
        )

    def test_large_sample_uses_isotonic(self):
        y_hat, outcomes = _make_regression_scores(n=300, rng_seed=77)
        _, result = fit_calibration_model(
            y_hat, outcomes, min_obs_platt=20, min_obs_isotonic=60, n_bootstrap=50
        )
        assert result.method == "isotonic"
        assert result.n_obs == 300

    def test_config_defaults_used_when_no_overrides(self):
        """fit_calibration_model uses config values when defaults are None."""
        import config
        y_hat, outcomes = _make_regression_scores(n=10, rng_seed=55)
        # n=10 < config.CALIBRATION_MIN_OBS_PLATT → should be uncalibrated
        model, result = fit_calibration_model(y_hat, outcomes)
        assert result.n_obs == 10
        # method depends on whether 10 < CALIBRATION_MIN_OBS_PLATT
        if len(np.unique(outcomes)) < 2 or 10 < config.CALIBRATION_MIN_OBS_PLATT:
            assert result.method == "uncalibrated"
