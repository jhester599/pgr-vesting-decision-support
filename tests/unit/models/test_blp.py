"""
Tests for the Beta-Transformed Linear Pool (BLP) module.

Validates:
  - Data-availability guard (InsufficientDataError below threshold)
  - Parameter fitting convergence and output shapes
  - Mathematical properties: output bounds, monotonicity, equal-weight identity
  - Config constant presence
  - is_fitted() / pre-fit guard
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.models.blp import BLPModel, BLPParams, InsufficientDataError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_oos_probs(
    T: int,
    n_models: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic OOS probability matrix in (0, 1)."""
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.uniform(0.3, 0.8, size=(T, n_models))


def _make_oos_outcomes(T: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate binary outcomes (0/1)."""
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=T).astype(float)


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

class TestBLPConfigConstants:
    def test_blp_min_oos_months_exists(self) -> None:
        assert hasattr(config, "BLP_MIN_OOS_MONTHS")
        assert isinstance(config.BLP_MIN_OOS_MONTHS, int)
        assert config.BLP_MIN_OOS_MONTHS > 0

    def test_blp_n_params_is_five(self) -> None:
        assert config.BLP_N_PARAMS == 5

    def test_blp_beta_init_values_are_positive(self) -> None:
        assert config.BLP_BETA_A_INIT > 0
        assert config.BLP_BETA_B_INIT > 0

    def test_blp_weight_init_is_quarter(self) -> None:
        assert abs(config.BLP_WEIGHT_INIT - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# InsufficientDataError guard
# ---------------------------------------------------------------------------

class TestBLPDataGuard:
    def test_fit_raises_below_min_obs(self, rng: np.random.Generator) -> None:
        """fit() must raise InsufficientDataError when T < BLP_MIN_OOS_MONTHS."""
        T_short = config.BLP_MIN_OOS_MONTHS - 1
        probs = _make_oos_probs(T_short, rng=rng)
        outcomes = _make_oos_outcomes(T_short, rng=rng)

        model = BLPModel()
        with pytest.raises(InsufficientDataError):
            model.fit(probs, outcomes)

    def test_fit_raises_at_zero_obs(self) -> None:
        """fit() must raise InsufficientDataError with empty arrays."""
        model = BLPModel()
        with pytest.raises(InsufficientDataError):
            model.fit(np.zeros((0, 4)), np.zeros(0))

    def test_fit_raises_at_boundary(self, rng: np.random.Generator) -> None:
        """T == BLP_MIN_OOS_MONTHS − 1 must still raise."""
        T = config.BLP_MIN_OOS_MONTHS - 1
        model = BLPModel()
        with pytest.raises(InsufficientDataError):
            model.fit(_make_oos_probs(T, rng=rng), _make_oos_outcomes(T, rng=rng))

    def test_fit_succeeds_at_min_obs(self, rng: np.random.Generator) -> None:
        """T == BLP_MIN_OOS_MONTHS must succeed without raising."""
        T = config.BLP_MIN_OOS_MONTHS
        model = BLPModel()
        model.fit(_make_oos_probs(T, rng=rng), _make_oos_outcomes(T, rng=rng))
        assert model.is_fitted()

    def test_insufficient_data_error_message_contains_count(
        self, rng: np.random.Generator
    ) -> None:
        """Error message must state how many months remain."""
        T = 5
        model = BLPModel()
        with pytest.raises(InsufficientDataError, match=str(config.BLP_MIN_OOS_MONTHS - T)):
            model.fit(_make_oos_probs(T, rng=rng), _make_oos_outcomes(T, rng=rng))

    def test_custom_min_obs_respected(self, rng: np.random.Generator) -> None:
        """BLPModel(min_obs=5) should accept T >= 5 regardless of config value."""
        T = 5
        model = BLPModel(min_obs=T)
        model.fit(_make_oos_probs(T, rng=rng), _make_oos_outcomes(T, rng=rng))
        assert model.is_fitted()


# ---------------------------------------------------------------------------
# Fitting output properties
# ---------------------------------------------------------------------------

class TestBLPFitOutputs:
    def _fitted_model(self, rng: np.random.Generator) -> BLPModel:
        T = config.BLP_MIN_OOS_MONTHS
        model = BLPModel()
        model.fit(_make_oos_probs(T, rng=rng), _make_oos_outcomes(T, rng=rng))
        return model

    def test_params_a_b_positive(self, rng: np.random.Generator) -> None:
        model = self._fitted_model(rng)
        assert model.params_.a > 0
        assert model.params_.b > 0

    def test_weights_sum_to_one(self, rng: np.random.Generator) -> None:
        model = self._fitted_model(rng)
        assert abs(sum(model.params_.weights) - 1.0) < 1e-6

    def test_weights_all_positive(self, rng: np.random.Generator) -> None:
        """Softmax parameterisation guarantees strictly positive weights."""
        model = self._fitted_model(rng)
        assert all(w > 0 for w in model.params_.weights)

    def test_weights_length_equals_n_models(self, rng: np.random.Generator) -> None:
        model = self._fitted_model(rng)
        assert len(model.params_.weights) == model.n_models

    def test_blp_params_dataclass(self, rng: np.random.Generator) -> None:
        model = self._fitted_model(rng)
        assert isinstance(model.params_, BLPParams)
        assert model.params_.n_models == 4

    def test_is_fitted_true_after_fit(self, rng: np.random.Generator) -> None:
        model = self._fitted_model(rng)
        assert model.is_fitted() is True

    def test_is_fitted_false_before_fit(self) -> None:
        model = BLPModel()
        assert model.is_fitted() is False

    def test_n_models_default_is_four(self) -> None:
        assert BLPModel().n_models == 4

    def test_wrong_n_models_raises_value_error(self, rng: np.random.Generator) -> None:
        """oos_probs with K ≠ n_models must raise ValueError."""
        T = config.BLP_MIN_OOS_MONTHS
        model = BLPModel(n_models=3)
        with pytest.raises(ValueError, match="n_models"):
            model.fit(_make_oos_probs(T, n_models=4, rng=rng), _make_oos_outcomes(T, rng=rng))


# ---------------------------------------------------------------------------
# Transform output properties
# ---------------------------------------------------------------------------

class TestBLPTransform:
    def _fitted_model(self, rng: np.random.Generator) -> BLPModel:
        T = config.BLP_MIN_OOS_MONTHS
        model = BLPModel()
        model.fit(_make_oos_probs(T, rng=rng), _make_oos_outcomes(T, rng=rng))
        return model

    def test_transform_raises_before_fit(self) -> None:
        model = BLPModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.transform(np.array([0.5, 0.6, 0.4, 0.7]))

    def test_transform_scalar_input_returns_float(
        self, rng: np.random.Generator
    ) -> None:
        model = self._fitted_model(rng)
        result = model.transform(np.array([0.5, 0.6, 0.4, 0.7]))
        assert isinstance(result, float)

    def test_transform_batch_input_returns_array(
        self, rng: np.random.Generator
    ) -> None:
        model = self._fitted_model(rng)
        T = 10
        result = model.transform(_make_oos_probs(T, rng=rng))
        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_transform_output_in_unit_interval(
        self, rng: np.random.Generator
    ) -> None:
        """BLP output must always be in [0, 1]."""
        model = self._fitted_model(rng)
        T = 50
        result = model.transform(_make_oos_probs(T, rng=rng))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_transform_monotone_in_input(self, rng: np.random.Generator) -> None:
        """Higher input probabilities should generally map to higher BLP output.

        Under equal weights (a=b=1 initial → identity transform), a higher
        linear pool value directly maps to a higher BLP value.  This property
        should hold after fitting on any reasonable dataset.
        """
        model = self._fitted_model(rng)
        low_probs = np.full((1, 4), 0.3)
        high_probs = np.full((1, 4), 0.7)
        assert model.transform(high_probs) > model.transform(low_probs)

    def test_transform_equal_inputs_deterministic(
        self, rng: np.random.Generator
    ) -> None:
        """Same input must produce the same output on repeated calls."""
        model = self._fitted_model(rng)
        x = np.array([0.55, 0.60, 0.50, 0.65])
        assert model.transform(x) == model.transform(x)

    def test_equal_weight_beta11_approx_identity(self) -> None:
        """With a=b=1 (uniform Beta) and equal weights, BLP ≈ linear pool.

        Beta(1, 1) is the uniform distribution; its CDF is the identity
        function on [0, 1].  The BLP with equal weights and Beta(1,1) should
        therefore return approximately the mean of the input probabilities.
        """
        model = BLPModel(n_models=4)
        model.params_ = BLPParams(
            a=1.0,
            b=1.0,
            weights=[0.25, 0.25, 0.25, 0.25],
        )
        x = np.array([0.40, 0.60, 0.50, 0.70])
        expected_mean = float(np.mean(x))
        result = model.transform(x)
        assert abs(result - expected_mean) < 1e-6

    def test_transform_clips_extreme_inputs(
        self, rng: np.random.Generator
    ) -> None:
        """transform() must not raise on 0.0 or 1.0 probability inputs."""
        model = self._fitted_model(rng)
        extreme = np.array([0.0, 1.0, 0.0, 1.0])
        result = model.transform(extreme)
        assert 0.0 <= result <= 1.0
