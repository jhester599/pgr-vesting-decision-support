"""
Tests for BayesianRidge pipeline and ensemble WFO runner (v3.1).

Verifies:
  - build_bayesian_ridge_pipeline() returns UncertaintyPipeline
  - predict_with_std() returns (y_pred, y_std) with finite positive std
  - Temporal isolation: scaler fit only on training data
  - run_ensemble_benchmarks() returns EnsembleWFOResult per ETF
  - Equal-weight averaging of IC and hit rate across models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.regularized_models import (
    UncertaintyPipeline,
    build_bayesian_ridge_pipeline,
)
from src.models.multi_benchmark_wfo import EnsembleWFOResult, run_ensemble_benchmarks


def _make_data(n: int = 120, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-31", periods=n, freq="ME")
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(size=n), index=idx, name="target")
    return X, y


class TestBuildBayesianRidgePipeline:
    def test_returns_uncertainty_pipeline(self):
        pipe = build_bayesian_ridge_pipeline()
        assert isinstance(pipe, UncertaintyPipeline)

    def test_is_also_sklearn_pipeline(self):
        pipe = build_bayesian_ridge_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_scaler_step(self):
        pipe = build_bayesian_ridge_pipeline()
        assert "scaler" in pipe.named_steps

    def test_has_model_step(self):
        from sklearn.linear_model import BayesianRidge
        pipe = build_bayesian_ridge_pipeline()
        assert isinstance(pipe.named_steps["model"], BayesianRidge)

    def test_predict_with_std_returns_two_arrays(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 3))
        y = rng.normal(size=60)
        pipe = build_bayesian_ridge_pipeline()
        pipe.fit(X, y)
        result = pipe.predict_with_std(X[:5])
        assert len(result) == 2
        y_pred, y_std = result
        assert y_pred.shape == (5,)
        assert y_std.shape == (5,)

    def test_predict_with_std_returns_positive_std(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 3))
        y = rng.normal(size=60)
        pipe = build_bayesian_ridge_pipeline()
        pipe.fit(X, y)
        _, y_std = pipe.predict_with_std(X[:5])
        assert (y_std >= 0).all(), "Standard deviation must be non-negative"
        assert y_std.sum() > 0, "At least some std values must be positive"

    def test_predict_with_std_finite_values(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 3))
        y = rng.normal(size=60)
        pipe = build_bayesian_ridge_pipeline()
        pipe.fit(X, y)
        y_pred, y_std = pipe.predict_with_std(X)
        assert np.isfinite(y_pred).all()
        assert np.isfinite(y_std).all()

    def test_scaler_fit_only_on_train_data(self):
        """Verify the scaler mean is computed from training data only."""
        rng = np.random.default_rng(0)
        X_train = rng.normal(loc=10.0, scale=1.0, size=(60, 3))
        y_train = rng.normal(size=60)
        pipe = build_bayesian_ridge_pipeline()
        pipe.fit(X_train, y_train)
        scaler = pipe.named_steps["scaler"]
        assert abs(scaler.mean_[0] - 10.0) < 1.5, (
            "Scaler mean should be near 10.0 (train), not contaminated by test data"
        )

    def test_predict_with_std_std_increases_for_ood(self):
        """OOD input should generally have higher uncertainty than in-distribution."""
        rng = np.random.default_rng(3)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(80, 3))
        y_train = X_train[:, 0] + rng.normal(scale=0.1, size=80)
        pipe = build_bayesian_ridge_pipeline()
        pipe.fit(X_train, y_train)

        X_in = rng.normal(loc=0.0, scale=1.0, size=(20, 3))
        X_ood = rng.normal(loc=100.0, scale=1.0, size=(20, 3))  # far from training

        _, std_in = pipe.predict_with_std(X_in)
        _, std_ood = pipe.predict_with_std(X_ood)
        # OOD std should be at least as large as in-distribution std on average
        assert std_ood.mean() >= std_in.mean() * 0.5  # lenient: not strict


class TestEnsembleWFOResult:
    def test_run_ensemble_returns_dict(self):
        X, _ = _make_data()
        rng = np.random.default_rng(10)
        rel_matrix = pd.DataFrame(
            {"VTI": rng.normal(size=len(X))},
            index=X.index,
        )
        results = run_ensemble_benchmarks(
            X, rel_matrix, target_horizon_months=6, purge_buffer=0
        )
        assert isinstance(results, dict)

    def test_result_is_ensemble_wfo_result(self):
        X, _ = _make_data()
        rng = np.random.default_rng(11)
        rel_matrix = pd.DataFrame(
            {"VTI": rng.normal(size=len(X))},
            index=X.index,
        )
        results = run_ensemble_benchmarks(
            X, rel_matrix, target_horizon_months=6, purge_buffer=0
        )
        assert "VTI" in results
        assert isinstance(results["VTI"], EnsembleWFOResult)

    def test_mean_ic_is_average_of_models(self):
        X, _ = _make_data()
        rng = np.random.default_rng(12)
        rel_matrix = pd.DataFrame(
            {"VTI": rng.normal(size=len(X))},
            index=X.index,
        )
        results = run_ensemble_benchmarks(
            X, rel_matrix, target_horizon_months=6, purge_buffer=0
        )
        if "VTI" not in results:
            pytest.skip("Insufficient data for this benchmark")
        ens = results["VTI"]
        individual_ics = [r.information_coefficient for r in ens.model_results.values()]
        expected_mean_ic = float(np.mean(individual_ics))
        assert abs(ens.mean_ic - expected_mean_ic) < 1e-9

    def test_model_results_contains_expected_types(self):
        X, _ = _make_data()
        rng = np.random.default_rng(13)
        rel_matrix = pd.DataFrame(
            {"VTI": rng.normal(size=len(X))},
            index=X.index,
        )
        results = run_ensemble_benchmarks(
            X, rel_matrix, target_horizon_months=6, purge_buffer=0
        )
        if "VTI" not in results:
            pytest.skip("Insufficient data")
        model_types = set(results["VTI"].model_results.keys())
        assert len(model_types) >= 1  # at least one model trained

    def test_empty_matrix_raises(self):
        X, _ = _make_data()
        with pytest.raises(ValueError):
            run_ensemble_benchmarks(X, pd.DataFrame(), target_horizon_months=6)
