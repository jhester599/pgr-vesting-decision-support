"""
Tests for ElasticNetCV pipeline in src/models/regularized_models.py (v3.0).

Verifies pipeline structure, l1_ratio grid, temporal isolation of StandardScaler,
and that the model type is accepted by run_wfo().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.regularized_models import (
    build_elasticnet_pipeline,
    get_feature_importances,
)
from src.models.wfo_engine import run_wfo


class TestBuildElasticnetPipeline:
    def test_returns_pipeline(self):
        pipe = build_elasticnet_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_scaler_step(self):
        pipe = build_elasticnet_pipeline()
        assert "scaler" in pipe.named_steps
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)

    def test_has_model_step(self):
        pipe = build_elasticnet_pipeline()
        assert "model" in pipe.named_steps
        assert isinstance(pipe.named_steps["model"], ElasticNetCV)

    def test_default_l1_ratios(self):
        pipe = build_elasticnet_pipeline()
        enet = pipe.named_steps["model"]
        assert list(enet.l1_ratio) == [0.1, 0.5, 0.9, 0.95, 1.0]

    def test_custom_l1_ratios(self):
        pipe = build_elasticnet_pipeline(l1_ratios=[0.5, 1.0])
        enet = pipe.named_steps["model"]
        assert list(enet.l1_ratio) == [0.5, 1.0]

    def test_default_alphas_count(self):
        pipe = build_elasticnet_pipeline()
        enet = pipe.named_steps["model"]
        assert len(enet.alphas) == 50

    def test_pipeline_fits_and_predicts(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 5))
        y = rng.normal(size=80)

        pipe = build_elasticnet_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X[:5])
        assert preds.shape == (5,)

    def test_scaler_fit_only_on_train_data(self):
        """Verify the scaler mean is computed from training data only."""
        rng = np.random.default_rng(0)
        X_train = rng.normal(loc=10.0, scale=1.0, size=(60, 3))
        X_test = rng.normal(loc=0.0, scale=1.0, size=(10, 3))
        y_train = rng.normal(size=60)

        pipe = build_elasticnet_pipeline()
        pipe.fit(X_train, y_train)

        scaler = pipe.named_steps["scaler"]
        # Scaler mean should be near 10.0 (from training), not near 0.0 (from test)
        assert abs(scaler.mean_[0] - 10.0) < 1.5


class TestElasticnetInWFO:
    def test_run_wfo_accepts_elasticnet(self):
        """run_wfo() should accept model_type='elasticnet' without error."""
        rng = np.random.default_rng(7)
        n = 100
        idx = pd.date_range("2015-01-31", periods=n, freq="ME")
        X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx,
                         columns=["f1", "f2", "f3", "f4"])
        y = pd.Series(rng.normal(size=n), index=idx, name="target")

        result = run_wfo(X, y, model_type="elasticnet",
                         target_horizon_months=6, purge_buffer=0)
        assert result.model_type == "elasticnet"
        assert len(result.folds) >= 1

    def test_elasticnet_result_has_coefficients(self):
        rng = np.random.default_rng(3)
        n = 100
        idx = pd.date_range("2015-01-31", periods=n, freq="ME")
        X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx,
                         columns=["a", "b", "c"])
        y = pd.Series(rng.normal(size=n), index=idx, name="target")

        result = run_wfo(X, y, model_type="elasticnet",
                         target_horizon_months=6, purge_buffer=0)

        for fold in result.folds:
            assert isinstance(fold.feature_importances, dict)
            assert len(fold.feature_importances) == 3


class TestGetFeatureImportances:
    def test_returns_sorted_by_abs(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 3))
        y = X[:, 0] * 2 + rng.normal(size=60) * 0.1  # first feature dominates

        pipe = build_elasticnet_pipeline()
        pipe.fit(X, y)

        importances = get_feature_importances(pipe, ["strong", "weak1", "weak2"])
        keys = list(importances.keys())
        vals = [abs(importances[k]) for k in keys]
        assert vals == sorted(vals, reverse=True)
