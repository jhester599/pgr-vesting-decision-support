"""Tests for scripts/target_experiments.py and binary research metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts.target_experiments import _transform_target, summarize_target_experiments
from src.research.evaluation import summarize_binary_predictions


def test_transform_target_binary_outperform():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.Series([-0.01, 0.00, 0.02, 0.05], index=idx, name="rel")
    result = _transform_target(y, "binary_outperform")
    assert result.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_transform_target_thresholded_outperform():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.Series([-0.01, 0.02, 0.03, 0.05], index=idx, name="rel")
    result = _transform_target(y, "binary_outperform_3pct")
    assert result.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_summarize_binary_predictions_returns_expected_metrics():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    y_score = pd.Series([0.1, 0.7, 0.9, 0.4, 0.8], index=idx)
    y_true = pd.Series([0, 1, 1, 0, 1], index=idx)
    summary = summarize_binary_predictions(y_score, y_true)
    assert summary.n_obs == 5
    assert summary.accuracy >= 0.8
    assert 0.0 <= summary.brier_score <= 1.0


def test_summarize_target_experiments_aggregates_binary_fields():
    detail = pd.DataFrame(
        [
            {
                "target_variant": "binary_outperform",
                "item_type": "model",
                "item_name": "elasticnet",
                "benchmark": "VTI",
                "horizon_months": 6,
                "ic": 0.06,
                "hit_rate": 0.55,
                "oos_r2": 0.01,
                "mae": 0.44,
                "brier_score": 0.21,
                "accuracy": 0.58,
                "balanced_accuracy": 0.57,
                "precision": 0.60,
                "recall": 0.56,
                "base_rate": 0.52,
                "predicted_positive_rate": 0.54,
                "gate_status": "MARGINAL",
            }
        ]
    )
    summary = summarize_target_experiments(detail)
    assert "mean_brier_score" in summary.columns
    assert summary.iloc[0]["target_variant"] == "binary_outperform"

