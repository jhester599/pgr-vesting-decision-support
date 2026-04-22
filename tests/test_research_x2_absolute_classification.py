"""Tests for x2 absolute PGR classification research utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _feature_frame(n_rows: int = 96) -> pd.DataFrame:
    dates = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    x1 = np.linspace(-1.0, 1.0, n_rows)
    x2 = np.sin(np.linspace(0.0, 8.0, n_rows))
    return pd.DataFrame({"signal": x1, "cycle": x2}, index=dates)


def _binary_target(X: pd.DataFrame) -> pd.Series:
    values = ((X["signal"] + 0.25 * X["cycle"]) > 0.0).astype(int)
    values.name = "target_1m_up"
    return values


def test_evaluate_absolute_classifier_returns_chronological_probabilities() -> None:
    from src.research.x2_absolute_classification import evaluate_absolute_classifier

    X = _feature_frame()
    y = _binary_target(X)

    predictions, metrics = evaluate_absolute_classifier(
        X,
        y,
        model_name="logistic_l2_balanced",
        feature_columns=["signal", "cycle"],
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert not predictions.empty
    assert predictions["date"].is_monotonic_increasing
    assert predictions["y_prob"].between(0.0, 1.0).all()
    assert metrics["model_name"] == "logistic_l2_balanced"
    assert metrics["n_obs"] == len(predictions)
    assert metrics["balanced_accuracy"] >= 0.5
    assert metrics["fold_count"] > 0


def test_evaluate_absolute_baseline_uses_fold_local_history() -> None:
    from src.research.x2_absolute_classification import evaluate_absolute_baseline

    X = _feature_frame()
    y = _binary_target(X)

    predictions, metrics = evaluate_absolute_baseline(
        X,
        y,
        baseline_name="base_rate",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert not predictions.empty
    assert predictions["y_prob"].between(0.0, 1.0).all()
    assert predictions["y_prob"].nunique() > 1
    assert metrics["model_name"] == "base_rate"
    assert metrics["n_obs"] == len(predictions)


def test_always_up_baseline_is_constant_positive_probability() -> None:
    from src.research.x2_absolute_classification import evaluate_absolute_baseline

    X = _feature_frame()
    y = _binary_target(X)

    predictions, metrics = evaluate_absolute_baseline(
        X,
        y,
        baseline_name="always_up",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert predictions["y_prob"].eq(1.0).all()
    assert metrics["predicted_positive_rate"] == 1.0


def test_summary_ranks_model_rows_before_weaker_baseline() -> None:
    from src.research.x2_absolute_classification import (
        summarize_absolute_classification_results,
    )

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 1,
                "model_name": "base_rate",
                "n_obs": 30,
                "balanced_accuracy": 0.50,
                "brier_score": 0.25,
                "log_loss": 0.70,
            },
            {
                "horizon_months": 1,
                "model_name": "logistic_l2_balanced",
                "n_obs": 30,
                "balanced_accuracy": 0.62,
                "brier_score": 0.20,
                "log_loss": 0.60,
            },
        ]
    )

    summary = summarize_absolute_classification_results(detail)

    assert summary.iloc[0]["model_name"] == "logistic_l2_balanced"
    assert summary.iloc[0]["rank"] == 1
    assert bool(summary.iloc[0]["beats_base_rate"]) is True


def test_unsupported_classifier_name_raises() -> None:
    from src.research.x2_absolute_classification import (
        build_absolute_classifier_pipeline,
    )

    with pytest.raises(ValueError, match="Unsupported classifier"):
        build_absolute_classifier_pipeline("wide_net")
