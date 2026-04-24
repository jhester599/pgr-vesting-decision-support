"""Tests for x15 P/B regime overlay utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_pb_regime_targets_creates_up_down_and_neutral_labels() -> None:
    from src.research.x15_pb_regime_overlay import build_pb_regime_targets

    current_pb = pd.Series(
        [2.0, 2.0, 2.0],
        index=pd.date_range("2024-01-31", periods=3, freq="BME"),
        name="current_pb",
    )
    future_pb = pd.Series(
        [2.3, 1.8, 2.06],
        index=current_pb.index,
        name="future_pb",
    )

    targets = build_pb_regime_targets(
        current_pb,
        future_pb,
        hurdle=0.05,
    )

    assert targets["target_up"].tolist() == [1, 0, 0]
    assert targets["target_down"].tolist() == [0, 1, 0]
    assert targets["target_regime"].tolist() == ["up", "down", "neutral"]


def test_apply_pb_overlay_uses_bounded_fold_local_shifts() -> None:
    from src.research.x15_pb_regime_overlay import apply_pb_overlay

    predicted_pb, action = apply_pb_overlay(
        current_pb=2.0,
        up_prob=0.72,
        down_prob=0.18,
        positive_shift=0.10,
        negative_shift=-0.08,
        confidence_threshold=0.55,
    )

    assert predicted_pb == pytest.approx(2.2)
    assert action == "up"

    predicted_pb, action = apply_pb_overlay(
        current_pb=2.0,
        up_prob=0.28,
        down_prob=0.67,
        positive_shift=0.10,
        negative_shift=-0.08,
        confidence_threshold=0.55,
    )

    assert predicted_pb == pytest.approx(1.84)
    assert action == "down"

    predicted_pb, action = apply_pb_overlay(
        current_pb=2.0,
        up_prob=0.51,
        down_prob=0.43,
        positive_shift=0.10,
        negative_shift=-0.08,
        confidence_threshold=0.55,
    )

    assert predicted_pb == pytest.approx(2.0)
    assert action == "neutral"


def test_summarize_x15_results_ranks_lower_pb_mae_first() -> None:
    from src.research.x15_pb_regime_overlay import summarize_x15_results

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 6,
                "model_name": "no_change_pb_overlay",
                "pb_mae": 0.40,
                "pb_rmse": 0.55,
                "overlay_action_rate": 0.0,
            },
            {
                "horizon_months": 6,
                "model_name": "logistic_regime_overlay",
                "pb_mae": 0.35,
                "pb_rmse": 0.56,
                "overlay_action_rate": 0.4,
            },
        ]
    )

    summary = summarize_x15_results(detail)

    assert summary.iloc[0]["model_name"] == "logistic_regime_overlay"
    assert bool(summary.iloc[0]["beats_no_change_pb"]) is False


def test_evaluate_x15_overlay_runs_wfo_path() -> None:
    from scripts.research.x15_pb_regime_overlay import evaluate_x15_overlay

    dates = pd.date_range("2017-01-31", periods=84, freq="BME")
    X = pd.DataFrame(
        {
            "pb_ratio": pd.Series(range(84), index=dates) / 100.0 + 2.0,
            "real_rate_10y": pd.Series(range(84), index=dates) / 200.0,
            "current_pb": 2.0,
        },
        index=dates,
    )
    future_pb = pd.Series(
        [2.2 if idx % 3 == 0 else 1.8 if idx % 3 == 1 else 2.02 for idx in range(84)],
        index=dates,
    )
    regime_targets = pd.DataFrame(
        {
            "current_pb": 2.0,
            "future_pb": future_pb,
            "normalized_delta": future_pb / 2.0 - 1.0,
            "target_up": (future_pb > 2.1).astype(int),
            "target_down": (future_pb < 1.9).astype(int),
            "target_regime": [
                "up" if value > 2.1 else "down" if value < 1.9 else "neutral"
                for value in future_pb
            ],
        },
        index=dates,
    )

    predictions, metrics = evaluate_x15_overlay(
        X,
        regime_targets,
        model_name="logistic_l2_balanced",
        feature_columns=["pb_ratio", "real_rate_10y", "current_pb"],
        target_horizon_months=3,
    )

    assert not predictions.empty
    assert metrics["fold_count"] > 0
    assert set(predictions["overlay_action"]).issubset({"up", "down", "neutral"})
