from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "research"
        / "v125_portfolio_target_classifier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v125_portfolio_target_classifier",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_resolve_n_splits_respects_rolling_window_capacity() -> None:
    resolved = MODULE._resolve_n_splits(
        152,
        requested_n_splits=15,
        train_window_months=60,
        gap=8,
        test_size=6,
    )
    assert resolved == 14


def test_aggregate_probability_panel_renormalizes_missing_benchmarks() -> None:
    probability_df = pd.DataFrame(
        {
            "VOO": [0.60, 0.40],
            "VGT": [0.30, None],
            "BND": [0.20, 0.10],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    aggregated = MODULE._aggregate_probability_panel(
        probability_df,
        {"VOO": 0.40, "VGT": 0.20, "BND": 0.05},
    )

    first = aggregated.iloc[0]
    second = aggregated.iloc[1]

    expected_first = (0.60 * 0.40 + 0.30 * 0.20 + 0.20 * 0.05) / (0.40 + 0.20 + 0.05)
    expected_second = (0.40 * 0.40 + 0.10 * 0.05) / (0.40 + 0.05)

    assert first["path_a_prob"] == pytest.approx(expected_first)
    assert second["path_a_prob"] == pytest.approx(expected_second)
    assert int(second["path_a_available_benchmarks"]) == 2


def test_run_wfo_uses_max_train_window() -> None:
    X = pd.DataFrame(
        {"feature": list(range(20))},
        index=pd.date_range("2020-01-31", periods=20, freq="ME"),
    )
    y = pd.Series(
        [0, 1] * 10,
        index=X.index,
        name="target",
    )

    fold_df = MODULE.run_wfo(
        X,
        y,
        n_splits=99,
        gap=2,
        test_size=2,
        min_train_obs=5,
        train_window_months=5,
    )

    assert not fold_df.empty
    assert fold_df["fold"].nunique() == 6
    assert fold_df["train_obs"].max() == 5
    assert fold_df["train_obs"].min() == 5


def test_verdict_text_blocks_promotion_when_calibration_worsens() -> None:
    verdict = MODULE._verdict_text(
        {
            "balanced_accuracy_covered": 0.55,
            "brier_score": 0.18,
            "ece": 0.08,
        },
        {
            "balanced_accuracy_covered": 0.61,
            "brier_score": 0.24,
            "ece": 0.18,
        },
    )
    assert "secondary research track" in verdict
