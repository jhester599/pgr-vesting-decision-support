"""Tests for x6 special-dividend sidecar utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _annual_frame() -> pd.DataFrame:
    dates = pd.to_datetime([f"{year}-11-30" for year in range(2008, 2024)])
    signal = np.linspace(-1.0, 1.0, len(dates))
    occurred = (signal > -0.2).astype(int)
    excess = np.where(occurred == 1, 0.5 + signal.clip(min=0.0), 0.0)
    return pd.DataFrame(
        {
            "capital_signal": signal,
            "profitability_signal": signal[::-1],
            "special_dividend_occurred": occurred,
            "special_dividend_excess": excess,
        },
        index=dates,
    )


def test_annual_expanding_splits_use_only_prior_years() -> None:
    from src.research.x6_special_dividend import iter_annual_expanding_splits

    frame = _annual_frame()
    splits = list(iter_annual_expanding_splits(frame, min_train_years=6))

    assert splits
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)
        assert len(test_idx) == 1
        assert frame.index[train_idx].max() < frame.index[test_idx].min()


def test_two_stage_historical_baseline_is_fold_local() -> None:
    from src.research.x6_special_dividend import (
        evaluate_special_dividend_two_stage,
    )

    frame = _annual_frame()

    predictions, metrics = evaluate_special_dividend_two_stage(
        frame,
        feature_columns=["capital_signal", "profitability_signal"],
        stage1_model_name="historical_rate",
        stage2_model_name="historical_positive_mean",
        min_train_years=8,
    )

    first = predictions.iloc[0]
    train = frame.iloc[:8]
    assert first["stage1_probability"] == pytest.approx(
        train["special_dividend_occurred"].mean()
    )
    positives = train[train["special_dividend_occurred"] == 1]
    assert first["conditional_size_prediction"] == pytest.approx(
        positives["special_dividend_excess"].mean()
    )
    assert first["expected_value_prediction"] == pytest.approx(
        first["stage1_probability"] * first["conditional_size_prediction"]
    )
    assert metrics["stage1_model_name"] == "historical_rate"
    assert metrics["stage2_model_name"] == "historical_positive_mean"


def test_two_stage_model_outputs_bounded_probabilities_and_nonnegative_size() -> None:
    from src.research.x6_special_dividend import (
        evaluate_special_dividend_two_stage,
    )

    frame = _annual_frame()

    predictions, metrics = evaluate_special_dividend_two_stage(
        frame,
        feature_columns=["capital_signal", "profitability_signal"],
        stage1_model_name="logistic_l2_balanced",
        stage2_model_name="ridge_positive_excess",
        min_train_years=8,
    )

    assert predictions["stage1_probability"].between(0.0, 1.0).all()
    assert predictions["conditional_size_prediction"].ge(0.0).all()
    assert predictions["expected_value_prediction"].ge(0.0).all()
    assert metrics["n_obs"] == len(predictions)


def test_stage2_ridge_size_is_capped_to_prior_positive_range() -> None:
    from src.research.x6_special_dividend import (
        evaluate_special_dividend_two_stage,
    )

    frame = _annual_frame()

    predictions, _ = evaluate_special_dividend_two_stage(
        frame,
        feature_columns=["capital_signal", "profitability_signal"],
        stage1_model_name="historical_rate",
        stage2_model_name="ridge_positive_excess",
        min_train_years=8,
    )

    for _, row in predictions.iterrows():
        prior = frame.loc[
            (frame.index >= row["train_start_date"])
            & (frame.index <= row["train_end_date"])
        ]
        prior_positive = prior.loc[
            prior["special_dividend_occurred"] == 1,
            "special_dividend_excess",
        ]
        if not prior_positive.empty:
            assert row["conditional_size_prediction"] <= prior_positive.max() + 1e-12


def test_special_dividend_summary_ranks_lower_expected_value_mae() -> None:
    from src.research.x6_special_dividend import summarize_special_dividend_results

    detail = pd.DataFrame(
        [
            {
                "stage1_model_name": "historical_rate",
                "stage2_model_name": "historical_positive_mean",
                "expected_value_mae": 0.4,
                "stage1_brier": 0.30,
            },
            {
                "stage1_model_name": "logistic_l2_balanced",
                "stage2_model_name": "ridge_positive_excess",
                "expected_value_mae": 0.3,
                "stage1_brier": 0.35,
            },
        ]
    )

    summary = summarize_special_dividend_results(detail)

    assert summary.iloc[0]["stage1_model_name"] == "logistic_l2_balanced"
    assert summary.iloc[0]["rank"] == 1
