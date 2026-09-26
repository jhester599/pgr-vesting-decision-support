from __future__ import annotations

import pandas as pd
import pytest

from scripts.classifier_feature_selection import (
    _choose_final_recommendation,
    summarize_classifier_feature_selection,
)


def test_summarize_classifier_feature_selection_groups_rows():
    detail = pd.DataFrame(
        [
            {
                "benchmark": "VXUS",
                "experiment_mode": "single_feature",
                "step": 1,
                "candidate_feature": "combined_ratio_ttm",
                "policy_name": "hybrid_confirm",
                "threshold": 0.55,
                "feature_set_key": "combined_ratio_ttm",
                "n_features": 1,
                "feature_columns": "combined_ratio_ttm",
                "regression_ic": 0.10,
                "regression_oos_r2": -0.20,
                "brier_score": 0.24,
                "accuracy": 0.60,
                "balanced_accuracy": 0.61,
                "precision": 0.62,
                "recall": 0.58,
                "mean_policy_return": 0.03,
                "uplift_vs_sell_50": 0.01,
                "uplift_vs_regression_sign": -0.01,
                "avg_hold_fraction": 0.35,
            },
            {
                "benchmark": "BND",
                "experiment_mode": "single_feature",
                "step": 1,
                "candidate_feature": "combined_ratio_ttm",
                "policy_name": "hybrid_confirm",
                "threshold": 0.55,
                "feature_set_key": "combined_ratio_ttm",
                "n_features": 1,
                "feature_columns": "combined_ratio_ttm",
                "regression_ic": 0.12,
                "regression_oos_r2": -0.18,
                "brier_score": 0.22,
                "accuracy": 0.62,
                "balanced_accuracy": 0.63,
                "precision": 0.64,
                "recall": 0.60,
                "mean_policy_return": 0.04,
                "uplift_vs_sell_50": 0.02,
                "uplift_vs_regression_sign": -0.005,
                "avg_hold_fraction": 0.40,
            },
        ]
    )
    summary = summarize_classifier_feature_selection(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["mean_balanced_accuracy"] == pytest.approx(0.62)
    assert row["mean_uplift_vs_regression_sign"] == pytest.approx(-0.0075)


def test_choose_final_recommendation_prefers_smaller_near_best_set():
    trace = pd.DataFrame(
        [
            {
                "step": 1,
                "n_features": 1,
                "mean_uplift_vs_regression_sign": -0.010,
                "mean_balanced_accuracy": 0.61,
                "mean_brier_score": 0.24,
                "feature_columns": "combined_ratio_ttm",
                "threshold": 0.55,
            },
            {
                "step": 2,
                "n_features": 2,
                "mean_uplift_vs_regression_sign": -0.009,
                "mean_balanced_accuracy": 0.615,
                "mean_brier_score": 0.23,
                "feature_columns": "combined_ratio_ttm,npw_growth_yoy",
                "threshold": 0.55,
            },
            {
                "step": 3,
                "n_features": 3,
                "mean_uplift_vs_regression_sign": -0.008,
                "mean_balanced_accuracy": 0.619,
                "mean_brier_score": 0.22,
                "feature_columns": "combined_ratio_ttm,npw_growth_yoy,credit_spread_hy",
                "threshold": 0.55,
            },
        ]
    )
    winner = _choose_final_recommendation(trace)
    assert winner["n_features"] == 1
