from __future__ import annotations

import pandas as pd
import pytest

from scripts.confirmatory_classifier_experiments import summarize_confirmatory_classifier_results


def test_summarize_confirmatory_classifier_results_groups_rows():
    detail = pd.DataFrame(
        [
            {
                "benchmark": "VXUS",
                "candidate_name": "ridge_lean_v1",
                "model_type": "ridge",
                "n_features": 12,
                "policy_name": "hybrid_confirm_0.55",
                "threshold": 0.55,
                "notes": "lean",
                "regression_ic": 0.10,
                "regression_oos_r2": -0.20,
                "brier_score": 0.24,
                "accuracy": 0.60,
                "balanced_accuracy": 0.58,
                "precision": 0.62,
                "recall": 0.56,
                "avg_hold_fraction": 0.40,
                "mean_policy_return": 0.03,
                "uplift_vs_sell_50": 0.01,
                "uplift_vs_regression_sign": 0.005,
            },
            {
                "benchmark": "BND",
                "candidate_name": "ridge_lean_v1",
                "model_type": "ridge",
                "n_features": 12,
                "policy_name": "hybrid_confirm_0.55",
                "threshold": 0.55,
                "notes": "lean",
                "regression_ic": 0.12,
                "regression_oos_r2": -0.18,
                "brier_score": 0.22,
                "accuracy": 0.64,
                "balanced_accuracy": 0.61,
                "precision": 0.66,
                "recall": 0.58,
                "avg_hold_fraction": 0.35,
                "mean_policy_return": 0.04,
                "uplift_vs_sell_50": 0.02,
                "uplift_vs_regression_sign": 0.007,
            },
        ]
    )
    summary = summarize_confirmatory_classifier_results(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["candidate_name"] == "ridge_lean_v1"
    assert row["n_benchmarks"] == 2
    assert row["mean_brier_score"] == pytest.approx(0.23)
    assert row["mean_policy_return"] == pytest.approx(0.035)
    assert row["mean_uplift_vs_regression_sign"] == pytest.approx(0.006)
