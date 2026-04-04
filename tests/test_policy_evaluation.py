from __future__ import annotations

import pandas as pd

from scripts.policy_evaluation import summarize_policy_evaluation
from src.research.policy_metrics import (
    evaluate_hold_fraction_series,
    evaluate_policy_series,
    hold_fraction_from_policy,
)


def test_hold_fraction_from_policy_maps_expected_actions():
    predicted = pd.Series([0.20, 0.08, -0.03], index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    hold_fraction = hold_fraction_from_policy(predicted, "tiered_25_50_100")
    assert hold_fraction.tolist() == [0.75, 0.5, 0.0]


def test_evaluate_policy_series_reports_uplift_vs_sell_all():
    predicted = pd.Series([0.2, -0.1], index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    realized = pd.Series([0.1, -0.2], index=predicted.index)
    summary = evaluate_policy_series(predicted, realized, "sign_hold_vs_sell")
    assert summary.n_obs == 2
    assert summary.mean_policy_return == 0.05
    assert summary.uplift_vs_sell_all == 0.05


def test_evaluate_hold_fraction_series_supports_custom_gate():
    hold_fraction = pd.Series([1.0, 0.0], index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    realized = pd.Series([0.08, -0.04], index=hold_fraction.index)
    summary = evaluate_hold_fraction_series(hold_fraction, realized)
    assert summary.n_obs == 2
    assert summary.mean_policy_return == 0.04
    assert summary.avg_hold_fraction == 0.5
    assert summary.uplift_vs_sell_50 == 0.03


def test_summarize_policy_evaluation_groups_rows():
    detail = pd.DataFrame(
        [
            {
                "universe_name": "full21",
                "benchmark": "VTI",
                "horizon_months": 6,
                "candidate_type": "model",
                "candidate_name": "gbt",
                "policy_name": "sign_hold_vs_sell",
                "mean_policy_return": 0.02,
                "positive_utility_rate": 0.60,
                "regret_vs_oracle": 0.04,
                "uplift_vs_sell_all": 0.02,
                "uplift_vs_sell_50": -0.01,
                "uplift_vs_hold_all": -0.03,
                "capture_ratio": 0.50,
                "avg_hold_fraction": 0.40,
            },
            {
                "universe_name": "full21",
                "benchmark": "BND",
                "horizon_months": 6,
                "candidate_type": "model",
                "candidate_name": "gbt",
                "policy_name": "sign_hold_vs_sell",
                "mean_policy_return": 0.04,
                "positive_utility_rate": 0.70,
                "regret_vs_oracle": 0.03,
                "uplift_vs_sell_all": 0.04,
                "uplift_vs_sell_50": 0.00,
                "uplift_vs_hold_all": -0.02,
                "capture_ratio": 0.55,
                "avg_hold_fraction": 0.45,
            },
        ]
    )
    summary = summarize_policy_evaluation(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_benchmarks"] == 2
    assert row["mean_policy_return"] == 0.03
