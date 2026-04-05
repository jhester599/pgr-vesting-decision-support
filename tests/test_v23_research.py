from __future__ import annotations

import pandas as pd

from src.research.v23 import (
    V23_REVIEW_PATHS,
    choose_v23_decision,
    fit_proxy_blend_weights,
    stitch_proxy_series,
)


def test_v23_review_paths_include_leading_candidates() -> None:
    assert V23_REVIEW_PATHS == (
        "live_production_ensemble_reduced",
        "ensemble_ridge_gbt_v18",
        "ensemble_ridge_gbt_v20_best",
    )


def test_fit_proxy_blend_weights_returns_non_negative_sum_to_one() -> None:
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    vea = pd.Series([0.02, 0.01, 0.03, 0.02], index=idx)
    vwo = pd.Series([0.04, 0.00, 0.02, 0.05], index=idx)
    target = 0.7 * vea + 0.3 * vwo

    weights = fit_proxy_blend_weights(target, {"VEA": vea, "VWO": vwo})
    assert weights["VEA"] >= 0.0
    assert weights["VWO"] >= 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_stitch_proxy_series_only_fills_pre_inception_gap() -> None:
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    actual = pd.Series([float("nan"), float("nan"), 0.02, 0.03], index=idx)
    proxy = pd.Series([0.01, 0.02, 0.50, 0.60], index=idx)

    stitched = stitch_proxy_series(actual, proxy)
    assert stitched.iloc[0] == 0.01
    assert stitched.iloc[1] == 0.02
    assert stitched.iloc[2] == 0.02
    assert stitched.iloc[3] == 0.03


def test_choose_v23_decision_confirms_candidate_when_extended_history_holds() -> None:
    metric_summary = pd.DataFrame(
        [
            {
                "candidate_name": "baseline_historical_mean",
                "mean_policy_return_sign": 0.070,
                "mean_oos_r2": -0.210,
                "mean_ic": 0.10,
            },
            {
                "candidate_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.064,
                "mean_oos_r2": -0.730,
                "mean_ic": 0.11,
            },
            {
                "candidate_name": "ensemble_ridge_gbt_v18",
                "mean_policy_return_sign": 0.076,
                "mean_oos_r2": -0.190,
                "mean_ic": 0.22,
            },
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "live_production_ensemble_reduced",
                "signal_agreement_with_shadow_rate": 0.65,
            },
            {
                "path_name": "ensemble_ridge_gbt_v18",
                "signal_agreement_with_shadow_rate": 0.80,
            },
        ]
    )

    decision = choose_v23_decision(metric_summary, review_summary)
    assert decision.status == "extended_history_confirms_candidate"
    assert decision.recommended_path == "ensemble_ridge_gbt_v18"
