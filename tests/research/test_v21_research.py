from __future__ import annotations

import pandas as pd

from src.research.v21 import (
    V21_REVIEW_PATHS,
    choose_v21_decision,
    common_historical_dates,
    summarize_v21_slices,
    v21_review_ensemble_specs,
)


def test_v21_review_paths_include_live_and_best_candidate() -> None:
    assert "live_production_ensemble_reduced" in V21_REVIEW_PATHS
    assert "ensemble_ridge_gbt_v20_best" in V21_REVIEW_PATHS


def test_v21_review_ensemble_specs_is_narrow_subset() -> None:
    specs = v21_review_ensemble_specs()
    assert set(specs) == set(V21_REVIEW_PATHS)


def test_common_historical_dates_intersects_all_paths() -> None:
    frame_a = pd.DataFrame({"y_hat": [1.0, 2.0], "y_true": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    frame_b = pd.DataFrame({"y_hat": [3.0], "y_true": [3.0]}, index=pd.to_datetime(["2020-02-29"]))
    prediction_map = {
        "shadow_baseline": {"VOO": frame_a, "VXUS": frame_b},
        "live_production_ensemble_reduced": {"VOO": frame_b, "VXUS": frame_b},
    }
    dates = common_historical_dates(prediction_map, ["VOO", "VXUS"])
    assert dates == [pd.Timestamp("2020-02-29")]


def test_summarize_v21_slices_builds_named_slice_rows() -> None:
    detail = pd.DataFrame(
        [
            {
                "as_of": "2019-12-31",
                "path_name": "live_production_ensemble_reduced",
                "consensus": "OUTPERFORM",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "sell_pct": 0.5,
                "mean_predicted": 0.05,
                "mean_ic": 0.10,
                "mean_hit_rate": 0.57,
                "aggregate_oos_r2": -0.10,
                "signal_agrees_with_shadow": True,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
            },
            {
                "as_of": "2020-01-31",
                "path_name": "live_production_ensemble_reduced",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "sell_pct": 0.5,
                "mean_predicted": -0.03,
                "mean_ic": 0.08,
                "mean_hit_rate": 0.56,
                "aggregate_oos_r2": -0.20,
                "signal_agrees_with_shadow": False,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
            },
        ]
    )
    summary = summarize_v21_slices(
        detail,
        {"pre_2020": (None, "2019-12-31"), "post_2020": ("2020-01-01", None)},
    )
    assert set(summary["slice_name"]) == {"pre_2020", "post_2020"}


def test_choose_v21_decision_keeps_live_when_candidate_is_still_biased() -> None:
    metric_summary = pd.DataFrame(
        [
            {"candidate_name": "baseline_historical_mean", "mean_policy_return_sign": 0.070, "mean_oos_r2": -0.205, "mean_ic": -0.14},
            {"candidate_name": "live_production_ensemble_reduced", "mean_policy_return_sign": 0.064, "mean_oos_r2": -0.729, "mean_ic": 0.10},
            {"candidate_name": "ensemble_ridge_gbt_v18", "mean_policy_return_sign": 0.077, "mean_oos_r2": -0.199, "mean_ic": 0.22},
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "live_production_ensemble_reduced",
                "signal_agreement_with_shadow_rate": 0.30,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 4,
                "underperform_rate": 0.40,
            },
            {
                "path_name": "ensemble_ridge_gbt_v18",
                "signal_agreement_with_shadow_rate": 0.00,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 0,
                "underperform_rate": 1.00,
            },
        ]
    )
    slice_summary = pd.DataFrame(
        [
            {"slice_name": "post_2020", "path_name": "live_production_ensemble_reduced", "signal_agreement_with_shadow_rate": 0.35},
            {"slice_name": "post_2020", "path_name": "ensemble_ridge_gbt_v18", "signal_agreement_with_shadow_rate": 0.00},
        ]
    )
    decision = choose_v21_decision(metric_summary, review_summary, slice_summary)
    assert decision.status == "keep_current_live_cross_check"


def test_choose_v21_decision_promotes_when_candidate_is_cleaner_historically() -> None:
    metric_summary = pd.DataFrame(
        [
            {"candidate_name": "baseline_historical_mean", "mean_policy_return_sign": 0.070, "mean_oos_r2": -0.205, "mean_ic": -0.14},
            {"candidate_name": "live_production_ensemble_reduced", "mean_policy_return_sign": 0.064, "mean_oos_r2": -0.729, "mean_ic": 0.10},
            {"candidate_name": "ensemble_ridge_gbt_v20_best", "mean_policy_return_sign": 0.073, "mean_oos_r2": -0.180, "mean_ic": 0.24},
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "live_production_ensemble_reduced",
                "signal_agreement_with_shadow_rate": 0.20,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 5,
                "underperform_rate": 0.35,
            },
            {
                "path_name": "ensemble_ridge_gbt_v20_best",
                "signal_agreement_with_shadow_rate": 0.28,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 2,
                "underperform_rate": 0.45,
            },
        ]
    )
    slice_summary = pd.DataFrame(
        [
            {"slice_name": "post_2020", "path_name": "live_production_ensemble_reduced", "signal_agreement_with_shadow_rate": 0.20},
            {"slice_name": "post_2020", "path_name": "ensemble_ridge_gbt_v20_best", "signal_agreement_with_shadow_rate": 0.25},
        ]
    )
    decision = choose_v21_decision(metric_summary, review_summary, slice_summary)
    assert decision.status == "promote_candidate_cross_check"
