from __future__ import annotations

import pandas as pd

from src.research.v18 import choose_best_v18_swaps, choose_v18_decision, v18_swap_candidates


def test_v18_swap_candidates_cover_both_models() -> None:
    swaps = v18_swap_candidates()
    model_names = {swap.candidate_name for swap in swaps}
    assert "ridge_lean_v1__v16" in model_names
    assert "gbt_lean_plus_two__v16" in model_names


def test_choose_best_v18_swaps_selects_one_per_model() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "ridge_lean_v1__v16",
                "candidate_feature": "a",
                "replace_feature": "x",
                "mean_policy_return_sign_delta": 0.01,
                "mean_oos_r2_delta": 0.02,
                "mean_ic_delta": 0.00,
            },
            {
                "candidate_name": "ridge_lean_v1__v16",
                "candidate_feature": "b",
                "replace_feature": "y",
                "mean_policy_return_sign_delta": 0.02,
                "mean_oos_r2_delta": 0.01,
                "mean_ic_delta": 0.00,
            },
            {
                "candidate_name": "gbt_lean_plus_two__v16",
                "candidate_feature": "c",
                "replace_feature": "z",
                "mean_policy_return_sign_delta": 0.03,
                "mean_oos_r2_delta": -0.01,
                "mean_ic_delta": 0.01,
            },
        ]
    )
    best = choose_best_v18_swaps(summary)
    assert len(best) == 2
    assert set(best["candidate_name"]) == {"ridge_lean_v1__v16", "gbt_lean_plus_two__v16"}


def test_choose_v18_decision_advances_when_bias_improves() -> None:
    metric_summary = pd.DataFrame(
        [
            {"candidate_name": "ensemble_ridge_gbt_v16", "mean_policy_return_sign": 0.074, "mean_oos_r2": -0.20},
            {"candidate_name": "ensemble_ridge_gbt_v18", "mean_policy_return_sign": 0.075, "mean_oos_r2": -0.19},
            {"candidate_name": "baseline_historical_mean", "mean_policy_return_sign": 0.070, "mean_oos_r2": -0.20},
        ]
    )
    shadow_summary = pd.DataFrame(
        [
            {"path_name": "candidate_v16", "signal_agreement_with_shadow_rate": 0.00},
            {"path_name": "candidate_v18", "signal_agreement_with_shadow_rate": 0.33},
        ]
    )
    decision = choose_v18_decision(metric_summary, shadow_summary)
    assert decision.status == "advance_to_v19"
    assert decision.recommended_candidate == "ensemble_ridge_gbt_v18"


def test_choose_v18_decision_blocks_when_bias_stays_bad() -> None:
    metric_summary = pd.DataFrame(
        [
            {"candidate_name": "ensemble_ridge_gbt_v16", "mean_policy_return_sign": 0.074, "mean_oos_r2": -0.20},
            {"candidate_name": "ensemble_ridge_gbt_v18", "mean_policy_return_sign": 0.073, "mean_oos_r2": -0.22},
            {"candidate_name": "baseline_historical_mean", "mean_policy_return_sign": 0.070, "mean_oos_r2": -0.20},
        ]
    )
    shadow_summary = pd.DataFrame(
        [
            {"path_name": "candidate_v16", "signal_agreement_with_shadow_rate": 0.00},
            {"path_name": "candidate_v18", "signal_agreement_with_shadow_rate": 0.00},
        ]
    )
    decision = choose_v18_decision(metric_summary, shadow_summary)
    assert decision.status == "keep_v16_as_research_only"
