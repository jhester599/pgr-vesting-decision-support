from __future__ import annotations

import pandas as pd

from src.research.v16 import choose_v16_promotion, v16_ensemble_specs, v16_model_specs


def test_v16_model_specs_apply_confirmed_v15_swaps() -> None:
    specs = v16_model_specs()

    ridge_features = specs["ridge_lean_v1__v16"].features
    assert "book_value_per_share_growth_yoy" in ridge_features
    assert "roe_net_income_ttm" not in ridge_features

    gbt_features = specs["gbt_lean_plus_two__v16"].features
    assert "rate_adequacy_gap_yoy" in gbt_features
    assert "vmt_yoy" not in gbt_features


def test_v16_ensemble_specs_include_live_and_modified_pair() -> None:
    ensembles = v16_ensemble_specs()
    assert "live_production_ensemble_reduced" in ensembles
    assert ensembles["ensemble_ridge_gbt_v16"]["members"] == [
        "ridge_lean_v1__v16",
        "gbt_lean_plus_two__v16",
    ]


def test_choose_v16_promotion_prefers_shadow_when_candidate_beats_live_only() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.060,
                "mean_oos_r2": -0.60,
            },
            {
                "candidate_name": "ensemble_ridge_gbt_v16",
                "mean_policy_return_sign": 0.070,
                "mean_oos_r2": -0.40,
            },
            {
                "candidate_name": "baseline_historical_mean",
                "mean_policy_return_sign": 0.071,
                "mean_oos_r2": -0.20,
            },
        ]
    )

    decision = choose_v16_promotion(summary)
    assert decision.status == "shadow_for_v17"
    assert decision.recommended_candidate == "ensemble_ridge_gbt_v16"


def test_choose_v16_promotion_blocks_candidate_without_enough_edge() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.068,
                "mean_oos_r2": -0.30,
            },
            {
                "candidate_name": "ensemble_ridge_gbt_v16",
                "mean_policy_return_sign": 0.067,
                "mean_oos_r2": -0.32,
            },
            {
                "candidate_name": "baseline_historical_mean",
                "mean_policy_return_sign": 0.070,
                "mean_oos_r2": -0.20,
            },
        ]
    )

    decision = choose_v16_promotion(summary)
    assert decision.status == "do_not_promote"
