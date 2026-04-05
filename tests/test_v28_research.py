from __future__ import annotations

import pandas as pd

from src.research.v28 import choose_v28_decision, v28_universe_manifest, v28_universe_specs


def test_v28_universe_specs_include_expected_candidates() -> None:
    specs = v28_universe_specs()
    assert set(specs) == {"current_reduced", "buyable_only", "buyable_plus_context"}
    assert specs["buyable_only"].benchmarks == ["VOO", "VGT", "SCHD", "VXUS", "VWO", "BND"]


def test_v28_universe_manifest_reflects_buyable_share() -> None:
    manifest = v28_universe_manifest()
    assert not manifest.empty
    buyable_only = manifest.loc[manifest["universe_name"] == "buyable_only"].iloc[0]
    assert float(buyable_only["buyable_share"]) == 1.0
    buyable_plus_context = manifest.loc[manifest["universe_name"] == "buyable_plus_context"].iloc[0]
    assert int(buyable_plus_context["n_contextual_only"]) == 2


def test_choose_v28_decision_prefers_buyable_only_when_it_clears_gate() -> None:
    summary = pd.DataFrame(
        [
            {
                "universe_name": "current_reduced",
                "path_name": "ensemble_ridge_gbt_v18",
                "mean_policy_return_sign": 0.075,
                "mean_oos_r2": -0.20,
                "signal_agreement_with_shadow_rate": 0.80,
            },
            {
                "universe_name": "buyable_only",
                "path_name": "ensemble_ridge_gbt_v18",
                "mean_policy_return_sign": 0.074,
                "mean_oos_r2": -0.21,
                "signal_agreement_with_shadow_rate": 0.79,
            },
            {
                "universe_name": "buyable_only",
                "path_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.070,
                "mean_oos_r2": -0.24,
                "signal_agreement_with_shadow_rate": 0.73,
            },
            {
                "universe_name": "buyable_only",
                "path_name": "shadow_baseline",
                "mean_policy_return_sign": 0.075,
                "mean_oos_r2": -0.19,
                "signal_agreement_with_shadow_rate": 1.0,
            },
        ]
    )

    decision = choose_v28_decision(summary)
    assert decision.status == "prune_forecast_universe"
    assert decision.recommended_universe == "buyable_only"


def test_choose_v28_decision_falls_back_to_current_when_narrower_universes_weaken() -> None:
    summary = pd.DataFrame(
        [
            {
                "universe_name": "current_reduced",
                "path_name": "ensemble_ridge_gbt_v18",
                "mean_policy_return_sign": 0.075,
                "mean_oos_r2": -0.20,
                "signal_agreement_with_shadow_rate": 0.80,
            },
            {
                "universe_name": "buyable_only",
                "path_name": "ensemble_ridge_gbt_v18",
                "mean_policy_return_sign": 0.068,
                "mean_oos_r2": -0.32,
                "signal_agreement_with_shadow_rate": 0.70,
            },
            {
                "universe_name": "buyable_only",
                "path_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.069,
                "mean_oos_r2": -0.28,
                "signal_agreement_with_shadow_rate": 0.72,
            },
            {
                "universe_name": "buyable_only",
                "path_name": "shadow_baseline",
                "mean_policy_return_sign": 0.074,
                "mean_oos_r2": -0.19,
                "signal_agreement_with_shadow_rate": 1.0,
            },
            {
                "universe_name": "buyable_plus_context",
                "path_name": "ensemble_ridge_gbt_v18",
                "mean_policy_return_sign": 0.070,
                "mean_oos_r2": -0.31,
                "signal_agreement_with_shadow_rate": 0.74,
            },
            {
                "universe_name": "buyable_plus_context",
                "path_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.071,
                "mean_oos_r2": -0.29,
                "signal_agreement_with_shadow_rate": 0.75,
            },
            {
                "universe_name": "buyable_plus_context",
                "path_name": "shadow_baseline",
                "mean_policy_return_sign": 0.074,
                "mean_oos_r2": -0.19,
                "signal_agreement_with_shadow_rate": 1.0,
            },
        ]
    )

    decision = choose_v28_decision(summary)
    assert decision.status == "keep_current_forecast_universe"
    assert decision.recommended_universe == "current_reduced"
