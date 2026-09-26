from __future__ import annotations

import pandas as pd

from src.research.v20 import choose_v20_decision, summarize_v20_review, v20_ensemble_specs, v20_model_specs


def test_v20_model_specs_include_new_variants() -> None:
    specs = v20_model_specs()
    assert "ridge_lean_v1__v20_value" in specs
    assert "gbt_lean_plus_two__v20_usd" in specs
    assert "gbt_lean_plus_two__v20_pricing" in specs


def test_v20_ensemble_specs_include_best_of_confirmed_stack() -> None:
    ensembles = v20_ensemble_specs()
    assert "ensemble_ridge_gbt_v20_best" in ensembles
    assert ensembles["ensemble_ridge_gbt_v20_best"]["members"] == [
        "ridge_lean_v1__v20_value",
        "gbt_lean_plus_two__v18",
    ]


def test_summarize_v20_review_captures_signal_bias_and_agreement() -> None:
    detail = pd.DataFrame(
        [
            {
                "as_of": "2026-01-31",
                "path_name": "ensemble_ridge_gbt_v20_best",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "sell_pct": 0.5,
                "mean_predicted": -0.02,
                "mean_ic": 0.12,
                "mean_hit_rate": 0.59,
                "aggregate_oos_r2": -0.15,
                "signal_agrees_with_shadow": False,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
                "signal_agrees_with_live": False,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
            },
            {
                "as_of": "2026-02-29",
                "path_name": "ensemble_ridge_gbt_v20_best",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "sell_pct": 0.5,
                "mean_predicted": -0.01,
                "mean_ic": 0.10,
                "mean_hit_rate": 0.57,
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
    summary = summarize_v20_review(detail)
    row = summary.iloc[0]
    assert row["path_name"] == "ensemble_ridge_gbt_v20_best"
    assert float(row["underperform_rate"]) == 1.0
    assert float(row["signal_agreement_with_shadow_rate"]) == 0.0


def test_choose_v20_decision_promotes_when_metrics_and_behavior_clear_gate() -> None:
    metric_summary = pd.DataFrame(
        [
            {"candidate_name": "baseline_historical_mean", "mean_policy_return_sign": 0.070, "mean_oos_r2": -0.205, "mean_ic": -0.14},
            {"candidate_name": "live_production_ensemble_reduced", "mean_policy_return_sign": 0.064, "mean_oos_r2": -0.729, "mean_ic": 0.10},
            {"candidate_name": "ensemble_ridge_gbt_v20_best", "mean_policy_return_sign": 0.073, "mean_oos_r2": -0.180, "mean_ic": 0.22},
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "live_production_ensemble_reduced",
                "signal_agreement_with_shadow_rate": 0.20,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 4,
                "underperform_rate": 0.25,
            },
            {
                "path_name": "ensemble_ridge_gbt_v20_best",
                "signal_agreement_with_shadow_rate": 0.30,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 2,
                "underperform_rate": 0.40,
            },
        ]
    )
    decision = choose_v20_decision(metric_summary, review_summary)
    assert decision.status == "promote_candidate_cross_check"
    assert decision.recommended_candidate == "ensemble_ridge_gbt_v20_best"


def test_choose_v20_decision_keeps_current_cross_check_when_bias_persists() -> None:
    metric_summary = pd.DataFrame(
        [
            {"candidate_name": "baseline_historical_mean", "mean_policy_return_sign": 0.070, "mean_oos_r2": -0.205, "mean_ic": -0.14},
            {"candidate_name": "live_production_ensemble_reduced", "mean_policy_return_sign": 0.064, "mean_oos_r2": -0.729, "mean_ic": 0.10},
            {"candidate_name": "ensemble_ridge_gbt_v20_best", "mean_policy_return_sign": 0.076, "mean_oos_r2": -0.190, "mean_ic": 0.24},
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "live_production_ensemble_reduced",
                "signal_agreement_with_shadow_rate": 0.20,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 4,
                "underperform_rate": 0.25,
            },
            {
                "path_name": "ensemble_ridge_gbt_v20_best",
                "signal_agreement_with_shadow_rate": 0.00,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 0,
                "underperform_rate": 1.00,
            },
        ]
    )
    decision = choose_v20_decision(metric_summary, review_summary)
    assert decision.status == "continue_research_keep_current_cross_check"
