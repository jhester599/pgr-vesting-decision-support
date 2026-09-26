from __future__ import annotations

import pandas as pd

from src.research.v17 import choose_v17_promotion, count_label_changes, summarize_shadow_review


def test_count_label_changes_counts_transitions() -> None:
    assert count_label_changes(["A", "A", "B", "B", "C"]) == 2
    assert count_label_changes(["A"]) == 0


def test_summarize_shadow_review_aggregates_by_path() -> None:
    detail = pd.DataFrame(
        [
            {
                "as_of": "2026-01-31",
                "path_name": "candidate_v16",
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
            },
            {
                "as_of": "2026-02-28",
                "path_name": "candidate_v16",
                "consensus": "NEUTRAL",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "sell_pct": 0.5,
                "mean_predicted": 0.01,
                "mean_ic": 0.08,
                "mean_hit_rate": 0.56,
                "aggregate_oos_r2": -0.08,
                "signal_agrees_with_shadow": False,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
            },
        ]
    )
    summary = summarize_shadow_review(detail)
    row = summary.iloc[0]
    assert row["path_name"] == "candidate_v16"
    assert int(row["signal_changes"]) == 1
    assert float(row["signal_agreement_with_shadow_rate"]) == 0.5


def test_choose_v17_promotion_promotes_cross_check_when_candidate_is_cleaner() -> None:
    v16_summary = pd.DataFrame(
        [
            {
                "candidate_name": "ensemble_ridge_gbt_v16",
                "mean_policy_return_sign": 0.075,
                "mean_oos_r2": -0.20,
            },
            {
                "candidate_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.064,
                "mean_oos_r2": -0.73,
            },
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "candidate_v16",
                "signal_agreement_with_shadow_rate": 0.90,
                "mode_agreement_with_shadow_rate": 1.00,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 1,
                "mode_changes": 0,
            },
            {
                "path_name": "live_production",
                "signal_agreement_with_shadow_rate": 0.70,
                "mode_agreement_with_shadow_rate": 0.80,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 3,
                "mode_changes": 1,
            },
        ]
    )

    decision = choose_v17_promotion(v16_summary, review_summary)
    assert decision.status == "promote_cross_check_candidate"
    assert decision.recommended_path == "candidate_v16"


def test_choose_v17_promotion_keeps_live_when_candidate_not_cleaner() -> None:
    v16_summary = pd.DataFrame(
        [
            {
                "candidate_name": "ensemble_ridge_gbt_v16",
                "mean_policy_return_sign": 0.075,
                "mean_oos_r2": -0.20,
            },
            {
                "candidate_name": "live_production_ensemble_reduced",
                "mean_policy_return_sign": 0.064,
                "mean_oos_r2": -0.73,
            },
        ]
    )
    review_summary = pd.DataFrame(
        [
            {
                "path_name": "candidate_v16",
                "signal_agreement_with_shadow_rate": 0.50,
                "mode_agreement_with_shadow_rate": 0.70,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 4,
                "mode_changes": 2,
            },
            {
                "path_name": "live_production",
                "signal_agreement_with_shadow_rate": 0.70,
                "mode_agreement_with_shadow_rate": 0.80,
                "sell_agreement_with_shadow_rate": 1.00,
                "signal_changes": 3,
                "mode_changes": 1,
            },
        ]
    )

    decision = choose_v17_promotion(v16_summary, review_summary)
    assert decision.status == "keep_current_live_cross_check"
