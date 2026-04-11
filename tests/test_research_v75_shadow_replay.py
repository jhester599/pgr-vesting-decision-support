from __future__ import annotations

from datetime import date

import pandas as pd

from src.research.v75 import choose_v75_decision, holdout_monthly_review_dates, summarize_v75_review


def test_holdout_monthly_review_dates_cover_holdout_window() -> None:
    dates = holdout_monthly_review_dates(date(2026, 4, 10), holdout_start="2024-04-01")
    assert dates[0].isoformat() == "2024-04-30"
    assert dates[-1].isoformat() == "2026-03-31"
    assert len(dates) == 24


def test_summarize_v75_review_keeps_both_paths() -> None:
    detail_df = pd.DataFrame(
        [
            {
                "as_of": "2025-01-31",
                "path_name": "live_equal_weight",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "Reduce",
                "sell_pct": 0.50,
                "mean_predicted": -0.02,
                "mean_ic": 0.08,
                "mean_hit_rate": 0.57,
                "mean_prob_outperform": 0.41,
                "top_benchmark_weight": 0.125,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
                "abs_sell_pct_diff_vs_live": 0.0,
            },
            {
                "as_of": "2025-02-28",
                "path_name": "live_equal_weight",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "Reduce",
                "sell_pct": 0.50,
                "mean_predicted": -0.01,
                "mean_ic": 0.07,
                "mean_hit_rate": 0.56,
                "mean_prob_outperform": 0.43,
                "top_benchmark_weight": 0.125,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
                "abs_sell_pct_diff_vs_live": 0.0,
            },
            {
                "as_of": "2025-01-31",
                "path_name": "v74_quality_weighted",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "Reduce",
                "sell_pct": 0.50,
                "mean_predicted": -0.01,
                "mean_ic": 0.10,
                "mean_hit_rate": 0.58,
                "mean_prob_outperform": 0.39,
                "top_benchmark_weight": 0.22,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
                "abs_sell_pct_diff_vs_live": 0.0,
            },
            {
                "as_of": "2025-02-28",
                "path_name": "v74_quality_weighted",
                "consensus": "UNDERPERFORM",
                "recommendation_mode": "Reduce",
                "sell_pct": 0.50,
                "mean_predicted": -0.005,
                "mean_ic": 0.11,
                "mean_hit_rate": 0.59,
                "mean_prob_outperform": 0.40,
                "top_benchmark_weight": 0.24,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
                "abs_sell_pct_diff_vs_live": 0.0,
            },
        ]
    )

    summary_df = summarize_v75_review(detail_df)

    assert {"live_equal_weight", "v74_quality_weighted"} == set(summary_df["path_name"])
    assert "mode_agreement_with_live_rate" in summary_df.columns
    assert "max_top_benchmark_weight" in summary_df.columns


def test_choose_v75_decision_advances_when_shadow_is_stable() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "path_name": "live_equal_weight",
                "review_months": 24,
                "mean_predicted": -0.020,
                "mean_ic": 0.080,
                "mean_hit_rate": 0.57,
                "mean_prob_outperform": 0.42,
                "mean_sell_pct": 0.50,
                "mean_top_benchmark_weight": 0.125,
                "max_top_benchmark_weight": 0.125,
                "signal_changes": 3,
                "mode_changes": 2,
                "signal_agreement_with_live_rate": 1.0,
                "mode_agreement_with_live_rate": 1.0,
                "sell_agreement_with_live_rate": 1.0,
                "mean_abs_sell_pct_diff_vs_live": 0.0,
                "underperform_rate": 0.50,
                "neutral_rate": 0.50,
                "outperform_rate": 0.0,
            },
            {
                "path_name": "v74_quality_weighted",
                "review_months": 24,
                "mean_predicted": -0.010,
                "mean_ic": 0.110,
                "mean_hit_rate": 0.59,
                "mean_prob_outperform": 0.40,
                "mean_sell_pct": 0.50,
                "mean_top_benchmark_weight": 0.22,
                "max_top_benchmark_weight": 0.28,
                "signal_changes": 3,
                "mode_changes": 2,
                "signal_agreement_with_live_rate": 0.92,
                "mode_agreement_with_live_rate": 0.92,
                "sell_agreement_with_live_rate": 0.92,
                "mean_abs_sell_pct_diff_vs_live": 0.02,
                "underperform_rate": 0.45,
                "neutral_rate": 0.55,
                "outperform_rate": 0.0,
            },
        ]
    )

    decision = choose_v75_decision(summary_df)

    assert decision.status == "advance_to_promotion_check"
    assert decision.recommended_path == "v74_quality_weighted"


def test_choose_v75_decision_keeps_shadow_when_concentration_is_too_high() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "path_name": "live_equal_weight",
                "review_months": 24,
                "mean_predicted": -0.020,
                "mean_ic": 0.080,
                "mean_hit_rate": 0.57,
                "mean_prob_outperform": 0.42,
                "mean_sell_pct": 0.50,
                "mean_top_benchmark_weight": 0.125,
                "max_top_benchmark_weight": 0.125,
                "signal_changes": 3,
                "mode_changes": 2,
                "signal_agreement_with_live_rate": 1.0,
                "mode_agreement_with_live_rate": 1.0,
                "sell_agreement_with_live_rate": 1.0,
                "mean_abs_sell_pct_diff_vs_live": 0.0,
                "underperform_rate": 0.50,
                "neutral_rate": 0.50,
                "outperform_rate": 0.0,
            },
            {
                "path_name": "v74_quality_weighted",
                "review_months": 24,
                "mean_predicted": -0.010,
                "mean_ic": 0.110,
                "mean_hit_rate": 0.59,
                "mean_prob_outperform": 0.40,
                "mean_sell_pct": 0.50,
                "mean_top_benchmark_weight": 0.33,
                "max_top_benchmark_weight": 0.52,
                "signal_changes": 3,
                "mode_changes": 2,
                "signal_agreement_with_live_rate": 0.92,
                "mode_agreement_with_live_rate": 0.92,
                "sell_agreement_with_live_rate": 0.92,
                "mean_abs_sell_pct_diff_vs_live": 0.02,
                "underperform_rate": 0.45,
                "neutral_rate": 0.55,
                "outperform_rate": 0.0,
            },
        ]
    )

    decision = choose_v75_decision(summary_df)

    assert decision.status == "keep_shadow_only"
    assert decision.recommended_path == "keep_v38_live"
