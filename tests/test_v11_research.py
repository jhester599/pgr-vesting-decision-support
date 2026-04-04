from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.v11_autonomous_loop import _combine_prediction_frames
from src.research.v11 import (
    add_destination_roles,
    choose_forecast_universe,
    choose_recommendation_universe,
    diversification_adjusted_policy_utility,
    summarize_existing_holdings_actions,
)
from src.tax.capital_gains import TaxLot


def _sample_scoreboard() -> pd.DataFrame:
    return add_destination_roles(
        pd.DataFrame(
            [
                {
                    "benchmark": "VOO",
                    "family": "broad_equity",
                    "corr_bucket": "moderately_correlated",
                    "diversification_score": 0.45,
                    "composite_score": 2.0,
                    "ensemble_ic": 0.10,
                },
                {
                    "benchmark": "VXUS",
                    "family": "broad_equity",
                    "corr_bucket": "diversifying",
                    "diversification_score": 0.60,
                    "composite_score": 1.5,
                    "ensemble_ic": 0.12,
                },
                {
                    "benchmark": "BND",
                    "family": "fixed_income",
                    "corr_bucket": "diversifying",
                    "diversification_score": 0.70,
                    "composite_score": 1.2,
                    "ensemble_ic": 0.08,
                },
                {
                    "benchmark": "GLD",
                    "family": "real_asset",
                    "corr_bucket": "diversifying",
                    "diversification_score": 0.65,
                    "composite_score": 1.8,
                    "ensemble_ic": 0.07,
                },
                {
                    "benchmark": "VFH",
                    "family": "sector",
                    "corr_bucket": "highly_correlated",
                    "diversification_score": 0.20,
                    "composite_score": 0.9,
                    "ensemble_ic": 0.20,
                },
            ]
        )
    )


def test_universe_selection_keeps_diversifiers_and_limits_context() -> None:
    scoreboard = _sample_scoreboard()
    recommendation = choose_recommendation_universe(scoreboard)
    forecast = choose_forecast_universe(scoreboard, recommendation)

    assert "VXUS" in recommendation
    assert "BND" in recommendation
    assert "GLD" in recommendation
    assert "VFH" not in recommendation
    assert "VFH" in forecast


def test_diversification_adjusted_policy_utility_penalizes_context() -> None:
    scoreboard = _sample_scoreboard()
    candidate_rows = pd.DataFrame(
        [
            {"benchmark": "VXUS", "policy_return_sign": 0.06},
            {"benchmark": "BND", "policy_return_sign": 0.05},
            {"benchmark": "VFH", "policy_return_sign": 0.08},
        ]
    )
    metrics = diversification_adjusted_policy_utility(candidate_rows, scoreboard)
    assert metrics["weighted_policy_return"] > 0
    assert metrics["contextual_penalty"] > 0
    assert metrics["diversification_aware_utility"] < metrics["weighted_policy_return"]


def test_existing_holdings_actions_prioritize_losses_then_ltcg() -> None:
    lots = [
        TaxLot(vest_date=date(2025, 1, 1), rsu_type="time", shares=1, cost_basis_per_share=250.0),
        TaxLot(vest_date=date(2023, 1, 1), rsu_type="time", shares=1, cost_basis_per_share=100.0),
        TaxLot(vest_date=date(2026, 1, 1), rsu_type="time", shares=1, cost_basis_per_share=180.0),
    ]
    actions = summarize_existing_holdings_actions(lots, current_price=190.0, sell_date=date(2026, 4, 1))
    assert actions[0].tax_bucket == "LOSS"
    assert actions[1].tax_bucket == "LTCG"
    assert actions[2].tax_bucket == "STCG"


def test_combine_prediction_frames_handles_shared_y_true() -> None:
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    frame_a = pd.DataFrame({"pred_a__0.1": [0.1, 0.2, 0.3], "y_true": [0.05, -0.02, 0.04]}, index=index)
    frame_b = pd.DataFrame({"pred_b__0.2": [0.0, 0.1, 0.2], "y_true": [0.05, -0.02, 0.04]}, index=index)
    y_hat, y_true = _combine_prediction_frames([frame_a, frame_b])
    assert list(y_true) == [0.05, -0.02, 0.04]
    assert len(y_hat) == 3
