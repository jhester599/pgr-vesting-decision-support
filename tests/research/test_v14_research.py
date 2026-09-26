from __future__ import annotations

import pandas as pd

from src.research.v14 import (
    choose_feature_surgery_candidates,
    count_signal_changes,
    select_best_universe,
)


def test_count_signal_changes_counts_only_transitions() -> None:
    assert count_signal_changes(["NEUTRAL", "NEUTRAL", "OUTPERFORM", "OUTPERFORM", "UNDERPERFORM"]) == 2
    assert count_signal_changes(["OUTPERFORM"]) == 0


def test_select_best_universe_prefers_policy_then_fewer_names() -> None:
    summary = pd.DataFrame(
        [
            {
                "universe_name": "core9",
                "benchmarks": "A,B,C",
                "n_benchmarks": 9,
                "best_nonbaseline_policy_return": 0.0600,
                "best_nonbaseline_oos_r2": -0.30,
            },
            {
                "universe_name": "core7",
                "benchmarks": "D,E",
                "n_benchmarks": 7,
                "best_nonbaseline_policy_return": 0.0600,
                "best_nonbaseline_oos_r2": -0.20,
            },
        ]
    )
    selection = select_best_universe(summary)
    assert selection.universe_name == "core7"
    assert selection.benchmarks == ["D", "E"]


def test_choose_feature_surgery_candidates_prefers_models_over_live() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "live_production_ensemble",
                "candidate_type": "live_ensemble",
                "diversification_aware_utility": 0.050,
                "mean_policy_return_neutral_3pct": 0.051,
                "mean_oos_r2": -0.40,
            },
            {
                "candidate_name": "baseline_historical_mean",
                "candidate_type": "baseline",
                "diversification_aware_utility": 0.060,
                "mean_policy_return_neutral_3pct": 0.061,
                "mean_oos_r2": -0.10,
            },
            {
                "candidate_name": "ridge_lean_v1",
                "candidate_type": "model",
                "diversification_aware_utility": 0.059,
                "mean_policy_return_neutral_3pct": 0.060,
                "mean_oos_r2": -0.18,
            },
            {
                "candidate_name": "gbt_lean_plus_two",
                "candidate_type": "model",
                "diversification_aware_utility": 0.048,
                "mean_policy_return_neutral_3pct": 0.049,
                "mean_oos_r2": -0.35,
            },
        ]
    )
    assert choose_feature_surgery_candidates(summary) == ["ridge_lean_v1"]
