from __future__ import annotations

import pandas as pd

from src.research.v24 import (
    V24_CURRENT_UNIVERSE,
    V24_VTI_UNIVERSE,
    choose_v24_decision,
    summarize_v24_scenarios,
)


def test_v24_universe_replaces_only_voo_with_vti() -> None:
    assert "VOO" in V24_CURRENT_UNIVERSE
    assert "VTI" not in V24_CURRENT_UNIVERSE
    assert "VTI" in V24_VTI_UNIVERSE
    assert "VOO" not in V24_VTI_UNIVERSE
    assert sorted(t for t in V24_CURRENT_UNIVERSE if t != "VOO") == sorted(
        t for t in V24_VTI_UNIVERSE if t != "VTI"
    )


def test_summarize_v24_scenarios_combines_scenario_frames() -> None:
    metric_df = pd.DataFrame(
        [
            {"scenario_name": "current_voo_actual", "candidate_name": "ensemble_ridge_gbt_v18", "mean_policy_return_sign": 0.07, "mean_oos_r2": -0.20, "mean_ic": 0.20},
            {"scenario_name": "vti_replacement_actual", "candidate_name": "ensemble_ridge_gbt_v18", "mean_policy_return_sign": 0.071, "mean_oos_r2": -0.19, "mean_ic": 0.21},
        ]
    )
    review_df = pd.DataFrame(
        [
            {"scenario_name": "current_voo_actual", "path_name": "ensemble_ridge_gbt_v18", "signal_agreement_with_shadow_rate": 0.75, "signal_changes": 20},
            {"scenario_name": "vti_replacement_actual", "path_name": "ensemble_ridge_gbt_v18", "signal_agreement_with_shadow_rate": 0.80, "signal_changes": 18},
        ]
    )
    window_df = pd.DataFrame(
        [
            {"scenario_name": "current_voo_actual", "common_start": "2016-10-31", "common_end": "2025-09-30", "n_common_dates": 108},
            {"scenario_name": "vti_replacement_actual", "common_start": "2016-10-31", "common_end": "2025-09-30", "n_common_dates": 108},
        ]
    )
    summary = summarize_v24_scenarios(metric_df, review_df, window_df)
    assert "mean_policy_return_sign" in summary.columns
    assert len(summary) == 2


def test_choose_v24_decision_prefers_vti_when_it_improves() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "scenario_name": "current_voo_actual",
                "mean_policy_return_sign": 0.0700,
                "mean_oos_r2": -0.2000,
                "mean_ic": 0.2000,
                "signal_agreement_with_shadow_rate": 0.75,
                "signal_changes": 20,
            },
            {
                "scenario_name": "vti_replacement_actual",
                "mean_policy_return_sign": 0.0710,
                "mean_oos_r2": -0.1950,
                "mean_ic": 0.2100,
                "signal_agreement_with_shadow_rate": 0.81,
                "signal_changes": 18,
            },
        ]
    )
    decision = choose_v24_decision(summary_df)
    assert decision.status == "prefer_vti"
    assert decision.recommended_universe == "vti_replacement_actual"
