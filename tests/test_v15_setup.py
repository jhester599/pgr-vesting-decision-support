from __future__ import annotations

import pandas as pd

from src.research.v15 import (
    apply_one_for_one_swap,
    base_model_specs,
    build_confirmation_queue,
    build_inventory_template,
    build_swap_queue,
    choose_best_confirmed_swaps,
    choose_phase0_winners,
    deployed_model_specs,
    normalize_inventory,
)


def test_build_inventory_template_has_required_columns() -> None:
    template = build_inventory_template()
    assert "feature_name" in template.columns
    assert "target_model" in template.columns
    assert "priority_rank" in template.columns


def test_normalize_inventory_standardizes_target_model() -> None:
    inventory = pd.DataFrame(
        [
            {
                "feature_name": "candidate_feature",
                "category": "PGR_specific",
                "replace_or_compete_with": "mom_3m, mom_6m",
                "definition": "test",
                "economic_rationale": "test",
                "expected_direction": "+",
                "likely_frequency": "monthly",
                "likely_source": "edgar",
                "implementation_difficulty": "low",
                "likely_signal_quality": "medium",
                "why_it_might_outperform_existing_feature": "better economics",
                "key_risks": "coverage",
                "target_model": "ridge+gbt",
                "priority_rank": 1,
            }
        ]
    )
    normalized = normalize_inventory(inventory)
    assert normalized.iloc[0]["target_model"] == "both"


def test_build_swap_queue_expands_replacements_by_model() -> None:
    specs = base_model_specs()
    inventory = pd.DataFrame(
        [
            {
                "feature_name": "candidate_feature",
                "category": "shared_regime",
                "replace_or_compete_with": "mom_12m,credit_spread_hy",
                "definition": "test",
                "economic_rationale": "test",
                "expected_direction": "+",
                "likely_frequency": "monthly",
                "likely_source": "fred",
                "implementation_difficulty": "low",
                "likely_signal_quality": "medium",
                "why_it_might_outperform_existing_feature": "better economics",
                "key_risks": "coverage",
                "target_model": "ridge",
                "priority_rank": 2,
            }
        ]
    )
    queue = build_swap_queue(inventory, specs, {"candidate_feature"})
    assert set(queue["candidate_name"]) == {"ridge_lean_v1"}
    assert set(queue["replace_feature"]) == {"mom_12m", "credit_spread_hy"}
    assert queue["candidate_available_now"].all()


def test_apply_one_for_one_swap_replaces_only_target_feature() -> None:
    spec = base_model_specs()["gbt_lean_plus_two"]
    swapped = apply_one_for_one_swap(spec, "mom_3m", "candidate_feature")
    assert "candidate_feature" in swapped
    assert "mom_3m" not in swapped
    assert len(swapped) == len(spec.features)


def test_deployed_model_specs_include_all_deployed_models() -> None:
    specs = deployed_model_specs()
    assert set(specs) == {
        "elasticnet_current",
        "ridge_lean_v1",
        "bayesian_ridge_current",
        "gbt_lean_plus_two",
    }


def test_choose_phase0_winners_returns_top_rows_per_model() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_type": "ridge",
                "candidate_feature": "feature_a",
                "replace_feature": "combined_ratio_ttm",
                "candidate_available_now": True,
                "mean_policy_return_sign_delta": 0.010,
                "mean_oos_r2_delta": 0.001,
                "mean_ic_delta": 0.002,
                "priority_rank": 2,
            },
            {
                "model_type": "ridge",
                "candidate_feature": "feature_b",
                "replace_feature": "combined_ratio_ttm",
                "candidate_available_now": True,
                "mean_policy_return_sign_delta": 0.005,
                "mean_oos_r2_delta": 0.000,
                "mean_ic_delta": 0.001,
                "priority_rank": 3,
            },
            {
                "model_type": "gbt",
                "candidate_feature": "feature_c",
                "replace_feature": "vmt_yoy",
                "candidate_available_now": True,
                "mean_policy_return_sign_delta": 0.008,
                "mean_oos_r2_delta": 0.000,
                "mean_ic_delta": 0.003,
                "priority_rank": 1,
            },
        ]
    )
    winners = choose_phase0_winners(summary, max_per_model=1)
    assert set(winners["candidate_feature"]) == {"feature_a", "feature_c"}


def test_build_confirmation_queue_expands_winners_to_other_models() -> None:
    winners = pd.DataFrame(
        [
            {
                "model_type": "ridge",
                "candidate_feature": "feature_a",
                "replace_feature": "credit_spread_hy",
                "priority_rank": 1,
                "research_source": "test",
            }
        ]
    )
    queue = build_confirmation_queue(winners, deployed_model_specs())
    assert set(queue["candidate_name"]) == {
        "elasticnet_current",
        "ridge_lean_v1",
        "bayesian_ridge_current",
        "gbt_lean_plus_two",
    }


def test_choose_best_confirmed_swaps_selects_one_per_model() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_type": "ridge",
                "candidate_feature": "feature_a",
                "mean_policy_return_sign_delta": 0.004,
                "mean_oos_r2_delta": 0.001,
                "mean_ic_delta": 0.002,
                "priority_rank": 2,
            },
            {
                "model_type": "ridge",
                "candidate_feature": "feature_b",
                "mean_policy_return_sign_delta": 0.002,
                "mean_oos_r2_delta": 0.000,
                "mean_ic_delta": 0.001,
                "priority_rank": 1,
            },
            {
                "model_type": "gbt",
                "candidate_feature": "feature_c",
                "mean_policy_return_sign_delta": 0.003,
                "mean_oos_r2_delta": 0.001,
                "mean_ic_delta": 0.001,
                "priority_rank": 1,
            },
        ]
    )
    winners = choose_best_confirmed_swaps(summary)
    assert set(winners["candidate_feature"]) == {"feature_a", "feature_c"}
