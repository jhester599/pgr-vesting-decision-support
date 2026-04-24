"""Tests for x24 indicator contract helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_x24_indicator_contract_keeps_research_only_bundle() -> None:
    from src.research.x24_indicator_contract import build_x24_indicator_contract

    contract = build_x24_indicator_contract(
        x16_package={
            "indicator_name": "adjusted_structural_bvps_pb_6m",
            "signal_family": "adjusted_structural_bvps_pb",
            "horizon_months": 6,
            "model_name": "ridge_bridge__no_change_pb",
            "status": "research_indicator_candidate",
            "pb_anchor_policy": "retain_no_change_pb",
        },
        x23_package={
            "recommendation": {"status": "research_size_indicator_candidate"},
            "dividend_lane_shape": {
                "best_size_target_scale": "to_current_bvps",
                "best_size_model_name": "ridge_scaled",
                "best_size_feature_set": "x10_capital_generation",
                "occurrence_status": "underidentified_post_policy",
                "policy_change_date": "2018-12-01",
            },
        },
        x8_summary={"shadow_readiness": {"status": "not_ready"}},
    )

    assert contract["status"] == "research_indicator_bundle_candidate"
    assert contract["production_changes"] is False
    assert contract["shadow_changes"] is False
    assert contract["structural_signal"]["indicator_name"] == "adjusted_structural_bvps_pb_6m"
    assert contract["dividend_signal"]["target_scale"] == "to_current_bvps"


def test_build_x24_indicator_contract_carries_shadow_caveat() -> None:
    from src.research.x24_indicator_contract import build_x24_indicator_contract

    contract = build_x24_indicator_contract(
        x16_package={
            "indicator_name": "adjusted_structural_bvps_pb_6m",
            "signal_family": "adjusted_structural_bvps_pb",
            "horizon_months": 6,
            "model_name": "ridge_bridge__no_change_pb",
            "status": "research_indicator_candidate",
            "pb_anchor_policy": "retain_no_change_pb",
        },
        x23_package={
            "recommendation": {"status": "continue_research"},
            "dividend_lane_shape": {
                "best_size_target_scale": "to_current_bvps",
                "best_size_model_name": "ridge_scaled",
                "best_size_feature_set": "x10_capital_generation",
                "occurrence_status": "underidentified_post_policy",
                "policy_change_date": "2018-12-01",
            },
        },
        x8_summary={"shadow_readiness": {"status": "not_ready"}},
    )

    assert "shadow_not_ready" in contract["caveats"]
    assert contract["status"] == "research_indicator_bundle_candidate"


def test_build_x24_peer_review_prompt_mentions_both_signals() -> None:
    from src.research.x24_indicator_contract import build_x24_peer_review_prompt

    prompt = build_x24_peer_review_prompt(
        contract={
            "structural_signal": {
                "indicator_name": "adjusted_structural_bvps_pb_6m",
                "model_name": "ridge_bridge__no_change_pb",
            },
            "dividend_signal": {
                "indicator_name": "special_dividend_size_watch",
                "target_scale": "to_current_bvps",
                "occurrence_status": "underidentified_post_policy",
            },
        },
        x8_summary={"shadow_readiness": {"status": "not_ready"}},
        x11_summary={"recommendation": {"status": "continue_research"}},
        x23_package={"recommendation": {"status": "research_size_indicator_candidate"}},
    )

    assert "adjusted_structural_bvps_pb_6m" in prompt
    assert "special_dividend_size_watch" in prompt
    assert "underidentified_post_policy" in prompt
