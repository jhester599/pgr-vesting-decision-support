"""Tests for x16 indicator packaging utilities."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_indicator_package_keeps_research_only_anchor() -> None:
    from src.research.x16_indicator_package import build_indicator_package

    package = build_indicator_package(
        x14_summary={
            "indicator_candidate": {
                "status": "candidate",
                "signal_family": "adjusted_structural_bvps_pb",
                "horizon_months": 6,
                "model_name": "ridge_bridge__no_change_pb",
            },
            "recommendation": {"status": "research_indicator_candidate"},
        },
        x15_summary={
            "ranked_rows": [
                {
                    "horizon_months": 6,
                    "model_name": "no_change_pb_overlay",
                    "beats_no_change_pb": False,
                    "rank": 1,
                }
            ]
        },
    )

    assert package["status"] == "research_indicator_candidate"
    assert package["production_changes"] is False
    assert package["pb_anchor_policy"] == "retain_no_change_pb"


def test_build_peer_review_prompt_mentions_candidate_and_null_overlay() -> None:
    from src.research.x16_indicator_package import build_peer_review_prompt

    prompt = build_peer_review_prompt(
        indicator_package={
            "indicator_name": "adjusted_structural_bvps_pb_6m",
            "signal_family": "adjusted_structural_bvps_pb",
            "horizon_months": 6,
            "pb_anchor_policy": "retain_no_change_pb",
        },
        x14_summary={"indicator_candidate": {"model_name": "ridge_bridge__no_change_pb"}},
        x15_summary={"ranked_rows": [{"horizon_months": 6, "model_name": "no_change_pb_overlay"}]},
    )

    assert "ridge_bridge__no_change_pb" in prompt
    assert "no_change_pb_overlay" in prompt
    assert "missing P/B features" in prompt
