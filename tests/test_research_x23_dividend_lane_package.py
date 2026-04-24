"""Tests for x23 dividend-lane package helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_x23_recommendation_marks_occurrence_underidentified() -> None:
    from src.research.x23_dividend_lane_package import build_x23_recommendation

    recommendation = build_x23_recommendation(
        overlap_positive_rate=1.0,
        x22_best_row={"feature_set": "x10_capital_generation", "target_scale": "to_current_bvps"},
    )

    assert recommendation["occurrence_status"] == "underidentified_post_policy"
    assert recommendation["status"] == "research_size_indicator_candidate"


def test_build_x23_recommendation_stays_research_only() -> None:
    from src.research.x23_dividend_lane_package import build_x23_recommendation

    recommendation = build_x23_recommendation(
        overlap_positive_rate=1.0,
        x22_best_row={"feature_set": "baseline_only", "target_scale": "to_current_bvps"},
    )

    assert recommendation["production_changes"] is False
    assert recommendation["shadow_changes"] is False
    assert recommendation["status"] == "continue_research"
