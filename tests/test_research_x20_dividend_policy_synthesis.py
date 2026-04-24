"""Tests for x20 dividend-policy synthesis helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_compare_overlap_leaders_prefers_lower_overlap_mae() -> None:
    from src.research.x20_dividend_policy_synthesis import compare_overlap_leaders

    x10_detail = [
        {
            "snapshot_date": "2023-11-30",
            "feature_set": "x10",
            "model_name": "a",
            "actual_excess": 1.0,
            "expected_value_prediction": 0.0,
        },
        {
            "snapshot_date": "2024-11-29",
            "feature_set": "x10",
            "model_name": "a",
            "actual_excess": 3.0,
            "expected_value_prediction": 1.0,
        },
    ]
    x19_detail = [
        {
            "snapshot_date": "2023-11-30",
            "feature_set": "x19",
            "model_name": "b",
            "actual_excess": 1.0,
            "expected_value_prediction": 0.8,
        },
        {
            "snapshot_date": "2024-11-29",
            "feature_set": "x19",
            "model_name": "b",
            "actual_excess": 3.0,
            "expected_value_prediction": 2.8,
        },
    ]

    result = compare_overlap_leaders(x10_detail, x19_detail)

    assert result["x19_beats_x10_overlap"] is True
    assert result["overlap_n_obs"] == 2
    assert result["x19_overlap_ev_mae"] < result["x10_overlap_ev_mae"]


def test_build_x20_recommendation_flags_size_only_when_all_overlap_positive() -> None:
    from src.research.x20_dividend_policy_synthesis import build_x20_recommendation

    recommendation = build_x20_recommendation(
        x19_beats_x10_overlap=True,
        overlap_n_obs=3,
        overlap_positive_rate=1.0,
    )

    assert recommendation["status"] == "continue_research_size_only"
    assert recommendation["production_changes"] is False
    assert recommendation["shadow_changes"] is False


def test_compare_sample_scope_keeps_low_confidence_for_tiny_post_policy_sample() -> None:
    from src.research.x20_dividend_policy_synthesis import compare_sample_scope

    result = compare_sample_scope(
        x10_summary_rows=[{"n_obs": 18}],
        x19_summary_rows=[{"n_obs": 3}],
        post_policy_snapshot_count=8,
    )

    assert result["x19_oos_n_obs"] == 3
    assert result["post_policy_snapshot_count"] == 8
    assert result["confidence"] == "low"
