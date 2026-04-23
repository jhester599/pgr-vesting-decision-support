"""Tests for x11 capital synthesis utilities."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_compare_bvps_leaders_flags_x9_horizon_improvement() -> None:
    from src.research.x11_capital_synthesis import compare_bvps_leaders

    x4_rows = [
        {"horizon_months": 1, "rank": 1, "model_name": "x4", "future_bvps_mae": 0.7}
    ]
    x9_rows = [
        {
            "horizon_months": 1,
            "rank": 1,
            "model_name": "x9",
            "feature_block": "bridge",
            "future_bvps_mae": 0.6,
        }
    ]

    comparison = compare_bvps_leaders(x4_rows, x9_rows)

    assert comparison[0]["x9_beats_x4"] is True
    assert comparison[0]["x9_feature_block"] == "bridge"
    assert comparison[0]["future_bvps_mae_delta"] == -0.1


def test_compare_dividend_leaders_keeps_small_sample_caveat() -> None:
    from src.research.x11_capital_synthesis import compare_dividend_leaders

    result = compare_dividend_leaders(
        [{"model_name": "x6", "expected_value_mae": 1.50, "n_obs": 18}],
        [{"model_name": "x10", "expected_value_mae": 1.40, "n_obs": 18}],
    )

    assert result["x10_beats_x6"] is True
    assert result["confidence"] == "low"


def test_build_x11_recommendation_remains_research_only() -> None:
    from src.research.x11_capital_synthesis import build_x11_recommendation

    recommendation = build_x11_recommendation(
        x9_beating_horizons=2,
        x10_beats_x6=True,
        dividend_confidence="low",
    )

    assert recommendation["production_changes"] is False
    assert recommendation["shadow_changes"] is False
    assert recommendation["status"] == "continue_research"
