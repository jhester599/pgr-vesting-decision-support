"""Tests for x14 indicator synthesis utilities."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_select_indicator_candidate_picks_consistent_6m_adjusted_path() -> None:
    from src.research.x14_indicator_synthesis import select_indicator_candidate

    candidate = select_indicator_candidate(
        evidence_rows=[
            {
                "horizon_months": 3,
                "x12_adjusted_beats_raw": False,
                "x5_uses_no_change_pb": True,
            },
            {
                "horizon_months": 6,
                "x12_adjusted_beats_raw": True,
                "x5_uses_no_change_pb": True,
            },
        ],
        x13_comparison=[
            {
                "horizon_months": 3,
                "adjusted_beats_raw": False,
                "mae_delta": 0.8,
                "adjusted_model_name": "adj3",
            },
            {
                "horizon_months": 6,
                "adjusted_beats_raw": True,
                "mae_delta": -0.4,
                "adjusted_model_name": "adj6",
            },
        ],
        x8_status="not_ready",
        x11_status="continue_research",
    )

    assert candidate["status"] == "candidate"
    assert candidate["horizon_months"] == 6
    assert candidate["model_name"] == "adj6"


def test_select_indicator_candidate_returns_none_when_no_horizon_improves() -> None:
    from src.research.x14_indicator_synthesis import select_indicator_candidate

    candidate = select_indicator_candidate(
        evidence_rows=[
            {
                "horizon_months": 3,
                "x12_adjusted_beats_raw": False,
                "x5_uses_no_change_pb": True,
            }
        ],
        x13_comparison=[
            {
                "horizon_months": 3,
                "adjusted_beats_raw": False,
                "mae_delta": 0.8,
                "adjusted_model_name": "adj3",
            }
        ],
        x8_status="not_ready",
        x11_status="continue_research",
    )

    assert candidate["status"] == "no_candidate"


def test_build_x14_recommendation_remains_research_only() -> None:
    from src.research.x14_indicator_synthesis import build_x14_recommendation

    recommendation = build_x14_recommendation(
        candidate_status="candidate",
        x8_status="not_ready",
        x11_status="continue_research",
    )

    assert recommendation["production_changes"] is False
    assert recommendation["shadow_changes"] is False


def test_select_indicator_candidate_respects_broader_gate() -> None:
    from src.research.x14_indicator_synthesis import select_indicator_candidate

    candidate = select_indicator_candidate(
        evidence_rows=[
            {
                "horizon_months": 6,
                "x12_adjusted_beats_raw": True,
                "x5_uses_no_change_pb": True,
            }
        ],
        x13_comparison=[
            {
                "horizon_months": 6,
                "adjusted_beats_raw": True,
                "mae_delta": -0.4,
                "adjusted_model_name": "adj6",
            }
        ],
        x8_status="ready",
        x11_status="continue_research",
    )

    assert candidate["status"] == "no_candidate"


def test_summarize_horizon_evidence_reconciles_prior_artifacts() -> None:
    from src.research.x14_indicator_synthesis import summarize_horizon_evidence

    evidence = summarize_horizon_evidence(
        x3_rows=[
            {
                "horizon_months": 6,
                "rank": 1,
                "model_name": "x3_leader",
                "beats_no_change": False,
                "target_kind": "return",
            },
            {
                "horizon_months": 6,
                "rank": 1,
                "model_name": "x3_log_leader",
                "beats_no_change": True,
                "target_kind": "log_return",
            }
        ],
        x5_rows=[
            {
                "horizon_months": 6,
                "rank": 1,
                "model_name": "x5_leader",
                "pb_model_name": "no_change_pb",
            }
        ],
        x12_rows=[
            {
                "horizon_months": 6,
                "rank": 1,
                "target_variant": "raw",
                "model_name": "x12_raw",
                "future_bvps_mae": 2.1,
            },
            {
                "horizon_months": 6,
                "rank": 1,
                "target_variant": "adjusted",
                "model_name": "x12_adj",
                "future_bvps_mae": 1.9,
            },
        ],
        x13_comparison=[
            {
                "horizon_months": 6,
                "raw_model_name": "x13_raw",
                "adjusted_model_name": "x13_adj",
                "adjusted_beats_raw": True,
                "mae_delta": -0.3,
            }
        ],
    )

    assert evidence[0]["x5_uses_no_change_pb"] is True
    assert evidence[0]["x12_supports_adjusted_target_family"] is True
    assert evidence[0]["x13_adjusted_beats_raw"] is True
    assert evidence[0]["x3_return_leader_model_name"] == "x3_leader"
    assert evidence[0]["x3_log_return_leader_model_name"] == "x3_log_leader"
