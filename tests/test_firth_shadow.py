"""Tests for the v159 Firth logistic shadow reporting module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_firth_shadow_constants() -> None:
    from src.reporting.firth_shadow import (
        FIRTH_BENCHMARKS,
        FIRTH_SHADOW_VARIANT_LABEL,
        FIRTH_SHADOW_VARIANT_NAME,
    )
    assert "VMBS" in FIRTH_BENCHMARKS
    assert "BND" in FIRTH_BENCHMARKS
    assert FIRTH_SHADOW_VARIANT_NAME == "firth_shadow_v159"
    assert FIRTH_SHADOW_VARIANT_LABEL == "Firth Logistic Shadow"


def test_load_firth_candidate_reads_v154_file() -> None:
    from src.reporting.firth_shadow import load_firth_candidate
    candidate = load_firth_candidate()
    assert "firth_winners" in candidate
    assert "VMBS" in candidate["firth_winners"]
    assert "BND" in candidate["firth_winners"]
    assert candidate["recommendation"] == "adopt_firth_for_thin_benchmarks"


def test_build_firth_shadow_payload_is_reporting_only() -> None:
    from src.reporting.firth_shadow import (
        FIRTH_SHADOW_VARIANT_NAME,
        build_firth_shadow_payload,
    )
    payload = build_firth_shadow_payload(
        probability_actionable_sell=0.40,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )
    assert payload["variant"] == FIRTH_SHADOW_VARIANT_NAME
    assert payload["label"] == "Firth Logistic Shadow"
    assert payload["reporting_only"] is True
    assert payload["probability_actionable_sell"] == 0.40
    assert payload["confidence_tier"] == "LOW"
    assert payload["stance"] == "NEUTRAL"


def test_build_firth_shadow_payload_includes_research_findings() -> None:
    from src.reporting.firth_shadow import build_firth_shadow_payload
    payload = build_firth_shadow_payload(
        probability_actionable_sell=0.36,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )
    assert "VMBS" in payload["firth_winners"]
    assert "BND" in payload["firth_winners"]
    assert payload["firth_winner_deltas"]["VMBS"] > 0.02
    assert payload["firth_winner_deltas"]["BND"] > 0.02
    assert payload["firth_recommendation"] == "adopt_firth_for_thin_benchmarks"
    assert "firth_benchmarks" in payload
    assert sorted(payload["firth_benchmarks"]) == ["BND", "VMBS"]


def test_build_firth_shadow_payload_none_probability() -> None:
    from src.reporting.firth_shadow import build_firth_shadow_payload
    payload = build_firth_shadow_payload(
        probability_actionable_sell=None,
        confidence_tier=None,
        stance=None,
    )
    assert payload["probability_actionable_sell"] is None
    assert payload["reporting_only"] is True
    assert "firth_winners" in payload


def test_build_firth_shadow_payload_no_recommended_sell_pct() -> None:
    """Firth shadow must not contain a recommended_sell_pct — reporting only."""
    from src.reporting.firth_shadow import build_firth_shadow_payload
    payload = build_firth_shadow_payload(
        probability_actionable_sell=0.75,
        confidence_tier="HIGH",
        stance="ACTIONABLE-SELL",
    )
    assert "recommended_sell_pct" not in payload
