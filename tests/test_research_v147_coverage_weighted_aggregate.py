"""Tests for the v147 coverage-weighted aggregate harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v147_coverage_weighted_aggregate import DEFAULT_CANDIDATE_PATH, evaluate_coverage_weighted_aggregate


def test_multiplier_below_floor_raises() -> None:
    with pytest.raises(ValueError, match="path_b_multiplier must be in"):
        evaluate_coverage_weighted_aggregate(0.10)


def test_candidate_file_is_bounded() -> None:
    value = float(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8").strip())
    assert 0.25 <= value <= 4.0


def test_default_multiplier_produces_reasonable_metrics() -> None:
    metrics = evaluate_coverage_weighted_aggregate(1.0)
    assert 0.0 <= metrics["covered_ba"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
