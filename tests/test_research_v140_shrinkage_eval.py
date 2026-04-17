"""Tests for the v140 shrinkage evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v140_shrinkage_eval import DEFAULT_CANDIDATE_PATH, evaluate_shrinkage


@pytest.mark.slow
def test_default_shrinkage_produces_reasonable_metrics() -> None:
    metrics = evaluate_shrinkage(0.50, benchmarks=["VOO", "BND"])
    assert -5.0 <= metrics["pooled_oos_r2"] <= 0.50
    assert 0.0 <= metrics["pooled_hit_rate"] <= 1.0


def test_shrinkage_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="shrinkage must be in"):
        evaluate_shrinkage(-0.1)


def test_candidate_file_is_bounded() -> None:
    value = float(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8").strip())
    assert 0.0 <= value <= 1.0
