"""Tests for the v143 correlation-pruning harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v143_corr_prune_eval import DEFAULT_CANDIDATE_PATH, evaluate_corr_pruning


@pytest.mark.slow
def test_default_corr_pruning_produces_reasonable_metrics() -> None:
    metrics = evaluate_corr_pruning(0.95, benchmarks=["VOO", "BND"])
    assert -5.0 <= metrics["pooled_oos_r2"] <= 0.50
    assert 0.0 <= metrics["pooled_hit_rate"] <= 1.0


def test_rho_below_floor_raises() -> None:
    with pytest.raises(ValueError, match="rho_threshold must be in"):
        evaluate_corr_pruning(0.40)


def test_candidate_file_is_bounded() -> None:
    value = float(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8").strip())
    assert 0.50 <= value <= 0.999
