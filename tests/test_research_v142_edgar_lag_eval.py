"""Tests for the v142 EDGAR lag evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v142_edgar_lag_eval import DEFAULT_CANDIDATE_PATH, evaluate_edgar_lag


@pytest.mark.slow
def test_default_edgar_lag_produces_reasonable_metrics() -> None:
    metrics = evaluate_edgar_lag(2, benchmarks=["VOO", "BND"])
    assert -5.0 <= metrics["pooled_oos_r2"] <= 0.50
    assert 0.0 <= metrics["pooled_hit_rate"] <= 1.0


def test_negative_lag_raises() -> None:
    with pytest.raises(ValueError, match="lag must be in"):
        evaluate_edgar_lag(-1)


def test_candidate_file_is_bounded() -> None:
    value = int(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8").strip())
    assert 0 <= value <= 3
