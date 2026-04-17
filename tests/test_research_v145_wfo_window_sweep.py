"""Tests for the v145 WFO window sweep harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v145_wfo_window_sweep import DEFAULT_CANDIDATE_PATH, evaluate_wfo_windows


@pytest.mark.slow
def test_default_window_pair_produces_reasonable_metrics() -> None:
    metrics = evaluate_wfo_windows(60, 6, benchmarks=["VOO", "BND"])
    assert -5.0 <= metrics["pooled_oos_r2"] <= 0.50
    assert 0.0 <= metrics["pooled_hit_rate"] <= 1.0


def test_test_months_above_ceiling_raises() -> None:
    with pytest.raises(ValueError, match="test_months must be in"):
        evaluate_wfo_windows(60, 36)


def test_candidate_file_has_expected_keys() -> None:
    payload = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {"train", "test"}
