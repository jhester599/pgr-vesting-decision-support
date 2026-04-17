"""Tests for the v149 Kelly replay proxy harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v149_kelly_eval import DEFAULT_CANDIDATE_PATH, evaluate_kelly_params


def test_fraction_below_floor_raises() -> None:
    with pytest.raises(ValueError, match="fraction must be in"):
        evaluate_kelly_params(0.0, 0.2)


def test_candidate_file_has_expected_keys() -> None:
    payload = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {"fraction", "cap"}


def test_default_candidate_produces_reasonable_metrics() -> None:
    metrics = evaluate_kelly_params(0.25, 0.20)
    assert -1.0 <= metrics["utility_score"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
