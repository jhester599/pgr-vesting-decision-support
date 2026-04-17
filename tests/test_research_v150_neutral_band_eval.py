"""Tests for the v150 neutral-band replay proxy harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v150_neutral_band_eval import DEFAULT_CANDIDATE_PATH, evaluate_neutral_band


def test_band_above_ceiling_raises() -> None:
    with pytest.raises(ValueError, match="neutral_band must be in"):
        evaluate_neutral_band(0.20)


def test_candidate_file_is_bounded() -> None:
    value = float(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8").strip())
    assert 0.0 <= value <= 0.10


def test_default_band_produces_reasonable_metrics() -> None:
    metrics = evaluate_neutral_band(0.015)
    assert -1.0 <= metrics["utility_score"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
