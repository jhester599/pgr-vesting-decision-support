"""Tests for the v138 Black-Litterman replay proxy harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v138_bl_param_eval import (
    DEFAULT_CANDIDATE_PATH,
    _parse_params,
    evaluate_bl_params,
)


def test_default_params_produce_reasonable_accuracy() -> None:
    """The baseline BL proxy should land in a broad sanity range."""
    metrics = evaluate_bl_params(0.05, 1.0)
    assert 0.40 <= metrics["recommendation_accuracy"] <= 0.90
    assert 0.10 <= metrics["coverage"] <= 1.0


def test_tau_below_floor_raises() -> None:
    """Tau below the guard floor should fail fast."""
    with pytest.raises(ValueError, match="tau must be in"):
        evaluate_bl_params(0.001, 1.0)


def test_view_confidence_below_floor_raises() -> None:
    """View-confidence scalar below the floor should fail fast."""
    with pytest.raises(ValueError, match="view_confidence_scalar must be in"):
        evaluate_bl_params(0.05, 0.10)


def test_candidate_file_parses_to_expected_pair() -> None:
    """The candidate JSON should round-trip through the CLI helper."""
    tau, confidence = _parse_params(None, None, str(DEFAULT_CANDIDATE_PATH))
    assert 0.01 <= tau <= 0.50
    assert 0.25 <= confidence <= 4.0
