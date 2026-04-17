"""Tests for the v144 conformal coverage harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v144_conformal_eval import DEFAULT_CANDIDATE_PATH, evaluate_conformal_config


@pytest.mark.slow
def test_default_conformal_config_produces_reasonable_coverage() -> None:
    metrics = evaluate_conformal_config(0.80, 0.05, benchmarks=["VOO", "BND"])
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.50 <= metrics["target_coverage"] < 1.0


def test_gamma_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="gamma must be in"):
        evaluate_conformal_config(0.80, 1.20)


def test_candidate_file_has_expected_keys() -> None:
    payload = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {"coverage", "aci_gamma"}
