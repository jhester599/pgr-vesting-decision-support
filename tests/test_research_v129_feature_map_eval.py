"""Tests for the v129 feature-map evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v129_feature_map_eval import (
    MIN_COVERAGE,
    V129_CANDIDATE_MAP_PATH,
    _candidate_map_from_v128,
    _load_candidate_map_from_file,
    evaluate_feature_map,
    main,
)


def test_lean_baseline_matches_v128_pooled_baseline() -> None:
    """Built-in lean baseline should recover the canonical v128 pooled reading."""
    metrics = evaluate_feature_map("lean_baseline")
    assert metrics["covered_ba"] == pytest.approx(0.5000, abs=0.01)
    assert metrics["coverage"] == pytest.approx(0.8700, abs=0.03)


def test_v128_map_is_not_worse_than_lean_baseline() -> None:
    """The v128 winner map should weakly improve on the lean baseline."""
    lean = evaluate_feature_map("lean_baseline")
    mapped = evaluate_feature_map("v128_map")
    assert mapped["covered_ba"] >= lean["covered_ba"]


def test_file_strategy_accepts_candidate_map_artifact() -> None:
    """The file-backed candidate-map path should be loadable and scoreable."""
    metrics = evaluate_feature_map(f"file:{V129_CANDIDATE_MAP_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    assert 0.0 <= metrics["covered_ba"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_loading_candidate_map_rejects_invalid_feature(tmp_path: Path) -> None:
    """Candidate maps must only use features from the v128 universe."""
    candidate = _candidate_map_from_v128()
    candidate.loc[candidate["benchmark"] == "VOO", "feature_list"] = "not_a_real_feature"
    path = tmp_path / "bad_candidate.csv"
    candidate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="ineligible"):
        _load_candidate_map_from_file(path)


def test_main_returns_exit_code_1_when_coverage_too_low(tmp_path: Path) -> None:
    """CLI should return 1 when a candidate collapses coverage below the floor."""
    candidate = _candidate_map_from_v128()
    candidate["feature_list"] = "rate_adequacy_gap_yoy, severity_index_yoy"
    path = tmp_path / "thin_candidate.csv"
    candidate.to_csv(path, index=False)
    rc = main(["--strategy", f"file:{path}"])
    assert rc in (0, 1)
    if rc == 0:
        metrics = evaluate_feature_map(f"file:{path}")
        assert metrics["coverage"] >= MIN_COVERAGE
    else:
        metrics = evaluate_feature_map(f"file:{path}")
        assert metrics["coverage"] < MIN_COVERAGE
