"""Tests for the v146 threshold sweep harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v146_threshold_sweep import DEFAULT_CANDIDATE_PATH, evaluate_threshold_candidate, main


def _write_fold_detail(tmp_path: Path, y_true: np.ndarray, path_b_prob: np.ndarray) -> Path:
    df = pd.DataFrame(
        {
            "fold": 1,
            "train_start": "2011-02-28",
            "train_end": "2016-01-29",
            "train_obs": 60,
            "test_date": pd.date_range("2016-01-01", periods=len(y_true), freq="ME").strftime("%Y-%m-%d"),
            "y_true": y_true,
            "path_b_prob": path_b_prob,
            "path_a_prob": path_b_prob,
            "path_a_available_benchmarks": 6,
            "path_a_weight_sum": 1.0,
        }
    )
    csv_path = tmp_path / "v125_portfolio_target_fold_detail.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_low_greater_than_high_raises() -> None:
    with pytest.raises(ValueError, match="low must be < high"):
        evaluate_threshold_candidate(0.50, 0.50)


def test_candidate_file_has_expected_keys() -> None:
    payload = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {"low", "high"}


def test_main_returns_exit_code_1_when_coverage_too_low(tmp_path: Path) -> None:
    y_true = np.array([0, 1] * 30, dtype=int)
    path_b_prob = np.full(len(y_true), 0.50, dtype=float)
    csv_path = _write_fold_detail(tmp_path, y_true, path_b_prob)
    with patch("results.research.v146_threshold_sweep.FOLD_DETAIL_PATH", csv_path):
        rc = main(["--low", "0.15", "--high", "0.70"])
    assert rc == 1
