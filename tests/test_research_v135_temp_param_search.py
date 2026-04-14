"""Tests for v135 Path B temperature-parameter search harness."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v135_temp_param_search import (
    DEFAULT_HIGH_THRESH,
    DEFAULT_LOW_THRESH,
    DEFAULT_N_TEMPS,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_WARMUP,
    MAX_WARMUP,
    MIN_WARMUP,
    _build_temperature_grid,
    evaluate_temperature_config,
    main,
)


def _write_fold_detail(
    tmp_path: Path,
    y_true: np.ndarray,
    path_b_prob: np.ndarray,
) -> Path:
    """Write a minimal fold-detail CSV for v135 tests."""
    df = pd.DataFrame(
        {
            "fold": 1,
            "train_start": "2011-02-28",
            "train_end": "2016-01-29",
            "train_obs": 60,
            "test_date": pd.date_range("2016-01-01", periods=len(y_true), freq="ME").strftime(
                "%Y-%m-%d"
            ),
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


def test_default_config_matches_current_v131_threshold_baseline() -> None:
    """The default v135 config should reproduce the current v131 search baseline."""
    metrics = evaluate_temperature_config(
        temp_min=DEFAULT_TEMP_MIN,
        temp_max=DEFAULT_TEMP_MAX,
        n_temps=DEFAULT_N_TEMPS,
        warmup=DEFAULT_WARMUP,
        low_thresh=DEFAULT_LOW_THRESH,
        high_thresh=DEFAULT_HIGH_THRESH,
    )
    assert metrics["covered_ba"] == pytest.approx(0.6322, abs=0.01)
    assert metrics["coverage"] == pytest.approx(0.4524, abs=0.02)


def test_temp_min_greater_than_temp_max_raises() -> None:
    """Invalid temperature bounds should fail fast."""
    with pytest.raises(ValueError, match="temp_min must be <="):
        _build_temperature_grid(3.0, 0.5, 51)


def test_warmup_below_minimum_raises() -> None:
    """Warmup below the stability floor should raise ValueError."""
    with pytest.raises(ValueError, match="warmup must be >="):
        evaluate_temperature_config(temp_min=0.5, temp_max=3.0, warmup=MIN_WARMUP - 1)


def test_warmup_above_maximum_raises() -> None:
    """Warmup above the evaluation ceiling should raise ValueError."""
    with pytest.raises(ValueError, match="warmup must be <="):
        evaluate_temperature_config(temp_min=0.5, temp_max=3.0, warmup=MAX_WARMUP + 1)


def test_log_grid_used_when_ratio_exceeds_ten() -> None:
    """Large temp ranges should use log-spaced search grids."""
    grid = _build_temperature_grid(0.5, 6.0, 11)
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0])


def test_main_returns_exit_code_1_when_coverage_too_low(tmp_path: Path) -> None:
    """CLI returns 1 when the fixed thresholds abstain nearly everything."""
    y_true = np.array([0, 1] * 30, dtype=int)
    path_b_prob = np.full(len(y_true), 0.50, dtype=float)
    csv_path = _write_fold_detail(tmp_path, y_true, path_b_prob)

    with patch(
        "results.research.v135_temp_param_search.FOLD_DETAIL_PATH",
        csv_path,
    ):
        rc = main(
            [
                "--temp-min",
                "0.5",
                "--temp-max",
                "3.0",
                "--warmup",
                "24",
                "--low",
                "0.15",
                "--high",
                "0.70",
            ]
        )
    assert rc == 1
