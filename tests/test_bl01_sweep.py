"""Unit tests for the BL-01 tau/risk_aversion sweep harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_make_scenario_returns_correct_shapes() -> None:
    from results.research.bl01_tau_sweep_eval import BENCHMARKS, _make_scenario
    returns_df, signals = _make_scenario(seed=0, benchmarks=BENCHMARKS, n_months=60)
    assert returns_df.shape == (60, len(BENCHMARKS))
    assert set(signals.keys()) == set(BENCHMARKS)


def test_make_scenario_different_seeds_differ() -> None:
    from results.research.bl01_tau_sweep_eval import BENCHMARKS, _make_scenario
    returns_a, signals_a = _make_scenario(seed=0, benchmarks=BENCHMARKS)
    returns_b, signals_b = _make_scenario(seed=1, benchmarks=BENCHMARKS)
    assert not returns_a.equals(returns_b)
    ics_a = [signals_a[b].mean_ic for b in BENCHMARKS]
    ics_b = [signals_b[b].mean_ic for b in BENCHMARKS]
    assert ics_a != ics_b


def test_rank_corr_perfect_positive() -> None:
    from results.research.bl01_tau_sweep_eval import _rank_corr
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_rank_corr(x, x) - 1.0) < 1e-9


def test_rank_corr_perfect_negative() -> None:
    from results.research.bl01_tau_sweep_eval import _rank_corr
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_rank_corr(x, x[::-1]) + 1.0) < 1e-9


def test_rank_corr_short_input_returns_zero() -> None:
    from results.research.bl01_tau_sweep_eval import _rank_corr
    assert _rank_corr(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 0.0


def test_compute_ic_rank_correlation_positive_alignment() -> None:
    """Weights concentrating in high-IC benchmarks → high rank correlation."""
    from results.research.bl01_tau_sweep_eval import (
        _compute_ic_rank_correlation,
        _make_signal,
    )
    benchmarks = ["A", "B", "C", "D"]
    ics = [0.20, 0.15, 0.10, 0.05]
    weights = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}
    signals = {bm: _make_signal(bm, ic) for bm, ic in zip(benchmarks, ics)}
    corr = _compute_ic_rank_correlation(weights, signals)
    assert corr > 0.9


def test_compute_ic_rank_correlation_negative_alignment() -> None:
    """Weights concentrating in low-IC benchmarks → low (negative) rank correlation."""
    from results.research.bl01_tau_sweep_eval import (
        _compute_ic_rank_correlation,
        _make_signal,
    )
    benchmarks = ["A", "B", "C", "D"]
    ics = [0.20, 0.15, 0.10, 0.05]
    weights = {"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}
    signals = {bm: _make_signal(bm, ic) for bm, ic in zip(benchmarks, ics)}
    corr = _compute_ic_rank_correlation(weights, signals)
    assert corr < -0.9


def test_select_winner_keeps_incumbent_when_delta_below_threshold() -> None:
    from results.research.bl01_tau_sweep_eval import select_winner
    rows = [
        {"tau": 0.05, "risk_aversion": 2.5, "mean_rank_corr": 0.50, "fallback_rate": 0.05},
        {"tau": 0.10, "risk_aversion": 2.0, "mean_rank_corr": 0.54, "fallback_rate": 0.05},
    ]
    result = select_winner(rows, incumbent_tau=0.05, incumbent_ra=2.5, win_threshold=0.05)
    assert result["bl_tau_winner"] == 0.05
    assert result["bl_risk_aversion_winner"] == 2.5
    assert result["recommendation"] == "keep_incumbent"


def test_select_winner_returns_winner_when_delta_exceeds_threshold() -> None:
    from results.research.bl01_tau_sweep_eval import select_winner
    rows = [
        {"tau": 0.05, "risk_aversion": 2.5, "mean_rank_corr": 0.50, "fallback_rate": 0.05},
        {"tau": 0.10, "risk_aversion": 2.0, "mean_rank_corr": 0.60, "fallback_rate": 0.05},
    ]
    result = select_winner(rows, incumbent_tau=0.05, incumbent_ra=2.5, win_threshold=0.05)
    assert result["bl_tau_winner"] == 0.10
    assert result["bl_risk_aversion_winner"] == 2.0
    assert result["recommendation"] == "update_bl_params"


def test_select_winner_output_keys() -> None:
    from results.research.bl01_tau_sweep_eval import select_winner
    rows = [
        {"tau": 0.05, "risk_aversion": 2.5, "mean_rank_corr": 0.50, "fallback_rate": 0.05},
    ]
    result = select_winner(rows, incumbent_tau=0.05, incumbent_ra=2.5, win_threshold=0.05)
    for key in [
        "bl_tau_winner",
        "bl_risk_aversion_winner",
        "incumbent_tau",
        "incumbent_risk_aversion",
        "winner_rank_corr",
        "incumbent_rank_corr",
        "delta_rank_corr",
        "win_threshold",
        "recommendation",
    ]:
        assert key in result, f"Missing key: {key}"
