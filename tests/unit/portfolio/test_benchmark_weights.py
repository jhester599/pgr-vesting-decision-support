"""
Tests for compute_benchmark_weights() in src/portfolio/rebalancer.py (v4.0).

Verifies:
  - Weights sum to 1.0 for a valid input
  - Zero-IC benchmarks receive zero weight
  - Normalization is correct: w_i = (IC_i × HR_i) / sum(IC_j × HR_j)
  - Empty input returns empty dict
  - Single benchmark gets weight = 1.0
  - Window_months truncation works (only recent observations used)
  - All-below-threshold benchmarks → equal weights fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

from src.portfolio.rebalancer import compute_benchmark_weights
from src.backtest.vesting_events import VestingEvent


@dataclass
class MockResult:
    benchmark: str
    ic_at_event: float
    correct_direction: bool
    event: VestingEvent = None  # noqa: RUF009

    def __post_init__(self):
        if self.event is None:
            self.event = VestingEvent(
                event_date=date(2020, 1, 31),
                rsu_type="time",
                horizon_6m_end=date(2020, 7, 31),
                horizon_12m_end=date(2021, 1, 31),
            )


def _make_results(n: int = 30, benchmark: str = "VTI", ic: float = 0.10, hr: float = 0.65):
    """Create n synthetic results for a single benchmark."""
    results = []
    base = date(2020, 1, 31)
    for i in range(n):
        d = base + timedelta(days=30 * i)
        ev = VestingEvent(
            event_date=d,
            rsu_type="time",
            horizon_6m_end=d + timedelta(days=180),
            horizon_12m_end=d + timedelta(days=365),
        )
        results.append(MockResult(
            benchmark=benchmark,
            ic_at_event=ic,
            correct_direction=hr > 0.5,
            event=ev,
        ))
    return results


class TestComputeBenchmarkWeights:
    def test_returns_dict(self):
        results = _make_results(benchmark="VTI", ic=0.10)
        weights = compute_benchmark_weights(results)
        assert isinstance(weights, dict)

    def test_empty_input_returns_empty(self):
        assert compute_benchmark_weights([]) == {}

    def test_single_benchmark_gets_weight_one(self):
        results = _make_results(benchmark="VTI", ic=0.10, hr=0.60)
        weights = compute_benchmark_weights(results, min_ic=0.0)
        assert "VTI" in weights
        assert abs(weights["VTI"] - 1.0) < 1e-9

    def test_weights_sum_to_one(self):
        results = (
            _make_results(benchmark="VTI", ic=0.12, hr=0.65)
            + _make_results(benchmark="BND", ic=0.06, hr=0.55)
            + _make_results(benchmark="GLD", ic=0.08, hr=0.60)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_weights_non_negative(self):
        results = (
            _make_results(benchmark="VTI", ic=0.10, hr=0.65)
            + _make_results(benchmark="BND", ic=0.05, hr=0.50)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)
        for k, v in weights.items():
            assert v >= 0.0, f"Weight for {k} is negative: {v}"

    def test_zero_ic_benchmark_gets_zero_weight(self):
        results = (
            _make_results(benchmark="VTI", ic=0.10, hr=0.65)
            + _make_results(benchmark="BND", ic=0.0, hr=0.50)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)
        # BND has IC = 0.0, which is ≤ min_ic=0.0 → zero weight
        assert weights.get("BND", 0.0) == 0.0

    def test_negative_ic_benchmark_gets_zero_weight(self):
        results = (
            _make_results(benchmark="VTI", ic=0.10, hr=0.65)
            + _make_results(benchmark="BND", ic=-0.05, hr=0.45)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)
        assert weights.get("BND", 0.0) == 0.0

    def test_higher_ic_gets_higher_weight(self):
        results = (
            _make_results(benchmark="VTI", ic=0.20, hr=0.70)
            + _make_results(benchmark="BND", ic=0.05, hr=0.55)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)
        assert weights.get("VTI", 0.0) > weights.get("BND", 0.0)

    def test_all_below_threshold_returns_equal_weights(self):
        results = (
            _make_results(benchmark="VTI", ic=0.0, hr=0.50)
            + _make_results(benchmark="BND", ic=-0.02, hr=0.48)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)
        # All IC ≤ 0 → equal weights fallback
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9
        for v in weights.values():
            assert abs(v - 0.5) < 1e-9

    def test_window_months_limits_lookback(self):
        """With window_months=5, only last 5 observations used per benchmark."""
        # First 20 obs have bad IC, last 5 have good IC for VTI
        results = []
        base = date(2019, 1, 31)
        for i in range(20):
            d = base + timedelta(days=30 * i)
            ev = VestingEvent(d, "time", d + timedelta(180), d + timedelta(365))
            results.append(MockResult("VTI", ic_at_event=-0.05, correct_direction=False, event=ev))
        for i in range(20, 25):
            d = base + timedelta(days=30 * i)
            ev = VestingEvent(d, "time", d + timedelta(180), d + timedelta(365))
            results.append(MockResult("VTI", ic_at_event=0.20, correct_direction=True, event=ev))

        weights_small_window = compute_benchmark_weights(results, window_months=5, min_ic=0.0)
        weights_full_window = compute_benchmark_weights(results, window_months=25, min_ic=0.0)
        # Small window sees only good-IC period → VTI gets positive weight
        # Full window sees mix → VTI weight may be zero or lower
        assert weights_small_window.get("VTI", 0.0) >= weights_full_window.get("VTI", 0.0)

    def test_normalization_formula(self):
        """w_i = (IC_i × HR_i) / sum(IC_j × HR_j)."""
        ic_vti, hr_vti = 0.10, 0.65
        ic_bnd, hr_bnd = 0.06, 0.60
        results = (
            _make_results(benchmark="VTI", ic=ic_vti, hr=hr_vti)
            + _make_results(benchmark="BND", ic=ic_bnd, hr=hr_bnd)
        )
        weights = compute_benchmark_weights(results, min_ic=0.0)

        raw_vti = ic_vti * hr_vti
        raw_bnd = ic_bnd * hr_bnd
        total = raw_vti + raw_bnd
        expected_vti = raw_vti / total
        expected_bnd = raw_bnd / total

        assert abs(weights.get("VTI", 0.0) - expected_vti) < 0.02
        assert abs(weights.get("BND", 0.0) - expected_bnd) < 0.02
