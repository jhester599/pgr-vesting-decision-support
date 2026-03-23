"""
Tests for generate_regime_breakdown() in src/reporting/backtest_report.py (v3.1).

Verifies:
  - Returns DataFrame with 4 quadrant rows
  - OOS R² per quadrant is between -1 and 1 (approximately)
  - Hit rate per quadrant is between 0 and 1
  - n_obs sums to total number of input results
  - Works without optional vix_series and sp500_returns (proxy mode)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass
from datetime import date, timedelta

from src.backtest.vesting_events import VestingEvent
from src.reporting.backtest_report import generate_regime_breakdown


@dataclass
class MockResult:
    event: VestingEvent
    benchmark: str
    target_horizon: int
    predicted_relative_return: float
    realized_relative_return: float
    signal_direction: str
    correct_direction: bool
    predicted_sell_pct: float
    ic_at_event: float
    hit_rate_at_event: float
    n_train_observations: int
    proxy_fill_fraction: float


def _make_results(n: int = 48) -> list[MockResult]:
    rng = np.random.default_rng(99)
    base = date(2019, 1, 31)
    results = []
    for i in range(n):
        d = base + timedelta(days=30 * i)
        ev = VestingEvent(
            event_date=d,
            rsu_type="time",
            horizon_6m_end=d + timedelta(days=180),
            horizon_12m_end=d + timedelta(days=365),
        )
        results.append(MockResult(
            event=ev,
            benchmark="VTI",
            target_horizon=6,
            predicted_relative_return=float(rng.normal(0, 0.05)),
            realized_relative_return=float(rng.normal(0, 0.05)),
            signal_direction="OUTPERFORM",
            correct_direction=bool(rng.integers(2)),
            predicted_sell_pct=0.5,
            ic_at_event=float(rng.uniform(0, 0.2)),
            hit_rate_at_event=float(rng.uniform(0.4, 0.7)),
            n_train_observations=80,
            proxy_fill_fraction=0.0,
        ))
    return results


class TestGenerateRegimeBreakdown:
    def test_returns_dataframe(self):
        results = _make_results()
        df = generate_regime_breakdown(results)
        assert isinstance(df, pd.DataFrame)

    def test_has_four_quadrants(self):
        results = _make_results(48)
        df = generate_regime_breakdown(results)
        assert len(df) == 4

    def test_quadrant_labels(self):
        results = _make_results()
        df = generate_regime_breakdown(results)
        expected = {"bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"}
        assert set(df.index) == expected

    def test_has_required_columns(self):
        results = _make_results()
        df = generate_regime_breakdown(results)
        for col in ["n_obs", "hit_rate", "mean_ic", "oos_r2"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_n_obs_sums_to_total(self):
        n = 48
        results = _make_results(n)
        df = generate_regime_breakdown(results)
        total_obs = df["n_obs"].sum(skipna=True)
        assert int(total_obs) == n, (
            f"Total n_obs ({total_obs}) should equal input length ({n})"
        )

    def test_hit_rate_between_zero_and_one(self):
        results = _make_results()
        df = generate_regime_breakdown(results)
        valid = df["hit_rate"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_empty_results_returns_empty(self):
        df = generate_regime_breakdown([])
        assert df.empty or len(df) == 0

    def test_with_vix_series(self):
        """Regime classification with real VIX series should work."""
        results = _make_results(48)
        dates = pd.DatetimeIndex([pd.Timestamp(r.event.event_date) for r in results])
        rng = np.random.default_rng(5)
        vix_series = pd.Series(
            rng.uniform(10, 40, len(results)), index=dates
        ).sort_index()
        df = generate_regime_breakdown(results, vix_series=vix_series, vix_threshold=20.0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_with_sp500_returns(self):
        """Regime classification with SP500 returns should work."""
        results = _make_results(48)
        dates = pd.DatetimeIndex([pd.Timestamp(r.event.event_date) for r in results])
        rng = np.random.default_rng(6)
        sp500 = pd.Series(
            rng.normal(0.10, 0.20, len(results)), index=dates
        ).sort_index()
        df = generate_regime_breakdown(results, sp500_returns=sp500)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_oos_r2_is_finite_or_nan(self):
        """OOS R² per quadrant should be finite or NaN (never Inf)."""
        results = _make_results(60)
        df = generate_regime_breakdown(results)
        for val in df["oos_r2"].dropna():
            assert np.isfinite(val), f"OOS R² should be finite, got {val}"
