from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.weekly_snapshot_experiments import (
    _asof_calendar_return,
    _build_snapshot_dates,
    _high_52w_ratio,
    _make_weekly_splitter,
    summarize_weekly_snapshot_results,
)


def test_build_snapshot_dates_uses_period_last_observation():
    idx = pd.bdate_range("2024-01-01", periods=15, freq="B")
    price_history = pd.DataFrame({"close": np.arange(len(idx), dtype=float) + 100.0}, index=idx)
    snapshot_dates = _build_snapshot_dates(price_history, snapshot_rule="W-FRI")
    assert list(snapshot_dates.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-12", "2024-01-19"]


def test_asof_calendar_return_is_point_in_time_safe():
    idx = pd.bdate_range("2023-01-02", "2024-12-31", freq="B")
    daily_close = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    snapshots = pd.DatetimeIndex([pd.Timestamp("2024-07-05")], name="date")
    result = _asof_calendar_return(daily_close, snapshots, months=6)
    lag_value = daily_close.asof(pd.Timestamp("2024-01-05"))
    expected = daily_close.asof(pd.Timestamp("2024-07-05")) / lag_value - 1.0
    assert result.iloc[0] == pytest.approx(expected)


def test_high_52w_ratio_is_bounded_by_one_when_at_high():
    idx = pd.bdate_range("2023-01-02", periods=300, freq="B")
    daily_close = pd.Series(np.linspace(100.0, 150.0, len(idx)), index=idx)
    snapshots = pd.DatetimeIndex([idx[-1]])
    ratio = _high_52w_ratio(daily_close, snapshots)
    assert ratio.iloc[0] == pytest.approx(1.0)


def test_make_weekly_splitter_has_expected_gap():
    splitter = _make_weekly_splitter(n_obs=400, train_weeks=156, test_weeks=13, gap_weeks=26)
    assert splitter.gap == 26
    assert splitter.test_size == 13


def test_summarize_weekly_snapshot_results_groups_candidates():
    detail = pd.DataFrame(
        [
            {
                "benchmark": "VXUS",
                "candidate_name": "ridge_lean_v1",
                "model_type": "ridge",
                "n_features": 12,
                "ic": 0.10,
                "hit_rate": 0.60,
                "oos_r2": -0.20,
                "mae": 0.12,
                "policy_return_sign": 0.05,
                "policy_return_tiered": 0.03,
                "policy_uplift_vs_sell_50_sign": 0.01,
                "policy_uplift_vs_sell_50_tiered": -0.01,
                "notes": "lean",
                "status": "ok",
            },
            {
                "benchmark": "BND",
                "candidate_name": "ridge_lean_v1",
                "model_type": "ridge",
                "n_features": 12,
                "ic": 0.12,
                "hit_rate": 0.58,
                "oos_r2": -0.10,
                "mae": 0.10,
                "policy_return_sign": 0.04,
                "policy_return_tiered": 0.02,
                "policy_uplift_vs_sell_50_sign": 0.02,
                "policy_uplift_vs_sell_50_tiered": 0.00,
                "notes": "lean",
                "status": "ok",
            },
        ]
    )
    summary = summarize_weekly_snapshot_results(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["candidate_name"] == "ridge_lean_v1"
    assert row["n_skipped"] == 0
    assert row["mean_ic"] == pytest.approx(0.11)
