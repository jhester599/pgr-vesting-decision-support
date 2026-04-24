"""Tests for x17 persistent BVPS utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_persistent_bvps_series_adds_cumulative_dividends() -> None:
    from src.research.x17_persistent_bvps import build_persistent_bvps_series

    index = pd.date_range("2024-01-31", periods=4, freq="BME")
    raw_bvps = pd.Series([100.0, 99.0, 101.0, 100.5], index=index)
    monthly_dividends = pd.Series([0.0, 1.0, 0.0, 0.5], index=index)

    result = build_persistent_bvps_series(raw_bvps, monthly_dividends)

    assert result.tolist() == pytest.approx([100.0, 100.0, 102.0, 102.0])


def test_build_persistent_bvps_targets_uses_future_persistent_growth() -> None:
    from src.research.x17_persistent_bvps import build_persistent_bvps_targets

    index = pd.date_range("2024-01-31", periods=4, freq="BME")
    persistent = pd.Series([100.0, 101.0, 103.0, 106.0], index=index)

    result = build_persistent_bvps_targets(persistent, horizons=(1, 3))

    assert result.loc[index[0], "target_1m_persistent_bvps"] == pytest.approx(101.0)
    assert result.loc[index[0], "target_1m_persistent_bvps_growth"] == pytest.approx(0.01)
    assert result.loc[index[0], "target_3m_persistent_bvps"] == pytest.approx(106.0)
    assert result.loc[index[0], "target_3m_persistent_bvps_growth"] == pytest.approx(0.06)


def test_summarize_x17_results_separates_variants() -> None:
    from src.research.x17_persistent_bvps import summarize_x17_results

    detail = pd.DataFrame(
        [
            {
                "target_variant": "raw",
                "horizon_months": 3,
                "model_name": "raw_model",
                "future_bvps_mae": 1.4,
                "growth_rmse": 0.06,
            },
            {
                "target_variant": "persistent",
                "horizon_months": 3,
                "model_name": "persistent_model",
                "future_bvps_mae": 1.1,
                "growth_rmse": 0.05,
            },
        ]
    )

    summary = summarize_x17_results(detail)

    assert set(summary["target_variant"]) == {"raw", "persistent"}
    assert summary.loc[summary["target_variant"] == "persistent", "rank"].iloc[0] == 1
