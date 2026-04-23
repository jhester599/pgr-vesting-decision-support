"""Tests for x12 BVPS target audit utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_monthly_dividend_series_aggregates_to_bme() -> None:
    from src.research.x12_bvps_target_audit import build_monthly_dividend_series

    dividends = pd.DataFrame(
        {"amount": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-10", "2024-01-18"]),
    )
    monthly_index = pd.date_range("2024-01-31", periods=2, freq="BME")

    result = build_monthly_dividend_series(dividends, monthly_index)

    assert result.loc[pd.Timestamp("2024-01-31")] == 3.0
    assert result.loc[pd.Timestamp("2024-02-29")] == 0.0


def test_build_adjusted_bvps_targets_adds_forward_window_dividends() -> None:
    from src.research.x12_bvps_target_audit import build_adjusted_bvps_targets

    idx = pd.date_range("2024-10-31", periods=4, freq="BME")
    bvps = pd.Series([50.0, 52.0, 40.0, 41.0], index=idx)
    dividends = pd.Series([0.0, 0.0, 12.0, 0.0], index=idx, name="dividend")

    result = build_adjusted_bvps_targets(bvps, dividends, horizons=(1,))

    assert result.loc[idx[1], "target_1m_bvps"] == 40.0
    assert result.loc[idx[1], "target_1m_adjusted_bvps"] == 52.0
    assert result.loc[idx[1], "target_1m_adjusted_bvps_growth"] == 0.0


def test_identify_bvps_discontinuities_flags_large_drop_with_dividend() -> None:
    from src.research.x12_bvps_target_audit import identify_bvps_discontinuities

    idx = pd.date_range("2024-10-31", periods=4, freq="BME")
    bvps = pd.Series([50.0, 52.0, 40.0, 41.0], index=idx)
    dividends = pd.Series([0.0, 0.0, 12.0, 0.0], index=idx, name="dividend")

    result = identify_bvps_discontinuities(
        bvps,
        dividends,
        pct_threshold=-0.10,
    )

    assert len(result) == 1
    assert result.iloc[0]["date"] == idx[2]
    assert result.iloc[0]["has_dividend_support"] is True


def test_summarize_x12_results_separates_raw_and_adjusted_variants() -> None:
    from src.research.x12_bvps_target_audit import summarize_x12_results

    detail = pd.DataFrame(
        [
            {
                "target_variant": "raw",
                "horizon_months": 1,
                "model_name": "baseline",
                "future_bvps_mae": 1.0,
                "growth_rmse": 0.2,
            },
            {
                "target_variant": "adjusted",
                "horizon_months": 1,
                "model_name": "baseline",
                "future_bvps_mae": 0.8,
                "growth_rmse": 0.1,
            },
        ]
    )

    result = summarize_x12_results(detail)

    assert set(result["target_variant"]) == {"raw", "adjusted"}
    assert result.loc[result["target_variant"] == "adjusted", "rank"].iloc[0] == 1
