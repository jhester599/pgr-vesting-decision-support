"""Tests for x1 absolute PGR and special-dividend target utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_forward_return_targets_align_future_months_and_mask_tail() -> None:
    from src.research.x1_targets import build_forward_return_targets

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    prices = pd.Series([100.0, 110.0, 121.0, 100.0, 90.0, 99.0], index=dates)

    targets = build_forward_return_targets(prices, horizons=(1, 3))

    assert abs(targets.loc[dates[0], "target_1m_return"] - 0.10) < 1e-12
    assert targets.loc[dates[0], "target_1m_up"] == 1
    assert targets.loc[dates[0], "target_3m_return"] == 0.0
    assert targets.loc[dates[0], "target_3m_up"] == 0
    assert abs(targets.loc[dates[1], "target_1m_log_return"] - np.log(1.10)) < 1e-12
    assert pd.isna(targets.loc[dates[-1], "target_1m_return"])
    assert pd.isna(targets.loc[dates[-3], "target_3m_return"])


def test_decomposition_targets_use_future_bvps_and_future_pb() -> None:
    from src.research.x1_targets import build_decomposition_targets

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    prices = pd.Series([100.0, 110.0, 132.0, 156.0], index=dates)
    bvps = pd.Series([20.0, 22.0, 24.0, 26.0], index=dates)

    targets = build_decomposition_targets(prices, bvps, horizons=(2,))

    assert targets.loc[dates[0], "target_2m_bvps"] == 24.0
    assert abs(targets.loc[dates[0], "target_2m_bvps_growth"] - 0.20) < 1e-12
    log_growth = targets.loc[dates[0], "target_2m_log_bvps_growth"]
    assert abs(log_growth - np.log(1.20)) < 1e-12
    assert targets.loc[dates[0], "target_2m_pb"] == 132.0 / 24.0
    assert abs(targets.loc[dates[0], "target_2m_log_pb"] - np.log(132.0 / 24.0)) < 1e-12
    assert pd.isna(targets.loc[dates[-2], "target_2m_bvps"])


def test_special_dividend_targets_use_november_snapshot_and_inferred_baseline() -> None:
    from src.research.x1_targets import build_special_dividend_targets

    monthly_dates = pd.to_datetime(
        [
            "2020-11-30",
            "2021-11-30",
            "2022-11-30",
        ]
    )
    monthly = pd.DataFrame(
        {
            "book_value_per_share": [20.0, 25.0, 30.0],
            "close_price": [90.0, 100.0, 150.0],
            "net_income_ttm_per_share": [5.0, 8.0, 10.0],
        },
        index=monthly_dates,
    )
    dividends = pd.DataFrame(
        {
            "amount": [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                1.50,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
            ]
        },
        index=pd.to_datetime(
            [
                "2020-03-01",
                "2020-06-01",
                "2020-09-01",
                "2020-12-01",
                "2021-03-01",
                "2021-06-01",
                "2022-01-15",
                "2022-03-01",
                "2022-06-01",
                "2022-09-01",
                "2022-12-01",
                "2023-03-01",
            ]
        ),
    )

    targets = build_special_dividend_targets(monthly, dividends)

    first = targets.loc[pd.Timestamp("2020-11-30")]
    assert first["target_year"] == 2021
    assert first["regular_baseline_dividend"] == 0.10
    assert first["q1_dividend_total"] == 0.10
    assert first["special_dividend_occurred"] == 0
    assert first["special_dividend_excess"] == 0.0

    second = targets.loc[pd.Timestamp("2021-11-30")]
    assert second["target_year"] == 2022
    assert second["regular_baseline_dividend"] == 0.10
    assert second["q1_dividend_total"] == 1.60
    assert second["special_dividend_occurred"] == 1
    assert abs(second["special_dividend_excess"] - 1.50) < 1e-12
    assert abs(second["special_dividend_excess_to_bvps"] - 1.50 / 25.0) < 1e-12
    assert abs(second["special_dividend_excess_to_price"] - 1.50 / 100.0) < 1e-12
    assert abs(second["special_dividend_excess_to_net_income"] - 1.50 / 8.0) < 1e-12


def test_special_dividend_targets_do_not_use_post_november_features() -> None:
    from src.research.x1_targets import build_special_dividend_targets

    monthly = pd.DataFrame(
        {
            "book_value_per_share": [20.0, 999.0],
            "close_price": [100.0, 999.0],
        },
        index=pd.to_datetime(["2021-11-30", "2021-12-31"]),
    )
    dividends = pd.DataFrame(
        {"amount": [0.10, 1.10]},
        index=pd.to_datetime(["2021-09-01", "2022-01-15"]),
    )

    targets = build_special_dividend_targets(monthly, dividends)

    row = targets.loc[pd.Timestamp("2021-11-30")]
    assert row["snapshot_month"] == 11
    assert row["snapshot_bvps"] == 20.0
    assert row["snapshot_price"] == 100.0
