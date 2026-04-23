"""Tests for x21 dividend target-scale helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_scaled_size_targets_back_transforms_to_raw_dollars() -> None:
    from src.research.x21_dividend_target_scales import (
        back_transform_scaled_prediction,
        build_scaled_size_targets,
    )

    annual = pd.DataFrame(
        {
            "special_dividend_excess": [2.0, 3.0],
            "current_bvps": [20.0, 30.0],
            "persistent_bvps": [22.0, 33.0],
            "close_price": [100.0, 120.0],
        },
        index=pd.to_datetime(["2023-11-30", "2024-11-29"]),
    )

    result = build_scaled_size_targets(annual)

    assert abs(result.loc[pd.Timestamp("2023-11-30"), "target_to_current_bvps"] - 0.1) < 1e-12
    assert (
        abs(
            back_transform_scaled_prediction(
                result.loc[pd.Timestamp("2023-11-30"), "target_to_current_bvps"],
                result.loc[pd.Timestamp("2023-11-30"), "current_bvps"],
            )
            - 2.0
        )
        < 1e-12
    )


def test_summarize_scaled_size_results_ranks_by_dollar_mae() -> None:
    from src.research.x21_dividend_target_scales import summarize_scaled_size_results

    detail = pd.DataFrame(
        [
            {"target_scale": "raw_dollars", "model_name": "a", "dollar_mae": 2.0},
            {"target_scale": "to_price", "model_name": "b", "dollar_mae": 1.0},
        ]
    )

    summary = summarize_scaled_size_results(detail)

    assert list(summary["target_scale"]) == ["to_price", "raw_dollars"]
    assert list(summary["rank"]) == [1, 2]


def test_build_scaled_size_targets_handles_zero_or_missing_scales_conservatively() -> None:
    from src.research.x21_dividend_target_scales import build_scaled_size_targets

    annual = pd.DataFrame(
        {
            "special_dividend_excess": [2.0, 3.0],
            "current_bvps": [0.0, 30.0],
            "persistent_bvps": [None, 33.0],
            "close_price": [100.0, 0.0],
        },
        index=pd.to_datetime(["2023-11-30", "2024-11-29"]),
    )

    result = build_scaled_size_targets(annual)

    assert pd.isna(result.loc[pd.Timestamp("2023-11-30"), "target_to_current_bvps"])
    assert pd.isna(result.loc[pd.Timestamp("2023-11-30"), "target_to_persistent_bvps"])
    assert pd.isna(result.loc[pd.Timestamp("2024-11-29"), "target_to_price"])
