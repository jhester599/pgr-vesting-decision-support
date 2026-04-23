"""Tests for x13 adjusted decomposition utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_combine_adjusted_decomposition_predictions_uses_adjusted_bvps() -> None:
    from src.research.x13_adjusted_decomposition import (
        combine_adjusted_decomposition_predictions,
    )

    bvps = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "fold_idx": [0],
            "current_bvps": [50.0],
            "current_pb": [2.0],
            "implied_future_bvps": [52.0],
            "true_future_bvps": [53.0],
        }
    )
    pb = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "fold_idx": [0],
            "y_pred_pb": [2.1],
            "y_true_pb": [2.2],
        }
    )

    _, metrics = combine_adjusted_decomposition_predictions(
        bvps,
        pb,
        horizon_months=3,
        bvps_model_name="adjusted_bvps",
        pb_model_name="no_change_pb",
        target_variant="adjusted",
    )

    assert metrics["target_variant"] == "adjusted"
    assert metrics["implied_price_mae"] == pytest.approx(7.4)


def test_summarize_x13_results_separates_raw_and_adjusted_variants() -> None:
    from src.research.x13_adjusted_decomposition import summarize_x13_results

    detail = pd.DataFrame(
        [
            {
                "target_variant": "raw",
                "horizon_months": 3,
                "model_name": "raw_model",
                "implied_price_mae": 8.0,
                "implied_price_rmse": 10.0,
                "directional_hit_rate": 0.5,
            },
            {
                "target_variant": "adjusted",
                "horizon_months": 3,
                "model_name": "adj_model",
                "implied_price_mae": 7.0,
                "implied_price_rmse": 9.0,
                "directional_hit_rate": 0.6,
            },
        ]
    )

    result = summarize_x13_results(detail)

    assert set(result["target_variant"]) == {"raw", "adjusted"}
    assert result.loc[result["target_variant"] == "adjusted", "rank"].iloc[0] == 1


def test_x13_horizons_are_bounded_to_medium_term() -> None:
    from src.research.x13_adjusted_decomposition import X13_HORIZONS

    assert X13_HORIZONS == (3, 6)
