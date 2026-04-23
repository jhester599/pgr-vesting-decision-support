"""Tests for x9 BVPS bridge research utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _monthly_frame(n_rows: int = 36) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2020-01-31", periods=n_rows, freq="BME")
    bvps = pd.Series(np.linspace(20.0, 30.0, n_rows), index=dates)
    frame = pd.DataFrame(
        {
            "npw_growth_yoy": np.linspace(0.02, 0.18, n_rows),
            "underwriting_margin_ttm": np.linspace(0.01, 0.08, n_rows),
            "combined_ratio_ttm": np.linspace(97.0, 92.0, n_rows),
            "monthly_combined_ratio_delta": np.linspace(2.0, -3.0, n_rows),
            "pgr_premium_to_surplus": np.linspace(2.0, 2.5, n_rows),
            "unrealized_gain_pct_equity": np.linspace(-0.1, 0.05, n_rows),
            "real_rate_10y": np.linspace(-0.01, 0.02, n_rows),
            "buyback_yield": np.linspace(0.0, 0.03, n_rows),
            "pb_ratio": np.linspace(2.0, 3.0, n_rows),
            "investment_book_yield": np.linspace(0.03, 0.05, n_rows),
        },
        index=dates,
    )
    return frame, bvps


def test_build_bvps_bridge_features_uses_only_current_and_lagged_values() -> None:
    from src.research.x9_bvps_bridge import build_bvps_bridge_features

    frame, bvps = _monthly_frame()

    result = build_bvps_bridge_features(frame, bvps)

    assert result.loc[bvps.index[6], "current_bvps"] == bvps.iloc[6]
    assert result.loc[bvps.index[6], "bvps_growth_1m"] == (
        bvps.iloc[6] / bvps.iloc[5] - 1.0
    )
    assert result.loc[bvps.index[12], "bvps_yoy_dollar_change"] == (
        bvps.iloc[12] - bvps.iloc[0]
    )
    assert result.loc[bvps.index[0], "month_of_year"] == 1
    assert result.loc[bvps.index[11], "q4_flag"] == 1


def test_build_bvps_interactions_is_pre_registered_and_bounded() -> None:
    from src.research.x9_bvps_bridge import build_bvps_interactions

    frame, bvps = _monthly_frame()
    features = frame.copy()
    features["current_bvps"] = bvps
    features["bvps_growth_3m"] = bvps.pct_change(3)

    result = build_bvps_interactions(features)

    assert "premium_growth_x_underwriting_margin" in result.columns
    assert "premium_to_surplus_x_cr_delta" in result.columns
    assert "buyback_yield_x_pb_ratio" in result.columns
    assert len(result.columns) <= 10


def test_evaluate_x9_baseline_uses_fold_local_trailing_growth() -> None:
    from src.research.x9_bvps_bridge import evaluate_x9_bvps_baseline

    frame, bvps = _monthly_frame(72)
    y = bvps.pct_change(1).shift(-1).rename("target_1m_bvps_growth")

    predictions, metrics = evaluate_x9_bvps_baseline(
        frame,
        y,
        current_bvps=bvps,
        baseline_name="trailing_3m_growth",
        target_kind="growth",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert not predictions.empty
    assert predictions["date"].is_monotonic_increasing
    assert metrics["model_name"] == "trailing_3m_growth"
    assert metrics["n_features"] == 0


def test_evaluate_x9_regularized_model_reports_feature_stability() -> None:
    from src.research.x9_bvps_bridge import evaluate_x9_bvps_model

    frame, bvps = _monthly_frame(84)
    X = build_test_feature_matrix(frame, bvps)
    y = (0.5 * X["bvps_growth_1m"].fillna(0.0)).shift(-1)
    y = y.rename("target_1m_bvps_growth")

    _, metrics, stability = evaluate_x9_bvps_model(
        X,
        y,
        current_bvps=bvps,
        model_name="elastic_net_bridge",
        feature_columns=["current_bvps", "bvps_growth_1m", "npw_growth_yoy"],
        target_kind="growth",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert metrics["model_name"] == "elastic_net_bridge"
    assert set(stability["feature"]) == {
        "current_bvps",
        "bvps_growth_1m",
        "npw_growth_yoy",
    }
    assert stability["fold_count"].max() > 0


def build_test_feature_matrix(frame: pd.DataFrame, bvps: pd.Series) -> pd.DataFrame:
    """Build a compact test matrix without importing production data."""
    X = frame.copy()
    X["current_bvps"] = bvps
    X["bvps_growth_1m"] = bvps.pct_change(1)
    return X
