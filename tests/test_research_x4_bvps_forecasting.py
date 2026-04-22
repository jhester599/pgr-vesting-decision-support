"""Tests for x4 BVPS forecasting research utilities."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _feature_frame(n_rows: int = 96) -> pd.DataFrame:
    dates = pd.date_range("2016-01-29", periods=n_rows, freq="BME")
    growth = np.linspace(-0.02, 0.12, n_rows)
    margin = np.sin(np.linspace(0.0, 7.0, n_rows))
    return pd.DataFrame(
        {"book_value_per_share_growth_yoy": growth, "margin": margin},
        index=dates,
    )


def _current_bvps(X: pd.DataFrame) -> pd.Series:
    values = 25.0 + np.linspace(0.0, 20.0, len(X))
    return pd.Series(values, index=X.index, name="current_bvps")


def _bvps_growth_target(X: pd.DataFrame) -> pd.Series:
    values = 0.01 + 0.15 * X["book_value_per_share_growth_yoy"]
    return pd.Series(values, index=X.index, name="target_1m_bvps_growth")


def test_normalize_bvps_to_business_month_end() -> None:
    from src.research.x4_bvps_forecasting import normalize_bvps_monthly

    raw = pd.DataFrame(
        {"book_value_per_share": [30.0, 31.0]},
        index=pd.to_datetime(["2024-03-31", "2024-04-30"]),
    )

    result = normalize_bvps_monthly(raw, filing_lag_months=2)

    assert list(result.index) == [
        pd.Timestamp("2024-05-31"),
        pd.Timestamp("2024-06-28"),
    ]
    assert result.loc[pd.Timestamp("2024-05-31")] == 30.0


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_evaluate_bvps_regressor_outputs_implied_future_bvps(
    horizon: int,
) -> None:
    from src.research.x4_bvps_forecasting import evaluate_bvps_regressor

    X = _feature_frame(144)
    y = _bvps_growth_target(X)
    current_bvps = _current_bvps(X)

    predictions, metrics = evaluate_bvps_regressor(
        X,
        y,
        current_bvps=current_bvps,
        model_name="ridge_bvps_growth",
        feature_columns=["book_value_per_share_growth_yoy", "margin"],
        target_kind="growth",
        target_horizon_months=horizon,
        purge_buffer=0,
        train_window_months=48,
        test_window_months=6,
    )

    assert not predictions.empty
    assert predictions["date"].is_monotonic_increasing
    expected_bvps = (
        predictions["current_bvps"] * (1.0 + predictions["y_pred_growth"])
    )
    pd.testing.assert_series_equal(
        predictions["implied_future_bvps"],
        expected_bvps,
        check_names=False,
    )
    assert (
        predictions["date"]
        > predictions["train_end_date"] + pd.offsets.MonthEnd(horizon)
    ).all()
    assert metrics["model_name"] == "ridge_bvps_growth"
    assert metrics["target_kind"] == "growth"


def test_log_growth_predictions_convert_to_simple_growth() -> None:
    from src.research.x4_bvps_forecasting import evaluate_bvps_baseline

    X = _feature_frame(72)
    growth = _bvps_growth_target(X)
    log_growth = np.log1p(growth).rename("target_1m_log_bvps_growth")
    current_bvps = _current_bvps(X)

    predictions, metrics = evaluate_bvps_baseline(
        X,
        log_growth,
        current_bvps=current_bvps,
        baseline_name="drift_bvps_growth",
        target_kind="log_growth",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    np.testing.assert_allclose(
        predictions["y_pred_growth"],
        np.expm1(predictions["y_pred_target"]),
    )
    first_fold = predictions[predictions["fold_idx"] == 0]
    train_window = growth.loc[
        first_fold["train_start_date"].iloc[0]:first_fold["train_end_date"].iloc[0]
    ]
    assert first_fold["y_pred_growth"].iloc[0] == pytest.approx(
        train_window.mean()
    )
    assert metrics["target_kind"] == "log_growth"


def test_no_change_bvps_baseline_predicts_current_bvps() -> None:
    from src.research.x4_bvps_forecasting import evaluate_bvps_baseline

    X = _feature_frame(72)
    y = _bvps_growth_target(X)
    current_bvps = _current_bvps(X)

    predictions, metrics = evaluate_bvps_baseline(
        X,
        y,
        current_bvps=current_bvps,
        baseline_name="no_change_bvps",
        target_kind="growth",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert predictions["y_pred_growth"].eq(0.0).all()
    pd.testing.assert_series_equal(
        predictions["implied_future_bvps"],
        predictions["current_bvps"],
        check_names=False,
    )
    assert metrics["future_bvps_mae"] > 0.0


def test_bvps_regressor_treats_nonfinite_features_as_missing() -> None:
    from src.research.x4_bvps_forecasting import evaluate_bvps_regressor

    X = _feature_frame(72)
    X["empty_feature"] = np.nan
    X["infinite_feature"] = np.inf
    y = _bvps_growth_target(X)
    current_bvps = _current_bvps(X)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predictions, metrics = evaluate_bvps_regressor(
            X,
            y,
            current_bvps=current_bvps,
            model_name="ridge_bvps_growth",
            feature_columns=["empty_feature", "infinite_feature", "margin"],
            target_kind="growth",
            target_horizon_months=1,
            purge_buffer=0,
            train_window_months=36,
            test_window_months=6,
        )

    assert not predictions.empty
    assert np.isfinite(predictions["y_pred_growth"]).all()
    assert metrics["n_features"] == 3


def test_bvps_summary_ranks_lower_future_bvps_mae_first() -> None:
    from src.research.x4_bvps_forecasting import summarize_bvps_results

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 1,
                "model_name": "no_change_bvps",
                "target_kind": "log_growth",
                "future_bvps_mae": 1.2,
                "growth_rmse": 0.05,
                "directional_hit_rate": 0.50,
            },
            {
                "horizon_months": 1,
                "model_name": "no_change_bvps",
                "target_kind": "growth",
                "future_bvps_mae": 1.2,
                "growth_rmse": 0.05,
                "directional_hit_rate": 0.50,
            },
            {
                "horizon_months": 1,
                "model_name": "ridge_bvps_growth",
                "target_kind": "growth",
                "future_bvps_mae": 0.8,
                "growth_rmse": 0.04,
                "directional_hit_rate": 0.60,
            },
        ]
    )

    summary = summarize_bvps_results(detail)

    assert summary.iloc[0]["model_name"] == "ridge_bvps_growth"
    assert summary.iloc[0]["rank"] == 1
    assert bool(summary.iloc[0]["beats_no_change_bvps"]) is True
    tied_baselines = summary[summary["model_name"] == "no_change_bvps"]
    assert tied_baselines.iloc[0]["target_kind"] == "growth"


def test_unsupported_bvps_regressor_name_raises() -> None:
    from src.research.x4_bvps_forecasting import build_bvps_regressor

    with pytest.raises(ValueError, match="Unsupported BVPS regressor"):
        build_bvps_regressor("wide_net")
