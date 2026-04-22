"""Tests for x5 P/B decomposition research utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _feature_frame(n_rows: int = 84) -> pd.DataFrame:
    dates = pd.date_range("2017-01-31", periods=n_rows, freq="BME")
    value = np.linspace(-0.5, 0.5, n_rows)
    rates = np.cos(np.linspace(0.0, 5.0, n_rows))
    return pd.DataFrame({"pb_ratio": 2.5 + value, "rates": rates}, index=dates)


def _current_pb(X: pd.DataFrame) -> pd.Series:
    values = 2.2 + 0.2 * np.sin(np.linspace(0.0, 4.0, len(X)))
    return pd.Series(values, index=X.index, name="current_pb")


def _future_pb_target(X: pd.DataFrame) -> pd.Series:
    values = 2.1 + 0.15 * X["pb_ratio"] - 0.05 * X["rates"]
    return pd.Series(values, index=X.index, name="target_1m_pb")


def test_pb_regressor_outputs_positive_pb_predictions() -> None:
    from src.research.x5_pb_decomposition import evaluate_pb_regressor

    X = _feature_frame()
    y = np.log(_future_pb_target(X)).rename("target_1m_log_pb")
    current_pb = _current_pb(X)

    predictions, metrics = evaluate_pb_regressor(
        X,
        y,
        current_pb=current_pb,
        model_name="ridge_log_pb",
        feature_columns=["pb_ratio", "rates"],
        target_kind="log_pb",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert not predictions.empty
    assert predictions["y_pred_pb"].gt(0.0).all()
    np.testing.assert_allclose(
        predictions["y_pred_pb"],
        np.exp(predictions["y_pred_target"]),
    )
    assert metrics["model_name"] == "ridge_log_pb"


def test_no_change_pb_baseline_predicts_current_pb() -> None:
    from src.research.x5_pb_decomposition import evaluate_pb_baseline

    X = _feature_frame()
    y = _future_pb_target(X)
    current_pb = _current_pb(X)

    predictions, metrics = evaluate_pb_baseline(
        X,
        y,
        current_pb=current_pb,
        baseline_name="no_change_pb",
        target_kind="pb",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    pd.testing.assert_series_equal(
        predictions["y_pred_pb"],
        predictions["current_pb"],
        check_names=False,
    )
    assert metrics["pb_mae"] > 0.0


def test_drift_pb_baseline_uses_fold_local_history() -> None:
    from src.research.x5_pb_decomposition import evaluate_pb_baseline

    X = _feature_frame()
    y = _future_pb_target(X)
    current_pb = _current_pb(X)

    predictions, _ = evaluate_pb_baseline(
        X,
        y,
        current_pb=current_pb,
        baseline_name="drift_pb",
        target_kind="pb",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    first_fold = predictions[predictions["fold_idx"] == 0]
    train_window = y.loc[
        first_fold["train_start_date"].iloc[0]:first_fold["train_end_date"].iloc[0]
    ]
    assert first_fold["y_pred_pb"].iloc[0] == pytest.approx(train_window.mean())


def test_combine_decomposition_predictions_aligns_by_date_and_fold() -> None:
    from src.research.x5_pb_decomposition import combine_decomposition_predictions

    dates = pd.date_range("2022-01-31", periods=3, freq="BME")
    bvps_predictions = pd.DataFrame(
        {
            "date": dates,
            "fold_idx": [0, 0, 1],
            "current_bvps": [18.0, 20.0, 21.0],
            "implied_future_bvps": [20.0, 21.0, 22.0],
            "true_future_bvps": [19.0, 22.0, 23.0],
        }
    )
    pb_predictions = pd.DataFrame(
        {
            "date": dates,
            "fold_idx": [0, 0, 1],
            "current_pb": [2.0, 2.0, 2.0],
            "y_pred_pb": [2.0, 2.1, 2.2],
            "y_true_pb": [2.1, 2.0, 2.3],
        }
    )

    combined, metrics = combine_decomposition_predictions(
        bvps_predictions,
        pb_predictions,
        horizon_months=1,
        bvps_model_name="bvps_model",
        pb_model_name="pb_model",
    )

    assert combined["implied_future_price"].tolist() == [
        pytest.approx(40.0),
        pytest.approx(44.1),
        pytest.approx(48.4),
    ]
    assert combined["true_future_price"].tolist() == [
        pytest.approx(39.9),
        pytest.approx(44.0),
        pytest.approx(52.9),
    ]
    assert metrics["model_name"] == "bvps_model__pb_model"
    assert metrics["n_obs"] == 3


def test_decomposition_summary_ranks_lower_price_mae_first() -> None:
    from src.research.x5_pb_decomposition import summarize_decomposition_results

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 1,
                "model_name": "baseline",
                "implied_price_mae": 5.0,
                "implied_price_rmse": 6.0,
                "directional_hit_rate": 0.50,
            },
            {
                "horizon_months": 1,
                "model_name": "model",
                "implied_price_mae": 4.0,
                "implied_price_rmse": 6.5,
                "directional_hit_rate": 0.60,
            },
        ]
    )

    summary = summarize_decomposition_results(detail)

    assert summary.iloc[0]["model_name"] == "model"
    assert summary.iloc[0]["rank"] == 1
