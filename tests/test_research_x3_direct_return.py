"""Tests for x3 direct-return regression research utilities."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _feature_frame(n_rows: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=n_rows, freq="ME")
    signal = np.linspace(-1.0, 1.0, n_rows)
    cycle = np.cos(np.linspace(0.0, 6.0, n_rows))
    return pd.DataFrame({"signal": signal, "cycle": cycle}, index=dates)


def _return_target(X: pd.DataFrame) -> pd.Series:
    values = 0.03 + 0.08 * X["signal"] - 0.02 * X["cycle"]
    result = pd.Series(values, index=X.index, name="target_1m_return")
    return result


def _current_price(X: pd.DataFrame) -> pd.Series:
    values = 100.0 + np.arange(len(X), dtype=float)
    return pd.Series(values, index=X.index, name="current_price")


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_evaluate_direct_return_regressor_outputs_implied_prices(
    horizon: int,
) -> None:
    from src.research.x3_direct_return import evaluate_direct_return_regressor

    X = _feature_frame(120)
    y = _return_target(X)
    current_price = _current_price(X)

    predictions, metrics = evaluate_direct_return_regressor(
        X,
        y,
        current_price=current_price,
        model_name="ridge_return",
        feature_columns=["signal", "cycle"],
        target_kind="return",
        target_horizon_months=horizon,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert not predictions.empty
    assert predictions["date"].is_monotonic_increasing
    assert metrics["model_name"] == "ridge_return"
    assert metrics["target_kind"] == "return"
    assert metrics["n_obs"] == len(predictions)
    expected_price = (
        predictions["current_price"] * (1.0 + predictions["y_pred_return"])
    )
    pd.testing.assert_series_equal(
        predictions["implied_future_price"],
        expected_price,
        check_names=False,
    )
    assert (
        predictions["date"]
        > predictions["train_end_date"] + pd.offsets.MonthEnd(horizon)
    ).all()


def test_log_return_predictions_convert_back_to_simple_returns() -> None:
    from src.research.x3_direct_return import evaluate_direct_return_baseline

    X = _feature_frame(48)
    simple_return = _return_target(X)
    log_return = np.log1p(simple_return).rename("target_1m_log_return")
    current_price = _current_price(X)

    predictions, metrics = evaluate_direct_return_baseline(
        X,
        log_return,
        current_price=current_price,
        baseline_name="drift",
        target_kind="log_return",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=24,
        test_window_months=6,
    )

    transformed = np.expm1(predictions["y_pred_target"])
    np.testing.assert_allclose(predictions["y_pred_return"], transformed)
    expected_price = current_price.reindex(predictions["date"]).to_numpy() * (
        1.0 + transformed
    )
    np.testing.assert_allclose(
        predictions["implied_future_price"],
        expected_price,
    )
    assert metrics["target_kind"] == "log_return"
    first_fold = predictions[predictions["fold_idx"] == 0]
    train_window = simple_return.loc[
        first_fold["train_start_date"].iloc[0]:first_fold["train_end_date"].iloc[0]
    ]
    assert first_fold["y_pred_return"].iloc[0] == pytest.approx(
        train_window.mean()
    )


def test_drift_baseline_uses_fold_local_target_history() -> None:
    from src.research.x3_direct_return import evaluate_direct_return_baseline

    X = _feature_frame(48)
    y = pd.Series(np.arange(48, dtype=float), index=X.index, name="target_1m_return")
    current_price = _current_price(X)

    predictions, metrics = evaluate_direct_return_baseline(
        X,
        y,
        current_price=current_price,
        baseline_name="drift",
        target_kind="return",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=24,
        test_window_months=6,
    )

    first_fold = predictions[predictions["fold_idx"] == 0]
    assert not first_fold.empty
    assert first_fold["y_pred_target"].nunique() == 1
    train_window = y.loc[
        first_fold["train_start_date"].iloc[0]:first_fold["train_end_date"].iloc[0]
    ]
    assert first_fold["y_pred_target"].iloc[0] == pytest.approx(
        train_window.mean()
    )
    assert predictions["y_pred_target"].nunique() > 1
    assert metrics["model_name"] == "drift"


def test_no_change_baseline_predicts_current_price() -> None:
    from src.research.x3_direct_return import evaluate_direct_return_baseline

    X = _feature_frame(48)
    y = _return_target(X)
    current_price = _current_price(X)

    predictions, metrics = evaluate_direct_return_baseline(
        X,
        y,
        current_price=current_price,
        baseline_name="no_change",
        target_kind="return",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=24,
        test_window_months=6,
    )

    assert predictions["y_pred_return"].eq(0.0).all()
    pd.testing.assert_series_equal(
        predictions["implied_future_price"],
        predictions["current_price"],
        check_names=False,
    )
    assert metrics["implied_price_mae"] > 0.0


def test_regressor_imputes_all_nan_feature_without_warning() -> None:
    from src.research.x3_direct_return import evaluate_direct_return_regressor

    X = _feature_frame(72)
    X["empty_feature"] = np.nan
    y = _return_target(X)
    current_price = _current_price(X)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predictions, metrics = evaluate_direct_return_regressor(
            X,
            y,
            current_price=current_price,
            model_name="ridge_return",
            feature_columns=["signal", "empty_feature"],
            target_kind="return",
            target_horizon_months=1,
            purge_buffer=0,
            train_window_months=36,
            test_window_months=6,
        )

    assert not predictions.empty
    assert metrics["n_features"] == 2


def test_regressor_treats_infinite_features_as_missing() -> None:
    from src.research.x3_direct_return import evaluate_direct_return_regressor

    X = _feature_frame(72)
    X["signal"] = np.inf
    X.loc[X.index[::2], "cycle"] = -np.inf
    y = _return_target(X.replace([np.inf, -np.inf], 0.0))
    current_price = _current_price(X)

    predictions, metrics = evaluate_direct_return_regressor(
        X,
        y,
        current_price=current_price,
        model_name="ridge_return",
        feature_columns=["signal", "cycle"],
        target_kind="return",
        target_horizon_months=1,
        purge_buffer=0,
        train_window_months=36,
        test_window_months=6,
    )

    assert not predictions.empty
    assert np.isfinite(predictions["y_pred_return"]).all()
    assert np.isfinite(metrics["return_rmse"])


def test_direct_return_summary_ranks_lower_price_mae_first() -> None:
    from src.research.x3_direct_return import summarize_direct_return_results

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 1,
                "model_name": "drift",
                "target_kind": "return",
                "implied_price_mae": 4.0,
                "return_rmse": 0.09,
                "directional_hit_rate": 0.60,
            },
            {
                "horizon_months": 1,
                "model_name": "ridge_return",
                "target_kind": "return",
                "implied_price_mae": 3.0,
                "return_rmse": 0.12,
                "directional_hit_rate": 0.55,
            },
        ]
    )

    summary = summarize_direct_return_results(detail)

    assert summary.iloc[0]["model_name"] == "ridge_return"
    assert summary.iloc[0]["rank"] == 1
    assert bool(summary.iloc[0]["beats_no_change"]) is False


def test_unsupported_regressor_name_raises() -> None:
    from src.research.x3_direct_return import build_direct_return_regressor

    with pytest.raises(ValueError, match="Unsupported regressor"):
        build_direct_return_regressor("wide_net")


def test_ridge_regressor_uses_strong_regularization() -> None:
    from src.research.x3_direct_return import build_direct_return_regressor

    pipeline = build_direct_return_regressor("ridge_return")

    assert pipeline.named_steps["regressor"].alpha == 5000.0
