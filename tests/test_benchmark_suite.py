"""Tests for scripts/benchmark_suite.py and v9 evaluation helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts.benchmark_suite import run_benchmark_suite, summarize_benchmark_suite
from src.research.evaluation import evaluate_baseline_strategy, summarize_predictions


def _sample_xy(n_rows: int = 84) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    X = pd.DataFrame(
        {
            "mom_3m": np.random.randn(n_rows),
            "mom_6m": np.random.randn(n_rows),
        },
        index=idx,
    )
    y = pd.Series(np.random.randn(n_rows) * 0.05, index=idx, name="target")
    return X, y


def test_summarize_predictions_returns_metrics():
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    y_hat = pd.Series(np.linspace(-0.02, 0.03, 10), index=idx)
    y_true = pd.Series(np.linspace(-0.01, 0.04, 10), index=idx)
    summary = summarize_predictions(y_hat, y_true, target_horizon_months=6)
    assert summary.n_obs == 10
    assert isinstance(summary.hit_rate, float)
    assert isinstance(summary.oos_r2, float)


def test_evaluate_baseline_strategy_returns_expected_keys():
    X, y = _sample_xy()
    result = evaluate_baseline_strategy(X, y, strategy="historical_mean", target_horizon_months=6)
    assert result["strategy"] == "historical_mean"
    for key in ["n_obs", "ic", "hit_rate", "mae", "oos_r2", "nw_ic", "nw_p_value"]:
        assert key in result


def test_summarize_benchmark_suite_aggregates_rows():
    detail = pd.DataFrame(
        [
            {"item_type": "model", "item_name": "elasticnet", "horizon_months": 6, "benchmark": "VTI", "ic": 0.10, "hit_rate": 0.56, "oos_r2": 0.03, "mae": 0.04, "gate_status": "PASS"},
            {"item_type": "model", "item_name": "elasticnet", "horizon_months": 6, "benchmark": "VOO", "ic": 0.04, "hit_rate": 0.53, "oos_r2": 0.00, "mae": 0.05, "gate_status": "MARGINAL"},
        ]
    )
    summary = summarize_benchmark_suite(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["item_name"] == "elasticnet"
    assert row["n_benchmarks"] == 2


def test_run_benchmark_suite_produces_detail_and_summary(tmp_path):
    idx = pd.date_range("2015-01-31", periods=84, freq="ME")
    fake_df = pd.DataFrame({"mom_3m": np.random.randn(84), "mom_6m": np.random.randn(84)}, index=idx)
    fake_series = pd.Series(np.random.randn(84) * 0.05, index=idx, name="VTI_6m")
    fake_wfo = MagicMock()
    fake_wfo.folds = [MagicMock(feature_importances={"mom_3m": 0.4})]
    fake_wfo.y_true_all = np.random.randn(24)
    fake_wfo.y_hat_all = np.random.randn(24)
    fake_wfo.mean_absolute_error = 0.05
    fake_wfo.information_coefficient = 0.07
    fake_wfo.hit_rate = 0.55

    fake_ensemble_fold = MagicMock()
    fake_ensemble_fold.y_true = np.array([0.01, -0.02])
    fake_ensemble_fold.y_hat = np.array([0.02, -0.01])
    fake_ensemble_fold._test_dates = list(pd.date_range("2022-01-31", periods=2, freq="ME"))
    fake_model_result = MagicMock()
    fake_model_result.mean_absolute_error = 0.05
    fake_model_result.folds = [fake_ensemble_fold]
    fake_ensemble = MagicMock()
    fake_ensemble.benchmark = "VTI"
    fake_ensemble.model_results = {"elasticnet": fake_model_result}

    with (
        patch("scripts.benchmark_suite.build_feature_matrix_from_db", return_value=fake_df),
        patch("scripts.benchmark_suite.load_relative_return_matrix", return_value=fake_series),
        patch("scripts.benchmark_suite.get_X_y_relative", return_value=(fake_df, fake_series)),
        patch("src.research.evaluation.run_wfo", return_value=fake_wfo),
        patch("scripts.benchmark_suite.run_ensemble_benchmarks", return_value={"VTI": fake_ensemble}),
    ):
        detail_df, summary_df = run_benchmark_suite(
            conn=MagicMock(),
            benchmarks=["VTI"],
            horizons=[6],
            model_types=["elasticnet"],
            baseline_strategies=["historical_mean"],
            output_dir=str(tmp_path),
        )

    assert not detail_df.empty
    assert not summary_df.empty
    assert set(detail_df["item_type"]) == {"baseline", "model", "ensemble"}
