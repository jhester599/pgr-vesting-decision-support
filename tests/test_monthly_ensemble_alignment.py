from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.monthly_decision as md
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.models.policy_metrics import evaluate_policy_series
from src.models.wfo_engine import FoldResult, WFOResult


def _make_wfo_result(
    model_type: str,
    y_hat: list[float],
    y_true: list[float],
    start: str = "2020-01-31",
) -> WFOResult:
    fold = FoldResult(
        fold_idx=0,
        train_start=pd.Timestamp("2015-01-31"),
        train_end=pd.Timestamp("2019-12-31"),
        test_start=pd.Timestamp(start),
        test_end=pd.Timestamp(start) + pd.offsets.MonthEnd(len(y_hat) - 1),
        y_true=np.array(y_true, dtype=float),
        y_hat=np.array(y_hat, dtype=float),
        optimal_alpha=0.01,
        feature_importances={"f1": 1.0},
        n_train=60,
        n_test=len(y_hat),
    )
    fold._test_dates = list(pd.date_range(start, periods=len(y_hat), freq="ME"))
    return WFOResult(
        folds=[fold],
        benchmark="VTI",
        target_horizon=6,
        model_type=model_type,
    )


def _make_two_model_ensemble() -> EnsembleWFOResult:
    y_true = [0.06, -0.04, 0.05, -0.03, 0.02, -0.01]
    ridge = _make_wfo_result("ridge", [0.10, -0.08, 0.09, -0.07, 0.04, -0.03], y_true)
    gbt = _make_wfo_result("gbt", [-0.02, 0.02, -0.01, 0.01, -0.01, 0.01], y_true)
    return EnsembleWFOResult(
        benchmark="VTI",
        target_horizon=6,
        mean_ic=0.10,
        mean_hit_rate=0.60,
        mean_mae=0.05,
        model_results={"ridge": ridge, "gbt": gbt},
    )


def test_compute_aggregate_health_uses_ensemble_reconstruction() -> None:
    ensemble = _make_two_model_ensemble()
    expected_predicted, expected_realized = reconstruct_ensemble_oos_predictions(ensemble)
    result = md._compute_aggregate_health({"VTI": ensemble})
    assert result is not None
    pd.testing.assert_series_equal(result["agg_predicted"], expected_predicted)
    pd.testing.assert_series_equal(result["agg_realized"], expected_realized)


def test_compute_policy_summary_uses_ensemble_reconstruction() -> None:
    ensemble = _make_two_model_ensemble()
    expected_predicted, expected_realized = reconstruct_ensemble_oos_predictions(ensemble)
    expected = evaluate_policy_series(
        predicted=expected_predicted,
        realized_relative_return=expected_realized,
        policy_name="neutral_band_3pct",
    )
    result = md._compute_policy_summary({"VTI": ensemble})
    assert result is not None
    assert result["neutral_band_3pct"].mean_policy_return == expected.mean_policy_return


def test_diagnostic_report_mentions_clark_west(tmp_path) -> None:
    ensemble = _make_two_model_ensemble()
    md._write_diagnostic_report(tmp_path, pd.Timestamp("2026-04-10").date(), {"VTI": ensemble})
    text = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
    assert "Clark-West" in text
