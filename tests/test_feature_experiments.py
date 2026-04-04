"""Tests for scripts/feature_experiments.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts.feature_experiments import _experiment_specs, run_feature_experiments, summarize_feature_experiments


def test_experiment_specs_for_baseline_feature_include_drop():
    specs = _experiment_specs(
        feature="mom_3m",
        model_type="elasticnet",
        baseline_features=["mom_3m", "mom_6m"],
    )
    modes = [mode for mode, _, _ in specs]
    assert "single_feature" in modes
    assert "drop_from_baseline" in modes


def test_experiment_specs_for_nonbaseline_feature_include_add():
    specs = _experiment_specs(
        feature="buyback_yield",
        model_type="elasticnet",
        baseline_features=["mom_3m", "mom_6m"],
    )
    modes = [mode for mode, _, _ in specs]
    assert "single_feature" in modes
    assert "add_to_baseline" in modes


def test_summarize_feature_experiments_computes_deltas():
    detail = pd.DataFrame(
        [
            {"experiment_mode": "baseline", "feature": "__baseline__", "feature_in_baseline": True, "benchmark": "VTI", "horizon_months": 6, "model_type": "elasticnet", "n_features": 2, "ic": 0.05, "hit_rate": 0.54, "oos_r2": 0.01, "mae": 0.05, "gate_status": "MARGINAL"},
            {"experiment_mode": "add_to_baseline", "feature": "buyback_yield", "feature_in_baseline": False, "benchmark": "VTI", "horizon_months": 6, "model_type": "elasticnet", "n_features": 3, "ic": 0.08, "hit_rate": 0.56, "oos_r2": 0.03, "mae": 0.04, "gate_status": "PASS"},
        ]
    )
    summary = summarize_feature_experiments(detail)
    row = summary.iloc[0]
    assert "mean_delta_ic" in summary.columns
    assert row["n_benchmarks"] == 1


def test_run_feature_experiments_produces_rows(tmp_path):
    idx = pd.date_range("2015-01-31", periods=84, freq="ME")
    fake_df = pd.DataFrame(
        {
            "mom_3m": np.random.randn(84),
            "mom_6m": np.random.randn(84),
            "buyback_yield": np.random.randn(84),
            "target_6m_return": np.random.randn(84),
        },
        index=idx,
    )
    fake_series = pd.Series(np.random.randn(84) * 0.05, index=idx, name="VTI_6m")
    fake_wfo = MagicMock()
    fake_wfo.folds = [MagicMock(feature_importances={"mom_3m": 0.4})]
    fake_wfo.y_true_all = np.random.randn(24)
    fake_wfo.y_hat_all = np.random.randn(24)
    fake_wfo.mean_absolute_error = 0.05
    fake_wfo.information_coefficient = 0.07
    fake_wfo.hit_rate = 0.55

    with (
        patch("scripts.feature_experiments.build_feature_matrix_from_db", return_value=fake_df),
        patch("scripts.feature_experiments.load_relative_return_matrix", return_value=fake_series),
        patch("scripts.feature_experiments.get_X_y_relative", return_value=(fake_df.drop(columns=["target_6m_return"]), fake_series)),
        patch("src.research.evaluation.run_wfo", return_value=fake_wfo),
    ):
        detail_df, summary_df = run_feature_experiments(
            conn=MagicMock(),
            benchmarks=["VTI"],
            horizons=[6],
            model_types=["elasticnet"],
            output_dir=str(tmp_path),
            features=["mom_3m", "buyback_yield"],
        )

    assert not detail_df.empty
    assert not summary_df.empty
    assert "baseline" in set(detail_df["experiment_mode"])
