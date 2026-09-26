"""Tests for scripts/regime_slice_backtest.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts.regime_slice_backtest import _build_slice_labels, run_regime_slice_backtest


def test_build_slice_labels_adds_expected_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2020-06-30", "2023-01-31"]),
            "predicted": [0.01, -0.02, 0.03],
            "realized": [0.02, -0.01, 0.01],
        }
    )
    vix = pd.Series([18.0, 28.0, 17.0], index=pd.to_datetime(["2019-12-31", "2020-06-30", "2023-01-31"]))
    labeled = _build_slice_labels(df, vix)
    expected = {"slice_pre_2020", "slice_2020_2021", "slice_post_2022", "slice_trailing_36m", "slice_low_vix", "slice_high_vix"}
    assert expected.issubset(labeled.columns)


def test_run_regime_slice_backtest_writes_outputs(tmp_path):
    idx = pd.date_range("2015-01-31", periods=84, freq="ME")
    fake_df = pd.DataFrame({"mom_3m": np.random.randn(84), "vix": np.linspace(15, 30, 84)}, index=idx)
    fake_series = pd.Series(np.random.randn(84) * 0.05, index=idx, name="VTI_6m")

    fake_fold = MagicMock()
    fake_fold._test_dates = list(pd.date_range("2022-01-31", periods=2, freq="ME"))
    fake_fold.y_true = np.array([0.01, -0.02])
    fake_fold.y_hat = np.array([0.02, -0.01])
    fake_fold.feature_importances = {"mom_3m": 0.4}

    fake_result = MagicMock()
    fake_result.folds = [fake_fold]
    fake_result.y_true_all = np.array([0.01, -0.02])
    fake_result.y_hat_all = np.array([0.02, -0.01])
    fake_result.test_dates_all = pd.DatetimeIndex(fake_fold._test_dates)
    fake_result.mean_absolute_error = 0.05

    fake_ensemble = MagicMock()
    fake_ensemble.model_results = {"elasticnet": fake_result}
    fake_ensemble.benchmark = "VTI"

    with (
        patch("scripts.regime_slice_backtest.build_feature_matrix_from_db", return_value=fake_df),
        patch("scripts.regime_slice_backtest.load_relative_return_matrix", return_value=fake_series),
        patch("scripts.regime_slice_backtest.get_X_y_relative", return_value=(fake_df[["mom_3m"]], fake_series)),
        patch("src.models.evaluation.run_wfo", return_value=fake_result),
        patch("scripts.regime_slice_backtest.run_ensemble_benchmarks", return_value={"VTI": fake_ensemble}),
    ):
        detail_df, summary_df = run_regime_slice_backtest(
            conn=MagicMock(),
            benchmarks=["VTI"],
            model_types=["elasticnet"],
            output_dir=str(tmp_path),
        )

    assert not detail_df.empty
    assert not summary_df.empty
