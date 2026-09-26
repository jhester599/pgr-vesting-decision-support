from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.forecast_diagnostics import (
    compute_clark_west_result,
    expanding_mean_benchmark,
    summarize_prediction_diagnostics,
)


def test_expanding_mean_benchmark_uses_prior_history() -> None:
    realized = pd.Series([0.10, 0.20, -0.10, 0.00])
    benchmark = expanding_mean_benchmark(realized)
    expected = pd.Series([0.10, 0.10, 0.15, 0.0666666667])
    np.testing.assert_allclose(benchmark.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_clark_west_equal_forecasts_is_near_neutral() -> None:
    realized = pd.Series([0.02, 0.01, -0.01, 0.03, 0.00, 0.02])
    predicted = expanding_mean_benchmark(realized)
    result = compute_clark_west_result(predicted, realized, lags=2)
    assert abs(result.mean_adjusted_differential) < 1e-12


def test_clark_west_prefers_better_model() -> None:
    realized = pd.Series([0.03, 0.01, -0.02, 0.04, 0.01, -0.01, 0.05, 0.02])
    predicted = pd.Series([0.025, 0.015, -0.015, 0.03, 0.015, -0.005, 0.04, 0.015])
    result = compute_clark_west_result(predicted, realized, lags=2)
    assert result.t_stat > 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_clark_west_penalizes_worse_model() -> None:
    realized = pd.Series([0.03, 0.01, -0.02, 0.04, 0.01, -0.01, 0.05, 0.02])
    predicted = pd.Series([-0.03, -0.01, 0.02, -0.04, -0.01, 0.01, -0.05, -0.02])
    result = compute_clark_west_result(predicted, realized, lags=2)
    assert result.t_stat < 0.5


def test_summarize_prediction_diagnostics_includes_cw_fields() -> None:
    realized = pd.Series([0.03, 0.01, -0.02, 0.04, 0.01, -0.01, 0.05, 0.02])
    predicted = pd.Series([0.025, 0.015, -0.015, 0.03, 0.015, -0.005, 0.04, 0.015])
    summary = summarize_prediction_diagnostics(predicted, realized, target_horizon_months=6)
    assert summary["n_obs"] == 8
    assert "cw_t_stat" in summary
    assert "cw_p_value" in summary
