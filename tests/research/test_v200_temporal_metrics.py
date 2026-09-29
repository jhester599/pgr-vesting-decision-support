"""Independent expected outputs for the v200 temporal and metric contract."""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def temporal_module() -> ModuleType:
    """Import the installed namespace, without altering the import path."""
    return importlib.import_module("pgr_vds.research_lib.temporal")


def metric_module() -> ModuleType:
    """Import the installed namespace, without altering the import path."""
    return importlib.import_module("pgr_vds.research_lib.metrics")


def test_label_end_uses_calendar_horizon_and_last_weekday() -> None:
    module = temporal_module()
    assert module.label_end(pd.Timestamp("2023-08-31"), 6) == pd.Timestamp(
        "2024-02-29"
    )
    assert module.label_end(pd.Timestamp("2024-02-29"), 6) == pd.Timestamp(
        "2024-08-30"
    )


@pytest.mark.parametrize("horizon,window,gap", [(6, 60, 12), (12, 120, 24)])
def test_outer_splits_have_independent_expected_indices(
    horizon: int, window: int, gap: int,
) -> None:
    module = temporal_module()
    dates = pd.date_range("2000-01-31", periods=window + gap + 12, freq="BME")
    folds = module.chronological_splits(dates, horizon)
    assert len(folds) == 2
    np.testing.assert_array_equal(folds[0][0], np.arange(window))
    np.testing.assert_array_equal(
        folds[0][1], np.arange(window + gap, window + gap + 6)
    )
    np.testing.assert_array_equal(folds[1][0], np.arange(6, window + 6))
    assert module.chronological_splits(dates[:-1], horizon) == []


@pytest.mark.parametrize(
    "horizon,window,sizes",
    [(6, 60, [30, 36, 42]), (12, 120, [78, 84, 90])],
)
def test_inner_splits_use_explicit_six_month_tests(
    horizon: int, window: int, sizes: list[int],
) -> None:
    dates = pd.date_range("2000-01-31", periods=window, freq="BME")
    folds = temporal_module().chronological_splits(dates, horizon, inner=True)
    assert [len(train) for train, _ in folds] == sizes
    assert [len(test) for _, test in folds] == [6, 6, 6]
    gaps = [test[0] - train[-1] - 1 for train, test in folds]
    assert gaps == [2 * horizon] * 3


def test_splits_reject_duplicate_and_missing_months() -> None:
    dates = pd.date_range("2000-01-31", periods=90, freq="BME")
    with pytest.raises(ValueError, match="unique|duplicate"):
        temporal_module().chronological_splits(dates.insert(1, dates[0]), 6)
    with pytest.raises(ValueError, match="contiguous|missing"):
        temporal_module().chronological_splits(dates.delete(10), 6)


def test_training_checks_every_origin_and_availability_support() -> None:
    module = temporal_module()
    origins = pd.date_range("2010-01-29", periods=30, freq="BME")
    ends = pd.DatetimeIndex([module.label_end(date, 6) for date in origins])
    available = pd.Series(ends).copy()
    tests = pd.DatetimeIndex(["2012-06-29", "2012-07-31"])
    available.iloc[0] = pd.Timestamp("2012-07-01")
    expected = np.arange(1, 24)
    actual = module.eligible_training(origins, ends, available, tests, 23)
    np.testing.assert_array_equal(actual, expected)
    unsupported = module.eligible_training(origins, ends, available, tests, 24)
    assert unsupported.size == 0


def test_prevailing_mean_uses_actual_availability_and_earlier_labels() -> None:
    history = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"]),
        "y_true": [2.0, 8.0, 1000.0],
        "available": pd.to_datetime([
            "2020-07-31", "2020-09-30", "2020-08-31",
        ]),
    })
    origins = pd.to_datetime(["2020-06-30", "2020-07-31", "2020-09-30"])
    actual = metric_module().prevailing_mean(origins, history)
    np.testing.assert_allclose(
        actual, [np.nan, 2.0, 1010.0 / 3], equal_nan=True
    )
    current = history.iloc[[0]].copy()
    current["available"] = current["date"]
    forecast = metric_module().prevailing_mean(current["date"], current)
    assert np.isnan(forecast[0])


def oracle_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conditional-mean oracle with a fixed irreducible outcome-error path."""
    realized = np.arange(8, dtype=float)
    predicted = realized + 3.0
    naive = np.array([np.nan] * 6 + [0.0, 0.5])
    return realized, predicted, naive


def test_overlapping_oracle_beats_honest_prevailing_mean() -> None:
    realized, predicted, naive = oracle_fixture()
    expected = 1.0 - 18.0 / 78.25
    actual = metric_module().honest_r2(realized, predicted, naive)
    assert actual == pytest.approx(expected)
    assert expected > 0.0


def pooled_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two endpoints have identical errors, with different known past means."""
    realized = np.array([1.0, -1.0, 1.0, -1.0, 101.0, 99.0, 101.0, 99.0])
    return realized, realized + 2.0, np.array([0.0] * 4 + [100.0] * 4)


def test_pooling_two_benchmarks_does_not_inflate_r2() -> None:
    realized, predicted, naive = pooled_fixture()
    function = metric_module().honest_r2
    assert function(realized[:4], predicted[:4], naive[:4]) == -3.0
    assert function(realized[4:], predicted[4:], naive[4:]) == -3.0
    assert function(realized, predicted, naive) == -3.0


def test_r2_masks_all_three_arrays_together() -> None:
    function = metric_module().honest_r2
    assert function([1, 2, 3], [1, np.inf, 4], [0, 0, 0]) == pytest.approx(0.9)
    assert np.isnan(function([1], [1], [0]))
    assert np.isnan(function([1, 2], [1, 2], [1, 2]))
    with pytest.raises(ValueError):
        function([1, 2], [1], [0, 0])


@pytest.mark.xfail(strict=True, reason="v37 silently uses a one-month horizon")
def test_legacy_v37_overlapping_oracle_expected_failure() -> None:
    from src.research.v37_utils import compute_metrics

    realized, predicted, _ = oracle_fixture()
    assert compute_metrics(realized, predicted)["r2"] > 0.0


@pytest.mark.xfail(
    strict=True, reason="v37 pools distinct benchmark past means"
)
def test_legacy_v37_pooled_expected_failure() -> None:
    from src.research.v37_utils import compute_metrics

    realized, predicted, _ = pooled_fixture()
    assert compute_metrics(realized, predicted)["r2"] == pytest.approx(-3.0)


def test_panel_summary_preserves_dates_and_benchmark_weights() -> None:
    dates = pd.date_range("2010-01-29", periods=12, freq="BME")
    rows = []
    for offset, benchmark in [(0.0, "A"), (100.0, "B")]:
        for number, date in enumerate(dates):
            value = float(number + 1) + offset
            rows.append({"date": date, "benchmark": benchmark,
                         "y_true": value, "y_hat": value, "naive": offset,
                         "base_prediction": 1.0})
    frame = pd.DataFrame(rows)
    result = metric_module().panel_summary(frame, 6, replicates=100, seed=19)
    assert result["n_dates"] == 12
    assert result["n_rows"] == 24
    assert result["oos_r2"] == 1.0
    assert result["equal_weight_ic"] == pytest.approx(1.0)
    assert result["panel_ic"] == pytest.approx(1.0)
    assert result["hit_rate"] == 1.0
    assert result["base_hit_rate"] == 1.0
    assert result["directional_skill"] == 0.0
    assert result["directional_skill_p"] == 1.0
    assert result["block_length"] == 6
    repeated = metric_module().panel_summary(frame, 6, replicates=100, seed=19)
    assert result == repeated


def test_holm_adjustment_counts_all_38_campaign_slots() -> None:
    assert metric_module().holm38([0.001, 0.01, 1.0]) == pytest.approx(
        [0.038, 0.37, 1.0]
    )
    assert metric_module().holm38([]) == []
    with pytest.raises(ValueError):
        metric_module().holm38([0.5] * 39)


def test_date_bootstrap_constant_gain_has_exact_monte_carlo_p() -> None:
    """Every block has gain one; zero centered null draws exceed it."""
    dates = pd.date_range("2010-01-29", periods=12, freq="BME")
    values = np.tile([1.0, -1.0], 6)
    frame = pd.DataFrame({
        "date": dates,
        "benchmark": "A",
        "y_true": values,
        "y_hat": values,
        "naive": 0.0,
        "base_prediction": -values,
    })
    result = metric_module().panel_summary(frame, 6, replicates=100, seed=19)
    assert result["primary_p"] == pytest.approx(1.0 / 101.0)
    assert result["directional_skill_p"] == pytest.approx(1.0 / 101.0)
    assert result["directional_skill"] == 1.0
    assert result["oos_r2_ci"] == [1.0, 1.0]
    clone = frame.assign(benchmark="B")
    duplicated = pd.concat([frame, clone], ignore_index=True)
    pooled = metric_module().panel_summary(
        duplicated, 6, replicates=100, seed=19
    )
    assert pooled["primary_p"] == result["primary_p"]
    assert pooled["directional_skill_p"] == result["directional_skill_p"]
    assert pooled["n_dates"] == result["n_dates"]
