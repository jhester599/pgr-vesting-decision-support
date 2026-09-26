"""Property tests: WFO temporal integrity, checked on ``run_wfo`` itself.

Review 2026-09-25, F28: the v36 version of this file built ``FoldResult``
objects by hand and checked their own arithmetic, so it never called
production code. These properties run ``run_wfo`` and ``predict_current``
on generated monthly data and check the folds they actually produce:

- train rows precede test rows, separated by exactly
  ``target_horizon + purge_buffer`` rows (the embargo);
- every training window is exactly ``WFO_TRAIN_WINDOW_MONTHS`` rows;
- test windows are ``WFO_TEST_WINDOW_MONTHS`` long, disjoint, increasing,
  and the last one ends on the last row;
- ``y_true`` is the target on the test dates;
- a fold's predictions do not change when rows after its training window
  change (no look-ahead through fitting, imputation or scaling);
- the live refit uses only the most recent ``train_window_months`` rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import config
from src.models.wfo_engine import _min_required_observations, predict_current, run_wfo

_FEATURES = ["f0", "f1", "f2"]
_SETTINGS = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _panel(n_rows: int, seed: int, nan_rate: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly features and a target that depends on them, with optional NaNs."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n_rows, freq="BME")
    X = pd.DataFrame(rng.normal(size=(n_rows, len(_FEATURES))), index=idx, columns=_FEATURES)
    y = pd.Series(
        0.5 * X["f0"].to_numpy() - 0.3 * X["f1"].to_numpy() + rng.normal(0, 0.5, n_rows),
        index=idx,
        name="target",
    )
    if nan_rate > 0:
        mask = rng.random(X.shape) < nan_rate
        X = X.mask(mask)
    return X, y


_wfo_case = st.fixed_dictionaries(
    {
        "horizon": st.sampled_from([6, 12]),
        "purge_buffer": st.integers(min_value=0, max_value=3),
        "extra_rows": st.integers(min_value=0, max_value=40),
        "seed": st.integers(min_value=0, max_value=10_000),
        "nan_rate": st.sampled_from([0.0, 0.1]),
    }
)


def _run(case: dict) -> tuple[pd.DataFrame, pd.Series, object, int]:
    gap = case["horizon"] + case["purge_buffer"]
    n_rows = _min_required_observations(gap) + case["extra_rows"]
    X, y = _panel(n_rows, case["seed"], case["nan_rate"])
    result = run_wfo(
        X,
        y,
        model_type="ridge",
        target_horizon_months=case["horizon"],
        purge_buffer=case["purge_buffer"],
        feature_columns=_FEATURES,
    )
    return X, y, result, gap


@given(_wfo_case)
@_SETTINGS
def test_folds_are_embargoed_bounded_and_ordered(case: dict) -> None:
    X, y, result, gap = _run(case)
    dates = X.index
    assert result.folds, "run_wfo produced no folds"
    previous_test_end = None
    for fold in result.folds:
        train_end_pos = dates.get_loc(fold.train_end)
        test_start_pos = dates.get_loc(fold.test_start)
        train_start_pos = dates.get_loc(fold.train_start)
        # Strict order and an embargo of exactly horizon + purge_buffer rows.
        assert fold.train_start <= fold.train_end < fold.test_start <= fold.test_end
        assert test_start_pos - train_end_pos - 1 == gap
        # Rolling window: exactly WFO_TRAIN_WINDOW_MONTHS contiguous rows in
        # every fold (the oldest train set is train + available mod test
        # rows before the cap).
        assert fold.n_train == config.WFO_TRAIN_WINDOW_MONTHS
        assert train_end_pos - train_start_pos + 1 == fold.n_train
        assert fold.n_test == config.WFO_TEST_WINDOW_MONTHS
        if previous_test_end is not None:
            assert fold.test_start > previous_test_end
        previous_test_end = fold.test_end
        # y_true is the target on the fold's own test dates.
        np.testing.assert_allclose(fold.y_true, y.loc[fold._test_dates].to_numpy())
        assert len(fold.y_hat) == len(fold.y_true)
    assert result.folds[-1].test_end == dates[-1]


@given(_wfo_case, st.data())
@_SETTINGS
def test_fold_predictions_ignore_rows_after_training(case: dict, data: st.DataObject) -> None:
    X, y, result, _gap = _run(case)
    fold_idx = data.draw(st.integers(min_value=0, max_value=len(result.folds) - 1), label="fold")
    fold = result.folds[fold_idx]
    after_train = X.index > fold.train_end
    # Scramble every target after the training window, and every feature
    # outside this fold's train and test rows.
    rng = np.random.default_rng(case["seed"] + 1)
    y_perturbed = y.copy()
    y_perturbed[after_train] = rng.normal(5.0, 3.0, int(after_train.sum()))
    X_perturbed = X.copy()
    other = after_train & ~X.index.isin(fold._test_dates)
    X_perturbed.loc[other] = rng.normal(10.0, 5.0, (int(other.sum()), X.shape[1]))
    perturbed = run_wfo(
        X_perturbed,
        y_perturbed,
        model_type="ridge",
        target_horizon_months=case["horizon"],
        purge_buffer=case["purge_buffer"],
        feature_columns=_FEATURES,
    )
    match = [f for f in perturbed.folds if f.test_start == fold.test_start]
    assert len(match) == 1
    np.testing.assert_allclose(match[0].y_hat, fold.y_hat, rtol=1e-9, atol=1e-12)


@given(
    st.integers(min_value=0, max_value=10_000),
    # The inner RidgeCV needs a TimeSeriesSplit with gap 8 inside the window.
    st.integers(min_value=36, max_value=60),
    st.integers(min_value=1, max_value=40),
)
@_SETTINGS
def test_live_refit_uses_only_the_recent_window(seed: int, window: int, older_rows: int) -> None:
    X, y = _panel(window + older_rows, seed)
    wfo_stub = run_wfo(
        *_panel(_min_required_observations(8), seed),
        model_type="ridge",
        target_horizon_months=6,
        feature_columns=_FEATURES,
    )
    X_current = X.iloc[[-1]]
    base = predict_current(X, y, X_current, wfo_stub, model_type="ridge", train_window_months=window)
    # Rows older than the window must not matter.
    y_older = y.copy()
    y_older.iloc[:older_rows] = np.random.default_rng(seed + 2).normal(-4.0, 2.0, older_rows)
    X_older = X.copy()
    X_older.iloc[:older_rows] = np.random.default_rng(seed + 3).normal(8.0, 1.0, (older_rows, X.shape[1]))
    moved = predict_current(X_older, y_older, X_current, wfo_stub, model_type="ridge", train_window_months=window)
    assert np.isclose(moved["predicted_return"], base["predicted_return"], rtol=1e-9, atol=1e-12)
