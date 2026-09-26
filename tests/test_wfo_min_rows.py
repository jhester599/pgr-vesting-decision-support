"""run_wfo's minimum row count matches what TimeSeriesSplit accepts.

Found by the step 9 property tests (review 2026-09-25, WP12): the documented
minimum was train (60) + gap + one test window (6), but with only one test
window ``n_splits`` is 1 and sklearn's ``TimeSeriesSplit`` raises "n_splits=2
or more". Every row count in [train + gap + 6, train + gap + 12) failed with
that sklearn error. The minimum is now train + gap + two test windows, and
below it the WFO helpers raise their own "too small" ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.models.evaluation import iter_wfo_splits
from src.models.wfo_engine import _min_required_observations, run_wfo
from src.research.x2_absolute_classification import iter_absolute_wfo_splits

_TRAIN = config.WFO_TRAIN_WINDOW_MONTHS
_TEST = config.WFO_TEST_WINDOW_MONTHS
# (target horizon, total gap = horizon + default purge buffer)
_HORIZONS = [(6, 6 + config.WFO_PURGE_BUFFER_6M), (12, 12 + config.WFO_PURGE_BUFFER_12M)]


def _panel(n_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2005-01-31", periods=n_rows, freq="BME")
    X = pd.DataFrame(rng.normal(size=(n_rows, 2)), index=idx, columns=["a", "b"])
    y = pd.Series(X["a"].to_numpy() + rng.normal(0, 0.3, n_rows), index=idx, name="y")
    return X, y


@pytest.mark.parametrize(("horizon", "gap"), _HORIZONS)
def test_minimum_is_train_plus_gap_plus_two_test_windows(horizon: int, gap: int) -> None:
    assert _min_required_observations(gap) == _TRAIN + gap + 2 * _TEST


@pytest.mark.parametrize(("horizon", "gap"), _HORIZONS)
@pytest.mark.parametrize("extra", [0, 5])
def test_one_test_window_raises_the_wfo_error_not_sklearns(horizon: int, gap: int, extra: int) -> None:
    """At the old minimum (+0 and +5 rows) the error is run_wfo's own."""
    X, y = _panel(_TRAIN + gap + _TEST + extra)
    with pytest.raises(ValueError, match=r"Need at least \d+ \(TRAIN_WINDOW=60 \+ GAP=\d+ \+ 2 x TEST_WINDOW=6\)"):
        run_wfo(X, y, model_type="ridge", target_horizon_months=horizon, feature_columns=["a", "b"])


@pytest.mark.parametrize(("horizon", "gap"), _HORIZONS)
def test_exact_minimum_gives_two_folds(horizon: int, gap: int) -> None:
    X, y = _panel(_min_required_observations(gap))
    result = run_wfo(X, y, model_type="ridge", target_horizon_months=horizon, feature_columns=["a", "b"])
    assert len(result.folds) == 2
    assert result.folds[-1].test_end == X.index[-1]
    assert [f.n_train for f in result.folds] == [_TRAIN, _TRAIN]


@pytest.mark.parametrize(("horizon", "gap"), _HORIZONS)
@pytest.mark.parametrize("extra", [0, 5])
def test_iter_wfo_splits_raises_its_own_error_below_two_windows(horizon: int, gap: int, extra: int) -> None:
    X, y = _panel(_TRAIN + gap + _TEST + extra)
    with pytest.raises(ValueError, match="need at least"):
        list(iter_wfo_splits(X, y, target_horizon_months=horizon))
    X, y = _panel(_min_required_observations(gap))
    assert len(list(iter_wfo_splits(X, y, target_horizon_months=horizon))) == 2


@pytest.mark.parametrize("extra", [0, 5])
def test_x2_splitter_raises_its_own_error_below_two_windows(extra: int) -> None:
    gap = 6 + config.WFO_PURGE_BUFFER_6M
    with pytest.raises(ValueError, match="need at least"):
        iter_absolute_wfo_splits(_TRAIN + gap + _TEST + extra, target_horizon_months=6)
    _, splitter = iter_absolute_wfo_splits(_TRAIN + gap + 2 * _TEST, target_horizon_months=6)
    assert splitter.get_n_splits() == 2
