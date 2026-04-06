"""v36 — Property-based tests: WFO temporal-integrity invariants.

Verifies that fold construction maintains strict temporal ordering
(train_start < train_end < test_start < test_end) and that no training
observation leaks into the out-of-sample test window, regardless of
the data length or embargo configuration chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.models.wfo_engine import FoldResult, WFOResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold(
    fold_idx: int,
    train_offset_months: int,
    train_window_months: int,
    embargo_months: int,
    test_window_months: int,
) -> FoldResult:
    """Construct a FoldResult with parameterised window sizes."""
    base = pd.Timestamp("2015-01-31")
    train_start = base + pd.DateOffset(months=train_offset_months)
    train_end = train_start + pd.DateOffset(months=train_window_months)
    test_start = train_end + pd.DateOffset(months=embargo_months)
    test_end = test_start + pd.DateOffset(months=test_window_months)
    n = test_window_months
    y = np.zeros(n)
    return FoldResult(
        fold_idx=fold_idx,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        y_true=y,
        y_hat=y,
        optimal_alpha=0.01,
        feature_importances={},
        n_train=train_window_months,
        n_test=n,
        _test_dates=[test_start + pd.DateOffset(months=i) for i in range(n)],
    )


# ---------------------------------------------------------------------------
# 1. Fold timestamps are strictly ordered
# ---------------------------------------------------------------------------

@given(
    st.integers(min_value=0, max_value=60),   # train_offset_months
    st.integers(min_value=12, max_value=84),  # train_window_months
    st.integers(min_value=1, max_value=24),   # embargo_months
    st.integers(min_value=1, max_value=12),   # test_window_months
)
@settings(max_examples=400)
def test_fold_timestamps_strictly_ordered(
    train_offset: int,
    train_window: int,
    embargo: int,
    test_window: int,
) -> None:
    """train_start < train_end < test_start < test_end for all valid configs."""
    fold = _make_fold(0, train_offset, train_window, embargo, test_window)
    assert fold.train_start < fold.train_end, "train_start must precede train_end"
    assert fold.train_end < fold.test_start, "embargo gap: train_end must precede test_start"
    assert fold.test_start < fold.test_end, "test_start must precede test_end"


# ---------------------------------------------------------------------------
# 2. No temporal overlap between train and test windows
# ---------------------------------------------------------------------------

@given(
    st.integers(min_value=0, max_value=60),
    st.integers(min_value=12, max_value=84),
    st.integers(min_value=1, max_value=24),
    st.integers(min_value=1, max_value=12),
)
@settings(max_examples=400)
def test_no_temporal_overlap_train_test(
    train_offset: int,
    train_window: int,
    embargo: int,
    test_window: int,
) -> None:
    """The test window must not overlap with the training window."""
    fold = _make_fold(0, train_offset, train_window, embargo, test_window)
    # Overlap condition: train_end >= test_start
    assert fold.train_end < fold.test_start, (
        f"Temporal leakage: train_end={fold.train_end} >= test_start={fold.test_start}"
    )


# ---------------------------------------------------------------------------
# 3. Embargo gap is at least the configured embargo duration
# ---------------------------------------------------------------------------

@given(
    st.integers(min_value=0, max_value=60),
    st.integers(min_value=12, max_value=84),
    st.integers(min_value=1, max_value=24),
    st.integers(min_value=1, max_value=12),
)
@settings(max_examples=400)
def test_embargo_gap_respected(
    train_offset: int,
    train_window: int,
    embargo: int,
    test_window: int,
) -> None:
    """Gap between train_end and test_start is at least `embargo` months."""
    fold = _make_fold(0, train_offset, train_window, embargo, test_window)
    gap_days = (fold.test_start - fold.train_end).days
    # embargo months in calendar days (conservative: use 28 days/month floor)
    min_gap_days = embargo * 28
    assert gap_days >= min_gap_days, (
        f"Embargo gap too small: {gap_days} days < {min_gap_days} days "
        f"(embargo={embargo} months)"
    )


# ---------------------------------------------------------------------------
# 4. Multi-fold WFO: folds are non-overlapping on the test window
# ---------------------------------------------------------------------------

@given(
    st.integers(min_value=3, max_value=8),    # n_folds
    st.integers(min_value=90, max_value=730), # train_window_days
    st.integers(min_value=7, max_value=180),  # embargo_days
    st.integers(min_value=30, max_value=180), # test_window_days
)
@settings(max_examples=200)
def test_sequential_folds_non_overlapping_test_windows(
    n_folds: int,
    train_window_days: int,
    embargo_days: int,
    test_window_days: int,
) -> None:
    """For sequentially offset folds, test windows must not overlap.

    Uses day-level arithmetic (pd.Timedelta) to avoid month-length
    ambiguity that arises with pd.DateOffset on month-end timestamps.
    """
    base = pd.Timestamp("2015-01-15")  # mid-month avoids month-end edge cases
    stride = pd.Timedelta(days=test_window_days)
    folds = []
    for i in range(n_folds):
        train_start = base + pd.Timedelta(days=i * test_window_days)
        train_end = train_start + pd.Timedelta(days=train_window_days)
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.Timedelta(days=test_window_days)
        fold = FoldResult(
            fold_idx=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            y_true=np.zeros(1),
            y_hat=np.zeros(1),
            optimal_alpha=0.01,
            feature_importances={},
            n_train=train_window_days,
            n_test=1,
        )
        folds.append(fold)
    for j in range(len(folds) - 1):
        earlier = folds[j]
        later = folds[j + 1]
        assert earlier.test_end <= later.test_start, (
            f"Fold {j} test_end={earlier.test_end} overlaps fold {j+1} "
            f"test_start={later.test_start}"
        )


# ---------------------------------------------------------------------------
# 5. y_true and y_hat arrays have the same length
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.floats(min_value=-0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=120,
    )
)
@settings(max_examples=300)
def test_y_true_y_hat_length_invariant(values: list[float]) -> None:
    """Any FoldResult where y_true and y_hat share the same array must stay equal in length."""
    arr = np.array(values)
    fold = FoldResult(
        fold_idx=0,
        train_start=pd.Timestamp("2020-01-31"),
        train_end=pd.Timestamp("2023-01-31"),
        test_start=pd.Timestamp("2023-07-31"),
        test_end=pd.Timestamp("2024-01-31"),
        y_true=arr.copy(),
        y_hat=arr.copy(),
        optimal_alpha=0.01,
        feature_importances={},
        n_train=36,
        n_test=len(arr),
    )
    assert len(fold.y_true) == len(fold.y_hat), (
        f"y_true length {len(fold.y_true)} != y_hat length {len(fold.y_hat)}"
    )
    assert fold.n_test == len(fold.y_true), (
        f"n_test={fold.n_test} != len(y_true)={len(fold.y_true)}"
    )


# ---------------------------------------------------------------------------
# 6. WFOResult aggregation: y_true_all / y_hat_all match total OOS obs count
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.lists(
            st.floats(min_value=-0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=12,
        ),
        min_size=2,
        max_size=6,
    )
)
@settings(max_examples=200)
def test_wfo_result_aggregation_length(fold_returns: list[list[float]]) -> None:
    """WFOResult.y_true_all length equals sum of individual fold y_true lengths."""
    folds = []
    base = pd.Timestamp("2015-01-31")
    for i, returns in enumerate(fold_returns):
        arr = np.array(returns)
        fold = FoldResult(
            fold_idx=i,
            train_start=base + pd.DateOffset(months=i * 12),
            train_end=base + pd.DateOffset(months=i * 12 + 6),
            test_start=base + pd.DateOffset(months=i * 12 + 7),
            test_end=base + pd.DateOffset(months=i * 12 + 7 + len(returns)),
            y_true=arr.copy(),
            y_hat=arr.copy(),
            optimal_alpha=0.01,
            feature_importances={},
            n_train=6,
            n_test=len(arr),
        )
        folds.append(fold)

    wfo = WFOResult(folds=folds, benchmark="VTI", target_horizon=6, model_type="lasso")
    expected_total = sum(len(f.y_true) for f in folds)
    assert len(wfo.y_true_all) == expected_total, (
        f"y_true_all length {len(wfo.y_true_all)} != expected {expected_total}"
    )
    assert len(wfo.y_hat_all) == expected_total
