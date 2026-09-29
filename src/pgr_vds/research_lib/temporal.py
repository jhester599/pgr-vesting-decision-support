"""Explicit monthly chronology and availability rules for research folds."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import TimeSeriesSplit

DateInput = Iterable[object]
IndexArray = NDArray[np.int64]


def label_end(origin: pd.Timestamp, h: int) -> pd.Timestamp:
    """Return the last weekday of the calendar month h months after origin."""
    if h not in (6, 12):
        raise ValueError("The registered horizons are six and twelve months.")
    month = pd.Timestamp(origin).to_period("M") + h
    month_end = month.to_timestamp(how="end").normalize()
    return pd.offsets.BMonthEnd().rollback(month_end)


def chronological_splits(
    origins: DateInput, h: int, inner: bool = False,
) -> list[tuple[IndexArray, IndexArray]]:
    """Split unique contiguous monthly origins before any panel expansion.

    Missing calendar months must be represented by an explicit unsupported
    origin rather than silently shortening a month-based purge/embargo.
    Unconstructible folds return no evidence; gaps are never reduced.
    """
    if h not in (6, 12):
        raise ValueError("The registered horizons are six and twelve months.")
    dates = pd.DatetimeIndex(origins)
    if dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("Origins must be valid chronological dates.")
    months = dates.to_period("M")
    if months.has_duplicates:
        raise ValueError(
            "Origins must have unique monthly dates; duplicates found."
        )
    if len(months) > 1 and np.any(np.diff(months.asi8) != 1):
        raise ValueError(
            "Monthly origins must be contiguous; missing months found."
        )
    window = 60 if h == 6 else 120
    gap = 2 * h
    n_splits = 3 if inner else (len(dates) - window - gap) // 6
    if n_splits < 2 or len(dates) - n_splits * 6 - gap <= 0:
        return []
    splitter = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=window, test_size=6, gap=gap,
    )
    return list(splitter.split(np.arange(len(dates))))


def eligible_training(
    origins: DateInput,
    ends: DateInput,
    available: DateInput,
    test_origins: DateInput,
    min_months: int,
) -> IndexArray:
    """Keep labels and sources available before every origin in the test fold.

    ``available`` is the maximum required label/source availability for each
    training row. Unknown availability cannot satisfy the gate. Minimum
    support counts unique calendar months, never repeated panel rows.
    """
    dates = pd.DatetimeIndex(origins)
    label_ends = pd.DatetimeIndex(ends)
    arrivals = pd.DatetimeIndex(available)
    tests = pd.DatetimeIndex(test_origins)
    if len(dates) != len(label_ends) or len(dates) != len(arrivals):
        raise ValueError(
            "Origins, ends, and availability must have equal length."
        )
    if min_months < 1:
        raise ValueError("Minimum monthly support must be positive.")
    if tests.empty or tests.hasnans:
        return np.array([], dtype=np.int64)
    # Checking against the earliest origin is equivalent to checking each
    # test origin, including an unsorted test input.
    first_test = tests.min()
    keep = (
        (dates < first_test)
        & (label_ends <= first_test)
        & (arrivals <= first_test)
    )
    indices = np.flatnonzero(keep).astype(np.int64)
    support = dates[indices].to_period("M").nunique()
    return indices if support >= min_months else np.array([], dtype=np.int64)
