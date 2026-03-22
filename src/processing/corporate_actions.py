"""
Corporate action processing: forward-apply stock splits to share counts.

Strategy: unadjusted (raw) prices are always preserved. Instead of
backward-adjusting prices (which distorts capital required at each date),
we forward-adjust the *share count* held in the position. This is the
only mathematically correct approach for simulating a buy-and-hold
accumulation strategy.

For a split with ratio R on date t_s:
  shares_held[t >= t_s] *= R

This module also exposes a validator that asserts the three known PGR
splits are present in the loaded split history.
"""

import warnings
from typing import Sequence

import pandas as pd

import config


def apply_splits(
    shares_series: pd.Series,
    split_history: pd.DataFrame,
) -> pd.Series:
    """
    Forward-apply all stock splits to a date-indexed share series.

    Args:
        shares_series: DatetimeIndex Series of share counts (float64),
                       representing the number of shares held on each date.
                       Must be sorted ascending by date.
        split_history: DataFrame returned by split_loader.load(), with a
                       DatetimeIndex and a ``split_ratio`` column.

    Returns:
        New Series with the same index as ``shares_series``, with split
        adjustments applied forward in time. Values before each split date
        are unchanged; values on and after each split date are multiplied
        by the split ratio.
    """
    if split_history.empty:
        warnings.warn("split_history is empty; no splits applied.", stacklevel=2)
        return shares_series.copy()

    result = shares_series.astype("float64").copy()

    for split_date, row in split_history.iterrows():
        ratio = row["split_ratio"]
        mask = result.index >= split_date
        result.loc[mask] = result.loc[mask] * ratio

    return result


def validate_known_splits(split_history: pd.DataFrame) -> None:
    """
    Assert that the three known PGR historical splits are present in
    ``split_history``. Raises AssertionError if any are missing or have
    an unexpected ratio. Issues a warning for each matched split.

    Args:
        split_history: DataFrame from split_loader.load().

    Raises:
        AssertionError: If a known split is not found within a 5-day
                        date tolerance or has an incorrect ratio.
    """
    for known in config.PGR_KNOWN_SPLITS:
        expected_date = pd.Timestamp(known["date"])
        expected_ratio = known["ratio"]
        tolerance = pd.Timedelta(days=5)

        nearby = split_history[
            (split_history.index >= expected_date - tolerance)
            & (split_history.index <= expected_date + tolerance)
        ]

        assert not nearby.empty, (
            f"Known PGR split on {known['date']} (ratio {expected_ratio}) "
            "not found in split_history."
        )

        actual_ratio = nearby["split_ratio"].iloc[0]
        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"PGR split near {known['date']}: expected ratio {expected_ratio}, "
            f"got {actual_ratio}."
        )


def get_cum_split_multiplier(
    date: pd.Timestamp,
    split_history: pd.DataFrame,
) -> float:
    """
    Return the cumulative split multiplier from the start of history up to
    (but not including) ``date``.

    Useful for converting a share count at a historical grant date to the
    equivalent post-split share count in the present.

    Args:
        date: The reference date (splits strictly before this date are included).
        split_history: DataFrame from split_loader.load().

    Returns:
        Cumulative product of all split ratios prior to ``date``.
    """
    past_splits = split_history[split_history.index < date]
    if past_splits.empty:
        return 1.0
    multiplier: float = past_splits["split_ratio"].prod()
    return multiplier


def iter_splits_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    split_history: pd.DataFrame,
) -> Sequence[tuple[pd.Timestamp, float]]:
    """
    Yield (date, ratio) tuples for splits occurring in [start, end].

    Args:
        start: Inclusive start date.
        end:   Inclusive end date.
        split_history: DataFrame from split_loader.load().

    Returns:
        List of (split_date, ratio) tuples sorted ascending by date.
    """
    subset = split_history[
        (split_history.index >= start) & (split_history.index <= end)
    ]
    return [(idx, row["split_ratio"]) for idx, row in subset.iterrows()]
