"""
Tests for src/processing/corporate_actions.py

Validates split application logic and the known-split validator using
synthetic data — no API calls are made.
"""

import pytest
import pandas as pd
import numpy as np

from src.processing.corporate_actions import (
    apply_splits,
    validate_known_splits,
    get_cum_split_multiplier,
    iter_splits_between,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_split_history() -> pd.DataFrame:
    """Two splits: 2-for-1 on 2010-06-01, 3-for-1 on 2015-01-01."""
    data = {
        "numerator": [2.0, 3.0],
        "denominator": [1.0, 1.0],
        "split_ratio": [2.0, 3.0],
    }
    idx = pd.to_datetime(["2010-06-01", "2015-01-01"])
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="date"))


@pytest.fixture()
def daily_dates() -> pd.DatetimeIndex:
    """Business-day range spanning both splits."""
    return pd.bdate_range("2009-01-01", "2020-01-01")


@pytest.fixture()
def shares_series(daily_dates) -> pd.Series:
    """100 shares held uniformly across all dates (before any splits)."""
    return pd.Series(100.0, index=daily_dates, name="shares_held")


# ---------------------------------------------------------------------------
# apply_splits
# ---------------------------------------------------------------------------

class TestApplySplits:
    def test_pre_split_unchanged(self, shares_series, simple_split_history):
        result = apply_splits(shares_series, simple_split_history)
        pre_split = result[result.index < pd.Timestamp("2010-06-01")]
        assert (pre_split == 100.0).all(), "Shares before first split must be unchanged."

    def test_after_first_split(self, shares_series, simple_split_history):
        result = apply_splits(shares_series, simple_split_history)
        after_first = result[
            (result.index >= pd.Timestamp("2010-06-01"))
            & (result.index < pd.Timestamp("2015-01-01"))
        ]
        expected = 100.0 * 2.0
        assert np.allclose(after_first, expected), (
            f"After 2-for-1 split, expected {expected} shares."
        )

    def test_after_both_splits(self, shares_series, simple_split_history):
        result = apply_splits(shares_series, simple_split_history)
        after_both = result[result.index >= pd.Timestamp("2015-01-01")]
        expected = 100.0 * 2.0 * 3.0  # 600 shares
        assert np.allclose(after_both, expected), (
            f"After both splits, expected {expected} shares."
        )

    def test_empty_split_history(self, shares_series):
        empty = pd.DataFrame(columns=["numerator", "denominator", "split_ratio"])
        empty.index = pd.DatetimeIndex([], name="date")
        result = apply_splits(shares_series, empty)
        assert (result == 100.0).all(), "No splits: series must be unchanged."

    def test_returns_float64(self, shares_series, simple_split_history):
        result = apply_splits(shares_series, simple_split_history)
        assert result.dtype == np.float64

    def test_index_preserved(self, shares_series, simple_split_history):
        result = apply_splits(shares_series, simple_split_history)
        assert result.index.equals(shares_series.index)


# ---------------------------------------------------------------------------
# validate_known_splits — uses a mock containing the actual PGR splits
# ---------------------------------------------------------------------------

@pytest.fixture()
def pgr_split_history() -> pd.DataFrame:
    data = {
        "numerator": [3.0, 3.0, 4.0],
        "denominator": [1.0, 1.0, 1.0],
        "split_ratio": [3.0, 3.0, 4.0],
    }
    idx = pd.to_datetime(["1992-12-09", "2002-04-23", "2006-05-19"])
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="date"))


class TestValidateKnownSplits:
    def test_all_known_splits_present(self, pgr_split_history):
        validate_known_splits(pgr_split_history)  # Should not raise

    def test_missing_split_raises(self, pgr_split_history):
        missing = pgr_split_history.iloc[[0, 1]]  # Drop the 2006 4-for-1
        with pytest.raises(AssertionError, match="2006"):
            validate_known_splits(missing)

    def test_wrong_ratio_raises(self, pgr_split_history):
        wrong = pgr_split_history.copy()
        wrong.iloc[2, wrong.columns.get_loc("split_ratio")] = 2.0  # Should be 4.0
        with pytest.raises(AssertionError, match="ratio"):
            validate_known_splits(wrong)

    def test_date_tolerance(self, pgr_split_history):
        """A split date off by 3 days should still validate."""
        shifted = pgr_split_history.copy()
        new_idx = pgr_split_history.index.tolist()
        new_idx[2] = pd.Timestamp("2006-05-22")  # 3 days late
        shifted.index = pd.DatetimeIndex(new_idx, name="date")
        validate_known_splits(shifted)  # Should not raise


# ---------------------------------------------------------------------------
# get_cum_split_multiplier
# ---------------------------------------------------------------------------

class TestGetCumSplitMultiplier:
    def test_before_any_split(self, simple_split_history):
        mult = get_cum_split_multiplier(pd.Timestamp("2005-01-01"), simple_split_history)
        assert mult == 1.0

    def test_between_splits(self, simple_split_history):
        mult = get_cum_split_multiplier(pd.Timestamp("2012-01-01"), simple_split_history)
        assert mult == 2.0  # Only first split applies

    def test_after_both_splits(self, simple_split_history):
        mult = get_cum_split_multiplier(pd.Timestamp("2020-01-01"), simple_split_history)
        assert mult == 6.0  # 2 * 3

    def test_on_split_date_excluded(self, simple_split_history):
        """Splits strictly BEFORE the reference date; split on the date is excluded."""
        mult = get_cum_split_multiplier(pd.Timestamp("2010-06-01"), simple_split_history)
        assert mult == 1.0


# ---------------------------------------------------------------------------
# iter_splits_between
# ---------------------------------------------------------------------------

class TestIterSplitsBetween:
    def test_no_splits_in_range(self, simple_split_history):
        result = iter_splits_between(
            pd.Timestamp("2001-01-01"), pd.Timestamp("2009-01-01"), simple_split_history
        )
        assert len(result) == 0

    def test_one_split_in_range(self, simple_split_history):
        result = iter_splits_between(
            pd.Timestamp("2010-01-01"), pd.Timestamp("2011-01-01"), simple_split_history
        )
        assert len(result) == 1
        assert result[0][1] == 2.0

    def test_both_splits_in_range(self, simple_split_history):
        result = iter_splits_between(
            pd.Timestamp("2009-01-01"), pd.Timestamp("2020-01-01"), simple_split_history
        )
        assert len(result) == 2
