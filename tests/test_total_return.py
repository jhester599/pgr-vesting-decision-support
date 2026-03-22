"""
Tests for src/processing/total_return.py

Validates DRIP total return reconstruction using fully synthetic data.
No API calls are made. Tests verify:
  - Share counts increase on each ex-dividend date
  - Share counts multiply on each split date
  - Total return is consistent with portfolio value formula
  - No negative-price or negative-share artifacts
  - Forward return series has NaN in the final forward_months window
"""

import pytest
import pandas as pd
import numpy as np

from src.processing.total_return import (
    build_position_series,
    compute_total_return,
    build_monthly_returns,
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_prices() -> pd.DataFrame:
    """
    3 years of daily prices at a flat $100, so returns are purely from DRIP.
    Uses a business-day range.
    """
    dates = pd.bdate_range("2018-01-01", "2020-12-31")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return df


@pytest.fixture()
def quarterly_dividends() -> pd.DataFrame:
    """$1.00 per share quarterly dividend, paid on the first business day of each quarter."""
    div_dates = pd.to_datetime([
        "2018-01-02", "2018-04-02", "2018-07-02", "2018-10-01",
        "2019-01-02", "2019-04-01", "2019-07-01", "2019-10-01",
        "2020-01-02", "2020-04-01", "2020-07-01", "2020-10-01",
    ])
    df = pd.DataFrame({"dividend": 1.0}, index=pd.DatetimeIndex(div_dates, name="date"))
    return df


@pytest.fixture()
def one_split() -> pd.DataFrame:
    """Single 2-for-1 split on 2019-06-03."""
    df = pd.DataFrame(
        {"numerator": 2.0, "denominator": 1.0, "split_ratio": 2.0},
        index=pd.DatetimeIndex([pd.Timestamp("2019-06-03")], name="date"),
    )
    return df


@pytest.fixture()
def no_splits() -> pd.DataFrame:
    return pd.DataFrame(
        {"numerator": [], "denominator": [], "split_ratio": []},
        index=pd.DatetimeIndex([], name="date"),
    )


# ---------------------------------------------------------------------------
# build_position_series
# ---------------------------------------------------------------------------

class TestBuildPositionSeries:
    def test_shares_increase_on_each_div_date(
        self, flat_prices, quarterly_dividends, no_splits
    ):
        pos = build_position_series(flat_prices, quarterly_dividends, no_splits)
        # Check each ex-div date: shares_held the day after > shares_held the day before
        for div_date in quarterly_dividends.index:
            if div_date not in pos.index:
                continue
            day_of = pos["shares_held"].asof(div_date)
            next_bday = div_date + pd.offsets.BusinessDay(1)
            if next_bday in pos.index:
                day_after = pos.at[next_bday, "shares_held"]
                assert day_after >= day_of, (
                    f"Shares did not increase on or after dividend date {div_date}"
                )

    def test_shares_double_on_split(self, flat_prices, no_splits, quarterly_dividends):
        """Ignore dividends here; focus purely on the split multiplier."""
        no_div = pd.DataFrame(
            {"dividend": []}, index=pd.DatetimeIndex([], name="date")
        )
        pos = build_position_series(flat_prices, no_div, one_split())
        split_date = pd.Timestamp("2019-06-03")
        before = pos["shares_held"].asof(split_date - pd.offsets.BusinessDay(1))
        after = pos["shares_held"].asof(split_date + pd.offsets.BusinessDay(1))
        assert np.isclose(after, before * 2.0), (
            f"Expected shares to double after split: {before} -> {after}"
        )

    def test_portfolio_value_formula(self, flat_prices, quarterly_dividends, no_splits):
        pos = build_position_series(flat_prices, quarterly_dividends, no_splits)
        computed = pos["shares_held"] * pos["close_price"]
        assert np.allclose(computed, pos["portfolio_value"], rtol=1e-9)

    def test_no_negative_shares(self, flat_prices, quarterly_dividends, no_splits):
        pos = build_position_series(flat_prices, quarterly_dividends, no_splits)
        assert (pos["shares_held"] > 0).all(), "shares_held must always be positive."

    def test_no_negative_prices(self, flat_prices, quarterly_dividends, no_splits):
        pos = build_position_series(flat_prices, quarterly_dividends, no_splits)
        assert (pos["close_price"] > 0).all(), "close_price must always be positive."

    def test_initial_share_count(self, flat_prices, quarterly_dividends, no_splits):
        """First row should equal initial_shares before any events on that day."""
        pos = build_position_series(
            flat_prices, quarterly_dividends, no_splits, initial_shares=500.0
        )
        # First date has no div or split event
        assert pos["shares_held"].iloc[0] == 500.0


# ---------------------------------------------------------------------------
# compute_total_return
# ---------------------------------------------------------------------------

class TestComputeTotalReturn:
    def test_zero_return_flat_no_dividend(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        pos = pd.DataFrame(
            {"shares_held": 1.0, "close_price": 50.0, "portfolio_value": 50.0},
            index=pd.DatetimeIndex(dates, name="date"),
        )
        tr = compute_total_return(
            pos, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-29")
        )
        assert np.isclose(tr, 0.0), f"Expected 0.0 return, got {tr}"

    def test_drip_positive_return_flat_price(
        self, flat_prices, quarterly_dividends, no_splits
    ):
        """Even with flat prices, DRIP dividends produce a positive total return."""
        pos = build_position_series(flat_prices, quarterly_dividends, no_splits)
        tr = compute_total_return(
            pos, pd.Timestamp("2018-01-02"), pd.Timestamp("2020-12-30")
        )
        assert tr > 0.0, f"DRIP should produce positive return; got {tr}"

    def test_raises_on_zero_start_value(self):
        dates = pd.bdate_range("2020-01-01", "2020-03-31")
        pos = pd.DataFrame(
            {"shares_held": 1.0, "close_price": 50.0, "portfolio_value": 0.0},
            index=pd.DatetimeIndex(dates, name="date"),
        )
        pos.iloc[0, pos.columns.get_loc("portfolio_value")] = 0.0
        with pytest.raises(ValueError, match="zero or NaN"):
            compute_total_return(
                pos, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-03-30")
            )


# ---------------------------------------------------------------------------
# build_monthly_returns
# ---------------------------------------------------------------------------

class TestBuildMonthlyReturns:
    def test_final_window_is_nan(self, flat_prices, quarterly_dividends, no_splits):
        """
        The last ``forward_months`` months of the series must be NaN
        because no forward data is available — critical no-leakage check.
        """
        series = build_monthly_returns(
            flat_prices, quarterly_dividends, no_splits, forward_months=6
        )
        data_end = flat_prices.index.max()
        cutoff = data_end - pd.DateOffset(months=6)
        tail = series[series.index > cutoff]
        assert tail.isna().all(), (
            "Forward return must be NaN for the final forward_months window."
        )

    def test_non_nan_before_cutoff(self, flat_prices, quarterly_dividends, no_splits):
        """Observations well before the cutoff should have valid return values."""
        series = build_monthly_returns(
            flat_prices, quarterly_dividends, no_splits, forward_months=6
        )
        early = series[series.index < pd.Timestamp("2020-01-01")]
        assert early.notna().any(), "Expected some non-NaN returns before the cutoff."

    def test_index_frequency_is_monthly(
        self, flat_prices, quarterly_dividends, no_splits
    ):
        series = build_monthly_returns(
            flat_prices, quarterly_dividends, no_splits, forward_months=6
        )
        # All dates should be month-end-ish (last business day of the month)
        assert len(series) > 0
        # Gaps between consecutive dates should be approximately 1 month
        diffs = pd.Series(series.index).diff().dropna()
        assert (diffs.dt.days >= 25).all(), "Monthly frequency check failed."

    def test_series_name(self, flat_prices, quarterly_dividends, no_splits):
        series = build_monthly_returns(
            flat_prices, quarterly_dividends, no_splits, forward_months=6
        )
        assert series.name == "target_6m_return"


# ---------------------------------------------------------------------------
# Helper to access the one_split fixture outside a class
# ---------------------------------------------------------------------------

def one_split() -> pd.DataFrame:
    df = pd.DataFrame(
        {"numerator": 2.0, "denominator": 1.0, "split_ratio": 2.0},
        index=pd.DatetimeIndex([pd.Timestamp("2019-06-03")], name="date"),
    )
    return df
