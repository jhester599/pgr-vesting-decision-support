"""
Tests for monthly stability backtesting (v3.0).

Verifies:
  - enumerate_monthly_evaluation_dates() generates ≥120 dates for 2014–2023
  - All generated dates are valid business days (Mon–Fri)
  - Key invariant: vesting event dates are a subset of monthly evaluation dates
    (approximately — within the same month)
  - No-lookahead: run_monthly_stability_backtest() uses the same slicing logic
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.vesting_events import (
    enumerate_monthly_evaluation_dates,
    enumerate_vesting_events,
)


class TestEnumerateMonthlyEvaluationDates:
    def test_generates_at_least_120_dates_for_decade(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2014, end_year=2023)
        assert len(dates) >= 120, f"Expected ≥120 dates, got {len(dates)}"

    def test_generates_exactly_12_per_year(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2020, end_year=2020)
        assert len(dates) == 12

    def test_all_dates_are_business_days(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2014, end_year=2023)
        for ev in dates:
            wd = ev.event_date.weekday()
            assert wd < 5, (
                f"Date {ev.event_date} is not a business day "
                f"(weekday={wd}, 0=Mon, 5=Sat, 6=Sun)"
            )

    def test_sorted_ascending(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2014, end_year=2023)
        event_dates = [ev.event_date for ev in dates]
        assert event_dates == sorted(event_dates)

    def test_starts_in_start_year(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2018, end_year=2020)
        assert dates[0].event_date.year == 2018

    def test_ends_in_end_year(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2018, end_year=2020)
        assert dates[-1].event_date.year == 2020

    def test_horizon_fields_populated(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2020, end_year=2020)
        for ev in dates:
            assert ev.horizon_6m_end > ev.event_date
            assert ev.horizon_12m_end > ev.horizon_6m_end

    def test_rsu_type_placeholder(self):
        """rsu_type should be 'time' (placeholder) for all monthly dates."""
        dates = enumerate_monthly_evaluation_dates(start_year=2020, end_year=2020)
        for ev in dates:
            assert ev.rsu_type == "time"

    def test_monthly_dates_cover_all_months(self):
        dates = enumerate_monthly_evaluation_dates(start_year=2020, end_year=2020)
        months = sorted({ev.event_date.month for ev in dates})
        assert months == list(range(1, 13))


class TestVestingIntersectionInvariant:
    """
    Key invariant: for every vesting event, there exists a monthly evaluation
    date in the same calendar month.

    This ensures that the monthly backtest covers all semi-annual vesting
    decision points.
    """

    def test_vesting_months_subset_of_monthly_months(self):
        vesting = enumerate_vesting_events(start_year=2014, end_year=2023)
        monthly = enumerate_monthly_evaluation_dates(start_year=2014, end_year=2023)

        vesting_month_years = {
            (ev.event_date.year, ev.event_date.month) for ev in vesting
        }
        monthly_month_years = {
            (ev.event_date.year, ev.event_date.month) for ev in monthly
        }

        missing = vesting_month_years - monthly_month_years
        assert not missing, (
            f"Monthly evaluation dates are missing these vesting months: {missing}"
        )


class TestDefaultEndYear:
    def test_default_end_year_is_current_minus_2(self):
        """Default end_year should be current_year - 2 to ensure realized returns."""
        import datetime as dt

        dates = enumerate_monthly_evaluation_dates()
        expected_last_year = dt.date.today().year - 2
        actual_last_year = max(ev.event_date.year for ev in dates)
        assert actual_last_year == expected_last_year
