"""
Historical PGR RSU vesting event enumeration.

PGR issues RSUs on two annual schedules:
  - Time-based (January): nearest business day to January 19 each year.
  - Performance-based (July): nearest business day to July 17 each year.

``enumerate_vesting_events()`` generates every such event from ``start_year``
through the most recent year for which a fully realized 12-month forward
return is available (``current_year - 2`` is the safe default, ensuring no
look-ahead leakage in the backtest).

The ``end_year`` parameter can be overridden, e.g. to 2024 for a fixed-date
backtest, or to the current year - 1 to include 6-month-only events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, cast

import pandas as pd


@dataclass
class VestingEvent:
    """A single PGR RSU vesting date with forward-return windows."""
    event_date: date
    rsu_type: Literal["time", "performance"]
    horizon_6m_end: date
    horizon_12m_end: date


def _nearest_business_day(target: date) -> date:
    """
    Return the nearest business day to ``target``.

    Saturday → Friday (preceding) or Monday (following), whichever is closer.
    Sunday → Monday.

    Implementation: offset by -1 day while on a weekend; this matches the
    'preceding' convention most exchanges use.  Pandas business-day offset
    is not used to keep this dependency-free.
    """
    weekday = target.weekday()
    if weekday == 5:          # Saturday → previous Friday
        return target - timedelta(days=1)
    if weekday == 6:          # Sunday → next Monday
        return target + timedelta(days=1)
    return target


def _add_months(d: date, months: int) -> date:
    """Add ``months`` calendar months to date ``d`` (clamp day to month end)."""
    month = d.month + months
    year = d.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    # Clamp to the last valid day of the target month
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


def enumerate_vesting_events(
    start_year: int = 2014,
    end_year: int | None = None,
) -> list[VestingEvent]:
    """
    Enumerate all PGR RSU vesting events for the backtest window.

    Args:
        start_year: First year to include (default 2014; first year with
                    enough pre-event price history for a 60-month WFO train
                    window starting from the available data).
        end_year:   Last year to include (inclusive).  Defaults to
                    ``current_year - 2`` so that 12-month forward returns
                    are fully realized for all included events.  Pass
                    ``current_year - 1`` to include 6-month-only events.

    Returns:
        List of VestingEvent objects sorted by event_date ascending.
        January events precede July events within the same year.
    """
    if end_year is None:
        end_year = date.today().year - 2

    events: list[VestingEvent] = []

    for year in range(start_year, end_year + 1):
        for month, day, rsu_type in [
            (1, 19, "time"),
            (7, 17, "performance"),
        ]:
            target = date(year, month, day)
            event_date = _nearest_business_day(target)
            events.append(
                VestingEvent(
                    event_date=event_date,
                    rsu_type=cast(Literal["time", "performance"], rsu_type),
                    horizon_6m_end=_add_months(event_date, 6),
                    horizon_12m_end=_add_months(event_date, 12),
                )
            )

    return sorted(events, key=lambda e: e.event_date)


def enumerate_monthly_evaluation_dates(
    start_year: int = 2014,
    end_year: int | None = None,
) -> list[VestingEvent]:
    """
    Generate month-end evaluation dates for monthly stability analysis.

    Returns VestingEvent-compatible objects for every month-end between
    ``start_year`` and ``end_year``.  These are NOT actual vesting events —
    they exist purely for backtesting the model's predictive stability across
    the full calendar (120+ evaluation points vs. the ~20 from semi-annual
    vesting events alone).

    The ``rsu_type`` field is set to ``"time"`` as a placeholder; it carries
    no semantic meaning for monthly evaluation dates.

    Statistical note: Consecutive monthly predictions with a 6-month forward
    target share 5 of their 6 months of return window.  Use Newey-West
    standard errors (lag = 5 for 6M targets) when testing significance on
    the full monthly series.  Alternatively, subsample to every 6th month for
    independent observations.

    Key invariant: ``enumerate_monthly_evaluation_dates() ∩
    enumerate_vesting_events()`` must produce identical predictions when
    evaluated through either code path.  Any divergence indicates a bug in
    the temporal slicing logic.

    Args:
        start_year: First year to include (default 2014).
        end_year:   Last year to include (inclusive).  Defaults to
                    ``current_year - 2`` so that 12-month forward returns
                    are fully realized for all included dates.

    Returns:
        List of VestingEvent objects — one per month-end from
        ``start_year-01`` through ``end_year-12``, sorted ascending.
    """
    import calendar

    if end_year is None:
        end_year = date.today().year - 2

    events: list[VestingEvent] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Last calendar day of the month
            last_day = calendar.monthrange(year, month)[1]
            month_end = date(year, month, last_day)
            # Snap to last business day of the month (preceding convention).
            # Sat → Fri (-1), Sun → Fri (-2).  This always stays within the
            # same month because month-end is at least the 28th.
            weekday = month_end.weekday()
            if weekday == 5:      # Saturday → Friday
                event_date = month_end - timedelta(days=1)
            elif weekday == 6:    # Sunday → Friday
                event_date = month_end - timedelta(days=2)
            else:
                event_date = month_end

            events.append(
                VestingEvent(
                    event_date=event_date,
                    rsu_type="time",  # placeholder — not a real vesting event
                    horizon_6m_end=_add_months(event_date, 6),
                    horizon_12m_end=_add_months(event_date, 12),
                )
            )

    return sorted(events, key=lambda e: e.event_date)


def get_nearest_month_end(d: date) -> pd.Timestamp:
    """
    Return the month-end business day (BME) on or before date ``d``.

    Used to align a vesting event date to the feature matrix index, which
    uses month-end frequency.
    """
    ts = pd.Timestamp(d)
    # Go back to the last BME on or before ts
    bme = ts - pd.offsets.BusinessMonthEnd(0)
    # If ts itself is a BME, BusinessMonthEnd(0) returns ts; otherwise it
    # returns the prior BME.
    if bme > ts:
        bme = ts - pd.offsets.BusinessMonthEnd(1)
    return bme
