"""Derived fields for the PGR monthly EDGAR table: one definition each.

Review 2026-09-25 (F11, F16) found three copies of the Gainshare formula,
row-based ``pct_change(12)`` YoY that spans 13 months at data gaps, a
leap-year key bug, and a policies-in-force (PIF) total whose definition
changed in 2024.  Every producer of these fields
(``scripts/edgar_8k_fetcher.py``, ``src/ingestion/pgr_monthly_loader.py``,
``src/ingestion/edgar_8k_fetcher.py``) now calls this module.

Definitions
-----------
* **PIF total** (thousands of policies) = agency auto + direct auto +
  special lines + commercial lines.  Property PIF is excluded: PGR's printed
  "companywide total" added property from 2024-04 (and "total personal lines"
  from 2024-12), which made reported growth jump ~13 points with no change
  in the business.  Property PIF is stored separately in ``pif_property``.
* **PIF total personal lines** = agency auto + direct auto + special lines
  (property excluded, for the same reason).
* **YoY growth** compares a month with the same calendar month one year
  earlier (``Period("M") - 12``).  It is NaN when that base month is missing
  or non-positive; it never falls back to "12 rows earlier".
* **Gainshare estimate** (0-2) = 0.5 x clip((96 - CR) / 10, 0, 2)
  + 0.5 x clip(PIF growth / 0.10, 0, 2).  Both inputs are required; the
  estimate is NaN when either is missing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

GAINSHARE_CR_TARGET: float = 96.0
GAINSHARE_CR_SCALE: float = 10.0
GAINSHARE_PIF_GROWTH_FULL: float = 0.10
GAINSHARE_SCORE_CAP: float = 2.0

PIF_PERSONAL_LINES_COMPONENTS: tuple[str, ...] = (
    "pif_agency_auto",
    "pif_direct_auto",
    "pif_special_lines",
)
PIF_TOTAL_COMPONENTS: tuple[str, ...] = PIF_PERSONAL_LINES_COMPONENTS + (
    "pif_commercial_lines",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def month_key(month_end: Any) -> pd.Period:
    """Return the calendar month of a month-end date as ``Period("M")``."""
    return pd.Period(pd.Timestamp(month_end), freq="M")


def prior_year_month_end(month_end: str) -> str:
    """Return the month-end date string of the same month one year earlier.

    Uses period arithmetic, so 2025-02-28 maps to 2024-02-29 and
    2024-02-29 maps to 2023-02-28.
    """
    prior = month_key(month_end) - 12
    return prior.to_timestamp(how="end").strftime("%Y-%m-%d")


def gainshare_estimate(combined_ratio: Any, pif_growth_yoy: Any) -> float | None:
    """Return the Gainshare estimate for one month, or None if an input is missing."""
    if _is_missing(combined_ratio) or _is_missing(pif_growth_yoy):
        return None
    cr_score = min(
        max((GAINSHARE_CR_TARGET - float(combined_ratio)) / GAINSHARE_CR_SCALE, 0.0),
        GAINSHARE_SCORE_CAP,
    )
    pif_score = min(
        max(float(pif_growth_yoy) / GAINSHARE_PIF_GROWTH_FULL, 0.0),
        GAINSHARE_SCORE_CAP,
    )
    return 0.5 * cr_score + 0.5 * pif_score


def gainshare_series(combined_ratio: pd.Series, pif_growth_yoy: pd.Series) -> pd.Series:
    """Vectorised :func:`gainshare_estimate` (NaN where either input is NaN)."""
    cr_score = (
        (GAINSHARE_CR_TARGET - combined_ratio) / GAINSHARE_CR_SCALE
    ).clip(lower=0.0, upper=GAINSHARE_SCORE_CAP)
    pif_score = (pif_growth_yoy / GAINSHARE_PIF_GROWTH_FULL).clip(
        lower=0.0, upper=GAINSHARE_SCORE_CAP
    )
    return 0.5 * cr_score + 0.5 * pif_score


def pif_sum(record: Mapping[str, Any], components: Iterable[str]) -> float | None:
    """Sum PIF components; None unless every component is present and >= 0."""
    total = 0.0
    for name in components:
        value = record.get(name)
        if value is None or _is_missing(value) or float(value) < 0:
            return None
        total += float(value)
    return round(total, 1)


def yoy_growth_by_period(values: Mapping[Any, Any]) -> dict[pd.Period, float | None]:
    """Return calendar-month YoY growth for a {month-end: value} mapping.

    Growth is ``value[m] / value[m - 12 months] - 1`` and None when either
    value is missing or the base is not positive.
    """
    by_period: dict[pd.Period, float] = {}
    for month_end, value in values.items():
        if not _is_missing(value):
            by_period[month_key(month_end)] = float(value)
    out: dict[pd.Period, float | None] = {}
    for period, value in by_period.items():
        base = by_period.get(period - 12)
        out[period] = value / base - 1.0 if base is not None and base > 0 else None
    return out


def yoy_growth_series(series: pd.Series) -> pd.Series:
    """Calendar-month YoY growth of a date-indexed series (same index back).

    Unlike ``series.pct_change(12)``, a missing month yields NaN twelve months
    later instead of a 13-month change.
    """
    if series.empty:
        return series.astype(float)
    growth = yoy_growth_by_period(dict(zip(series.index, series.to_numpy())))
    return pd.Series(
        [growth.get(month_key(ts)) for ts in series.index],
        index=series.index,
        dtype=float,
        name=series.name,
    )
