"""
Detect stock splits from Alpha Vantage adjusted price series (review F03/F05).

Two payload shapes are supported:

* ``TIME_SERIES_DAILY_ADJUSTED`` ("Time Series (Daily)") carries an explicit
  ``8. split coefficient`` per bar; any value other than 1 is a split.
* ``TIME_SERIES_WEEKLY_ADJUSTED`` ("Weekly Adjusted Time Series", free tier)
  has no coefficient field, so the coefficient is recovered from the
  adjustment factor ``f_t = adjusted_close_t / close_t``.  Adjusted closes are
  back-adjusted for splits and dividends, so across a split of ratio R between
  bars t-1 and t, ``f_t / f_{t-1} = R`` up to that week's small dividend
  adjustment.  Ratios far from 1 are snapped to the nearest simple fraction.

Detections are reconciled against the canonical registry
(``config.KNOWN_SPLITS``) by :func:`reconcile_detected_splits`; anything
unknown must be verified and added to ``config/splits.py`` by hand, which
keeps one reviewed source of truth.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import pandas as pd

_DAILY_KEY = "Time Series (Daily)"
_WEEKLY_KEY = "Weekly Adjusted Time Series"

# |log(factor ratio)| above this is treated as a split candidate. Dividend
# adjustments move the factor by a few percent at most; the smallest common
# split (5-for-4) moves it by 25 %.
_MIN_SPLIT_LOG_RATIO = math.log(1.15)
# A candidate must be within this relative distance of a simple fraction.
_SNAP_TOLERANCE = 0.04
_MAX_SPLIT_TERM = 20
_MATCH_WINDOW_DAYS = 7


_SIMPLE_FRACTIONS: list[Fraction] = sorted(
    {
        Fraction(n, d)
        for n in range(1, _MAX_SPLIT_TERM + 1)
        for d in range(1, _MAX_SPLIT_TERM + 1)
        if n != d
    },
    key=lambda f: (f.numerator + f.denominator, f),
)


def _snap_ratio(raw_ratio: float) -> Fraction | None:
    """Return the simplest fraction n/d (n, d <= 20) within tolerance of ``raw_ratio``.

    Simplest means smallest ``n + d``, so 1.26 snaps to 5/4 rather than 19/15.
    """
    if raw_ratio <= 0 or not math.isfinite(raw_ratio):
        return None
    for candidate in _SIMPLE_FRACTIONS:
        if abs(float(candidate) / raw_ratio - 1.0) <= _SNAP_TOLERANCE:
            return candidate
    return None


def _split_record(ticker: str, bar_date: str, ratio: Fraction, raw: float) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "bar_date": bar_date,
        "split_ratio": float(ratio),
        "numerator": float(ratio.numerator),
        "denominator": float(ratio.denominator),
        "raw_ratio": raw,
    }


def detect_splits_from_adjusted_series(raw: dict, ticker: str) -> list[dict[str, Any]]:
    """Return the splits implied by an AV adjusted time-series payload.

    Args:
        raw:    Parsed JSON from ``TIME_SERIES_DAILY_ADJUSTED`` or
                ``TIME_SERIES_WEEKLY_ADJUSTED``.
        ticker: Ticker symbol written into each record.

    Returns:
        List of dicts with ``ticker``, ``bar_date`` (first bar that trades
        split-adjusted; for weekly data the split happened during that week),
        ``split_ratio``, ``numerator``, ``denominator`` and ``raw_ratio``,
        sorted by date.
    """
    if _DAILY_KEY in raw:
        records = []
        for bar_date, vals in sorted(raw[_DAILY_KEY].items()):
            try:
                coeff = float(vals.get("8. split coefficient", 1.0))
            except (TypeError, ValueError):
                continue
            if abs(coeff - 1.0) < 1e-9:
                continue
            snapped = _snap_ratio(coeff)
            if snapped is not None:
                records.append(_split_record(ticker, bar_date, snapped, coeff))
        return records

    series = raw.get(_WEEKLY_KEY, {})
    rows = []
    for bar_date, vals in series.items():
        try:
            close = float(vals["4. close"])
            adjusted = float(vals["5. adjusted close"])
        except (KeyError, TypeError, ValueError):
            continue
        if close > 0 and adjusted > 0:
            rows.append((bar_date, adjusted / close))
    if len(rows) < 2:
        return []
    factors = pd.Series(dict(rows)).sort_index()
    ratios = factors / factors.shift(1)
    records = []
    for bar_date, ratio in ratios.iloc[1:].items():
        if abs(math.log(ratio)) < _MIN_SPLIT_LOG_RATIO:
            continue
        snapped = _snap_ratio(float(ratio))
        if snapped is not None:
            records.append(_split_record(ticker, str(bar_date), snapped, float(ratio)))
    return records


def reconcile_detected_splits(
    detected: list[dict[str, Any]],
    known: list[dict[str, Any]],
    window_days: int = _MATCH_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Compare detected splits with the canonical registry.

    A detection matches a known split of the same ticker whose ``split_date``
    lies within ``window_days`` of the detection's ``bar_date``.

    Returns:
        One dict per problem, with ``ticker``, ``bar_date``, ``issue``
        (``"unknown_split"`` or ``"ratio_mismatch"``), ``detected_ratio`` and
        ``known_ratio`` (None when unknown).  Empty when everything matches.
    """
    issues: list[dict[str, Any]] = []
    window = pd.Timedelta(days=window_days)
    for det in detected:
        bar = pd.Timestamp(det["bar_date"])
        candidates = [
            k for k in known
            if k["ticker"] == det["ticker"]
            and abs(pd.Timestamp(k["split_date"]) - bar) <= window
        ]
        if not candidates:
            issues.append({
                "ticker": det["ticker"], "bar_date": det["bar_date"],
                "issue": "unknown_split", "detected_ratio": det["split_ratio"],
                "known_ratio": None,
            })
            continue
        known_ratio = float(candidates[0]["split_ratio"])
        if abs(known_ratio / float(det["split_ratio"]) - 1.0) > _SNAP_TOLERANCE:
            issues.append({
                "ticker": det["ticker"], "bar_date": det["bar_date"],
                "issue": "ratio_mismatch", "detected_ratio": det["split_ratio"],
                "known_ratio": known_ratio,
            })
    return issues
