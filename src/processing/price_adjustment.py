"""
Split-adjusted, weekly-bar price series for every price-derived feature.

``daily_prices`` holds one unadjusted bar per ISO week (Alpha Vantage
``TIME_SERIES_WEEKLY``; review 2026-09-25, F01). Features must therefore:

  - restate every close onto the latest share basis before comparing two
    dates (``split_adjusted_close``), so a split is not read as a price move;
  - define windows in calendar terms (months, or weeks of weekly bars), never
    as a count of "trading days".

Raw closes are kept only for DRIP total-return accounting
(``src.processing.total_return`` / ``multi_total_return``), which applies
splits to the share count instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.processing.valuation_multiples import (
    latest_share_basis_factor,
    share_basis_factor,
)

WEEKS_PER_YEAR: int = 52

# Median spacing (calendar days) accepted for weekly bars. A daily input is
# collapsed to weekly bars first, so anything outside this band means the
# source is coarser than weekly (or has lost most of its bars).
_WEEKLY_MEDIAN_GAP_DAYS: tuple[int, int] = (5, 9)


def split_adjusted_close(
    close: pd.Series,
    split_history: pd.DataFrame | None,
) -> pd.Series:
    """Restate raw closes onto the share basis after the last known split.

    ``adjusted = close × share_basis_factor(date) / latest_share_basis_factor``.
    For PGR's 4-for-1 split on 2006-05-19 a pre-split close of 107.20 becomes
    26.80 and the first post-split close is unchanged.

    Args:
        close: Raw (unadjusted) closes indexed by bar date.
        split_history: Splits indexed by split date with a ``split_ratio``
            column (new shares per old share), or None / empty.

    Returns:
        Float series with the same index and name as ``close``.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    index = pd.DatetimeIndex(pd.to_datetime(values.index))
    factor = share_basis_factor(index, split_history).to_numpy()
    latest = latest_share_basis_factor(split_history)
    adjusted = pd.Series(values.to_numpy() * factor / latest, index=index, name=close.name)
    return adjusted


def split_adjusted_ohlcv(
    prices: pd.DataFrame,
    split_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Restate open/high/low/close and volume onto the latest share basis.

    Prices are multiplied by ``factor(date) / latest`` and volume by the
    inverse, so dollar volume is unchanged. Other columns pass through.

    A weekly bar that spans a split mixes bases: Alpha Vantage takes its open
    (and often its high or low) from the pre-split days and its close from the
    post-split days (PGR 2006-05-19: high 108.63, low 26.79, close 27.24).
    On such a bar the open is restated on whichever basis (pre- or post-split)
    lies nearer, in log terms, to the previous adjusted close, and the high and
    low on whichever lies nearer to the bar's adjusted open-close midpoint.
    The close is always post-split.
    """
    result = prices.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    result = result.sort_index()
    factor = share_basis_factor(result.index, split_history).to_numpy()
    latest = latest_share_basis_factor(split_history)
    post = factor / latest
    pre = np.concatenate([post[:1], post[:-1]])
    spans_split = pre != post

    def _numeric(column: str) -> np.ndarray:
        return pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)

    def _nearest(raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
        as_pre, as_post = raw * pre, raw * post
        with np.errstate(divide="ignore", invalid="ignore"):
            use_pre = np.abs(np.log(as_pre / reference)) < np.abs(np.log(as_post / reference))
        return np.where(spans_split & use_pre, as_pre, as_post)

    # An ``adjusted_close`` column is already adjusted by its source.
    close_adj = _numeric("close") * post if "close" in result.columns else None
    if close_adj is not None:
        result["close"] = close_adj
    if "open" in result.columns:
        reference = (
            np.concatenate([close_adj[:1], close_adj[:-1]])
            if close_adj is not None
            else np.full(len(result), np.nan)
        )
        result["open"] = _nearest(_numeric("open"), reference)
    if close_adj is not None and "open" in result.columns:
        midpoint = np.sqrt(result["open"].to_numpy(dtype=float) * close_adj)
    else:
        midpoint = close_adj if close_adj is not None else np.full(len(result), np.nan)
    for column in ("high", "low"):
        if column in result.columns:
            result[column] = _nearest(_numeric(column), midpoint)
    if "volume" in result.columns:
        result["volume"] = (
            pd.to_numeric(result["volume"], errors="coerce").astype(float)
            * latest
            / factor
        )
    return result


def weekly_bars(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Keep the last bar of each Friday-ending week, at its own date.

    Weekly input passes through unchanged (bar dates are kept, so a
    holiday-shortened week stays on its Thursday). Daily input collapses to
    one bar per week. The result is validated with
    ``assert_weekly_bar_frequency``.
    """
    result = values.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    result = result.sort_index()
    if isinstance(result, pd.Series):
        result = result.dropna()
    else:
        result = result.dropna(how="all")
    week = result.index.to_period("W-FRI")
    result = result[~week.duplicated(keep="last")]
    assert_weekly_bar_frequency(result.index)
    return result


def assert_weekly_bar_frequency(index: pd.Index, label: str = "price series") -> None:
    """Raise ``ValueError`` unless ``index`` looks like weekly bars.

    Every weekly window in the feature builders (13-week volatility, 52-week
    high) counts bars, so the bar spacing must be checked where they are
    computed. Fewer than three bars are accepted (nothing to window).
    """
    dates = pd.DatetimeIndex(index).sort_values()
    if len(dates) < 3:
        return
    gaps = dates[1:] - dates[:-1]
    median_gap = float(gaps.median() / pd.Timedelta(days=1))
    low, high = _WEEKLY_MEDIAN_GAP_DAYS
    if not low <= median_gap <= high:
        raise ValueError(
            f"{label}: expected weekly bars (median spacing {low}-{high} days), "
            f"got a median spacing of {median_gap:.1f} days"
        )


def month_end_close(close: pd.Series) -> pd.Series:
    """Last available close in each calendar month, labelled by business month-end."""
    result = close.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    return result.sort_index().resample("BME").last()


def calendar_momentum(close: pd.Series, months: int) -> pd.Series:
    """Return ``close(t) / close(t − months) − 1`` on month-end closes.

    Both closes are the last bar on or before their business month-end, so
    the window is ``months`` calendar months whatever the bar frequency. Pass
    split-adjusted closes.
    """
    monthly = month_end_close(close)
    return monthly / monthly.shift(months) - 1.0


def weekly_realized_vol(
    weekly_close: pd.Series,
    n_weeks: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Annualised volatility of the last ``n_weeks`` weekly log returns (× √52).

    Pass split-adjusted weekly closes (``weekly_bars``). ``min_periods``
    defaults to three quarters of the window.
    """
    assert_weekly_bar_frequency(weekly_close.index, "weekly_realized_vol input")
    if min_periods is None:
        min_periods = max(2, math.ceil(0.75 * n_weeks))
    log_ret = np.log(weekly_close / weekly_close.shift(1))
    return log_ret.rolling(n_weeks, min_periods=min_periods).std() * math.sqrt(
        WEEKS_PER_YEAR
    )


def trailing_52w_high(close: pd.Series, min_history_weeks: int = 25) -> pd.Series:
    """Highest close in the trailing 364 days, inclusive of the current bar.

    On weekly bars the window holds exactly 52 bars (t, t − 1 week, …,
    t − 51 weeks). Values are NaN until ``min_history_weeks`` of history
    exist (25 weeks = 26 weekly bars). Pass split-adjusted closes.
    """
    values = close.copy()
    values.index = pd.DatetimeIndex(pd.to_datetime(values.index))
    values = values.sort_index()
    high = values.rolling("364D", min_periods=1).max()
    if values.empty:
        return high
    enough = (values.index - values.index[0]) >= pd.Timedelta(weeks=min_history_weeks)
    return high.where(enough)


def return_spread_6m(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """6-month calendar return of ``left`` minus that of ``right``.

    Both inputs must already be split-adjusted closes.
    """
    return calendar_momentum(left, 6) - calendar_momentum(right, 6)


# Per-share amounts and share counts in ``pgr_edgar_monthly``. Each row is on
# the share basis in effect at its report period.
EDGAR_PER_SHARE_COLUMNS: tuple[str, ...] = (
    "book_value_per_share",
    "eps_basic",
    "eps_diluted",
    "comprehensive_eps_diluted",
    "avg_cost_per_share",
)
EDGAR_SHARE_COUNT_COLUMNS: tuple[str, ...] = (
    "shares_repurchased",
    "common_shares_outstanding",
    "avg_shares_basic",
    "avg_shares_diluted",
    "avg_diluted_equivalent_shares",
)


def restate_to_latest_share_basis(
    frame: pd.DataFrame,
    split_history: pd.DataFrame | None,
    per_share_columns: tuple[str, ...] = EDGAR_PER_SHARE_COLUMNS,
    share_count_columns: tuple[str, ...] = EDGAR_SHARE_COUNT_COLUMNS,
) -> pd.DataFrame:
    """Restate per-share columns onto the latest share basis (review F15).

    ``frame`` must be indexed by report period, before any filing lag is
    applied, so that each row's factor is the one in effect when it was
    measured. Per-share amounts are multiplied by ``factor / latest`` and
    share counts by the inverse, so dollar totals (shares × price) are
    unchanged. Columns that are absent are skipped.
    """
    result = frame.copy()
    if result.empty:
        return result
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    factor = share_basis_factor(index, split_history).to_numpy()
    latest = latest_share_basis_factor(split_history)
    for column in per_share_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce") * factor / latest
    for column in share_count_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce") * latest / factor
    return result
