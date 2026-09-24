"""
Monthly PGR valuation multiples: price-to-book and trailing price-to-earnings.

Inputs are stored exactly as reported: unadjusted closing prices and the
per-share figures from Progressive's monthly 8-K supplements.  A period's
book value per share and a month's EPS are therefore on the share basis in
effect at that month-end, and P/B can be computed row by row.

Trailing-12-month EPS is different: the 12 monthly EPS figures being summed
can straddle a stock split (PGR's 4-for-1 on 2006-05-19), so each month is
first restated onto the share basis of the month being valued.

Output rows cover every calendar month from the first to the last EDGAR
month.  Months with no filing are kept as all-NaN rows so gaps stay visible,
and a TTM EPS is only produced when all 12 trailing months are present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS: list[str] = [
    "month_end",
    "filing_date",
    "price_date",
    "close",
    "book_value_per_share",
    "eps_basic",
    "eps_basic_ttm",
    "pb_ratio",
    "pe_ratio",
]


def share_basis_factor(
    dates: pd.DatetimeIndex,
    split_history: pd.DataFrame,
) -> pd.Series:
    """Return the cumulative split multiplier in effect on each date.

    A split is in effect on its own split date (prices on the split date
    already trade on the post-split basis).  Per-share amounts on two dates
    are made comparable by multiplying by ``factor(from) / factor(to)``.

    Args:
        dates: Dates to evaluate.
        split_history: DataFrame indexed by split date with a
            ``split_ratio`` column.

    Returns:
        Series of float multipliers indexed by ``dates``.
    """
    factors = pd.Series(1.0, index=dates, dtype=float)
    if split_history is None or split_history.empty:
        return factors
    for split_date, ratio in split_history["split_ratio"].items():
        factors[dates >= pd.Timestamp(split_date)] *= float(ratio)
    return factors


def trailing_eps_latest_basis(
    eps_monthly: pd.Series,
    split_history: pd.DataFrame,
) -> pd.Series:
    """Return trailing-12-month EPS restated onto the latest share basis.

    Each month's EPS is converted to the share basis in effect after the
    last split, then summed over 12 consecutive calendar months.  A TTM value
    is only produced when all 12 months are present, so a missing filing
    never lets the window silently span 13 months.

    Args:
        eps_monthly: Single-month EPS indexed by period month-end, each value
            on the share basis in effect at that month-end.
        split_history: Splits indexed by split date with ``split_ratio``.

    Returns:
        Series indexed by every calendar month-end from the first to the last
        period, rounded to 6 decimals to clear float residue (e.g. 4e-16
        instead of 0.0) that would otherwise explode a P/E ratio.
    """
    eps = pd.to_numeric(eps_monthly, errors="coerce")
    eps.index = pd.DatetimeIndex(eps.index) + pd.offsets.MonthEnd(0)
    eps = eps[~eps.index.duplicated(keep="last")].sort_index()
    months = pd.date_range(eps.index.min(), eps.index.max(), freq="ME")
    months.name = eps.index.name
    eps = eps.reindex(months)
    factor = share_basis_factor(months, split_history)
    latest_factor = latest_share_basis_factor(split_history)
    eps_latest = eps * factor / latest_factor
    return eps_latest.rolling(12, min_periods=12).sum().round(6)


def latest_share_basis_factor(split_history: pd.DataFrame) -> float:
    """Return the cumulative multiplier of every split in ``split_history``."""
    if split_history is None or split_history.empty:
        return 1.0
    return float(split_history["split_ratio"].astype(float).prod())


def _month_end_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Return the last available close in each calendar month.

    Args:
        prices: DataFrame indexed by date with a ``close`` column.

    Returns:
        DataFrame indexed by calendar month-end with ``price_date`` and
        ``close`` columns.
    """
    closes = prices["close"].dropna().sort_index()
    frame = pd.DataFrame({"price_date": closes.index, "close": closes.values})
    frame["month_end"] = closes.index + pd.offsets.MonthEnd(0)
    return frame.groupby("month_end").last()


def build_monthly_valuation_multiples(
    prices: pd.DataFrame,
    edgar_monthly: pd.DataFrame,
    split_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build the monthly P/B and trailing P/E series for PGR.

    Args:
        prices: Unadjusted PGR prices indexed by date with a ``close`` column
            (daily or weekly bars).
        edgar_monthly: Monthly 8-K data indexed by ``month_end`` with
            ``book_value_per_share``, ``eps_basic`` and ``filing_date``.
        split_history: PGR splits indexed by split date with ``split_ratio``.

    Returns:
        DataFrame with ``OUTPUT_COLUMNS``, one row per calendar month.
        ``pb_ratio`` is NaN when book value is missing or non-positive;
        ``pe_ratio`` is NaN when fewer than 12 trailing months of EPS exist
        or TTM EPS is non-positive.
    """
    edgar = edgar_monthly.copy()
    edgar.index = pd.DatetimeIndex(edgar.index) + pd.offsets.MonthEnd(0)
    edgar = edgar[~edgar.index.duplicated(keep="last")].sort_index()

    months = pd.date_range(edgar.index.min(), edgar.index.max(), freq="ME")
    months.name = "month_end"
    out = pd.DataFrame(index=months)
    out["filing_date"] = edgar["filing_date"].reindex(months)
    out["book_value_per_share"] = pd.to_numeric(
        edgar["book_value_per_share"], errors="coerce"
    ).reindex(months)
    out["eps_basic"] = pd.to_numeric(
        edgar["eps_basic"], errors="coerce"
    ).reindex(months)

    month_close = _month_end_close(prices).reindex(months)
    out["price_date"] = month_close["price_date"]

    month_factor = share_basis_factor(months, split_history)
    # Restate each close onto its month-end share basis (a no-op unless a
    # split falls between the last bar of the month and the month-end).
    price_dates = pd.DatetimeIndex(month_close["price_date"])
    price_factor = share_basis_factor(price_dates, split_history).to_numpy()
    out["close"] = (
        month_close["close"].to_numpy() * price_factor / month_factor.to_numpy()
    )

    # TTM EPS on the basis of the valuation month.
    latest_factor = latest_share_basis_factor(split_history)
    ttm_latest_basis = trailing_eps_latest_basis(
        edgar["eps_basic"], split_history
    ).reindex(months)
    out["eps_basic_ttm"] = (ttm_latest_basis * latest_factor / month_factor).round(6)

    bvps = out["book_value_per_share"].where(out["book_value_per_share"] > 0)
    ttm = out["eps_basic_ttm"].where(out["eps_basic_ttm"] > 0)
    out["pb_ratio"] = out["close"] / bvps
    out["pe_ratio"] = out["close"] / ttm

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.reset_index()
    out["month_end"] = out["month_end"].dt.strftime("%Y-%m-%d")
    out["price_date"] = pd.to_datetime(out["price_date"]).dt.strftime("%Y-%m-%d")
    return out[OUTPUT_COLUMNS]
