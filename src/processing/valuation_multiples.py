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
month, and a TTM EPS is only produced when all 12 trailing months are
present.  Missing EPS and book value can optionally be filled by
``fill_eps_and_bvps_gaps``; each value is tagged with its source and the
date its inputs were public, so as-of analyses never use a fill early.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS: list[str] = [
    "month_end",
    "filing_date",
    "data_available_date",
    "price_date",
    "close",
    "book_value_per_share",
    "book_value_source",
    "eps_basic",
    "eps_basic_source",
    "eps_basic_ttm",
    "pb_ratio",
    "pe_ratio",
]

SOURCE_REPORTED = "reported"
SOURCE_QUARTERLY_RESIDUAL = "quarterly_xbrl_less_reported_months"
SOURCE_INTERPOLATED = "earnings_aware_interpolation"

# 10-Q filings are due 40 days after quarter-end and the 10-K 60 days after
# year-end for large accelerated filers; quarterly XBRL rows carry no filing
# date, so fills that use them are treated as public only after these delays.
_TEN_Q_AVAILABLE_DAYS = 45
_TEN_K_AVAILABLE_DAYS = 60


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


def _quarterly_eps(quarterly: pd.DataFrame) -> pd.Series:
    """Return discrete-quarter EPS from XBRL rows whose Q4 rows are full-year.

    Q4 is derived as full-year EPS less Q1-Q3 when all three are present.
    """
    eps = pd.to_numeric(quarterly["eps"], errors="coerce").dropna()
    eps.index = pd.DatetimeIndex(eps.index) + pd.offsets.MonthEnd(0)
    discrete = eps[eps.index.month != 12].copy()
    for year_end, annual in eps[eps.index.month == 12].items():
        quarter_ends = [
            pd.Timestamp(year_end.year, month, 1) + pd.offsets.MonthEnd(0)
            for month in (3, 6, 9)
        ]
        if all(q in discrete.index for q in quarter_ends):
            discrete[year_end] = annual - discrete[quarter_ends].sum()
    return discrete.sort_index()


def fill_eps_and_bvps_gaps(
    edgar_monthly: pd.DataFrame,
    quarterly_fundamentals: pd.DataFrame | None,
    split_history: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing monthly EPS and book value per share where defensible.

    EPS: a missing month is the discrete-quarter XBRL EPS less the other two
    reported months of that quarter.  Only used when exactly one month of the
    quarter is missing and no split falls inside the quarter.

    Book value: a run of missing months between two reported months is
    rolled forward with each month's EPS, and the unexplained change
    (dividends, OCI, buybacks) is spread evenly across the run.  Only used
    when EPS is known for every month in the run and no split intervenes.

    Args:
        edgar_monthly: Monthly 8-K data indexed by period month-end with
            ``eps_basic``, ``book_value_per_share`` and ``filing_date``.
        quarterly_fundamentals: XBRL rows indexed by quarter-end with an
            ``eps`` column (Q4 rows hold full-year EPS), or None.
        split_history: Splits indexed by split date with ``split_ratio``.

    Returns:
        DataFrame indexed by every calendar month-end with ``eps_basic``,
        ``eps_basic_source``, ``eps_available_date``,
        ``book_value_per_share``, ``book_value_source`` and
        ``bvps_available_date``.  Unfilled gaps stay NaN with no source.
    """
    edgar = edgar_monthly.copy()
    edgar.index = pd.DatetimeIndex(edgar.index) + pd.offsets.MonthEnd(0)
    edgar = edgar[~edgar.index.duplicated(keep="last")].sort_index()
    months = pd.date_range(edgar.index.min(), edgar.index.max(), freq="ME")
    months.name = "month_end"

    filing = pd.to_datetime(edgar["filing_date"], errors="coerce").reindex(months)
    eps = pd.to_numeric(edgar["eps_basic"], errors="coerce").reindex(months)
    bvps = pd.to_numeric(edgar["book_value_per_share"], errors="coerce").reindex(months)
    factor = share_basis_factor(months, split_history)

    out = pd.DataFrame(index=months)
    out["eps_basic"] = eps
    out["eps_basic_source"] = np.where(eps.notna(), SOURCE_REPORTED, None)
    out["eps_available_date"] = filing.where(eps.notna())

    if quarterly_fundamentals is not None and not quarterly_fundamentals.empty:
        quarter_eps = _quarterly_eps(quarterly_fundamentals)
        for month in months[eps.isna().to_numpy()]:
            quarter_end = month + pd.offsets.QuarterEnd(0)
            if quarter_end not in quarter_eps.index:
                continue
            quarter_months = pd.date_range(
                quarter_end - pd.offsets.MonthEnd(2), quarter_end, freq="ME"
            )
            others = [m for m in quarter_months if m != month]
            if not all(m in eps.index and pd.notna(eps[m]) for m in others):
                continue
            if factor.reindex(quarter_months).nunique() != 1:
                continue
            delay = _TEN_K_AVAILABLE_DAYS if quarter_end.month == 12 else _TEN_Q_AVAILABLE_DAYS
            available = max(
                [filing[m] for m in others if pd.notna(filing[m])]
                + [quarter_end + pd.Timedelta(days=delay)]
            )
            out.loc[month, "eps_basic"] = round(
                float(quarter_eps[quarter_end] - eps[others].sum()), 2
            )
            out.loc[month, "eps_basic_source"] = SOURCE_QUARTERLY_RESIDUAL
            out.loc[month, "eps_available_date"] = available

    out["book_value_per_share"] = bvps
    out["book_value_source"] = np.where(bvps.notna(), SOURCE_REPORTED, None)
    out["bvps_available_date"] = filing.where(bvps.notna())

    known = np.flatnonzero(bvps.notna().to_numpy())
    for left, right in zip(known[:-1], known[1:]):
        if right - left < 2:
            continue
        run = months[left + 1 : right + 1]
        run_eps = out.loc[run, "eps_basic"]
        if run_eps.isna().any() or factor.iloc[left] != factor.iloc[right]:
            continue
        start, end = bvps.iloc[left], bvps.iloc[right]
        residual = (end - start) - run_eps.sum()
        steps = len(run)
        available = max(
            [filing.iloc[right]] + out.loc[run, "eps_available_date"].dropna().tolist()
        )
        for k, month in enumerate(run[:-1], start=1):
            estimate = start + run_eps.iloc[:k].sum() + residual * k / steps
            out.loc[month, "book_value_per_share"] = round(float(estimate), 2)
            out.loc[month, "book_value_source"] = SOURCE_INTERPOLATED
            out.loc[month, "bvps_available_date"] = available

    return out


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
    quarterly_fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the monthly P/B and trailing P/E series for PGR.

    Args:
        prices: Unadjusted PGR prices indexed by date with a ``close`` column
            (daily or weekly bars).
        edgar_monthly: Monthly 8-K data indexed by ``month_end`` with
            ``book_value_per_share``, ``eps_basic`` and ``filing_date``.
        split_history: PGR splits indexed by split date with ``split_ratio``.
        quarterly_fundamentals: Optional XBRL quarterly EPS used to fill
            missing months (see ``fill_eps_and_bvps_gaps``).

    Returns:
        DataFrame with ``OUTPUT_COLUMNS``, one row per calendar month.
        ``data_available_date`` is the first date every input behind that
        row's P/B and TTM EPS had been published.
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
    filled = fill_eps_and_bvps_gaps(edgar, quarterly_fundamentals, split_history)
    for col in (
        "book_value_per_share",
        "book_value_source",
        "eps_basic",
        "eps_basic_source",
    ):
        out[col] = filled[col]
    # Latest publication date across this month's book value and the 12
    # EPS months in its TTM window.
    eps_avail = pd.to_datetime(filled["eps_available_date"]).astype("datetime64[ns]")
    eps_avail_ns = eps_avail.astype("int64").where(eps_avail.notna())
    ttm_avail = pd.to_datetime(
        eps_avail_ns.rolling(12, min_periods=12).max(), unit="ns"
    )
    bvps_avail = pd.to_datetime(filled["bvps_available_date"]).astype("datetime64[ns]")
    out["data_available_date"] = pd.concat([bvps_avail, ttm_avail], axis=1).max(axis=1)

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
        out["eps_basic"], split_history
    ).reindex(months)
    out["eps_basic_ttm"] = (ttm_latest_basis * latest_factor / month_factor).round(6)

    bvps = out["book_value_per_share"].where(out["book_value_per_share"] > 0)
    ttm = out["eps_basic_ttm"].where(out["eps_basic_ttm"] > 0)
    out["pb_ratio"] = out["close"] / bvps
    out["pe_ratio"] = out["close"] / ttm

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.reset_index()
    out["month_end"] = out["month_end"].dt.strftime("%Y-%m-%d")
    for col in ("price_date", "data_available_date"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    return out[OUTPUT_COLUMNS]
