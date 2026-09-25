"""Data frames behind the recurring PGR capital-return charts.

``scripts/repurchase_timeseries_charts.py`` and
``scripts/capital_return_charts.py`` plot these frames; the tests check the
frames rather than pixels (review 2026-09-25, step 4b).

Data basis
----------
* EDGAR monthly 8-K table ``pgr_edgar_monthly`` as repaired in step 3b.
* PGR prices are Alpha Vantage weekly bars (``daily_prices``). A month's
  price is the **last weekly close** in that calendar month; nothing here is
  a true month-end close.
* As-reported values are kept on the share basis in effect at the time.
  ``*_split_adjusted`` / ``*_latest_basis`` columns restate them onto the
  basis after the last split in ``split_history``, using the step-4 helpers
  (``split_adjusted_close``, ``share_basis_factor``).
* Market cap = price × common shares outstanding of the same month. It is
  computed on the latest basis, where it does not depend on which side of a
  split a bar falls.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.database import db_client
from src.processing.pgr_edgar_validation import preferred_stock
from src.processing.price_adjustment import split_adjusted_close, weekly_bars
from src.processing.valuation_multiples import (
    latest_share_basis_factor,
    share_basis_factor,
)

# Common shares outstanding on the latest share basis, millions. PGR ranged
# from ~870M (2004, restated for the 2006 split) to ~580M (2026). The
# equity / BVPS fallback is used only inside this range; the F18 rows
# (equity mis-parsed as a ratio) give < 2M shares and are rejected.
SHARES_SANITY_RANGE_LATEST_BASIS: tuple[float, float] = (300.0, 2000.0)

# A month's repurchase above this fraction of shares outstanding is treated
# as a unit error (e.g. dollars stored as shares) and stops the charts. The
# largest real month is the Oct-2004 Dutch auction: 16.9M of 200M (8.4 %).
MAX_MONTHLY_REPURCHASE_FRACTION: float = 0.15

AVG_COST_REPORTED = "reported"
AVG_COST_ESTIMATED = "estimated_mean_weekly_close"
AVG_COST_NONE_REPURCHASED = "none_repurchased"

SHARES_REPORTED = "reported"
SHARES_EQUITY_OVER_BVPS = "equity_over_bvps"

# Columns that must be populated in a split month covered by EDGAR.
SPLIT_MONTH_REQUIRED_COLUMNS: tuple[str, ...] = (
    "price",
    "price_split_adjusted",
    "book_value_per_share",
    "book_value_per_share_split_adjusted",
    "price_to_book",
    "price_to_book_split_adjusted",
    "shares_outstanding",
    "market_cap",
    "repurchase_dollars",
)


@dataclass
class ChartInputs:
    """Raw tables read from the DB."""

    edgar: pd.DataFrame
    prices: pd.DataFrame
    splits: pd.DataFrame
    dividends: pd.DataFrame


def load_chart_inputs(conn: sqlite3.Connection) -> ChartInputs:
    """Read the EDGAR monthly table, PGR weekly bars, splits and dividends."""
    edgar = pd.read_sql_query(
        "SELECT * FROM pgr_edgar_monthly ORDER BY month_end", conn, parse_dates=["month_end"]
    ).set_index("month_end")
    edgar.index = pd.DatetimeIndex(edgar.index) + pd.offsets.MonthEnd(0)
    prices = pd.read_sql_query(
        "SELECT date, close FROM daily_prices WHERE ticker = 'PGR' ORDER BY date",
        conn,
        parse_dates=["date"],
    ).set_index("date")
    dividends = pd.read_sql_query(
        "SELECT ex_date, amount FROM daily_dividends WHERE ticker = 'PGR' ORDER BY ex_date",
        conn,
        parse_dates=["ex_date"],
    )
    return ChartInputs(edgar, prices, db_client.get_splits(conn, "PGR"), dividends)


def _basis_ratio(dates: pd.DatetimeIndex, splits: pd.DataFrame) -> np.ndarray:
    """``factor(date) / latest factor``: multiply a per-share amount by this."""
    factor = share_basis_factor(pd.DatetimeIndex(dates), splits).to_numpy(dtype=float)
    return factor / latest_share_basis_factor(splits)


def month_end_prices(prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Last weekly close of each calendar month, indexed by month-end.

    Columns: ``price_date`` (the bar's date), ``price`` (unadjusted close) and
    ``price_split_adjusted`` (latest share basis).
    """
    close = weekly_bars(pd.to_numeric(prices["close"], errors="coerce"))
    adjusted = split_adjusted_close(close, splits)
    month = close.index + pd.offsets.MonthEnd(0)
    last = ~pd.Index(month).duplicated(keep="last")
    return pd.DataFrame(
        {
            "price_date": close.index[last],
            "price": close.to_numpy()[last],
            "price_split_adjusted": adjusted.to_numpy()[last],
        },
        index=pd.DatetimeIndex(month[last], name="month_end"),
    )


def resolve_shares_outstanding(edgar: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Common shares outstanding (millions) for each EDGAR month.

    ``common_shares_outstanding`` as reported. Only where it is missing,
    (equity − preferred stock) / BVPS, and only if that lands inside
    ``SHARES_SANITY_RANGE_LATEST_BASIS`` once restated to the latest basis.
    """
    reported = pd.to_numeric(edgar["common_shares_outstanding"], errors="coerce")
    bvps = pd.to_numeric(edgar["book_value_per_share"], errors="coerce")
    equity = pd.to_numeric(edgar["shareholders_equity"], errors="coerce")
    implied = ((equity - preferred_stock(edgar.index)) / bvps).where(bvps > 0)
    to_latest = 1.0 / _basis_ratio(edgar.index, splits)
    low, high = SHARES_SANITY_RANGE_LATEST_BASIS
    implied = implied.where((implied * to_latest).between(low, high))
    shares = reported.fillna(implied)
    source = pd.Series(
        np.where(
            reported.notna(),
            SHARES_REPORTED,
            np.where(implied.notna(), SHARES_EQUITY_OVER_BVPS, None),
        ),
        index=edgar.index,
        dtype=object,
    )
    return pd.DataFrame(
        {
            "shares_outstanding": shares,
            "shares_outstanding_latest_basis": shares * to_latest,
            "shares_source": source,
        }
    )


def repurchases(
    edgar: pd.DataFrame,
    prices: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Monthly shares repurchased (M), average cost ($) and dollars ($M).

    Values are as reported, on the share basis of the report month. (The
    2006-05 split month comes from the Q2 2006 10-Q on the post-split basis;
    see ``scripts/repair_split_month_buybacks.py``.) Where shares were
    repurchased but no average cost is recorded, the cost is the mean of that
    month's weekly closes restated onto the month-end basis
    (``avg_cost_source`` = ``estimated_mean_weekly_close``).
    """
    shares = pd.to_numeric(edgar["shares_repurchased"], errors="coerce")
    cost = pd.to_numeric(edgar["avg_cost_per_share"], errors="coerce")
    source = pd.Series(np.where(cost.notna(), AVG_COST_REPORTED, None), index=edgar.index,
                       dtype=object)

    close = weekly_bars(pd.to_numeric(prices["close"], errors="coerce"))
    adjusted = split_adjusted_close(close, splits)
    for month in edgar.index[(shares > 0) & cost.isna()]:
        in_month = adjusted[(adjusted.index + pd.offsets.MonthEnd(0)) == month]
        if in_month.empty:
            continue
        cost[month] = float(in_month.mean()) / _basis_ratio(pd.DatetimeIndex([month]), splits)[0]
        source[month] = AVG_COST_ESTIMATED

    none_repurchased = (shares == 0) & cost.isna()
    source[none_repurchased] = AVG_COST_NONE_REPURCHASED
    dollars = (shares * cost).where(~none_repurchased, 0.0)
    return pd.DataFrame(
        {
            "shares_repurchased": shares,
            "avg_cost_per_share": cost,
            "avg_cost_source": source,
            "repurchase_dollars": dollars,
        }
    )


def build_monthly_frame(inputs: ChartInputs) -> pd.DataFrame:
    """One row per month: price, BVPS, P/B, shares, market cap, buybacks.

    Rows cover every EDGAR month and every month with a price bar;
    ``edgar_month`` marks the former.
    """
    edgar = inputs.edgar
    splits = inputs.splits
    prices = month_end_prices(inputs.prices, splits)
    frame = pd.DataFrame(index=edgar.index.union(prices.index))
    frame.index.name = "month_end"
    frame = frame.join(prices)

    bvps = pd.to_numeric(edgar["book_value_per_share"], errors="coerce").where(lambda s: s > 0)
    frame["book_value_per_share"] = bvps
    frame["book_value_per_share_split_adjusted"] = bvps * _basis_ratio(edgar.index, splits)
    frame = frame.join(resolve_shares_outstanding(edgar, splits))
    frame = frame.join(repurchases(edgar, inputs.prices, splits))
    for column in ("combined_ratio", "net_premiums_earned"):
        frame[column] = pd.to_numeric(edgar[column], errors="coerce")

    # As-reported P/B divides the last weekly close by month-end BVPS; the two
    # share the same basis unless a split falls between them, which the
    # split-adjusted P/B handles.
    frame["price_to_book"] = frame["price"] / frame["book_value_per_share"]
    frame["price_to_book_split_adjusted"] = (
        frame["price_split_adjusted"] / frame["book_value_per_share_split_adjusted"]
    )
    frame["market_cap"] = frame["price_split_adjusted"] * frame["shares_outstanding_latest_basis"]
    frame["edgar_month"] = frame.index.isin(edgar.index)

    too_large = frame["shares_repurchased"] > (
        MAX_MONTHLY_REPURCHASE_FRACTION * frame["shares_outstanding"]
    )
    if too_large.any():
        months = ", ".join(m.strftime("%Y-%m") for m in frame.index[too_large])
        raise ValueError(
            f"shares_repurchased exceeds {MAX_MONTHLY_REPURCHASE_FRACTION:.0%} of shares "
            f"outstanding in {months}; check the EDGAR units before charting"
        )
    verify_monthly_frame(frame, splits)
    return frame


def verify_monthly_frame(frame: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the chart frame is internally inconsistent.

    * Market cap = price × shares for every month where the price bar and the
      month-end share count are on the same basis (every month but one that
      contains a split after its last weekly bar).
    * Shares outstanding lie inside ``SHARES_SANITY_RANGE_LATEST_BASIS``.
    * A split month covered by EDGAR has every ``SPLIT_MONTH_REQUIRED_COLUMNS``.
    """
    has_cap = frame["market_cap"].notna()
    same_basis = pd.Series(False, index=frame.index)
    dates = pd.DatetimeIndex(frame.loc[has_cap, "price_date"])
    same_basis[has_cap] = np.isclose(
        _basis_ratio(dates, splits), _basis_ratio(frame.index[has_cap], splits)
    )
    reported = frame["price"] * frame["shares_outstanding"]
    mismatch = same_basis & ~np.isclose(frame["market_cap"], reported, rtol=1e-9, atol=0.0)
    if mismatch.any():
        raise ValueError(f"market cap != price × shares in {list(frame.index[mismatch])}")

    low, high = SHARES_SANITY_RANGE_LATEST_BASIS
    shares = frame["shares_outstanding_latest_basis"].dropna()
    outside = shares[(shares < low) | (shares > high)]
    if not outside.empty:
        raise ValueError(f"shares outstanding outside {low}–{high}M: {outside.to_dict()}")

    edgar_months = frame.index[frame["edgar_month"]]
    for split_date in ([] if splits is None or splits.empty else splits.index):
        month = pd.Timestamp(split_date) + pd.offsets.MonthEnd(0)
        if month not in edgar_months:
            continue
        missing = [c for c in SPLIT_MONTH_REQUIRED_COLUMNS if pd.isna(frame.at[month, c])]
        if missing:
            raise ValueError(f"split month {month:%Y-%m} is missing {missing}")


def split_markers(
    splits: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, str]]:
    """Split dates from ``split_history`` inside ``[start, end]``, with labels."""
    if splits is None or splits.empty:
        return []
    markers = []
    for split_date, row in splits.iterrows():
        date = pd.Timestamp(split_date)
        if not start <= date <= end:
            continue
        numerator = row.get("numerator")
        denominator = row.get("denominator")
        if pd.notna(numerator) and pd.notna(denominator):
            ratio = f"{numerator:g}-for-{denominator:g}"
        else:
            ratio = f"{row['split_ratio']:g}-for-1"
        markers.append((date, f"{ratio} split\n({date.day} {date:%b %Y})"))
    return markers


def _shares_on_basis_of(
    monthly: pd.DataFrame,
    month: pd.Timestamp,
    basis_date: pd.Timestamp,
    splits: pd.DataFrame,
) -> float | None:
    """Shares of ``month`` (or the latest earlier EDGAR month with a count),
    restated onto the share basis in effect on ``basis_date``."""
    shares = monthly.loc[monthly["edgar_month"], "shares_outstanding_latest_basis"].dropna()
    shares = shares[shares.index <= month]
    if shares.empty:
        return None
    return float(shares.iloc[-1]) * _basis_ratio(pd.DatetimeIndex([basis_date]), splits)[0]


def dividend_dollars(
    dividends: pd.DataFrame,
    monthly: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Dividend dollars ($M) per ex-date, attributed to a performance year.

    Amounts are as paid (unadjusted); shares are those of the ex-date's month,
    restated onto the ex-date's basis. Ex-dates before the first EDGAR month
    have no share count and are dropped. Q1 ex-dates are distributions of the
    prior year's earnings and are attributed to the prior calendar year.
    """
    columns = ["ex_date", "amount", "shares_outstanding", "dividend_dollars", "year"]
    first_month = monthly.index[monthly["edgar_month"]].min()
    rows = []
    for ex_date, amount in zip(dividends["ex_date"], dividends["amount"]):
        ex_date = pd.Timestamp(ex_date)
        month = ex_date + pd.offsets.MonthEnd(0)
        if month < first_month:
            continue
        shares = _shares_on_basis_of(monthly, month, ex_date, splits)
        if shares is None:
            continue
        rows.append(
            {
                "ex_date": ex_date,
                "amount": float(amount),
                "shares_outstanding": shares,
                "dividend_dollars": float(amount) * shares,
                "year": ex_date.year - 1 if ex_date.month <= 3 else ex_date.year,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def annual_capital_return(
    monthly: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Annual repurchase and dividend dollars, year-end market cap and CR.

    * Repurchases: sum of the monthly ``repurchase_dollars`` of the year.
    * Year-end market cap: last weekly close of the year's last EDGAR month
      (December, or the latest month of a partial year) × that month's shares
      (or, if missing, the latest earlier month's, on the same basis).
    * Combined ratio: NPE-weighted mean of the monthly ratios.
    * ``partial_label``: set for years with fewer than 12 EDGAR months.
    """
    edgar = monthly[monthly["edgar_month"]]
    rows = []
    for year, months in edgar.groupby(edgar.index.year):
        last_month = months.index.max()
        price = months.at[last_month, "price"]
        price_date = months.at[last_month, "price_date"]
        shares = (
            _shares_on_basis_of(monthly, last_month, pd.Timestamp(price_date), splits)
            if pd.notna(price_date) else None
        )
        cr_rows = months.dropna(subset=["combined_ratio", "net_premiums_earned"])
        cr = (
            float((cr_rows["combined_ratio"] * cr_rows["net_premiums_earned"]).sum()
                  / cr_rows["net_premiums_earned"].sum())
            if not cr_rows.empty else np.nan
        )
        label = None
        if len(months) < 12:
            label = f"{months.index.min():%b}–{last_month:%b} {year}"
        rows.append(
            {
                "year": int(year),
                "months_reported": len(months),
                "partial_label": label,
                "repurchase_dollars": float(months["repurchase_dollars"].sum()),
                "repurchase_months_missing": int(months["repurchase_dollars"].isna().sum()),
                "year_end_month": last_month,
                "year_end_price_date": price_date,
                "year_end_price": price,
                "year_end_shares": shares,
                "market_cap": np.nan if shares is None or pd.isna(price) else price * shares,
                "cr_months": len(cr_rows),
                "combined_ratio": cr,
            }
        )
    annual = pd.DataFrame(rows).set_index("year")
    by_year = dividends.groupby("year")["dividend_dollars"].sum() if not dividends.empty else {}
    annual["dividend_dollars"] = [float(by_year.get(y, 0.0)) for y in annual.index]
    annual["total_dollars"] = annual["repurchase_dollars"] + annual["dividend_dollars"]
    for part in ("repurchase", "dividend", "total"):
        annual[f"{part}_pct_market_cap"] = annual[f"{part}_dollars"] / annual["market_cap"] * 100
    return annual
