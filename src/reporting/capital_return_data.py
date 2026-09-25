"""Data frames behind the recurring PGR capital-return charts.

``scripts/repurchase_timeseries_charts.py`` and
``scripts/capital_return_charts.py`` plot these frames; the tests check the
frames rather than pixels.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.database import db_client

_SHARE_UNIT_ERROR_THRESHOLD = 25.0

SPLIT_MARKERS: tuple[tuple[pd.Timestamp, str], ...] = (
    (pd.Timestamp("2002-04-23"), "3-for-1 split\n(Apr 2002)"),
    (pd.Timestamp("2006-05-01"), "4-for-1 split\n(May 2006)"),
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


def month_end_prices(prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Last bar of each calendar month, indexed by calendar month-end."""
    close = prices["close"].dropna().sort_index()
    month = close.index + pd.offsets.MonthEnd(0)
    last = ~pd.Index(month).duplicated(keep="last")
    return pd.DataFrame(
        {"price_date": close.index[last], "price": close.to_numpy()[last]},
        index=pd.DatetimeIndex(month[last], name="month_end"),
    )


def resolve_shares_outstanding(edgar: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Common shares outstanding (millions) for each EDGAR month."""
    reported = pd.to_numeric(edgar["common_shares_outstanding"], errors="coerce")
    bvps = pd.to_numeric(edgar["book_value_per_share"], errors="coerce")
    equity = pd.to_numeric(edgar["shareholders_equity"], errors="coerce")
    implied = (equity / bvps).where(bvps > 0)
    shares = reported.fillna(implied)
    source = pd.Series(np.where(reported.notna(), "reported", np.where(implied.notna(), "equity_over_bvps", None)),
                       index=edgar.index, dtype=object)
    return pd.DataFrame({"shares_outstanding": shares, "shares_source": source})


def repurchases(edgar: pd.DataFrame, prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Monthly shares repurchased (M), average cost ($) and dollars ($M)."""
    shares = pd.to_numeric(edgar["shares_repurchased"], errors="coerce")
    cost = pd.to_numeric(edgar["avg_cost_per_share"], errors="coerce")
    unit_error = (shares > _SHARE_UNIT_ERROR_THRESHOLD) & (cost > 10)
    corrected_shares = shares.where(~unit_error, shares / cost)
    dollars = (shares * cost).where(~unit_error, shares)
    source = pd.Series(np.where(cost.notna(), "reported", None), index=edgar.index, dtype=object)
    return pd.DataFrame(
        {
            "shares_repurchased": corrected_shares,
            "avg_cost_per_share": cost,
            "avg_cost_source": source,
            "repurchase_dollars": dollars,
        }
    )


def build_monthly_frame(inputs: ChartInputs) -> pd.DataFrame:
    """One row per month: price, BVPS, P/B, shares, market cap, buybacks."""
    edgar = inputs.edgar.copy()
    edgar.index = pd.DatetimeIndex(edgar.index) + pd.offsets.MonthEnd(0)
    prices = month_end_prices(inputs.prices, inputs.splits)
    frame = pd.DataFrame(index=edgar.index.union(prices.index))
    frame.index.name = "month_end"
    frame = frame.join(prices)
    frame["book_value_per_share"] = pd.to_numeric(edgar["book_value_per_share"], errors="coerce")
    frame = frame.join(resolve_shares_outstanding(edgar, inputs.splits))
    frame = frame.join(repurchases(edgar, prices, inputs.splits))
    for column in ("combined_ratio", "net_premiums_earned"):
        frame[column] = pd.to_numeric(edgar[column], errors="coerce")
    frame["price_to_book"] = (frame["price"] / frame["book_value_per_share"]).where(
        frame["book_value_per_share"] > 0
    )
    frame["market_cap"] = frame["price"] * frame["shares_outstanding"]
    frame["edgar_month"] = frame.index.isin(edgar.index)
    return frame


def split_markers(
    splits: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, str]]:
    """Split dates (and labels) that fall inside ``[start, end]``."""
    return [(date, label) for date, label in SPLIT_MARKERS if start <= date <= end]


def _shares_at(monthly: pd.DataFrame, month: pd.Timestamp) -> float | None:
    shares = monthly["shares_outstanding"].dropna()
    if shares.empty:
        return None
    months = [int(m.strftime("%Y%m")) for m in shares.index]
    target = int(month.strftime("%Y%m"))
    for i, m in enumerate(months):
        if m >= target:
            if i > 0 and abs(target - months[i - 1]) < abs(m - target):
                return float(shares.iloc[i - 1])
            return float(shares.iloc[i])
    return float(shares.iloc[-1])


def dividend_dollars(
    dividends: pd.DataFrame,
    monthly: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Dividend dollars ($M) per ex-date, attributed to a performance year.

    Q1 ex-dates are distributions of the prior year's earnings and are
    attributed to the prior calendar year.
    """
    rows = []
    for ex_date, amount in zip(dividends["ex_date"], dividends["amount"]):
        ex_date = pd.Timestamp(ex_date)
        shares = _shares_at(monthly, ex_date)
        year = ex_date.year - 1 if ex_date.month <= 3 else ex_date.year
        rows.append(
            {
                "ex_date": ex_date,
                "amount": float(amount),
                "shares_outstanding": shares,
                "dividend_dollars": None if shares is None else float(amount) * shares,
                "year": year,
            }
        )
    return pd.DataFrame(rows)


def annual_capital_return(
    monthly: pd.DataFrame,
    dividends: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Annual repurchase and dividend dollars, year-end market cap and CR."""
    edgar_months = monthly[monthly["edgar_month"]]
    repurchase = edgar_months["repurchase_dollars"].groupby(edgar_months.index.year).sum(min_count=1)
    dividend = dividends.dropna(subset=["dividend_dollars"]).groupby("year")["dividend_dollars"].sum()
    years = sorted(set(repurchase.dropna().index) | set(dividend.index))
    years = [y for y in years if y >= 2004]
    close = prices["close"].dropna().sort_index()
    last_edgar = edgar_months.index.max()
    rows = []
    for year in years:
        in_year = close[close.index.year == year]
        december = in_year[in_year.index.month == 12]
        bucket = december if not december.empty else in_year
        price = float(bucket.iloc[-1]) if not bucket.empty else None
        month = pd.Timestamp(f"{year}-12-31") if year != last_edgar.year else last_edgar
        shares = _shares_at(monthly, month)
        cr_rows = edgar_months[edgar_months.index.year == year].dropna(
            subset=["combined_ratio", "net_premiums_earned"]
        )
        cr = (
            float((cr_rows["combined_ratio"] * cr_rows["net_premiums_earned"]).sum()
                  / cr_rows["net_premiums_earned"].sum())
            if not cr_rows.empty else None
        )
        rows.append(
            {
                "year": year,
                "repurchase_dollars": float(repurchase.get(year, 0.0) or 0.0),
                "dividend_dollars": float(dividend.get(year, 0.0)),
                "year_end_price": price,
                "year_end_shares": shares,
                "market_cap": None if price is None or shares is None else price * shares,
                "cr_months": len(cr_rows),
                "combined_ratio": cr,
            }
        )
    annual = pd.DataFrame(rows).set_index("year")
    annual["total_dollars"] = annual["repurchase_dollars"] + annual["dividend_dollars"]
    annual["repurchase_pct_market_cap"] = annual["repurchase_dollars"] / annual["market_cap"] * 100
    annual["dividend_pct_market_cap"] = annual["dividend_dollars"] / annual["market_cap"] * 100
    annual["total_pct_market_cap"] = annual["total_dollars"] / annual["market_cap"] * 100
    annual["partial_label"] = None
    if 2004 in annual.index:
        annual.loc[2004, "partial_label"] = "Aug–Dec 2004"
    if 2026 in annual.index:
        annual.loc[2026, "partial_label"] = "Jan–Apr 2026 (repurchases only)"
    return annual
