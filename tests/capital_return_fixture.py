"""Small PGR DB for the capital-return chart tests.

EDGAR months 2005-01 … 2007-08 around the 2006-05-19 4-for-1 split, weekly
unadjusted PGR bars 2004-12-31 … 2007-09-28, the canonical splits and the
quarterly dividends.  Special rows:

* 2005-10: ``common_shares_outstanding`` NULL, equity / BVPS valid
  (the fallback passes the sanity range).
* 2005-12: ``common_shares_outstanding`` NULL and equity mis-parsed as 30.0
  (the F18 pattern), so equity / BVPS = 0.97M shares fails the range.
* 2006-05: 2.3M shares repurchased with no average cost (the split month).
* 2005-11: no repurchase reported.
* 2007-09: a price bar but no EDGAR row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.database import db_client

SPLIT_DATE = pd.Timestamp("2006-05-19")
SPLIT_RATIO = 4.0


def _post_split(date: pd.Timestamp) -> bool:
    return date >= SPLIT_DATE


def weekly_closes() -> pd.Series:
    """Unadjusted weekly closes: 25 → ~30 on the post-split basis."""
    dates = pd.date_range("2004-12-31", "2007-09-28", freq="W-FRI")
    values = []
    for i, date in enumerate(dates):
        adjusted = round(25.0 + 0.035 * i + (0.4 if i % 3 == 0 else 0.0), 2)
        values.append(adjusted if _post_split(date) else round(adjusted * SPLIT_RATIO, 2))
    return pd.Series(values, index=dates, name="close")


def edgar_rows() -> list[dict]:
    rows = []
    months = pd.date_range("2005-01-31", "2007-08-31", freq="ME")
    for i, month in enumerate(months):
        post = _post_split(month)
        shares = (200.0 - 0.4 * i) * (SPLIT_RATIO if post else 1.0)
        bvps = round((26.0 + 0.25 * i) / (SPLIT_RATIO if post else 1.0), 4)
        repurchased = 0.5 * (SPLIT_RATIO if post else 1.0)
        cost = round((95.0 + 0.4 * i) / (SPLIT_RATIO if post else 1.0), 2)
        rows.append(
            {
                "month_end": month.strftime("%Y-%m-%d"),
                "common_shares_outstanding": round(shares, 1),
                "book_value_per_share": bvps,
                "shareholders_equity": round(shares * bvps, 1),
                "shares_repurchased": repurchased,
                "avg_cost_per_share": cost,
                "combined_ratio": 90.0 + (i % 5),
                "net_premiums_earned": 1000.0 + 5 * i,
            }
        )
    by_month = {r["month_end"][:7]: r for r in rows}
    by_month["2005-10"]["common_shares_outstanding"] = None
    by_month["2005-12"]["common_shares_outstanding"] = None
    by_month["2005-12"]["shareholders_equity"] = 30.0
    by_month["2006-05"]["shares_repurchased"] = 2.3
    by_month["2006-05"]["avg_cost_per_share"] = None
    by_month["2005-11"]["shares_repurchased"] = None
    by_month["2005-11"]["avg_cost_per_share"] = None
    return rows


DIVIDENDS: tuple[tuple[str, float], ...] = (
    ("2004-12-08", 0.03),  # before the first EDGAR month
    ("2005-03-09", 0.03),
    ("2005-06-08", 0.03),
    ("2005-09-07", 0.03),
    ("2005-12-07", 0.03),
    ("2006-03-08", 0.03),
    ("2006-06-07", 0.0075),
    ("2006-09-06", 0.00875),
    ("2006-12-06", 0.00875),
    ("2007-01-31", 2.00),
)


def build_fixture_db(path: Path, rows: list[dict] | None = None) -> Path:
    """Create the fixture DB at ``path`` and return the path."""
    conn = sqlite3.connect(path)
    db_client.initialize_schema(conn)
    for row in rows if rows is not None else edgar_rows():
        columns = ", ".join(row)
        placeholders = ", ".join(f":{key}" for key in row)
        conn.execute(f"INSERT INTO pgr_edgar_monthly ({columns}) VALUES ({placeholders})", row)
    conn.executemany(
        "INSERT INTO daily_prices"
        " (ticker, date, open, high, low, close, volume, source, proxy_fill)"
        " VALUES ('PGR', ?, ?, ?, ?, ?, 1000000, 'test', 0)",
        [
            (date.strftime("%Y-%m-%d"), close, close, close, close)
            for date, close in weekly_closes().items()
        ],
    )
    conn.executemany(
        "INSERT INTO split_history (ticker, split_date, split_ratio, numerator, denominator)"
        " VALUES ('PGR', ?, ?, ?, 1)",
        [("2002-04-23", 3.0, 3.0), ("2006-05-19", 4.0, 4.0)],
    )
    conn.executemany(
        "INSERT INTO daily_dividends (ticker, ex_date, amount, source)"
        " VALUES ('PGR', ?, ?, 'test')",
        DIVIDENDS,
    )
    conn.commit()
    conn.close()
    return path


def last_weekly_close(month: str) -> tuple[pd.Timestamp, float]:
    """Date and unadjusted close of the last weekly bar in ``month`` (YYYY-MM)."""
    closes = weekly_closes()
    in_month = closes[closes.index.strftime("%Y-%m") == month]
    return in_month.index[-1], float(in_month.iloc[-1])
