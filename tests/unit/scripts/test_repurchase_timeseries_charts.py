"""Plotted data of ``scripts/repurchase_timeseries_charts.py`` (step 4b).

Checks the frames the charts are drawn from, not pixels, on the fixture DB in
``tests/capital_return_fixture.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import repurchase_timeseries_charts as charts
from src.database import db_client
from tests.capital_return_fixture import (
    SPLIT_RATIO,
    build_fixture_db,
    edgar_rows,
    last_weekly_close,
    weekly_closes,
)

SPLIT_MONTH = pd.Timestamp("2006-05-31")


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return build_fixture_db(tmp_path / "fixture.db")


@pytest.fixture()
def frames(db_path: Path) -> dict:
    conn = db_client.get_connection(str(db_path), read_only=True)
    try:
        return charts.build_chart_frames(conn)
    finally:
        conn.close()


def test_market_cap_is_price_times_shares_every_month(frames) -> None:
    monthly = frames["monthly"]
    checked = 0
    for row in edgar_rows():
        month = row["month_end"][:7]
        _, close = last_weekly_close(month)
        shares = row["common_shares_outstanding"]
        if shares is None and month == "2005-10":
            shares = row["shareholders_equity"] / row["book_value_per_share"]
        got = monthly.loc[pd.Timestamp(row["month_end"]), "market_cap"]
        if shares is None:
            # 2005-12: equity mis-parsed, equity / BVPS = 0.97M fails the range.
            assert math.isnan(got), month
            continue
        assert got == pytest.approx(close * shares, rel=1e-9), month
        checked += 1
    assert checked == len(edgar_rows()) - 1
    # A price month without an EDGAR row has no share count, so no market cap.
    assert math.isnan(monthly.loc[pd.Timestamp("2007-09-30"), "market_cap"])


def test_market_cap_is_split_invariant(frames) -> None:
    monthly = frames["monthly"].dropna(subset=["market_cap"])
    adjusted = monthly["price_split_adjusted"] * monthly["shares_outstanding_latest_basis"]
    np.testing.assert_allclose(adjusted, monthly["market_cap"], rtol=1e-9)


def test_shares_fallback_only_within_sanity_range(frames) -> None:
    monthly = frames["monthly"]
    october = monthly.loc[pd.Timestamp("2005-10-31")]
    assert october["shares_source"] == "equity_over_bvps"
    december = monthly.loc[pd.Timestamp("2005-12-31")]
    assert math.isnan(december["shares_outstanding"])
    assert december["shares_source"] is None or pd.isna(december["shares_source"])


def test_no_nan_in_split_month(frames) -> None:
    row = frames["monthly"].loc[SPLIT_MONTH]
    for column in (
        "price",
        "price_split_adjusted",
        "book_value_per_share",
        "book_value_per_share_split_adjusted",
        "price_to_book",
        "price_to_book_split_adjusted",
        "shares_outstanding",
        "market_cap",
        "shares_repurchased",
        "avg_cost_per_share",
        "repurchase_dollars",
    ):
        assert not pd.isna(row[column]), column


def test_split_month_average_cost_is_derived_from_weekly_closes(frames) -> None:
    closes = weekly_closes()
    may = closes[closes.index.strftime("%Y-%m") == "2006-05"]
    # Restate the pre-split bars onto the month-end (post-split) basis.
    restated = [c / SPLIT_RATIO if d < pd.Timestamp("2006-05-19") else c for d, c in may.items()]
    row = frames["monthly"].loc[SPLIT_MONTH]
    assert row["avg_cost_per_share"] == pytest.approx(np.mean(restated))
    assert row["avg_cost_source"] == "estimated_mean_weekly_close"
    assert row["repurchase_dollars"] == pytest.approx(2.3 * np.mean(restated))


def test_split_marker_is_the_canonical_split_date(frames) -> None:
    dates = [date for date, _ in frames["split_markers"]]
    assert dates == [pd.Timestamp("2006-05-19")]


def test_monthly_buyback_totals(frames) -> None:
    monthly = frames["monthly"]
    # Every month that reports a repurchase has a dollar bar, the split month
    # included.
    reported = {
        pd.Timestamp(r["month_end"]) for r in edgar_rows() if r["shares_repurchased"] is not None
    }
    assert set(monthly["repurchase_dollars"].dropna().index) == reported
    for row in edgar_rows():
        month = pd.Timestamp(row["month_end"])
        got = monthly.loc[month, "repurchase_dollars"]
        if row["shares_repurchased"] is None:
            assert math.isnan(got), row["month_end"]
        elif row["avg_cost_per_share"] is not None:
            expected = row["shares_repurchased"] * row["avg_cost_per_share"]
            assert got == pytest.approx(expected), row["month_end"]


def test_split_adjusted_series_are_continuous_across_the_split(frames) -> None:
    monthly = frames["monthly"]
    april, may = pd.Timestamp("2006-04-30"), SPLIT_MONTH
    for column in ("price_split_adjusted", "book_value_per_share_split_adjusted"):
        change = abs(math.log(monthly.loc[may, column] / monthly.loc[april, column]))
        assert change < 0.1, column
    # As-reported P/B uses one share basis per month, so it equals the
    # split-adjusted P/B.
    pb = monthly.dropna(subset=["price_to_book"])
    np.testing.assert_allclose(pb["price_to_book"], pb["price_to_book_split_adjusted"])


def test_implausible_repurchase_raises(tmp_path: Path) -> None:
    rows = edgar_rows()
    rows[3]["shares_repurchased"] = 60.0  # 30 % of shares outstanding
    path = build_fixture_db(tmp_path / "bad.db", rows)
    conn = db_client.get_connection(str(path), read_only=True)
    try:
        with pytest.raises(ValueError, match="shares_repurchased"):
            charts.build_chart_frames(conn)
    finally:
        conn.close()


def test_main_writes_every_named_chart(db_path: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "charts"
    assert charts.main(["--db-path", str(db_path), "--out-dir", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(charts.CHART_FILES)
