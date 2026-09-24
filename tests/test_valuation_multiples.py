"""Tests for src/processing/valuation_multiples.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.valuation_multiples import (
    OUTPUT_COLUMNS,
    build_monthly_valuation_multiples,
    share_basis_factor,
)


def _edgar(months: pd.DatetimeIndex, eps: list[float], bvps: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eps_basic": eps,
            "book_value_per_share": bvps,
            "filing_date": [(m + pd.Timedelta(days=15)).strftime("%Y-%m-%d") for m in months],
        },
        index=pd.DatetimeIndex(months, name="month_end"),
    )


def _weekly_prices(start: str, end: str, close_fn) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="W-FRI")
    return pd.DataFrame({"close": [close_fn(d) for d in dates]}, index=dates)


NO_SPLITS = pd.DataFrame(columns=["split_ratio"], index=pd.DatetimeIndex([]))


def test_share_basis_factor_includes_split_on_its_own_date() -> None:
    splits = pd.DataFrame(
        {"split_ratio": [3.0, 4.0]},
        index=pd.to_datetime(["2002-04-23", "2006-05-19"]),
    )
    dates = pd.to_datetime(["2002-04-22", "2002-04-23", "2006-05-18", "2006-05-19"])
    factors = share_basis_factor(pd.DatetimeIndex(dates), splits)
    assert factors.tolist() == [1.0, 3.0, 3.0, 12.0]


def test_pb_and_pe_without_splits() -> None:
    months = pd.date_range("2020-01-31", periods=13, freq="ME")
    edgar = _edgar(months, eps=[0.5] * 13, bvps=[20.0] * 13)
    prices = _weekly_prices("2020-01-01", "2021-01-31", lambda d: 60.0)

    out = build_monthly_valuation_multiples(prices, edgar, NO_SPLITS)

    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 13
    assert out["pb_ratio"].tolist() == pytest.approx([3.0] * 13)
    # First 11 months lack a full trailing year.
    assert out["pe_ratio"].iloc[:11].isna().all()
    assert out["eps_basic_ttm"].iloc[11] == pytest.approx(6.0)
    assert out["pe_ratio"].iloc[11:].tolist() == pytest.approx([10.0, 10.0])


def test_month_close_uses_last_bar_in_month() -> None:
    months = pd.DatetimeIndex(["2020-01-31"])
    edgar = _edgar(months, eps=[1.0], bvps=[10.0])
    prices = pd.DataFrame(
        {"close": [40.0, 50.0, 70.0]},
        index=pd.to_datetime(["2020-01-17", "2020-01-24", "2020-02-07"]),
    )

    out = build_monthly_valuation_multiples(prices, edgar, NO_SPLITS)

    assert out.loc[0, "price_date"] == "2020-01-24"
    assert out.loc[0, "close"] == pytest.approx(50.0)
    assert out.loc[0, "pb_ratio"] == pytest.approx(5.0)


def test_ttm_eps_is_restated_across_split() -> None:
    # 4-for-1 split mid-May: EPS/BVPS/price are reported pre-split before
    # May and post-split from May onward, so the economics never change.
    months = pd.date_range("2005-05-31", "2006-12-31", freq="ME")
    split_date = pd.Timestamp("2006-05-19")
    post = months >= split_date
    eps = np.where(post, 0.2, 0.8).tolist()
    bvps = np.where(post, 8.0, 32.0).tolist()
    edgar = _edgar(months, eps=eps, bvps=bvps)
    prices = _weekly_prices(
        "2005-05-01",
        "2006-12-31",
        lambda d: 24.0 if d >= split_date else 96.0,
    )
    splits = pd.DataFrame({"split_ratio": [4.0]}, index=pd.DatetimeIndex([split_date]))

    out = build_monthly_valuation_multiples(prices, edgar, splits).set_index("month_end")

    # Pre-split TTM: 12 x 0.8; post-split TTM: 12 x 0.2 regardless of mix.
    assert out.loc["2006-04-30", "eps_basic_ttm"] == pytest.approx(9.6)
    assert out.loc["2006-05-31", "eps_basic_ttm"] == pytest.approx(2.4)
    assert out.loc["2006-10-31", "eps_basic_ttm"] == pytest.approx(2.4)
    assert out["pe_ratio"].dropna().tolist() == pytest.approx([10.0] * (len(months) - 11))
    assert out["pb_ratio"].tolist() == pytest.approx([3.0] * len(months))


def test_missing_month_breaks_ttm_and_nonpositive_values_are_nan() -> None:
    months = pd.date_range("2019-01-31", periods=14, freq="ME")
    eps = [1.0] * 14
    bvps = [10.0] * 14
    bvps[13] = 0.0
    edgar = _edgar(months, eps=eps, bvps=bvps).drop(months[5])
    prices = _weekly_prices("2019-01-01", "2020-02-29", lambda d: 30.0)

    out = build_monthly_valuation_multiples(prices, edgar, NO_SPLITS)

    # The dropped month is kept as a visible gap, not silently skipped.
    assert len(out) == 14
    assert np.isnan(out.loc[5, "book_value_per_share"])
    assert np.isnan(out.loc[5, "pb_ratio"])
    # Every 12-month window up to 2020-02 includes the gap.
    assert out["eps_basic_ttm"].isna().all()
    assert np.isnan(out.loc[13, "pb_ratio"])


def test_negative_or_zero_ttm_eps_gives_nan_pe() -> None:
    months = pd.date_range("2008-01-31", periods=13, freq="ME")
    eps = [0.1] * 11 + [-1.1, -0.5]
    edgar = _edgar(months, eps=eps, bvps=[5.0] * 13)
    prices = _weekly_prices("2008-01-01", "2009-01-31", lambda d: 15.0)

    out = build_monthly_valuation_multiples(prices, edgar, NO_SPLITS)

    assert out.loc[11, "eps_basic_ttm"] == 0.0
    assert np.isnan(out.loc[11, "pe_ratio"])
    assert out.loc[12, "eps_basic_ttm"] == pytest.approx(-0.6)
    assert np.isnan(out.loc[12, "pe_ratio"])
