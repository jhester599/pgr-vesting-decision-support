"""Tests for src/processing/valuation_multiples.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.valuation_multiples import (
    OUTPUT_COLUMNS,
    SOURCE_INTERPOLATED,
    SOURCE_QUARTERLY_RESIDUAL,
    SOURCE_REPORTED,
    build_monthly_valuation_multiples,
    fill_eps_and_bvps_gaps,
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


def _quarterly(rows: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"eps": list(rows.values())},
        index=pd.DatetimeIndex(pd.to_datetime(list(rows)), name="period_end"),
    )


def test_missing_month_eps_is_quarter_less_reported_months() -> None:
    months = pd.date_range("2015-01-31", "2015-12-31", freq="ME")
    eps = [0.30, 0.10, 0.20, 0.32, 0.99, 0.16, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
    bvps = [12.0 + 0.1 * i for i in range(12)]
    edgar = _edgar(months, eps=eps, bvps=bvps).drop(pd.Timestamp("2015-05-31"))
    quarterly = _quarterly(
        {"2015-03-31": 0.60, "2015-06-30": 0.62, "2015-09-30": 0.60, "2015-12-31": 2.00}
    )

    filled = fill_eps_and_bvps_gaps(edgar, quarterly, NO_SPLITS)

    may = filled.loc["2015-05-31"]
    assert may["eps_basic"] == pytest.approx(0.14)
    assert may["eps_basic_source"] == SOURCE_QUARTERLY_RESIDUAL
    # Public only once the Q2 10-Q is presumed filed (quarter-end + 45 days).
    assert may["eps_available_date"] == pd.Timestamp("2015-08-14")
    assert filled.loc["2015-04-30", "eps_basic_source"] == SOURCE_REPORTED


def test_q4_gap_uses_discrete_q4_less_reported_months() -> None:
    months = pd.date_range("2018-01-31", "2018-12-31", freq="ME")
    eps = [0.1] * 12
    edgar = _edgar(months, eps=eps, bvps=[10.0] * 12).drop(pd.Timestamp("2018-11-30"))
    # pgr_fundamentals_quarterly stores Q4 as a discrete quarter (FY − 9M,
    # derived in edgar_client; F09): 1.5 − 0.9 = 0.6.
    quarterly = _quarterly(
        {"2018-03-31": 0.3, "2018-06-30": 0.3, "2018-09-30": 0.3, "2018-12-31": 0.6}
    )

    filled = fill_eps_and_bvps_gaps(edgar, quarterly, NO_SPLITS)

    # November = Q4 0.6 - 0.1 - 0.1.
    assert filled.loc["2018-11-30", "eps_basic"] == pytest.approx(0.4)
    assert filled.loc["2018-11-30", "eps_available_date"] == pd.Timestamp("2019-03-01")


def test_eps_not_filled_when_two_months_of_quarter_missing() -> None:
    months = pd.date_range("2019-01-31", "2019-06-30", freq="ME")
    edgar = _edgar(months, eps=[0.1] * 6, bvps=[10.0] * 6).drop(
        [pd.Timestamp("2019-04-30"), pd.Timestamp("2019-05-31")]
    )
    quarterly = _quarterly({"2019-06-30": 0.5})

    filled = fill_eps_and_bvps_gaps(edgar, quarterly, NO_SPLITS)

    assert filled.loc["2019-04-30":"2019-05-31", "eps_basic"].isna().all()
    assert filled.loc["2019-04-30":"2019-05-31", "eps_basic_source"].isna().all()


def test_book_value_run_is_rolled_forward_with_eps_and_even_residual() -> None:
    months = pd.date_range("2005-01-31", "2005-04-30", freq="ME")
    edgar = _edgar(
        months,
        eps=[0.75, 0.64, 0.68, 0.75],
        bvps=[26.18, np.nan, np.nan, 27.19],
    )

    filled = fill_eps_and_bvps_gaps(edgar, None, NO_SPLITS)

    # residual = (27.19 - 26.18) - (0.64 + 0.68 + 0.75) = -1.06, spread over 3.
    residual = -1.06
    assert filled.loc["2005-02-28", "book_value_per_share"] == pytest.approx(
        round(26.18 + 0.64 + residual / 3, 2)
    )
    assert filled.loc["2005-03-31", "book_value_per_share"] == pytest.approx(
        round(26.18 + 0.64 + 0.68 + 2 * residual / 3, 2)
    )
    assert (filled.loc["2005-02-28":"2005-03-31", "book_value_source"] == SOURCE_INTERPOLATED).all()
    # Needs the April filing, so it is public on April's filing date.
    april_filing = pd.Timestamp(edgar.loc["2005-04-30", "filing_date"])
    assert (filled.loc["2005-02-28":"2005-03-31", "bvps_available_date"] == april_filing).all()


def test_book_value_not_interpolated_across_split() -> None:
    split_date = pd.Timestamp("2006-05-19")
    months = pd.date_range("2006-03-31", "2006-06-30", freq="ME")
    edgar = _edgar(months, eps=[0.8, 0.8, 0.2, 0.2], bvps=[32.0, np.nan, np.nan, 8.0])
    splits = pd.DataFrame({"split_ratio": [4.0]}, index=pd.DatetimeIndex([split_date]))

    filled = fill_eps_and_bvps_gaps(edgar, None, splits)

    assert filled.loc["2006-04-30":"2006-05-31", "book_value_per_share"].isna().all()


def test_availability_date_covers_fills_in_ttm_window() -> None:
    """Every TTM window that contains the filled May 2015 value waits for the
    Q2 10-Q (presumed filed 2015-08-14).

    Review F28: the old fixture started in 2015-01, so the first month with
    a TTM value was 2015-12, filed after 2015-08-14 anyway, and the check
    passed without the 12-month maximum of availability dates. Here the
    history starts in 2014-06, so June 2015 (its own 8-K filed 2015-07-15)
    has a TTM window containing May 2015 and must wait for the 10-Q.
    """
    months = pd.date_range("2014-06-30", "2016-06-30", freq="ME")
    edgar = _edgar(months, eps=[0.2] * len(months), bvps=[10.0] * len(months)).drop(
        pd.Timestamp("2015-05-31")
    )
    quarterly = _quarterly({"2015-03-31": 0.6, "2015-06-30": 0.6})
    prices = _weekly_prices("2014-06-01", "2016-06-30", lambda d: 24.0)

    out = build_monthly_valuation_multiples(prices, edgar, NO_SPLITS, quarterly).set_index(
        "month_end"
    )

    assert out.loc["2015-05-31", "eps_basic"] == pytest.approx(0.2)
    assert out["pe_ratio"].dropna().tolist() == pytest.approx([10.0] * 14)
    # June 2015 was filed 2015-07-15 but its TTM window holds May 2015.
    assert out.loc["2015-06-30", "filing_date"] == "2015-07-15"
    assert out.loc["2015-06-30", "data_available_date"] == "2015-08-14"
    # Every TTM window containing May 2015 (June 2015 to April 2016).
    window = out.loc["2015-06-30":"2016-04-30", "data_available_date"]
    assert (pd.to_datetime(window) >= pd.Timestamp("2015-08-14")).all()
    assert out.loc["2016-05-31", "data_available_date"] == out.loc["2016-05-31", "filing_date"]
