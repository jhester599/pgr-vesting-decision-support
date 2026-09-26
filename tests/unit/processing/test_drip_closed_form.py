"""Closed-form checks of the unadjusted-price DRIP total return (review F28).

* Flat price P with a $d dividend each quarter: the DRIP return over a window
  holding n ex-dates is prod(1 + d/P) - 1 = (1 + d/P)**n - 1.
* A 4:1 split with the matching 75 % price drop is not a return: ~0.

Both are checked on the low-level position series and through the DB-backed
monthly target pipeline on weekly bars.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.database import db_client
from src.processing.multi_total_return import build_etf_monthly_returns
from src.processing.total_return import (
    build_monthly_returns,
    build_position_series,
    compute_total_return,
)

_P = 50.0
_D = 1.0


def _weekly(start: str, end: str, price: float = _P) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="W-FRI")
    return pd.DataFrame({"close": price}, index=dates)


def _quarterly_divs(start: str, end: str) -> pd.DataFrame:
    # Mid-quarter ex-dates that are not bar dates (reinvested at prior close).
    dates = pd.date_range(start=start, end=end, freq="QS-FEB") + pd.Timedelta(days=14)
    return pd.DataFrame({"dividend": _D}, index=pd.DatetimeIndex(dates, name="ex_date"))


def _no_splits() -> pd.DataFrame:
    return pd.DataFrame(columns=["split_ratio"], index=pd.DatetimeIndex([], name="split_date"))


class TestFlatPriceQuarterlyDividend:
    def test_position_series_matches_product_formula(self) -> None:
        prices = _weekly("2020-01-03", "2022-12-30")
        divs = _quarterly_divs("2020-01-01", "2022-12-31")
        pos = build_position_series(prices, divs, _no_splits())
        start, end = pd.Timestamp("2020-12-31"), pd.Timestamp("2022-12-30")
        n_divs = int(((divs.index > start) & (divs.index <= end)).sum())
        assert n_divs == 8
        expected = (1.0 + _D / _P) ** n_divs - 1.0
        assert compute_total_return(pos, start, end) == pytest.approx(expected, rel=1e-12)

    def test_twelve_month_target_matches_product_formula(self) -> None:
        prices = _weekly("2020-01-03", "2022-12-30")
        divs = _quarterly_divs("2020-01-01", "2022-12-31")
        returns = build_monthly_returns(prices, divs, _no_splits(), forward_months=12)
        # Every complete 12M window on this calendar holds exactly 4 ex-dates.
        expected = (1.0 + _D / _P) ** 4 - 1.0
        valid = returns.dropna()
        assert len(valid) >= 20
        assert valid.to_numpy() == pytest.approx(expected, rel=1e-12)

    def test_db_pipeline_matches_product_formula(self, tmp_path) -> None:
        conn = db_client.get_connection(str(tmp_path / "t.db"))
        db_client.initialize_schema(conn)
        prices = _weekly("2020-01-03", "2022-12-30")
        divs = _quarterly_divs("2020-01-01", "2022-12-31")
        db_client.upsert_prices(conn, [
            {"ticker": "VTI", "date": d.strftime("%Y-%m-%d"), "close": _P}
            for d in prices.index
        ])
        db_client.upsert_dividends(conn, [
            {"ticker": "VTI", "ex_date": d.strftime("%Y-%m-%d"), "amount": _D}
            for d in divs.index
        ])
        returns = build_etf_monthly_returns(conn, "VTI", 6).dropna()
        expected = (1.0 + _D / _P) ** 2 - 1.0
        assert returns.to_numpy() == pytest.approx(expected, rel=1e-12)
        conn.close()


class TestFourForOneSplit:
    @staticmethod
    def _split_prices() -> pd.DataFrame:
        pre = _weekly("2020-01-03", "2021-06-11", price=100.0)
        post = _weekly("2021-06-18", "2022-12-30", price=25.0)
        return pd.concat([pre, post])

    @staticmethod
    def _splits() -> pd.DataFrame:
        return pd.DataFrame(
            {"split_ratio": [4.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2021-06-14")], name="split_date"),
        )

    def test_split_with_matching_price_drop_is_zero_return(self) -> None:
        pos = build_position_series(self._split_prices(), pd.DataFrame(
            columns=["dividend"], index=pd.DatetimeIndex([], name="ex_date")
        ), self._splits())
        ret = compute_total_return(pos, pd.Timestamp("2021-01-29"), pd.Timestamp("2021-12-31"))
        assert ret == pytest.approx(0.0, abs=1e-12)

    def test_monthly_targets_spanning_split_are_zero(self) -> None:
        returns = build_monthly_returns(
            self._split_prices(),
            pd.DataFrame(columns=["dividend"], index=pd.DatetimeIndex([], name="ex_date")),
            self._splits(),
            forward_months=12,
        ).dropna()
        assert len(returns) >= 12
        assert returns.abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_db_pipeline_split_is_zero_and_missing_split_is_not(self, tmp_path) -> None:
        conn = db_client.get_connection(str(tmp_path / "t.db"))
        db_client.initialize_schema(conn)
        prices = self._split_prices()
        db_client.upsert_prices(conn, [
            {"ticker": "VGT", "date": d.strftime("%Y-%m-%d"), "close": float(c)}
            for d, c in prices["close"].items()
        ])
        without = build_etf_monthly_returns(conn, "VGT", 6).dropna()
        assert without.min() == pytest.approx(-0.75)  # the defect the split row fixes
        db_client.upsert_splits(conn, [{
            "ticker": "VGT", "split_date": "2021-06-14", "split_ratio": 4.0,
            "numerator": 4.0, "denominator": 1.0,
        }])
        with_split = build_etf_monthly_returns(conn, "VGT", 6).dropna()
        assert with_split.abs().max() == pytest.approx(0.0, abs=1e-12)
        conn.close()
