"""Forward-return target windows end at the last bar on or before BMonthEnd(t + h).

Review F22: windows used to end at the last weekly close on or before
``t + DateOffset(months=h)``, so 6M windows ran 175-190 days and about a third
ended one week early (e.g. 2025-02-28 + 6M = Thu 2025-08-28 -> bar 2025-08-22
instead of the month-end bar 2025-08-29).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.database import db_client
from src.processing.multi_total_return import build_etf_monthly_returns
from src.processing.total_return import build_monthly_returns


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        ("2025-02-28", 6, "2025-08-29"),   # DateOffset lands on Thu 08-28
        ("2025-08-29", 6, "2026-02-27"),   # DateOffset lands on Sat 02-28
        ("2025-10-31", 6, "2026-04-30"),
        ("2025-03-31", 12, "2026-03-31"),
        ("2024-02-29", 12, "2025-02-28"),  # leap-day start
        ("2025-11-28", 6, "2026-05-29"),
    ],
)
def test_forward_window_end_is_business_month_end(start, months, expected) -> None:
    from src.processing.total_return import forward_window_end

    assert forward_window_end(pd.Timestamp(start), months) == pd.Timestamp(expected)


def _weekly_prices(start: str = "2024-01-05", periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="W-FRI")
    # Strictly increasing, distinct closes so the bar used is identifiable.
    closes = 100.0 + np.arange(periods, dtype=float)
    return pd.DataFrame({"close": closes}, index=dates)


def _empty_divs() -> pd.DataFrame:
    return pd.DataFrame(columns=["dividend"], index=pd.DatetimeIndex([], name="ex_date"))


def _empty_splits() -> pd.DataFrame:
    return pd.DataFrame(columns=["split_ratio"], index=pd.DatetimeIndex([], name="split_date"))


def test_window_ends_on_month_end_bar_not_a_week_early() -> None:
    prices = _weekly_prices()
    returns = build_monthly_returns(prices, _empty_divs(), _empty_splits(), forward_months=6)
    t = pd.Timestamp("2025-02-28")
    p0 = prices.loc["2025-02-28", "close"]
    p1 = prices.loc["2025-08-29", "close"]
    assert returns.loc[t] == pytest.approx(p1 / p0 - 1.0)


def test_window_needing_unavailable_month_end_is_nan() -> None:
    # Last bar 2026-03-20: a window ending BME 2026-03-31 is not yet complete.
    prices = _weekly_prices(start="2025-01-03", periods=64)
    assert prices.index[-1] == pd.Timestamp("2026-03-20")
    returns = build_monthly_returns(prices, _empty_divs(), _empty_splits(), forward_months=6)
    assert np.isnan(returns.loc[pd.Timestamp("2025-09-30")])
    assert not np.isnan(returns.loc[pd.Timestamp("2025-08-29")])


def test_db_pipeline_uses_business_month_end_window(tmp_path) -> None:
    conn = db_client.get_connection(str(tmp_path / "t.db"))
    db_client.initialize_schema(conn)
    prices = _weekly_prices()
    db_client.upsert_prices(conn, [
        {"ticker": "VTI", "date": d.strftime("%Y-%m-%d"), "close": float(c)}
        for d, c in prices["close"].items()
    ])
    returns = build_etf_monthly_returns(conn, "VTI", 6)
    p0 = prices.loc["2025-02-28", "close"]
    p1 = prices.loc["2025-08-29", "close"]
    assert returns.loc[pd.Timestamp("2025-02-28")] == pytest.approx(p1 / p0 - 1.0)
    conn.close()
