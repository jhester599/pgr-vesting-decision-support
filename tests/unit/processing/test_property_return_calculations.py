"""Property tests: DRIP total return, checked on ``build_position_series``.

Review 2026-09-25, F28: the v36 version of this file tested arithmetic
identities (compounding, log additivity, mean within min/max) without
calling production code. These properties run ``build_position_series`` on
generated unadjusted price paths with dividends and splits:

- without corporate actions, shares are constant and value tracks price;
- a split with the matching price drop leaves the position's value
  unchanged at every date (the split multiplies shares, not value);
- each ex-dividend date reinvests ``shares x dividend / close`` new shares,
  so the ending value equals the closed form
  ``initial_shares x P_T x prod(1 + d_i / P_i)``;
- with positive dividends and splits the share count never falls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.processing.total_return import build_position_series

_SETTINGS = settings(max_examples=60, deadline=None)


def _no_dividends() -> pd.DataFrame:
    return pd.DataFrame(columns=["dividend"], index=pd.DatetimeIndex([], name="ex_date"), dtype=float)


def _no_splits() -> pd.DataFrame:
    return pd.DataFrame(columns=["split_ratio"], index=pd.DatetimeIndex([], name="split_date"), dtype=float)


@st.composite
def price_paths(draw: st.DrawFn, min_len: int = 8, max_len: int = 80) -> pd.DataFrame:
    """Weekly unadjusted closes: a positive random walk."""
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    returns = draw(
        st.lists(
            st.floats(min_value=-0.15, max_value=0.15, allow_nan=False),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    start = draw(st.floats(min_value=5.0, max_value=500.0))
    closes = start * np.cumprod(np.concatenate([[1.0], 1.0 + np.asarray(returns)]))
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    return pd.DataFrame({"close": closes}, index=idx)


initial_shares = st.floats(min_value=0.5, max_value=5_000.0)


@given(price_paths(), initial_shares)
@_SETTINGS
def test_no_corporate_actions_value_tracks_price(prices: pd.DataFrame, shares: float) -> None:
    pos = build_position_series(prices, _no_dividends(), _no_splits(), initial_shares=shares)
    np.testing.assert_allclose(pos["shares_held"].to_numpy(), shares)
    np.testing.assert_allclose(pos["portfolio_value"].to_numpy(), shares * prices["close"].to_numpy())


@given(price_paths(), initial_shares, st.sampled_from([2.0, 3.0, 4.0, 0.5, 1.5]), st.data())
@_SETTINGS
def test_split_with_matching_price_drop_preserves_value(
    prices: pd.DataFrame, shares: float, ratio: float, data: st.DataObject
) -> None:
    """Unadjusted prices fall by the split ratio on the split date; the
    position's value must be the same as the unsplit history's."""
    k = data.draw(st.integers(min_value=1, max_value=len(prices) - 1), label="split_bar")
    split_date = prices.index[k]
    split_prices = prices.copy()
    split_prices.iloc[k:, 0] = prices["close"].iloc[k:] / ratio
    splits = pd.DataFrame({"split_ratio": [ratio]}, index=pd.DatetimeIndex([split_date], name="split_date"))

    unsplit = build_position_series(prices, _no_dividends(), _no_splits(), initial_shares=shares)
    split = build_position_series(split_prices, _no_dividends(), splits, initial_shares=shares)

    np.testing.assert_allclose(split["portfolio_value"].to_numpy(), unsplit["portfolio_value"].to_numpy(), rtol=1e-10)
    assert np.isclose(split["shares_held"].iloc[-1], shares * ratio, rtol=1e-12)


@given(price_paths(min_len=12), initial_shares, st.data())
@_SETTINGS
def test_drip_matches_the_closed_form(prices: pd.DataFrame, shares: float, data: st.DataObject) -> None:
    """Ending value = shares x P_T x prod(1 + d_i / P_i) over ex-dividend bars."""
    n = len(prices)
    bars = data.draw(
        st.lists(st.integers(min_value=0, max_value=n - 1), min_size=1, max_size=6, unique=True),
        label="ex_bars",
    )
    bars.sort()
    amounts = data.draw(
        st.lists(st.floats(min_value=0.01, max_value=5.0), min_size=len(bars), max_size=len(bars)),
        label="dividends",
    )
    dividends = pd.DataFrame(
        {"dividend": amounts}, index=pd.DatetimeIndex(prices.index[bars], name="ex_date")
    )
    pos = build_position_series(prices, dividends, _no_splits(), initial_shares=shares)

    closes = prices["close"].to_numpy()
    growth = np.prod([1.0 + d / closes[b] for b, d in zip(bars, amounts)])
    assert np.isclose(pos["shares_held"].iloc[-1], shares * growth, rtol=1e-10)
    assert np.isclose(pos["portfolio_value"].iloc[-1], shares * growth * closes[-1], rtol=1e-10)
    # Before the first ex-date the share count is untouched.
    assert np.allclose(pos["shares_held"].iloc[: bars[0]].to_numpy(), shares)


@given(price_paths(min_len=12), initial_shares, st.data())
@_SETTINGS
def test_share_count_never_falls_with_dividends_and_forward_splits(
    prices: pd.DataFrame, shares: float, data: st.DataObject
) -> None:
    n = len(prices)
    div_bars = sorted(data.draw(st.lists(st.integers(0, n - 1), min_size=0, max_size=5, unique=True)))
    split_bar = data.draw(st.integers(1, n - 1))
    ratio = data.draw(st.sampled_from([2.0, 3.0, 1.5]))
    dividends = pd.DataFrame(
        {"dividend": [0.25] * len(div_bars)},
        index=pd.DatetimeIndex(prices.index[div_bars], name="ex_date"),
    )
    split_prices = prices.copy()
    split_prices.iloc[split_bar:, 0] = prices["close"].iloc[split_bar:] / ratio
    splits = pd.DataFrame(
        {"split_ratio": [ratio]}, index=pd.DatetimeIndex([prices.index[split_bar]], name="split_date")
    )
    pos = build_position_series(split_prices, dividends, splits, initial_shares=shares)
    assert (np.diff(pos["shares_held"].to_numpy()) >= -1e-12).all()
    assert pos["shares_held"].iloc[-1] >= shares * ratio - 1e-9
