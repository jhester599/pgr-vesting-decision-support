"""Property tests: feature-engineering invariants on production functions.

Review 2026-09-25, F28: four of the v36 properties here recomputed a
formula inline (momentum sign, 52-week ratio, volatility, YoY growth) and
never called production code. They are replaced by properties of the
production price-feature helpers and of ``build_feature_matrix``:

- ``calendar_momentum`` equals the ratio of the last closes on or before
  the two business month-ends, found here by a plain loop;
- ``trailing_52w_high`` is the maximum close within 364 days, so the
  52-week-high ratio lies in (0, 1];
- ``weekly_realized_vol`` is non-negative and unchanged by rescaling
  prices;
- after ``split_adjusted_close``, a split with the matching price drop
  changes neither momentum nor volatility;
- ``build_feature_matrix`` is causal: rows up to month t do not change when
  prices after t change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.processing.feature_engineering import build_feature_matrix, compute_vif
from src.processing.price_adjustment import (
    calendar_momentum,
    split_adjusted_close,
    trailing_52w_high,
    weekly_realized_vol,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Realistic return-like floats (bounded to keep VIF regression stable).
bounded_float = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)


def _feature_matrix_strategy(
    min_rows: int = 10,
    max_rows: int = 60,
    n_cols: int = 3,
) -> st.SearchStrategy[pd.DataFrame]:
    """Strategy that produces a DataFrame with `n_cols` numeric columns."""
    return st.lists(
        st.lists(bounded_float, min_size=n_cols, max_size=n_cols),
        min_size=min_rows,
        max_size=max_rows,
    ).map(
        lambda rows: pd.DataFrame(rows, columns=[f"f{i}" for i in range(n_cols)])
    )


# ---------------------------------------------------------------------------
# 1. VIF values are always positive (or the Series is empty)
# ---------------------------------------------------------------------------

@given(_feature_matrix_strategy(min_rows=10, max_rows=60, n_cols=3))
@settings(max_examples=200, deadline=None)  # first call imports statsmodels
def test_vif_values_are_positive(df: pd.DataFrame) -> None:
    """All VIF values returned by compute_vif() must be >= 1.0 by construction."""
    result = compute_vif(df)
    if result.empty:
        return  # degenerate rank-deficient input — empty result is allowed
    for feature, vif_val in result.items():
        assert vif_val >= 1.0 - 1e-9, (
            f"VIF for '{feature}' = {vif_val:.4f} is below 1.0 (mathematical lower bound)"
        )


# ---------------------------------------------------------------------------
# 2. VIF index contains only feature names present in the DataFrame
# ---------------------------------------------------------------------------

@given(_feature_matrix_strategy(min_rows=10, max_rows=40, n_cols=4))
@settings(max_examples=150, deadline=None)
def test_vif_index_subset_of_columns(df: pd.DataFrame) -> None:
    """VIF result index must be a subset of the DataFrame columns."""
    result = compute_vif(df)
    for name in result.index:
        assert name in df.columns, (
            f"VIF returned feature '{name}' not present in DataFrame columns"
        )


# ---------------------------------------------------------------------------
# 3-7. Price-feature helpers and build_feature_matrix
# ---------------------------------------------------------------------------

@st.composite
def weekly_closes(draw: st.DrawFn, min_weeks: int = 30, max_weeks: int = 160) -> pd.Series:
    """Positive weekly closes (a random walk) on Fridays."""
    n = draw(st.integers(min_value=min_weeks, max_value=max_weeks))
    steps = draw(
        arrays(np.float64, n - 1, elements=st.floats(min_value=-0.12, max_value=0.12, allow_nan=False))
    )
    start = draw(st.floats(min_value=5.0, max_value=400.0))
    closes = start * np.cumprod(np.concatenate([[1.0], 1.0 + steps]))
    return pd.Series(closes, index=pd.date_range("2012-01-06", periods=n, freq="W-FRI"), name="close")


def _last_close_on_or_before(close: pd.Series, when: pd.Timestamp) -> float:
    eligible = close[close.index <= when]
    return float(eligible.iloc[-1]) if len(eligible) else float("nan")


@given(weekly_closes(), st.sampled_from([1, 3, 6, 12]))
@settings(max_examples=60, deadline=None)
def test_calendar_momentum_uses_month_end_closes(close: pd.Series, months: int) -> None:
    momentum = calendar_momentum(close, months).dropna()
    for t, value in momentum.items():
        start = pd.offsets.BMonthEnd().rollback(t - pd.DateOffset(months=months) + pd.offsets.MonthEnd(0))
        expected = _last_close_on_or_before(close, t) / _last_close_on_or_before(close, start) - 1.0
        assert value == pytest.approx(expected, rel=1e-12)


@given(weekly_closes())
@settings(max_examples=60, deadline=None)
def test_trailing_52w_high_is_the_max_within_364_days(close: pd.Series) -> None:
    high = trailing_52w_high(close)
    for t, value in high.dropna().items():
        window = close[(close.index > t - pd.Timedelta(days=364)) & (close.index <= t)]
        assert value == pytest.approx(float(window.max()))
        assert 0.0 < close.loc[t] / value <= 1.0 + 1e-12


@given(weekly_closes(), st.floats(min_value=0.01, max_value=100.0), st.sampled_from([4, 13]))
@settings(max_examples=60, deadline=None)
def test_weekly_vol_is_non_negative_and_scale_free(close: pd.Series, scale: float, n_weeks: int) -> None:
    vol = weekly_realized_vol(close, n_weeks)
    scaled = weekly_realized_vol(close * scale, n_weeks)
    assert (vol.dropna() >= 0).all()
    np.testing.assert_allclose(vol.to_numpy(), scaled.to_numpy(), rtol=1e-9, atol=1e-8)


@given(weekly_closes(min_weeks=60), st.sampled_from([2.0, 3.0, 4.0]), st.data())
@settings(max_examples=40, deadline=None)
def test_split_adjustment_removes_the_split_from_price_features(
    close: pd.Series, ratio: float, data: st.DataObject
) -> None:
    k = data.draw(st.integers(min_value=1, max_value=len(close) - 1), label="split_bar")
    raw = close.copy()
    raw.iloc[k:] = close.iloc[k:] / ratio
    splits = pd.DataFrame({"split_ratio": [ratio]}, index=pd.DatetimeIndex([close.index[k]], name="split_date"))
    adjusted = split_adjusted_close(raw, splits)
    # Latest-basis prices: the unsplit path divided by the ratio throughout.
    np.testing.assert_allclose(adjusted.to_numpy(), close.to_numpy() / ratio, rtol=1e-12)
    np.testing.assert_allclose(
        calendar_momentum(adjusted, 6).to_numpy(), calendar_momentum(close, 6).to_numpy(), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        weekly_realized_vol(adjusted, 13).to_numpy(), weekly_realized_vol(close, 13).to_numpy(), rtol=1e-9, atol=1e-8
    )


@given(weekly_closes(min_weeks=120, max_weeks=200), st.data())
@settings(max_examples=10, deadline=None)
def test_build_feature_matrix_is_causal(close: pd.Series, data: st.DataObject) -> None:
    empty_div = pd.DataFrame(columns=["dividend", "source"], index=pd.DatetimeIndex([], name="ex_date"))
    empty_split = pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"], index=pd.DatetimeIndex([], name="split_date")
    )
    prices = close.to_frame()
    base = build_feature_matrix(prices, empty_div, empty_split, force_refresh=True)
    cut_pos = data.draw(st.integers(min_value=12, max_value=len(base) - 2), label="cut_month")
    cut = base.index[cut_pos]
    changed = prices.copy()
    later = changed.index > cut
    changed.loc[later, "close"] = changed.loc[later, "close"] * np.linspace(0.5, 2.0, int(later.sum()))
    moved = build_feature_matrix(changed, empty_div, empty_split, force_refresh=True)
    features = [c for c in base.columns if not c.startswith("target")]
    pd.testing.assert_frame_equal(base.loc[:cut, features], moved.loc[:cut, features])
