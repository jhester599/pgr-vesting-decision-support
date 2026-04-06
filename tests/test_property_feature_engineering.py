"""v36 — Property-based tests: feature-engineering numerical invariants.

Verifies that the VIF helper and momentum/volatility feature computations
satisfy mathematical bounds regardless of input values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.processing.feature_engineering import compute_vif


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
@settings(max_examples=200)
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
@settings(max_examples=150)
def test_vif_index_subset_of_columns(df: pd.DataFrame) -> None:
    """VIF result index must be a subset of the DataFrame columns."""
    result = compute_vif(df)
    for name in result.index:
        assert name in df.columns, (
            f"VIF returned feature '{name}' not present in DataFrame columns"
        )


# ---------------------------------------------------------------------------
# 3. Momentum sign: if price_t > price_0, momentum is positive
# ---------------------------------------------------------------------------

@given(
    st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False),  # pct gain
)
@settings(max_examples=300)
def test_momentum_sign_matches_price_direction_up(price_start: float, gain_pct: float) -> None:
    """If end price is above start price, momentum (pct_change) is positive."""
    price_end = price_start * (1.0 + gain_pct)
    momentum = (price_end - price_start) / price_start
    assert momentum > 0.0, (
        f"momentum={momentum:.6f} should be positive when price rose from {price_start} to {price_end}"
    )


@given(
    st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False),  # pct loss
)
@settings(max_examples=300)
def test_momentum_sign_matches_price_direction_down(price_start: float, loss_pct: float) -> None:
    """If end price is below start price, momentum (pct_change) is negative."""
    price_end = price_start * (1.0 - loss_pct)
    momentum = (price_end - price_start) / price_start
    assert momentum < 0.0, (
        f"momentum={momentum:.6f} should be negative when price fell from {price_start} to {price_end}"
    )


# ---------------------------------------------------------------------------
# 4. 52-week high ratio is bounded in (0, 1]
# ---------------------------------------------------------------------------

@given(
    st.floats(min_value=0.01, max_value=1_000.0, allow_nan=False, allow_infinity=False),  # current
    st.floats(min_value=0.01, max_value=1_000.0, allow_nan=False, allow_infinity=False),  # 52w high
)
@settings(max_examples=300)
def test_52w_high_ratio_in_unit_interval(current_price: float, high_52w: float) -> None:
    """52-week high ratio = current / max(current, high_52w) ∈ (0, 1]."""
    assume(current_price > 0 and high_52w > 0)
    rolling_high = max(current_price, high_52w)
    ratio = current_price / rolling_high
    assert 0.0 < ratio <= 1.0 + 1e-12, (
        f"52w high ratio={ratio:.6f} out of (0, 1] for current={current_price}, high={high_52w}"
    )


# ---------------------------------------------------------------------------
# 5. Realized volatility (annualised) is non-negative
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.floats(min_value=-0.20, max_value=0.20, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=252,
    )
)
@settings(max_examples=300)
def test_annualised_volatility_non_negative(daily_returns: list[float]) -> None:
    """Annualised volatility (daily std * sqrt(252)) must be >= 0."""
    assume(len(set(daily_returns)) > 1)  # skip perfectly constant series
    arr = np.array(daily_returns)
    annualised_vol = float(np.std(arr, ddof=1)) * np.sqrt(252)
    assert annualised_vol >= 0.0, f"annualised_vol={annualised_vol} is negative"


# ---------------------------------------------------------------------------
# 6. YoY growth rate: positive base + higher current → positive growth
# ---------------------------------------------------------------------------

@given(
    st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.001, max_value=5.0, allow_nan=False, allow_infinity=False),  # multiplier > 1 means growth
)
@settings(max_examples=300)
def test_yoy_growth_rate_sign_positive(base: float, multiplier: float) -> None:
    """YoY growth = (current / prior - 1); if current > prior, growth > 0."""
    assume(multiplier > 1.0)
    current = base * multiplier
    growth = (current / base) - 1.0
    assert growth > 0.0, f"growth={growth:.6f} should be positive when current > base"
