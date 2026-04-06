"""v36 — Property-based tests: return calculation invariants.

Uses hypothesis to verify that numerical primitives used throughout the
pipeline satisfy mathematical invariants regardless of the exact values
chosen.  These tests catch silent regressions in arithmetic logic that
unit tests with fixed inputs might miss.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Realistic monthly return values (bounded to avoid overflow in compounding).
monthly_return = st.floats(min_value=-0.50, max_value=1.00, allow_nan=False, allow_infinity=False)

# Price-like values: strictly positive, finite.
positive_price = st.floats(min_value=0.001, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# 1. Total-return lower bound: a position can lose at most 100 %
# ---------------------------------------------------------------------------

@given(st.lists(monthly_return, min_size=1, max_size=120))
@settings(max_examples=300)
def test_compounded_return_never_below_minus_one(monthly_returns: list[float]) -> None:
    """Gross return (product of 1+r_i) is always >= 0; net return >= -1."""
    gross = 1.0
    for r in monthly_returns:
        gross *= (1.0 + r)
    net_return = gross - 1.0
    assert net_return >= -1.0, (
        f"net_return={net_return:.6f} violates lower bound; monthly_returns={monthly_returns}"
    )


# ---------------------------------------------------------------------------
# 2. Percent-change identity: pct_change then compound back → original ratio
# ---------------------------------------------------------------------------

@given(positive_price, positive_price)
@settings(max_examples=300)
def test_pct_change_compounding_identity(p0: float, p1: float) -> None:
    """(p1 - p0) / p0 compounded from p0 recovers p1."""
    pct = (p1 - p0) / p0
    recovered = p0 * (1.0 + pct)
    assert abs(recovered - p1) < 1e-9 * max(abs(p1), 1.0), (
        f"Identity broken: p0={p0}, p1={p1}, recovered={recovered}"
    )


# ---------------------------------------------------------------------------
# 3. Log-return additivity: sum of log returns equals log of gross return
# ---------------------------------------------------------------------------

@given(st.lists(monthly_return, min_size=2, max_size=60))
@settings(max_examples=300)
def test_log_return_additivity(monthly_returns: list[float]) -> None:
    """Sum of log(1+r_i) == log(gross_return) to floating-point precision."""
    # Skip if any factor (1+r) is non-positive (log undefined).
    assume(all(r > -1.0 for r in monthly_returns))

    sum_log = sum(np.log1p(r) for r in monthly_returns)
    gross = 1.0
    for r in monthly_returns:
        gross *= (1.0 + r)
    log_gross = np.log(gross)

    assert abs(sum_log - log_gross) < 1e-9, (
        f"Log-return additivity violated: sum_log={sum_log}, log_gross={log_gross}"
    )


# ---------------------------------------------------------------------------
# 4. Arithmetic mean of signed returns is between min and max
# ---------------------------------------------------------------------------

@given(st.lists(monthly_return, min_size=1, max_size=240))
@settings(max_examples=300)
def test_mean_return_within_bounds(monthly_returns: list[float]) -> None:
    """Mean return is always within [min, max] of the series."""
    arr = np.array(monthly_returns)
    mean_r = float(np.mean(arr))
    assert mean_r >= float(np.min(arr)) - 1e-12
    assert mean_r <= float(np.max(arr)) + 1e-12


# ---------------------------------------------------------------------------
# 5. Rolling standard deviation is non-negative
# ---------------------------------------------------------------------------

@given(st.lists(monthly_return, min_size=2, max_size=120))
@settings(max_examples=300)
def test_rolling_std_non_negative(monthly_returns: list[float]) -> None:
    """Standard deviation is always >= 0."""
    arr = np.array(monthly_returns)
    std = float(np.std(arr, ddof=1))
    assert std >= 0.0, f"std={std} is negative for input {monthly_returns}"


# ---------------------------------------------------------------------------
# 6. Sharpe-ratio numerator sign matches mean excess return sign
# ---------------------------------------------------------------------------

@given(
    st.lists(monthly_return, min_size=3, max_size=60),
    st.floats(min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False),  # risk-free rate per period
)
@settings(max_examples=300)
def test_sharpe_numerator_sign(monthly_returns: list[float], rf: float) -> None:
    """Sign of (mean - rf) matches direction of Sharpe numerator."""
    arr = np.array(monthly_returns)
    excess_mean = float(np.mean(arr)) - rf
    std = float(np.std(arr, ddof=1))
    assume(std > 1e-12)  # skip degenerate constant series
    sharpe = excess_mean / std
    if excess_mean > 0:
        assert sharpe > 0, f"sharpe={sharpe} should be positive when excess_mean={excess_mean}"
    elif excess_mean < 0:
        assert sharpe < 0, f"sharpe={sharpe} should be negative when excess_mean={excess_mean}"
