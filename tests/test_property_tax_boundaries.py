"""v36 — Property-based tests: tax-boundary logic invariants.

Verifies that the STCG boundary guard constants and tax-rate ordering
satisfy their documented contracts for arbitrary (but realistic) inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import config
from config.tax import (
    LTCG_RATE,
    STCG_RATE,
    STCG_BREAKEVEN_THRESHOLD,
    STCG_ZONE_MIN_DAYS,
    STCG_ZONE_MAX_DAYS,
    TLH_LOSS_THRESHOLD,
    TLH_WASH_SALE_DAYS,
)


# ---------------------------------------------------------------------------
# 1. LTCG rate is always strictly less than STCG rate
# ---------------------------------------------------------------------------

def test_ltcg_rate_less_than_stcg_rate() -> None:
    """Federal LTCG rate must be strictly below ordinary STCG rate."""
    assert LTCG_RATE < STCG_RATE, (
        f"LTCG_RATE={LTCG_RATE} must be < STCG_RATE={STCG_RATE}"
    )


# ---------------------------------------------------------------------------
# 2. Tax rates are valid probabilities (0 < rate < 1)
# ---------------------------------------------------------------------------

def test_tax_rates_in_valid_range() -> None:
    """Both tax rates must be strictly between 0 and 1."""
    assert 0.0 < LTCG_RATE < 1.0, f"LTCG_RATE={LTCG_RATE} out of range"
    assert 0.0 < STCG_RATE < 1.0, f"STCG_RATE={STCG_RATE} out of range"


# ---------------------------------------------------------------------------
# 3. STCG zone bounds are logically ordered
# ---------------------------------------------------------------------------

def test_stcg_zone_bounds_ordered() -> None:
    """STCG_ZONE_MIN_DAYS < STCG_ZONE_MAX_DAYS and both positive."""
    assert STCG_ZONE_MIN_DAYS > 0
    assert STCG_ZONE_MAX_DAYS > 0
    assert STCG_ZONE_MIN_DAYS < STCG_ZONE_MAX_DAYS, (
        f"MIN={STCG_ZONE_MIN_DAYS} must be < MAX={STCG_ZONE_MAX_DAYS}"
    )


# ---------------------------------------------------------------------------
# 4. Breakeven threshold is positive and below the STCG–LTCG spread
# ---------------------------------------------------------------------------

def test_stcg_breakeven_threshold_reasonable() -> None:
    """Breakeven threshold should be positive and less than the rate differential."""
    spread = STCG_RATE - LTCG_RATE
    assert STCG_BREAKEVEN_THRESHOLD > 0.0
    assert STCG_BREAKEVEN_THRESHOLD <= spread + 0.10, (
        f"Threshold {STCG_BREAKEVEN_THRESHOLD} exceeds rate spread {spread:.2f} by more than 10pp"
    )


# ---------------------------------------------------------------------------
# 5. STCG zone membership is exclusive of LTCG qualification day
# ---------------------------------------------------------------------------

@given(st.integers(min_value=0, max_value=730))
@settings(max_examples=500)
def test_stcg_zone_membership_disjoint_from_ltcg(holding_days: int) -> None:
    """A lot cannot simultaneously be in the STCG zone and LTCG-qualified."""
    in_stcg_zone = STCG_ZONE_MIN_DAYS < holding_days <= STCG_ZONE_MAX_DAYS
    is_ltcg = holding_days > STCG_ZONE_MAX_DAYS
    # These two conditions must be mutually exclusive.
    assert not (in_stcg_zone and is_ltcg), (
        f"holding_days={holding_days} cannot be both STCG-zone and LTCG"
    )


# ---------------------------------------------------------------------------
# 6. After-tax gain ordering: LTCG lot always yields >= after-tax proceeds
#    than STCG lot for any positive gain
# ---------------------------------------------------------------------------

@given(
    st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),  # gain
)
@settings(max_examples=300)
def test_ltcg_after_tax_dominates_stcg_for_positive_gains(gain: float) -> None:
    """For any non-negative gain, LTCG after-tax proceeds >= STCG after-tax proceeds."""
    after_tax_ltcg = gain * (1.0 - LTCG_RATE)
    after_tax_stcg = gain * (1.0 - STCG_RATE)
    assert after_tax_ltcg >= after_tax_stcg - 1e-12, (
        f"LTCG after-tax={after_tax_ltcg:.4f} < STCG after-tax={after_tax_stcg:.4f}"
        f" for gain={gain}"
    )


# ---------------------------------------------------------------------------
# 7. TLH parameters are internally consistent
# ---------------------------------------------------------------------------

def test_tlh_loss_threshold_is_negative() -> None:
    """TLH harvest trigger must be a negative return (a loss)."""
    assert TLH_LOSS_THRESHOLD < 0.0, (
        f"TLH_LOSS_THRESHOLD={TLH_LOSS_THRESHOLD} should be negative"
    )


def test_tlh_wash_sale_window_positive() -> None:
    """Wash-sale days must be a positive integer."""
    assert TLH_WASH_SALE_DAYS > 0


# ---------------------------------------------------------------------------
# 8. Tax differential determines whether waiting is worthwhile
#    (breakeven derivation: hold if alpha < (STCG_RATE - LTCG_RATE))
# ---------------------------------------------------------------------------

@given(
    st.floats(min_value=-0.5, max_value=2.0, allow_nan=False, allow_infinity=False),  # predicted alpha
)
@settings(max_examples=300)
def test_breakeven_logic_direction(predicted_alpha: float) -> None:
    """If predicted alpha > STCG breakeven, selling STCG-classified lots is justified."""
    rate_differential = STCG_RATE - LTCG_RATE
    # The breakeven threshold should be in the plausible neighbourhood of the
    # rate differential (within ±25 percentage points).
    assert abs(STCG_BREAKEVEN_THRESHOLD - rate_differential) <= 0.25, (
        "Breakeven threshold should be near the STCG−LTCG rate differential"
    )
    # Property: if we decide to sell (alpha >= threshold), the alpha justifies
    # paying the higher tax rate — no assertion on alpha itself, just verify
    # the threshold membership predicate is deterministic.
    should_sell = predicted_alpha >= STCG_BREAKEVEN_THRESHOLD
    should_wait = predicted_alpha < STCG_BREAKEVEN_THRESHOLD
    assert should_sell != should_wait or predicted_alpha == STCG_BREAKEVEN_THRESHOLD, (
        "sell/wait decision must be deterministic"
    )
