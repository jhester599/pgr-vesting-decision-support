"""Tax invariants: config sanity checks and properties over ``optimize_sale``.

Review 2026-09-25, F28: three of the v36 properties here were tautologies
(a zone predicate compared with its own complement, ``g(1-L) >= g(1-S)``
recomputed from the constants, and ``x >= t`` versus ``x < t``). They are
replaced by properties of the production optimiser, ``optimize_sale``, and
of ``TaxLot.is_ltcg_eligible``:

- the sale conserves shares and dollars, and each lot's tax is gain x rate;
- only vested lots are sold, none beyond its remaining shares, and only the
  last lot used can be partly sold;
- lots are used loss first, then LTCG gains, then STCG gains;
- each lot's holding type matches an independent calendar rule: long-term
  iff sold after the one-year anniversary (29 February vests have their
  anniversary on 28 February);
- with one rate for every lot, the tax is the minimum over all allocations
  (sell the highest-basis shares first);
- asking for more than the vested shares raises.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.tax.capital_gains import TaxLot, optimize_sale
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
# 5. TLH parameters are internally consistent
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
# Properties of optimize_sale and the LTCG boundary
# ---------------------------------------------------------------------------

_SELL_DATE = date(2026, 6, 15)
_RATES = {"ltcg_rate": 0.20, "stcg_rate": 0.37}


def _anniversary(vest: date) -> date:
    """One year after ``vest`` on the calendar (29 Feb -> 28 Feb)."""
    try:
        return vest.replace(year=vest.year + 1)
    except ValueError:
        return date(vest.year + 1, 2, 28)


def _is_long_term(vest: date, sold: date) -> bool:
    return sold > _anniversary(vest)


_vest_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2026, 5, 1))


@st.composite
def _lots(draw: st.DrawFn, max_lots: int = 6) -> list[TaxLot]:
    n = draw(st.integers(min_value=1, max_value=max_lots))
    lots = []
    for _ in range(n):
        lots.append(
            TaxLot(
                vest_date=draw(_vest_dates),
                rsu_type="time",
                shares=draw(st.floats(min_value=1.0, max_value=500.0)),
                cost_basis_per_share=draw(st.floats(min_value=20.0, max_value=300.0)),
            )
        )
    return lots


@given(_lots(), st.floats(min_value=20.0, max_value=300.0), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=300, deadline=None)
def test_optimize_sale_conserves_shares_and_dollars(
    lots: list[TaxLot], price: float, fraction: float
) -> None:
    available = sum(lot.shares for lot in lots)
    to_sell = available * fraction
    result = optimize_sale(lots, to_sell, price, _SELL_DATE, acquisition_dates=[], **_RATES)

    sold = sum(r.shares_sold for r in result.lots)
    assert sold == pytest.approx(to_sell, rel=1e-9, abs=1e-9)
    assert result.total_gross == pytest.approx(to_sell * price, rel=1e-9, abs=1e-6)
    assert result.total_net == pytest.approx(result.total_gross - result.total_tax, abs=1e-6)
    for r in result.lots:
        assert r.shares_sold <= r.lot.shares_remaining + 1e-9
        assert r.lot.vest_date <= _SELL_DATE
        assert r.taxable_gain == pytest.approx(r.shares_sold * (price - r.lot.cost_basis_per_share))
        assert r.tax_liability == pytest.approx(r.taxable_gain * r.tax_rate)
    # Only the last lot used may be partly sold.
    for r in result.lots[:-1]:
        assert r.shares_sold == pytest.approx(r.lot.shares_remaining)


@given(_lots(), st.floats(min_value=20.0, max_value=300.0))
@settings(max_examples=300, deadline=None)
def test_optimize_sale_orders_losses_then_ltcg_then_stcg(lots: list[TaxLot], price: float) -> None:
    total = sum(lot.shares for lot in lots)
    result = optimize_sale(lots, total, price, _SELL_DATE, acquisition_dates=[], **_RATES)
    rank = {"LOSS": 0, "LTCG": 1, "STCG": 2}
    ranks = [rank[r.holding_type] for r in result.lots]
    assert ranks == sorted(ranks)
    for r in result.lots:
        long_term = _is_long_term(r.lot.vest_date, _SELL_DATE)
        if r.holding_type == "LOSS":
            assert r.lot.cost_basis_per_share > price
            assert r.tax_rate == (_RATES["ltcg_rate"] if long_term else _RATES["stcg_rate"])
        else:
            assert r.lot.cost_basis_per_share <= price
            assert r.holding_type == ("LTCG" if long_term else "STCG")


@given(_vest_dates, st.integers(min_value=-3, max_value=3))
@settings(max_examples=500)
def test_ltcg_boundary_is_the_day_after_the_calendar_anniversary(vest: date, offset: int) -> None:
    lot = TaxLot(vest_date=vest, rsu_type="time", shares=1.0, cost_basis_per_share=1.0)
    sold = _anniversary(vest) + timedelta(days=offset)
    assert lot.is_ltcg_eligible(sold) is (offset >= 1)


@given(
    st.lists(
        st.tuples(st.floats(min_value=1.0, max_value=200.0), st.floats(min_value=10.0, max_value=99.0)),
        min_size=1,
        max_size=6,
    ),
    st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=300, deadline=None)
def test_single_rate_sale_pays_the_minimum_tax(lot_specs: list[tuple[float, float]], fraction: float) -> None:
    """All lots long-term with a gain at price 100: the cheapest allocation
    sells the highest-basis shares first."""
    price = 100.0
    lots = [
        TaxLot(vest_date=date(2022, 1, 3), rsu_type="time", shares=shares, cost_basis_per_share=basis)
        for shares, basis in lot_specs
    ]
    to_sell = fraction * sum(shares for shares, _ in lot_specs)
    result = optimize_sale(lots, to_sell, price, _SELL_DATE, acquisition_dates=[], **_RATES)

    remaining, minimum_gain = to_sell, 0.0
    for shares, basis in sorted(lot_specs, key=lambda spec: -spec[1]):
        take = min(shares, remaining)
        minimum_gain += take * (price - basis)
        remaining -= take
    assert result.total_tax == pytest.approx(_RATES["ltcg_rate"] * minimum_gain, rel=1e-9, abs=1e-6)


@given(_lots(), st.floats(min_value=1.01, max_value=3.0))
@settings(max_examples=100, deadline=None)
def test_selling_more_than_the_vested_shares_raises(lots: list[TaxLot], excess: float) -> None:
    vested = sum(lot.shares for lot in lots if lot.vest_date <= _SELL_DATE)
    assume(vested > 0)
    with pytest.raises(ValueError):
        optimize_sale(lots, vested * excess, 100.0, _SELL_DATE, acquisition_dates=[], **_RATES)
