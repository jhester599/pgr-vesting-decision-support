"""
Tests for Tax-Loss Harvesting functions in src/tax/capital_gains.py (v4.0).

Verifies:
  - identify_tlh_candidates() returns lots below the loss threshold
  - Harvest triggered at -10% threshold; larger losses appear first
  - compute_after_tax_expected_return() formula: return - max(0, gain × tax_rate)
  - suggest_tlh_replacement() returns correct substitute from TLH_REPLACEMENT_MAP
  - wash_sale_clear_date() returns harvest_date + 31 days
  - No candidates returned when all lots are above threshold
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import config
from src.tax.capital_gains import (
    TaxLot,
    compute_after_tax_expected_return,
    identify_tlh_candidates,
    suggest_tlh_replacement,
    wash_sale_clear_date,
)


def _make_lot(
    cost_basis: float,
    shares: float = 100.0,
    vest_date: date = date(2022, 1, 19),
    rsu_type: str = "time",
) -> TaxLot:
    return TaxLot(
        vest_date=vest_date,
        rsu_type=rsu_type,
        shares=shares,
        cost_basis_per_share=cost_basis,
    )


class TestIdentifyTlhCandidates:
    def test_returns_list(self):
        lots = [_make_lot(100.0), _make_lot(200.0)]
        result = identify_tlh_candidates(lots, current_price=90.0)
        assert isinstance(result, list)

    def test_loss_lot_identified(self):
        """Lot at -15% should be identified as a candidate."""
        lot = _make_lot(cost_basis=100.0)
        result = identify_tlh_candidates([lot], current_price=85.0)
        assert lot in result

    def test_gain_lot_not_identified(self):
        """Lot at +10% should not be harvested."""
        lot = _make_lot(cost_basis=100.0)
        result = identify_tlh_candidates([lot], current_price=110.0)
        assert lot not in result

    def test_exactly_at_threshold_not_identified(self):
        """Lot at exactly -10% is not below threshold."""
        lot = _make_lot(cost_basis=100.0)
        result = identify_tlh_candidates([lot], current_price=90.0, loss_threshold=-0.10)
        assert lot not in result  # must be strictly below threshold

    def test_just_below_threshold_identified(self):
        lot = _make_lot(cost_basis=100.0)
        result = identify_tlh_candidates([lot], current_price=89.9, loss_threshold=-0.10)
        assert lot in result

    def test_sorted_largest_loss_first(self):
        lot_big = _make_lot(cost_basis=200.0)   # -50% at price 100
        lot_small = _make_lot(cost_basis=120.0)  # -17% at price 100
        result = identify_tlh_candidates([lot_small, lot_big], current_price=100.0)
        assert len(result) == 2
        assert result[0].cost_basis_per_share == 200.0  # bigger loss first

    def test_empty_lots_returns_empty(self):
        assert identify_tlh_candidates([], current_price=100.0) == []

    def test_all_gains_returns_empty(self):
        lots = [_make_lot(50.0), _make_lot(60.0), _make_lot(70.0)]
        result = identify_tlh_candidates(lots, current_price=100.0)
        assert result == []

    def test_uses_config_threshold_by_default(self):
        """Default threshold should equal config.TLH_LOSS_THRESHOLD (-0.10)."""
        lot = _make_lot(cost_basis=100.0)
        result_default = identify_tlh_candidates([lot], current_price=89.0)
        result_explicit = identify_tlh_candidates(
            [lot], current_price=89.0, loss_threshold=config.TLH_LOSS_THRESHOLD
        )
        assert (lot in result_default) == (lot in result_explicit)


class TestComputeAfterTaxExpectedReturn:
    def test_no_embedded_gain_no_tax_drag(self):
        """Zero unrealized gain → after-tax return equals predicted return."""
        result = compute_after_tax_expected_return(
            predicted_return=0.10,
            unrealized_gain_fraction=0.0,
            tax_rate=0.20,
        )
        assert abs(result - 0.10) < 1e-9

    def test_embedded_loss_no_additional_drag(self):
        """Embedded loss → no tax drag (max(0, negative) = 0)."""
        result = compute_after_tax_expected_return(
            predicted_return=0.10,
            unrealized_gain_fraction=-0.30,  # embedded loss
            tax_rate=0.20,
        )
        assert abs(result - 0.10) < 1e-9

    def test_embedded_gain_reduces_return(self):
        """Embedded gain creates tax drag that reduces after-tax return."""
        result = compute_after_tax_expected_return(
            predicted_return=0.10,
            unrealized_gain_fraction=0.50,  # 50% gain
            tax_rate=0.20,
        )
        expected = 0.10 - 0.50 * 0.20  # = 0.10 - 0.10 = 0.0
        assert abs(result - expected) < 1e-9

    def test_formula_correct(self):
        """after_tax = predicted - max(0, unrealized_gain × tax_rate)."""
        pred = 0.15
        gain = 0.40
        rate = 0.238  # combined federal + state
        result = compute_after_tax_expected_return(pred, gain, rate)
        expected = pred - max(0.0, gain * rate)
        assert abs(result - expected) < 1e-9

    def test_uses_config_ltcg_rate_by_default(self):
        result1 = compute_after_tax_expected_return(0.10, 0.30, tax_rate=None)
        result2 = compute_after_tax_expected_return(0.10, 0.30, tax_rate=config.LTCG_RATE)
        assert abs(result1 - result2) < 1e-9


class TestSuggestTlhReplacement:
    def test_vti_replacement_is_itot(self):
        assert suggest_tlh_replacement("VTI") == "ITOT"

    def test_voo_replacement_is_ivv(self):
        assert suggest_tlh_replacement("VOO") == "IVV"

    def test_unknown_ticker_returns_none(self):
        assert suggest_tlh_replacement("PGR") is None

    def test_returns_string_for_known_tickers(self):
        for ticker in ["VTI", "BND", "GLD", "VGT"]:
            replacement = suggest_tlh_replacement(ticker)
            assert replacement is None or isinstance(replacement, str)

    def test_all_universe_tickers_have_replacements(self):
        """All ETFs in ETF_BENCHMARK_UNIVERSE should have TLH replacements."""
        for ticker in config.ETF_BENCHMARK_UNIVERSE:
            replacement = suggest_tlh_replacement(ticker)
            assert replacement is not None, (
                f"ETF {ticker} has no TLH replacement defined in TLH_REPLACEMENT_MAP"
            )


class TestWashSaleClearDate:
    def test_adds_31_days(self):
        harvest = date(2024, 3, 15)
        clear = wash_sale_clear_date(harvest)
        assert clear == harvest + timedelta(days=31)

    def test_uses_config_default(self):
        harvest = date(2024, 1, 1)
        clear = wash_sale_clear_date(harvest)
        assert (clear - harvest).days == config.TLH_WASH_SALE_DAYS

    def test_custom_days(self):
        harvest = date(2024, 6, 1)
        clear = wash_sale_clear_date(harvest, wash_sale_days=45)
        assert (clear - harvest).days == 45

    def test_result_after_harvest_date(self):
        harvest = date(2024, 11, 30)
        clear = wash_sale_clear_date(harvest)
        assert clear > harvest
