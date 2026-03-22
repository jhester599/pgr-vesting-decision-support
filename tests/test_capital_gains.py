"""
Tests for src/tax/capital_gains.py

Verifies:
  1. LTCG rate applied for lots held > 365 days.
  2. STCG rate applied for lots held <= 365 days.
  3. Net proceeds < gross proceeds (tax liability always positive on gains).
  4. Loss lots generate a tax benefit (negative tax liability).
  5. Lot selection priority: losses first, then LTCG, then STCG.
  6. ValueError raised when trying to sell more shares than available.
  7. Position summary accurately tracks LTCG vs. STCG split.
"""

import pytest
from datetime import date, timedelta
import math

import config
from src.tax.capital_gains import (
    TaxLot,
    optimize_sale,
    compute_position_summary,
    TotalSaleResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SELL_DATE = date(2026, 1, 15)
CURRENT_PRICE = 200.0

@pytest.fixture()
def ltcg_lot() -> TaxLot:
    """100 shares, $100 basis, held 730 days (well over 365)."""
    return TaxLot(
        vest_date=SELL_DATE - timedelta(days=730),
        rsu_type="time",
        shares=100.0,
        cost_basis_per_share=100.0,
    )


@pytest.fixture()
def stcg_lot() -> TaxLot:
    """100 shares, $100 basis, held only 180 days (under 365)."""
    return TaxLot(
        vest_date=SELL_DATE - timedelta(days=180),
        rsu_type="performance",
        shares=100.0,
        cost_basis_per_share=100.0,
    )


@pytest.fixture()
def loss_lot() -> TaxLot:
    """50 shares, $250 basis (above current price of $200 = embedded loss)."""
    return TaxLot(
        vest_date=SELL_DATE - timedelta(days=730),  # LTCG eligible but at a loss
        rsu_type="time",
        shares=50.0,
        cost_basis_per_share=250.0,
    )


# ---------------------------------------------------------------------------
# LTCG / STCG rate application
# ---------------------------------------------------------------------------

class TestTaxRateApplication:
    def test_ltcg_rate_applied_for_long_held_lot(self, ltcg_lot):
        result = optimize_sale(
            [ltcg_lot], shares_to_sell=10.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
            ltcg_rate=0.20, stcg_rate=0.37,
        )
        # Expected: gain = 10 * (200 - 100) = 1000; tax = 1000 * 0.20 = 200
        assert math.isclose(result.total_tax, 200.0, rel_tol=1e-6), (
            f"Expected LTCG tax of $200, got ${result.total_tax:.2f}"
        )
        assert result.lots[0].holding_type == "LTCG"

    def test_stcg_rate_applied_for_short_held_lot(self, stcg_lot):
        result = optimize_sale(
            [stcg_lot], shares_to_sell=10.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
            ltcg_rate=0.20, stcg_rate=0.37,
        )
        # Expected: gain = 10 * (200 - 100) = 1000; tax = 1000 * 0.37 = 370
        assert math.isclose(result.total_tax, 370.0, rel_tol=1e-6), (
            f"Expected STCG tax of $370, got ${result.total_tax:.2f}"
        )
        assert result.lots[0].holding_type == "STCG"

    def test_ltcg_tax_lower_than_stcg_tax(self, ltcg_lot, stcg_lot):
        """LTCG tax must always be less than STCG tax for the same gain."""
        ltcg_result = optimize_sale(
            [ltcg_lot], shares_to_sell=10.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        stcg_result = optimize_sale(
            [stcg_lot], shares_to_sell=10.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        assert ltcg_result.total_tax < stcg_result.total_tax, (
            "LTCG tax should be less than STCG tax for equivalent gains."
        )

    def test_net_less_than_gross_on_gain(self, ltcg_lot):
        result = optimize_sale(
            [ltcg_lot], shares_to_sell=50.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        assert result.total_net < result.total_gross, (
            "Net proceeds must be less than gross proceeds when there is a gain."
        )

    def test_loss_lot_generates_tax_benefit(self, loss_lot):
        """Selling a loss lot should produce negative tax liability (benefit)."""
        result = optimize_sale(
            [loss_lot], shares_to_sell=10.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
            ltcg_rate=0.20, stcg_rate=0.37,
        )
        # Gain = 10 * (200 - 250) = -500; tax_benefit = -500 * 0.20 = -100
        assert result.total_tax < 0, (
            f"Loss lot should generate negative tax (benefit), got ${result.total_tax:.2f}"
        )
        assert math.isclose(result.total_tax, -100.0, rel_tol=1e-6), (
            f"Expected tax benefit of -$100, got ${result.total_tax:.2f}"
        )
        assert result.lots[0].holding_type == "LOSS"


# ---------------------------------------------------------------------------
# Lot selection priority
# ---------------------------------------------------------------------------

class TestLotSelectionPriority:
    def test_loss_lot_sold_before_stcg_lot(self, loss_lot, stcg_lot):
        """
        Loss lots must be selected before STCG gain lots.
        We sell only enough to consume the loss lot first.
        """
        shares_to_sell = loss_lot.shares  # Exactly the loss lot size
        result = optimize_sale(
            [stcg_lot, loss_lot],    # Order: stcg first to test sorting
            shares_to_sell=shares_to_sell,
            sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        # All sold shares should come from the loss lot
        assert len(result.lots) == 1
        assert result.lots[0].holding_type == "LOSS"

    def test_ltcg_lot_sold_before_stcg_lot(self, ltcg_lot, stcg_lot):
        """
        LTCG gain lots must be selected before STCG gain lots.
        """
        shares_to_sell = ltcg_lot.shares  # Exactly the LTCG lot size
        result = optimize_sale(
            [stcg_lot, ltcg_lot],  # Order reversed to test sorting
            shares_to_sell=shares_to_sell,
            sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        assert len(result.lots) == 1
        assert result.lots[0].holding_type == "LTCG"

    def test_all_three_priorities_in_order(self, loss_lot, ltcg_lot, stcg_lot):
        """
        Sell all 250 shares (50 loss + 100 LTCG + 100 STCG).
        Result order should be: LOSS, LTCG, STCG.
        """
        total = loss_lot.shares + ltcg_lot.shares + stcg_lot.shares
        result = optimize_sale(
            [stcg_lot, loss_lot, ltcg_lot],  # Random order input
            shares_to_sell=total,
            sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        holding_types = [r.holding_type for r in result.lots]
        assert holding_types[0] == "LOSS", "Loss lot must be sold first."
        assert holding_types[1] == "LTCG", "LTCG lot must be sold second."
        assert holding_types[2] == "STCG", "STCG lot must be sold last."


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_raises_on_oversell(self, ltcg_lot):
        """Attempting to sell more shares than available must raise ValueError."""
        with pytest.raises(ValueError, match="only .* available"):
            optimize_sale(
                [ltcg_lot],
                shares_to_sell=ltcg_lot.shares + 1.0,
                sale_price=CURRENT_PRICE,
                sell_date=SELL_DATE,
            )

    def test_sell_zero_shares_returns_empty(self, ltcg_lot):
        result = optimize_sale(
            [ltcg_lot], shares_to_sell=0.0, sale_price=CURRENT_PRICE,
            sell_date=SELL_DATE,
        )
        assert result.total_gross == 0.0
        assert result.total_tax == 0.0
        assert result.total_net == 0.0


# ---------------------------------------------------------------------------
# compute_position_summary
# ---------------------------------------------------------------------------

class TestPositionSummary:
    def test_total_value(self, ltcg_lot, stcg_lot):
        lots = [ltcg_lot, stcg_lot]  # 200 shares total
        summary = compute_position_summary(lots, CURRENT_PRICE, SELL_DATE)
        assert math.isclose(summary["total_shares"], 200.0)
        assert math.isclose(summary["current_value"], 200.0 * CURRENT_PRICE)

    def test_ltcg_stcg_split(self, ltcg_lot, stcg_lot):
        lots = [ltcg_lot, stcg_lot]
        summary = compute_position_summary(lots, CURRENT_PRICE, SELL_DATE)
        assert math.isclose(summary["ltcg_eligible_shares"], 100.0)
        assert math.isclose(summary["stcg_shares"], 100.0)
        assert math.isclose(summary["ltcg_pct_of_position"], 0.5)

    def test_unrealized_gain_correct(self, ltcg_lot):
        """100 shares at $100 basis, current $200: gain = $10,000."""
        summary = compute_position_summary([ltcg_lot], CURRENT_PRICE, SELL_DATE)
        assert math.isclose(summary["total_unrealized_gain"], 10_000.0)

    def test_is_ltcg_eligible_boundary(self):
        """Exactly 365 days should NOT qualify (must be > 365)."""
        lot_366 = TaxLot(
            vest_date=SELL_DATE - timedelta(days=366),
            rsu_type="time", shares=100.0, cost_basis_per_share=100.0,
        )
        lot_365 = TaxLot(
            vest_date=SELL_DATE - timedelta(days=365),
            rsu_type="time", shares=100.0, cost_basis_per_share=100.0,
        )
        assert lot_366.is_ltcg_eligible(SELL_DATE) is True
        assert lot_365.is_ltcg_eligible(SELL_DATE) is False
