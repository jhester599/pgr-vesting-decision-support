"""
Tests for v7.1 Three-Scenario Tax Framework.

Covers:
  - compute_stcg_ltcg_breakeven()  (4 tests)
  - compute_three_scenarios() Scenario A  (4 tests)
  - compute_three_scenarios() Scenario B  (4 tests)
  - compute_three_scenarios() Scenario C  (3 tests)
  - Recommendation logic  (3 tests)
  - Integration  (1 test)
  Total: 19 tests
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from src.tax.capital_gains import (
    TaxScenario,
    ThreeScenarioResult,
    compute_stcg_ltcg_breakeven,
    compute_three_scenarios,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_kwargs(**overrides) -> dict:
    """Return a standard set of kwargs for compute_three_scenarios."""
    defaults = dict(
        vest_date=date(2026, 7, 17),
        rsu_type="performance",
        shares=500.0,
        cost_basis_per_share=116.08,
        current_price=270.0,
        predicted_6m_return=0.04,
        predicted_12m_return=0.07,
        prob_outperform_6m=0.58,
        prob_outperform_12m=0.60,
        stcg_rate=0.37,
        ltcg_rate=0.20,
    )
    defaults.update(overrides)
    return defaults


def _call(**overrides) -> ThreeScenarioResult:
    return compute_three_scenarios(**_base_kwargs(**overrides))


# ---------------------------------------------------------------------------
# 1–4: Breakeven tests
# ---------------------------------------------------------------------------

class TestBreakeven:
    def test_breakeven_default_rates(self):
        """With STCG=0.37, LTCG=0.20, breakeven ≈ 0.2125."""
        result = compute_stcg_ltcg_breakeven(stcg_rate=0.37, ltcg_rate=0.20)
        assert math.isclose(result, 0.17 / 0.80, rel_tol=1e-9)
        assert math.isclose(result, 0.2125, rel_tol=1e-4)

    def test_breakeven_zero_rate_difference(self):
        """STCG == LTCG → breakeven = 0.0."""
        result = compute_stcg_ltcg_breakeven(stcg_rate=0.20, ltcg_rate=0.20)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_breakeven_custom_rates(self):
        """STCG=0.40, LTCG=0.15 → verify formula manually."""
        # (0.40 - 0.15) / (1.0 - 0.15) = 0.25 / 0.85
        expected = 0.25 / 0.85
        result = compute_stcg_ltcg_breakeven(stcg_rate=0.40, ltcg_rate=0.15)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_breakeven_is_positive(self):
        """Breakeven is always > 0 when STCG > LTCG."""
        result = compute_stcg_ltcg_breakeven(stcg_rate=0.37, ltcg_rate=0.20)
        assert result > 0.0


# ---------------------------------------------------------------------------
# 5–8: Scenario A tests
# ---------------------------------------------------------------------------

class TestScenarioA:
    def _get_a(self, **overrides) -> TaxScenario:
        return _call(**overrides).scenarios[0]

    def test_scenario_a_immediate_gain(self):
        """current_price > cost_basis → positive tax_liability."""
        a = self._get_a(current_price=270.0, cost_basis_per_share=116.08)
        gain = (270.0 - 116.08) * 500.0
        assert a.tax_liability == pytest.approx(gain * 0.37, rel=1e-9)
        assert a.tax_liability > 0

    def test_scenario_a_immediate_loss(self):
        """current_price < cost_basis → negative tax_liability (tax benefit)."""
        a = self._get_a(current_price=100.0, cost_basis_per_share=116.08)
        loss = (100.0 - 116.08) * 500.0  # negative
        assert a.tax_liability == pytest.approx(loss * 0.37, rel=1e-9)
        assert a.tax_liability < 0

    def test_scenario_a_probability_is_one(self):
        """Scenario A always has probability = 1.0 (certain outcome)."""
        a = self._get_a()
        assert a.probability == pytest.approx(1.0)

    def test_scenario_a_net_equals_gross_minus_tax(self):
        """net_proceeds = gross_proceeds - tax_liability."""
        a = self._get_a()
        assert a.net_proceeds == pytest.approx(a.gross_proceeds - a.tax_liability, rel=1e-9)


# ---------------------------------------------------------------------------
# 9–12: Scenario B tests
# ---------------------------------------------------------------------------

class TestScenarioB:
    def _get_b(self, **overrides) -> TaxScenario:
        return _call(**overrides).scenarios[1]

    def test_scenario_b_uses_ltcg_rate(self):
        """Scenario B tax_rate field equals the LTCG rate."""
        b = self._get_b(ltcg_rate=0.20)
        assert b.tax_rate == pytest.approx(0.20)

    def test_scenario_b_holding_period_366_days(self):
        """Scenario B holding_period_days is always 366."""
        b = self._get_b()
        assert b.holding_period_days == 366

    def test_scenario_b_positive_predicted_return(self):
        """When model predicts +10%, Scenario B gross > Scenario A gross."""
        result = _call(predicted_12m_return=0.10, current_price=270.0)
        a = result.scenarios[0]
        b = result.scenarios[1]
        assert b.gross_proceeds > a.gross_proceeds

    def test_scenario_b_uses_12m_prediction(self):
        """predicted_price reflects predicted_12m_return."""
        b = self._get_b(current_price=270.0, predicted_12m_return=0.10)
        assert b.predicted_price == pytest.approx(270.0 * 1.10, rel=1e-9)


# ---------------------------------------------------------------------------
# 13–15: Scenario C tests
# ---------------------------------------------------------------------------

class TestScenarioC:
    def _get_c(self, **overrides) -> TaxScenario:
        return _call(**overrides).scenarios[2]

    def test_scenario_c_loss_harvest_benefit(self):
        """When predicted price is below cost basis, net_proceeds > gross_proceeds
        (negative tax_liability = benefit is subtracted, increasing net proceeds).

        Setup: cost_basis=250, current=270, predicted_6m=-0.15 →
        predicted_price=229.5 < 250 → real capital loss.
        """
        c = self._get_c(
            predicted_6m_return=-0.15,
            prob_outperform_6m=0.35,
            cost_basis_per_share=250.0,  # ensures predicted_price < cost_basis
        )
        # tax_liability is negative (benefit); net = gross - negative = gross + |benefit|
        assert c.net_proceeds > c.gross_proceeds

    def test_scenario_c_degenerate_when_positive(self):
        """When predicted_6m_return > 0, Scenario C probability = 0.0."""
        c = self._get_c(predicted_6m_return=0.04)
        assert c.probability == pytest.approx(0.0)

    def test_scenario_c_tax_benefit_uses_stcg_rate(self):
        """Loss offsets at the higher STCG rate.

        Uses cost_basis=250 so predicted_price (229.5) < cost_basis, producing
        a real capital loss.  tax_liability = loss_gain × stcg_rate (negative).
        """
        c = self._get_c(
            predicted_6m_return=-0.15,
            current_price=270.0,
            cost_basis_per_share=250.0,  # ensures real capital loss
            shares=500.0,
            stcg_rate=0.37,
        )
        predicted_loss_price = 270.0 * (1 - 0.15)  # = 229.5
        loss = (predicted_loss_price - 250.0) * 500.0  # negative
        expected_tax = loss * 0.37  # negative (benefit)
        assert c.tax_liability == pytest.approx(expected_tax, rel=1e-9)


# ---------------------------------------------------------------------------
# 16–18: Recommendation tests
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommended_scenario_is_highest_utility(self):
        """recommended_scenario is the one with max prob × net_proceeds."""
        result = _call()
        a, b, c = result.scenarios
        utility_a = a.probability * a.net_proceeds
        utility_b = b.probability * b.net_proceeds
        # C is degenerate here (positive predicted return)
        best_label = max(
            [("SELL_NOW_STCG", utility_a), ("HOLD_TO_LTCG", utility_b)],
            key=lambda x: x[1],
        )[0]
        assert result.recommended_scenario == best_label

    def test_always_three_scenarios(self):
        """len(result.scenarios) == 3 always."""
        for predicted_6m in [-0.20, 0.0, 0.10]:
            result = _call(predicted_6m_return=predicted_6m)
            assert len(result.scenarios) == 3

    def test_stcg_ltcg_breakeven_stored(self):
        """result.stcg_ltcg_breakeven matches compute_stcg_ltcg_breakeven()."""
        result = _call(stcg_rate=0.37, ltcg_rate=0.20)
        expected = compute_stcg_ltcg_breakeven(0.37, 0.20)
        assert result.stcg_ltcg_breakeven == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 19: Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_three_scenario_with_real_lot_data(self):
        """Real-world lot (July 2026 vest, 500 shares at $116.08).

        Verify:
          - Exactly 3 scenarios
          - All scenarios have positive gross_proceeds (shares > 0)
          - net_proceeds = gross_proceeds - tax_liability for each scenario
          - days_to_ltcg == 366
          - recommended_scenario is one of the three valid labels
        """
        result = compute_three_scenarios(
            vest_date=date(2026, 7, 17),
            rsu_type="performance",
            shares=500.0,
            cost_basis_per_share=116.08,
            current_price=270.0,
            predicted_6m_return=0.04,
            predicted_12m_return=0.07,
            prob_outperform_6m=0.58,
            prob_outperform_12m=0.60,
            stcg_rate=0.37,
            ltcg_rate=0.20,
        )

        assert len(result.scenarios) == 3

        valid_labels = {"SELL_NOW_STCG", "HOLD_TO_LTCG", "HOLD_FOR_LOSS"}
        assert result.recommended_scenario in valid_labels

        assert result.days_to_ltcg == 366

        for scenario in result.scenarios:
            assert scenario.gross_proceeds >= 0, (
                f"Scenario {scenario.label}: negative gross_proceeds"
            )
            assert scenario.net_proceeds == pytest.approx(
                scenario.gross_proceeds - scenario.tax_liability, rel=1e-9
            ), f"Scenario {scenario.label}: net != gross - tax"

        # With positive predicted return, Scenario C should be degenerate.
        c = result.scenarios[2]
        assert c.probability == pytest.approx(0.0)

        # Breakeven sanity check.
        assert result.stcg_ltcg_breakeven == pytest.approx(0.2125, rel=1e-3)
