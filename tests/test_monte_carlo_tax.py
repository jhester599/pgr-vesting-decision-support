"""
Tests for src/tax/monte_carlo.py (Tier 4.5, v35).

Coverage:
  - simulate_gbm_terminal_prices: shape, reproducibility, positivity, zero-vol
  - estimate_annual_vol: deterministic round-trip, minimum length guard
  - run_monte_carlo_tax_analysis: field types, probability bounds, sell-now reference,
    high-vol dominates sell-now less often than low-vol at positive drift
  - _build_mc_sensitivity_lines: renders when mc_analysis present, silent when None
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from src.tax.monte_carlo import (
    MonteCarloResult,
    MonteCarloTaxAnalysis,
    estimate_annual_vol,
    run_monte_carlo_tax_analysis,
    simulate_gbm_terminal_prices,
)


# ---------------------------------------------------------------------------
# simulate_gbm_terminal_prices
# ---------------------------------------------------------------------------

class TestSimulateGbmTerminalPrices:
    def test_output_shape(self):
        prices = simulate_gbm_terminal_prices(100.0, 0.08, 0.20, 366, 500)
        assert prices.shape == (500,)

    def test_all_positive(self):
        """GBM terminal prices must always be positive (log-normal distribution)."""
        prices = simulate_gbm_terminal_prices(50.0, -0.30, 0.50, 366, 1000, seed=1)
        assert np.all(prices > 0)

    def test_reproducible_with_seed(self):
        a = simulate_gbm_terminal_prices(100.0, 0.05, 0.20, 365, 100, seed=7)
        b = simulate_gbm_terminal_prices(100.0, 0.05, 0.20, 365, 100, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = simulate_gbm_terminal_prices(100.0, 0.05, 0.20, 365, 100, seed=1)
        b = simulate_gbm_terminal_prices(100.0, 0.05, 0.20, 365, 100, seed=2)
        assert not np.array_equal(a, b)

    def test_zero_volatility_returns_deterministic(self):
        """With vol=0, all paths should be identical (S0 * exp(mu*T))."""
        s0, mu, days = 100.0, 0.10, 365
        prices = simulate_gbm_terminal_prices(s0, mu, 0.0, days, 50, seed=0)
        expected = s0 * math.exp(mu * days / 365.0)
        np.testing.assert_allclose(prices, expected, rtol=1e-9)

    def test_negative_starting_price_raises(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_gbm_terminal_prices(-1.0, 0.0, 0.2, 365, 10)

    def test_zero_starting_price_raises(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_gbm_terminal_prices(0.0, 0.0, 0.2, 365, 10)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="[Vv]olatility"):
            simulate_gbm_terminal_prices(100.0, 0.0, -0.1, 365, 10)

    def test_zero_paths_raises(self):
        with pytest.raises(ValueError, match="n_paths"):
            simulate_gbm_terminal_prices(100.0, 0.0, 0.2, 365, 0)

    def test_positive_drift_mean_above_s0(self):
        """E[S(T)] = S0 * exp(mu*T) for GBM; with large n, sample mean should converge."""
        s0, mu, sigma, days = 100.0, 0.20, 0.25, 365
        prices = simulate_gbm_terminal_prices(s0, mu, sigma, days, n_paths=20_000, seed=42)
        expected_mean = s0 * math.exp(mu * days / 365.0)
        # Allow 3% tolerance given sampling noise at N=20_000
        assert abs(prices.mean() - expected_mean) / expected_mean < 0.03


# ---------------------------------------------------------------------------
# estimate_annual_vol
# ---------------------------------------------------------------------------

class TestEstimateAnnualVol:
    def test_constant_prices_zero_vol(self):
        """Constant price series has zero log-return variance."""
        prices = [100.0] * 50
        vol = estimate_annual_vol(prices)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_known_daily_vol(self):
        """Manually construct a daily vol series and verify annualisation."""
        rng = np.random.default_rng(0)
        daily_vol = 0.01
        log_rets = rng.normal(0.0, daily_vol, 500)
        # Reconstruct prices from log-returns
        prices = 100.0 * np.exp(np.cumsum(np.insert(log_rets, 0, 0.0)))
        estimated = estimate_annual_vol(prices)
        expected = daily_vol * math.sqrt(252)
        assert abs(estimated - expected) < 0.005  # within 50bp of target

    def test_minimum_length_raises(self):
        with pytest.raises(ValueError):
            estimate_annual_vol([100.0])  # only 1 price

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            estimate_annual_vol([])

    def test_returns_positive_float(self):
        prices = [100.0, 102.0, 99.0, 101.5, 103.0]
        vol = estimate_annual_vol(prices)
        assert isinstance(vol, float)
        assert vol > 0.0


# ---------------------------------------------------------------------------
# run_monte_carlo_tax_analysis
# ---------------------------------------------------------------------------

class TestRunMonteCarloTaxAnalysis:
    def _run(self, **overrides):
        defaults = dict(
            current_price=200.0,
            cost_basis_per_share=150.0,
            shares=100.0,
            annual_vol=0.20,
            annual_drift=0.08,
            n_paths=500,
            seed=42,
        )
        defaults.update(overrides)
        return run_monte_carlo_tax_analysis(**defaults)

    def test_returns_correct_type(self):
        result = self._run()
        assert isinstance(result, MonteCarloTaxAnalysis)
        assert isinstance(result.hold_ltcg, MonteCarloResult)

    def test_sell_now_reference_correct(self):
        """sell_now_net should match hand-computed STCG net proceeds."""
        import config
        result = self._run(current_price=200.0, cost_basis_per_share=150.0, shares=100.0)
        gain = (200.0 - 150.0) * 100.0
        expected_tax = gain * config.STCG_RATE
        expected_net = 200.0 * 100.0 - expected_tax
        assert result.sell_now_net == pytest.approx(expected_net, rel=1e-9)

    def test_sell_now_net_positive(self):
        result = self._run()
        assert result.sell_now_net > 0.0

    def test_probabilities_in_unit_interval(self):
        result = self._run()
        mc = result.hold_ltcg
        assert 0.0 <= mc.prob_beats_sell_now <= 1.0
        assert 0.0 <= mc.prob_positive_gain <= 1.0

    def test_percentile_ordering(self):
        result = self._run()
        mc = result.hold_ltcg
        assert mc.net_proceeds_p10 <= mc.net_proceeds_p25
        assert mc.net_proceeds_p25 <= mc.net_proceeds_p50
        assert mc.net_proceeds_p50 <= mc.net_proceeds_p75
        assert mc.net_proceeds_p75 <= mc.net_proceeds_p90

    def test_mean_between_p10_and_p90(self):
        result = self._run(n_paths=2000)
        mc = result.hold_ltcg
        assert mc.net_proceeds_p10 < mc.net_proceeds_mean < mc.net_proceeds_p90

    def test_n_paths_stored(self):
        result = self._run(n_paths=300)
        assert result.hold_ltcg.n_paths == 300

    def test_horizon_days_stored(self):
        result = self._run(horizon_days=366)
        assert result.horizon_days == 366

    def test_high_drift_beats_sell_now_more_often(self):
        """Strong positive drift → HOLD_TO_LTCG should beat sell-now more often."""
        low_drift = self._run(annual_drift=0.0, n_paths=2000)
        high_drift = self._run(annual_drift=0.40, n_paths=2000)
        assert high_drift.hold_ltcg.prob_beats_sell_now > low_drift.hold_ltcg.prob_beats_sell_now

    def test_zero_vol_prob_is_zero_or_one(self):
        """At zero vol the outcome is deterministic — prob should be 0 or 1."""
        result = self._run(annual_vol=0.0, annual_drift=0.08, n_paths=100)
        prob = result.hold_ltcg.prob_beats_sell_now
        assert prob == pytest.approx(0.0, abs=1e-9) or prob == pytest.approx(1.0, abs=1e-9)

    def test_cost_basis_above_price_still_runs(self):
        """Lots acquired above current price (unrealised loss) should not raise."""
        result = self._run(current_price=100.0, cost_basis_per_share=200.0, shares=50.0)
        assert result.sell_now_net > 0  # gross proceeds still positive


# ---------------------------------------------------------------------------
# _build_mc_sensitivity_lines (via decision_rendering)
# ---------------------------------------------------------------------------

class TestBuildMcSensitivityLines:
    def test_none_mc_returns_empty(self):
        from src.reporting.decision_rendering import _build_mc_sensitivity_lines
        assert _build_mc_sensitivity_lines(None) == []

    def test_renders_with_mc_analysis(self):
        from src.reporting.decision_rendering import _build_mc_sensitivity_lines
        mc = run_monte_carlo_tax_analysis(
            current_price=200.0,
            cost_basis_per_share=150.0,
            shares=100.0,
            annual_vol=0.20,
            annual_drift=0.08,
            n_paths=200,
            seed=0,
        )
        lines = _build_mc_sensitivity_lines(mc)
        combined = "\n".join(lines)
        assert "Monte Carlo" in combined
        assert "HOLD_TO_LTCG" in combined
        assert "Sell Now" in combined or "sell-now" in combined.lower() or "Sell Now" in combined
        # Should include a probability line
        assert "%" in combined
        # Percentile table rows
        assert "median" in combined.lower() or "P50" in combined

    def test_renders_n_paths_in_output(self):
        from src.reporting.decision_rendering import _build_mc_sensitivity_lines
        mc = run_monte_carlo_tax_analysis(
            current_price=180.0,
            cost_basis_per_share=120.0,
            shares=50.0,
            annual_vol=0.25,
            annual_drift=0.10,
            n_paths=750,
        )
        lines = "\n".join(_build_mc_sensitivity_lines(mc))
        assert "750" in lines
