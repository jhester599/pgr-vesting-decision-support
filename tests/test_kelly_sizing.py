"""
Tests for fractional Kelly sizing in src/portfolio/rebalancer.py (v3.1).

Verifies:
  - Kelly formula: f* = kelly_fraction × predicted_return / variance
  - Max position cap at KELLY_MAX_POSITION (0.30)
  - Zero or negative predicted return → 100% sell (f* ≤ 0)
  - Fallback to legacy logic when variance = 0
  - VestingRecommendation has new prediction_std and kelly_fraction_used fields
"""

from __future__ import annotations

import pytest

import config
from src.portfolio.rebalancer import (
    VestingRecommendation,
    _compute_sell_pct_kelly,
)


class TestComputeSellPctKelly:
    def test_zero_predicted_return_is_full_sell(self):
        sell_pct, _, _ = _compute_sell_pct_kelly(
            predicted_excess_return=0.0,
            prediction_variance=0.01,
        )
        assert sell_pct == 1.0

    def test_negative_predicted_return_is_full_sell(self):
        sell_pct, _, _ = _compute_sell_pct_kelly(
            predicted_excess_return=-0.10,
            prediction_variance=0.01,
        )
        assert sell_pct == 1.0

    def test_positive_return_reduces_sell_pct(self):
        sell_pct, _, _ = _compute_sell_pct_kelly(
            predicted_excess_return=0.20,
            prediction_variance=0.01,
        )
        assert sell_pct < 1.0

    def test_max_position_cap_limits_hold(self):
        """Very high predicted return should be capped at max_position."""
        sell_pct, _, _ = _compute_sell_pct_kelly(
            predicted_excess_return=100.0,    # absurdly high signal
            prediction_variance=0.001,
            max_position=0.30,
        )
        # With cap at 30%, minimum sell is 70%
        assert sell_pct >= 1.0 - 0.30 - 1e-9

    def test_kelly_formula_math(self):
        """f* = kelly_fraction × predicted / variance; sell = 1 - f*."""
        kelly_fraction = 0.25
        predicted = 0.10
        variance = 0.05
        max_pos = 1.0  # no cap

        sell_pct, kf_used, _ = _compute_sell_pct_kelly(
            predicted_excess_return=predicted,
            prediction_variance=variance,
            kelly_fraction=kelly_fraction,
            max_position=max_pos,
        )
        expected_f = kelly_fraction * predicted / variance  # = 0.5
        expected_sell = 1.0 - min(max(expected_f, 0.0), max_pos)
        assert abs(sell_pct - expected_sell) < 1e-9

    def test_returns_kelly_fraction_used(self):
        _, kf_used, _ = _compute_sell_pct_kelly(
            predicted_excess_return=0.10,
            prediction_variance=0.02,
        )
        assert kf_used == config.KELLY_FRACTION

    def test_returns_rationale_string(self):
        _, _, rationale = _compute_sell_pct_kelly(
            predicted_excess_return=0.10,
            prediction_variance=0.02,
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    def test_sell_pct_between_zero_and_one(self):
        for pred in [-0.5, 0.0, 0.05, 0.15, 0.50]:
            sell_pct, _, _ = _compute_sell_pct_kelly(
                predicted_excess_return=pred,
                prediction_variance=0.01,
            )
            assert 0.0 <= sell_pct <= 1.0, (
                f"sell_pct={sell_pct} out of range for predicted={pred}"
            )

    def test_zero_variance_falls_back_to_legacy(self):
        """When variance=0, should use legacy _compute_sell_pct fallback."""
        sell_pct, kf_used, rationale = _compute_sell_pct_kelly(
            predicted_excess_return=0.20,
            prediction_variance=0.0,
        )
        assert 0.0 <= sell_pct <= 1.0
        assert kf_used == 0.0  # no Kelly fraction applied
        assert "fallback" in rationale.lower()

    def test_high_uncertainty_increases_sell(self):
        """Higher variance (more uncertainty) → smaller Kelly position → more selling."""
        sell_low_var, _, _ = _compute_sell_pct_kelly(
            predicted_excess_return=0.10,
            prediction_variance=0.01,
        )
        sell_high_var, _, _ = _compute_sell_pct_kelly(
            predicted_excess_return=0.10,
            prediction_variance=0.10,
        )
        assert sell_high_var >= sell_low_var


class TestVestingRecommendationFields:
    def test_prediction_std_field_exists(self):
        """VestingRecommendation must have prediction_std field."""
        assert hasattr(VestingRecommendation, "__dataclass_fields__")
        assert "prediction_std" in VestingRecommendation.__dataclass_fields__

    def test_kelly_fraction_used_field_exists(self):
        assert "kelly_fraction_used" in VestingRecommendation.__dataclass_fields__

    def test_default_prediction_std_is_zero(self):
        fields = VestingRecommendation.__dataclass_fields__
        assert fields["prediction_std"].default == 0.0

    def test_default_kelly_fraction_used_is_zero(self):
        fields = VestingRecommendation.__dataclass_fields__
        assert fields["kelly_fraction_used"].default == 0.0

    def test_config_kelly_fraction_is_quarter(self):
        assert config.KELLY_FRACTION == 0.25

    def test_config_kelly_max_position(self):
        assert 0.0 < config.KELLY_MAX_POSITION <= 1.0
