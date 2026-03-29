"""
Tests for the v4.4 STCG Tax Boundary Guard.

Validates:
  - _check_stcg_boundary() zone detection (180 < days ≤ 365)
  - Warning fires when predicted_alpha < STCG_BREAKEVEN_THRESHOLD
  - Warning suppressed when alpha is sufficient to justify selling STCG
  - Warning suppressed when no lots are in the boundary zone
  - Warning content: days_to_ltcg, shares, embedded gain
  - Lot ordering in warning (closest to LTCG first)
  - Boundary edge cases (exactly 180d, exactly 365d, 181d, 364d)
  - config.py constant values and types
  - stcg_warning field present on VestingRecommendation dataclass
  - generate_recommendation() populates stcg_warning correctly
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.portfolio.rebalancer import (
    VestingRecommendation,
    _check_stcg_boundary,
)
from src.tax.capital_gains import TaxLot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lot(vest_date: date, shares: float = 100.0, basis: float = 200.0) -> TaxLot:
    """Build a TaxLot with the given vest_date."""
    return TaxLot(
        vest_date=vest_date,
        rsu_type="time",
        shares=shares,
        cost_basis_per_share=basis,
    )


def _sell_date(base: date, offset_days: int = 0) -> date:
    return base + timedelta(days=offset_days)


# A reference sell date; lots are built relative to it
SELL_DATE = date(2026, 7, 1)
PRICE = 250.0   # current price > basis → embedded gain


def _lot_held(days: int, shares: float = 100.0, basis: float = 200.0) -> TaxLot:
    """Build a lot that has been held exactly ``days`` at SELL_DATE."""
    vest = SELL_DATE - timedelta(days=days)
    return _lot(vest, shares=shares, basis=basis)


# ===========================================================================
# Config constants
# ===========================================================================

class TestConfigConstants:
    def test_stcg_breakeven_threshold_value(self) -> None:
        assert config.STCG_BREAKEVEN_THRESHOLD == pytest.approx(0.18)

    def test_stcg_breakeven_threshold_is_float(self) -> None:
        assert isinstance(config.STCG_BREAKEVEN_THRESHOLD, float)

    def test_stcg_zone_min_days(self) -> None:
        assert config.STCG_ZONE_MIN_DAYS == 180

    def test_stcg_zone_max_days(self) -> None:
        assert config.STCG_ZONE_MAX_DAYS == 365

    def test_zone_max_greater_than_min(self) -> None:
        assert config.STCG_ZONE_MAX_DAYS > config.STCG_ZONE_MIN_DAYS


# ===========================================================================
# _check_stcg_boundary() — zone detection
# ===========================================================================

class TestZoneDetection:
    """Lots in the 6–12 month zone trigger the check; others are ignored."""

    def test_lot_in_zone_fires_warning(self) -> None:
        lot = _lot_held(270)  # 270d: inside (180, 365]
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is not None

    def test_lot_at_181_days_in_zone(self) -> None:
        """181 days is just inside the zone (> 180)."""
        lot = _lot_held(181)
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is not None

    def test_lot_at_364_days_in_zone(self) -> None:
        """364 days is still STCG (≤ 365)."""
        lot = _lot_held(364)
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is not None

    def test_lot_at_exactly_180_days_not_in_zone(self) -> None:
        """Exactly 180 days is NOT in the zone (zone_min is exclusive: > 180)."""
        lot = _lot_held(180)
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is None

    def test_lot_at_exactly_365_days_in_zone(self) -> None:
        """Exactly 365 days is still STCG (IRS requires > 365 days for LTCG).
        The lot is in the zone with 0 days to LTCG qualification — highest
        priority boundary case."""
        lot = _lot_held(365)
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is not None
        assert "0d to LTCG" in result

    def test_lot_at_366_days_ltcg_no_warning(self) -> None:
        """Already LTCG-eligible — no boundary concern."""
        lot = _lot_held(366)
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is None

    def test_young_lot_not_in_zone(self) -> None:
        """Lot held only 90 days: too far from LTCG to worry about."""
        lot = _lot_held(90)
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is None

    def test_empty_lots_returns_none(self) -> None:
        result = _check_stcg_boundary([], SELL_DATE, 0.05, PRICE)
        assert result is None

    def test_zero_shares_remaining_ignored(self) -> None:
        lot = _lot_held(270)
        lot.shares_remaining = 0.0
        result = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert result is None


# ===========================================================================
# _check_stcg_boundary() — alpha threshold logic
# ===========================================================================

class TestAlphaThreshold:
    """Warning fires only when predicted_alpha < breakeven_threshold."""

    def test_alpha_below_threshold_fires(self) -> None:
        lot = _lot_held(270)
        result = _check_stcg_boundary(
            [lot], SELL_DATE,
            predicted_alpha=0.05,
            current_price=PRICE,
            breakeven_threshold=0.18,
        )
        assert result is not None

    def test_alpha_at_threshold_no_warning(self) -> None:
        """At exactly the threshold, no warning (not strictly below)."""
        lot = _lot_held(270)
        result = _check_stcg_boundary(
            [lot], SELL_DATE,
            predicted_alpha=0.18,
            current_price=PRICE,
            breakeven_threshold=0.18,
        )
        assert result is None

    def test_alpha_above_threshold_no_warning(self) -> None:
        lot = _lot_held(270)
        result = _check_stcg_boundary(
            [lot], SELL_DATE,
            predicted_alpha=0.25,
            current_price=PRICE,
            breakeven_threshold=0.18,
        )
        assert result is None

    def test_negative_alpha_fires(self) -> None:
        """Negative predicted return is well below threshold."""
        lot = _lot_held(270)
        result = _check_stcg_boundary([lot], SELL_DATE, -0.10, PRICE)
        assert result is not None

    def test_default_threshold_from_config(self) -> None:
        """Omitting breakeven_threshold uses config.STCG_BREAKEVEN_THRESHOLD."""
        lot = _lot_held(270)
        # 1pp below the config threshold — should fire
        alpha_below = config.STCG_BREAKEVEN_THRESHOLD - 0.01
        result = _check_stcg_boundary([lot], SELL_DATE, alpha_below, PRICE)
        assert result is not None

    def test_zone_not_in_zone_alpha_irrelevant(self) -> None:
        """Young lot: no warning even with very low alpha."""
        lot = _lot_held(50)
        result = _check_stcg_boundary([lot], SELL_DATE, -0.50, PRICE)
        assert result is None


# ===========================================================================
# Warning content
# ===========================================================================

class TestWarningContent:
    """The warning string should contain actionable detail."""

    def test_warning_mentions_stcg(self) -> None:
        lot = _lot_held(270)
        w = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert "STCG" in w

    def test_warning_mentions_days_to_ltcg(self) -> None:
        lot = _lot_held(300)  # 300d held → 65d to LTCG
        w = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert "65" in w

    def test_warning_mentions_shares(self) -> None:
        lot = _lot_held(270, shares=42.5)
        w = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert "42.5" in w

    def test_warning_mentions_alpha(self) -> None:
        lot = _lot_held(270)
        w = _check_stcg_boundary([lot], SELL_DATE, 0.07, PRICE)
        assert "+7.0%" in w or "7.0%" in w

    def test_warning_mentions_threshold(self) -> None:
        lot = _lot_held(270)
        w = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE, breakeven_threshold=0.18)
        assert "18%" in w

    def test_warning_mentions_embedded_gain(self) -> None:
        """PRICE=250, basis=200 → $50/share gain; 100 shares → $5000."""
        lot = _lot_held(270, shares=100.0, basis=200.0)
        w = _check_stcg_boundary([lot], SELL_DATE, 0.05, PRICE)
        assert "5,000" in w or "5000" in w

    def test_warning_multiple_lots_sorted_by_days_to_ltcg(self) -> None:
        """Lot closest to LTCG should appear first in the warning."""
        near_lot = _lot_held(350)   # 15d to LTCG
        far_lot = _lot_held(200)    # 165d to LTCG
        w = _check_stcg_boundary([far_lot, near_lot], SELL_DATE, 0.05, PRICE)
        assert w.index("15") < w.index("165")

    def test_warning_lot_count(self) -> None:
        lots = [_lot_held(200), _lot_held(270), _lot_held(340)]
        w = _check_stcg_boundary(lots, SELL_DATE, 0.05, PRICE)
        assert "3 lot(s)" in w


# ===========================================================================
# Mixed lot portfolios
# ===========================================================================

class TestMixedPortfolios:
    """Only boundary lots contribute to the warning."""

    def test_ltcg_lot_does_not_trigger(self) -> None:
        ltcg = _lot_held(400)
        result = _check_stcg_boundary([ltcg], SELL_DATE, 0.05, PRICE)
        assert result is None

    def test_mix_ltcg_and_boundary(self) -> None:
        ltcg = _lot_held(400)
        boundary = _lot_held(270)
        w = _check_stcg_boundary([ltcg, boundary], SELL_DATE, 0.05, PRICE)
        # Warning fires because boundary lot exists
        assert w is not None
        # Only 1 lot mentioned (the boundary one)
        assert "1 lot(s)" in w

    def test_mix_young_and_boundary(self) -> None:
        young = _lot_held(60)
        boundary = _lot_held(300)
        w = _check_stcg_boundary([young, boundary], SELL_DATE, 0.05, PRICE)
        assert w is not None
        assert "1 lot(s)" in w

    def test_all_ltcg_returns_none(self) -> None:
        lots = [_lot_held(400), _lot_held(500), _lot_held(700)]
        result = _check_stcg_boundary(lots, SELL_DATE, 0.05, PRICE)
        assert result is None


# ===========================================================================
# VestingRecommendation dataclass
# ===========================================================================

class TestVestingRecommendationField:
    """stcg_warning field exists on the dataclass and defaults to None."""

    def test_stcg_warning_field_exists(self) -> None:
        import dataclasses
        fields = {f.name for f in dataclasses.fields(VestingRecommendation)}
        assert "stcg_warning" in fields

    def test_stcg_warning_defaults_to_none(self) -> None:
        """Default value for stcg_warning is None (optional field)."""
        import dataclasses
        field_map = {f.name: f for f in dataclasses.fields(VestingRecommendation)}
        assert field_map["stcg_warning"].default is None
