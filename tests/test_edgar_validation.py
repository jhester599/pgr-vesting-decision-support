"""
Tests for v7.2 EDGAR 8-K Parser Hardening.

Tests _validate_parsed_record() directly (importable from edgar_8k_fetcher),
and the most-complete-wins deduplication logic via a small helper that mirrors
the inline _completeness function.

Tests (12):
  1.  test_validate_cr_consistent_with_components
  2.  test_validate_cr_inconsistent_nullifies
  3.  test_validate_cr_missing_components_no_check
  4.  test_validate_pif_below_floor
  5.  test_validate_pif_above_floor
  6.  test_validate_eps_out_of_range_high
  7.  test_validate_eps_out_of_range_low
  8.  test_validate_eps_in_range
  9.  test_validate_npw_segment_warning
  10. test_dedup_prefers_more_complete
  11. test_dedup_same_completeness_uses_last
  12. test_validation_rejects_after_both_nullified
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.edgar_8k_fetcher import _validate_parsed_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(**kwargs) -> dict:
    """Build a minimal record with all validatable fields defaulted to None."""
    base = {
        "month_end": "2026-01-31",
        "combined_ratio": None,
        "loss_lae_ratio": None,
        "expense_ratio": None,
        "net_premiums_written": None,
        "npw_agency": None,
        "npw_direct": None,
        "npw_commercial": None,
        "npw_property": None,
        "pif_total": None,
        "eps_basic": None,
    }
    base.update(kwargs)
    return base


def _completeness(rec: dict) -> int:
    """Mirror of the inline _completeness() in fetch_and_upsert()."""
    return sum(1 for k, v in rec.items() if v is not None and k != "month_end")


def _dedup_most_complete(records: list[dict]) -> list[dict]:
    """Mirror of the v7.2 dedup block in fetch_and_upsert()."""
    seen: dict[str, dict] = {}
    for rec in records:
        me = rec["month_end"]
        if me not in seen or _completeness(rec) >= _completeness(seen[me]):
            seen[me] = rec
    return sorted(seen.values(), key=lambda r: r["month_end"])


# ---------------------------------------------------------------------------
# Enhancement 1: _validate_parsed_record tests (1–9)
# ---------------------------------------------------------------------------

class TestValidateParsedRecord:

    def test_validate_cr_consistent_with_components(self):
        """CR=95, LR=65, ER=30 → delta=0 ≤ 5pp → CR preserved."""
        rec = _rec(combined_ratio=95.0, loss_lae_ratio=65.0, expense_ratio=30.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC001")
        assert result["combined_ratio"] == 95.0

    def test_validate_cr_inconsistent_nullifies(self):
        """CR=95, LR=65, ER=40 → delta=10 > 5pp → CR set to None."""
        rec = _rec(combined_ratio=95.0, loss_lae_ratio=65.0, expense_ratio=40.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC002")
        assert result["combined_ratio"] is None

    def test_validate_cr_missing_components_no_check(self):
        """CR present but LR=None → skip ratio check, CR unchanged."""
        rec = _rec(combined_ratio=95.0, loss_lae_ratio=None, expense_ratio=30.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC003")
        assert result["combined_ratio"] == 95.0

    def test_validate_pif_below_floor(self):
        """pif_total=5,000 < 10,000 canonical floor → set to None."""
        rec = _rec(pif_total=5_000.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC004")
        assert result["pif_total"] is None

    def test_validate_pif_above_floor(self):
        """pif_total=20,000 (thousands) ≥ 10,000 floor → preserved."""
        rec = _rec(pif_total=20_000.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC005")
        assert result["pif_total"] == 20_000.0

    def test_validate_eps_out_of_range_high(self):
        """eps_basic=25.0 > 15.0 → set to None."""
        rec = _rec(eps_basic=25.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC006")
        assert result["eps_basic"] is None

    def test_validate_eps_out_of_range_low(self):
        """eps_basic=-10.0 < -5.0 → set to None."""
        rec = _rec(eps_basic=-10.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC007")
        assert result["eps_basic"] is None

    def test_validate_eps_in_range(self):
        """eps_basic=3.50 within [-5, 15] → preserved."""
        rec = _rec(eps_basic=3.50)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC008")
        assert result["eps_basic"] == 3.50

    def test_validate_npw_segment_warning(self, caplog):
        """NPW total < 90% of sum of parts → warning logged; no nullification."""
        rec = _rec(
            net_premiums_written=1_000.0,
            npw_agency=600.0,
            npw_direct=600.0,   # sum = 1200 → total (1000) < 1200 * 0.9 = 1080
        )
        with caplog.at_level(logging.WARNING, logger="scripts.edgar_8k_fetcher"):
            result = _validate_parsed_record(rec, "2026-01-20", "ACC009")

        assert result["net_premiums_written"] == 1_000.0  # not nullified
        assert any("NPW_total" in m for m in caplog.messages), (
            "Expected NPW validation warning in logs"
        )

    def test_validate_cr_boundary_exactly_5pp_not_nullified(self):
        """CR=95, LR=65, ER=35 → delta=5.0, not > 5 → CR preserved."""
        rec = _rec(combined_ratio=95.0, loss_lae_ratio=65.0, expense_ratio=35.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC010")
        assert result["combined_ratio"] == 95.0

    def test_validate_pif_at_floor_boundary(self):
        """pif_total=10,000 exactly → not < 10,000 → preserved."""
        rec = _rec(pif_total=10_000.0)
        result = _validate_parsed_record(rec, "2026-01-20", "ACC011")
        assert result["pif_total"] == 10_000.0


# ---------------------------------------------------------------------------
# Enhancement 2: most-complete dedup (10–11)
# ---------------------------------------------------------------------------

class TestMostCompleteDedup:

    def test_dedup_prefers_more_complete(self):
        """Two records for same month_end: 5-field record loses to 10-field."""
        # sparse: 3 non-null fields
        sparse = _rec(
            month_end="2026-01-31",
            combined_ratio=95.0,
            pif_total=20_000_000.0,
            eps_basic=1.50,
        )
        # dense: 5 non-null fields
        dense = _rec(
            month_end="2026-01-31",
            combined_ratio=94.5,
            pif_total=21_000_000.0,
            eps_basic=1.55,
            net_premiums_written=5_000_000.0,
            loss_lae_ratio=64.0,
        )
        # sparse inserted first
        result = _dedup_most_complete([sparse, dense])
        assert len(result) == 1
        assert result[0]["combined_ratio"] == 94.5  # dense record wins

    def test_dedup_same_completeness_uses_last(self):
        """Two records with equal completeness → last one wins (iteration order)."""
        first = _rec(month_end="2026-02-28", combined_ratio=93.0, pif_total=20_000_000.0)
        last  = _rec(month_end="2026-02-28", combined_ratio=94.0, pif_total=21_000_000.0)
        result = _dedup_most_complete([first, last])
        assert len(result) == 1
        assert result[0]["combined_ratio"] == 94.0  # last wins on tie


# ---------------------------------------------------------------------------
# Enhancement 1+wiring: skip on both-nullified (12)
# ---------------------------------------------------------------------------

class TestBothNullifiedSkip:

    def test_validation_rejects_after_both_nullified(self):
        """After validation nullifies CR and PIF, the caller (fetch_and_upsert)
        should skip the record.  We verify the validate function returns a record
        where both are None, so the wired-in check fires correctly."""
        # Construct a record that will have CR nullified (inconsistent components)
        # and PIF nullified (below floor).
        rec = _rec(
            combined_ratio=95.0,
            loss_lae_ratio=65.0,
            expense_ratio=40.0,   # delta=10 > 5 → CR nullified
            pif_total=5_000.0,    # < 10,000 → PIF nullified
        )
        result = _validate_parsed_record(rec, "2026-01-20", "ACC_BOTH")

        assert result["combined_ratio"] is None, "CR should have been nullified"
        assert result["pif_total"] is None, "PIF should have been nullified"
        # Confirm the caller's skip condition would trigger:
        assert result["combined_ratio"] is None and result["pif_total"] is None

    def test_validate_accepts_both_legacy_and_current_pif_scales(self):
        """Legacy raw counts and current thousand-scale counts both survive after normalization."""
        legacy = _validate_parsed_record(_rec(pif_total=18_500.0), "2026-01-20", "ACC_LEGACY")
        current = _validate_parsed_record(_rec(pif_total=29_575.0), "2026-01-20", "ACC_CURRENT")
        assert legacy["pif_total"] == 18_500.0
        assert current["pif_total"] == 29_575.0
