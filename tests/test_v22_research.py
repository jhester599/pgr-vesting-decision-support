from __future__ import annotations

from datetime import date

from src.research.v12 import SnapshotSummary, build_shadow_check_lines
from src.research.v22 import v22_promoted_cross_check_spec


def test_v22_promoted_cross_check_spec_defaults_to_v21_winner() -> None:
    spec = v22_promoted_cross_check_spec()
    assert spec.candidate_name == "ensemble_ridge_gbt_v18"
    assert spec.members == ["ridge_lean_v1__v18", "gbt_lean_plus_two__v18"]


def test_build_shadow_check_lines_mentions_v22_promoted_cross_check() -> None:
    cross_check = SnapshotSummary(
        label="cross-check",
        as_of=date(2026, 4, 4),
        candidate_name="ensemble_ridge_gbt_v18",
        policy_name="v21_promoted_cross_check",
        consensus="OUTPERFORM",
        confidence_tier="MODERATE",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        sell_pct=0.5,
        mean_predicted=0.04,
        mean_ic=0.10,
        mean_hit_rate=0.57,
        aggregate_oos_r2=-0.20,
        aggregate_nw_ic=0.13,
    )
    shadow = SnapshotSummary(
        label="shadow",
        as_of=date(2026, 4, 4),
        candidate_name="baseline_historical_mean",
        policy_name="neutral_band_3pct",
        consensus="OUTPERFORM",
        confidence_tier="LOW",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        sell_pct=0.5,
        mean_predicted=0.05,
        mean_ic=-0.01,
        mean_hit_rate=0.62,
        aggregate_oos_r2=-0.16,
        aggregate_nw_ic=-0.01,
    )

    lines = build_shadow_check_lines(cross_check, shadow, active_path="shadow")
    joined = "\n".join(lines)
    assert "| Visible cross-check | `ensemble_ridge_gbt_v18`" in joined
    assert "v22 keeps the simpler diversification-first baseline as the active recommendation layer" in joined
