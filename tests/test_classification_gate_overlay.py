from __future__ import annotations

from src.models.classification_gate_overlay import (
    build_decision_overlay_frame,
    compute_shadow_gate_overlay,
)


def test_permission_overlay_only_increases_sell_pct() -> None:
    overlay = compute_shadow_gate_overlay(
        live_mode="DEFER-TO-TAX-DEFAULT",
        live_sell_pct=0.50,
        consensus="UNDERPERFORM",
        mean_predicted=-0.05,
        mean_ic=0.08,
        aggregate_oos_r2=0.03,
        classifier_prob_actionable_sell=0.75,
        variant="permission_overlay",
        gate_style="permission_to_deviate",
        threshold=0.70,
    )
    assert overlay.recommended_sell_pct >= 0.50


def test_veto_overlay_can_block_actionable_sell() -> None:
    overlay = compute_shadow_gate_overlay(
        live_mode="ACTIONABLE",
        live_sell_pct=1.00,
        consensus="UNDERPERFORM",
        mean_predicted=-0.06,
        mean_ic=0.08,
        aggregate_oos_r2=0.04,
        classifier_prob_actionable_sell=0.45,
        variant="veto_overlay",
        gate_style="veto_regression_sell",
        threshold=0.60,
    )
    assert overlay.recommendation_mode == "DEFER-TO-TAX-DEFAULT"
    assert overlay.would_change is True


def test_overlay_frame_has_live_and_shadow_rows() -> None:
    frame, overlay = build_decision_overlay_frame(
        live_mode="DEFER-TO-TAX-DEFAULT",
        live_sell_pct=0.50,
        consensus="NEUTRAL",
        mean_predicted=-0.01,
        mean_ic=0.04,
        aggregate_oos_r2=-0.01,
        classifier_prob_actionable_sell=0.51,
        variant="permission_overlay",
        gate_style="permission_to_deviate",
        threshold=0.70,
    )
    assert list(frame["variant"]) == ["live", "shadow_gate"]
    assert overlay.would_change is False
