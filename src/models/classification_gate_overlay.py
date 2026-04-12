"""Shadow-only classification gate overlay logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import config


@dataclass(frozen=True)
class ShadowGateOverlay:
    """One shadow gate decision row."""

    variant: str
    recommendation_mode: str
    recommended_sell_pct: float
    would_change: bool
    reason: str
    classifier_prob_actionable_sell: float | None
    gate_style: str
    threshold: float

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return asdict(self)


def _actionable_health_pass(
    mean_predicted: float,
    mean_ic: float,
    aggregate_oos_r2: float | None,
) -> bool:
    """Return whether the regression side is strong enough to allow an action."""
    return (
        mean_predicted < -0.03
        and mean_ic >= config.DIAG_MIN_IC
        and aggregate_oos_r2 is not None
        and aggregate_oos_r2 >= config.DIAG_MIN_OOS_R2
    )


def resolve_overlay_policy_variant(
    results_path: Path | None = None,
) -> tuple[str, str, float]:
    """Resolve the best shadow overlay candidate from the constrained ranking."""
    if results_path is None:
        results_path = (
            Path("results")
            / "research"
            / "v113_constrained_candidate_selection_results.csv"
        )
    if not results_path.exists():
        return ("permission_overlay", "permission_to_deviate", 0.70)

    results_df = pd.read_csv(results_path)
    if results_df.empty:
        return ("permission_overlay", "permission_to_deviate", 0.70)

    passing = results_df[results_df["promotion_eligible"].fillna(False).astype(bool)]
    selected = passing.iloc[0] if not passing.empty else results_df.iloc[0]
    return (
        str(selected.get("variant", "permission_overlay")),
        str(selected.get("gate_style", "permission_to_deviate")),
        float(selected.get("threshold", 0.70)),
    )


def compute_shadow_gate_overlay(
    *,
    live_mode: str,
    live_sell_pct: float,
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    aggregate_oos_r2: float | None,
    classifier_prob_actionable_sell: float | None,
    variant: str,
    gate_style: str,
    threshold: float,
) -> ShadowGateOverlay:
    """Compute the shadow-only classification gate overlay."""
    default = ShadowGateOverlay(
        variant=variant,
        recommendation_mode=live_mode,
        recommended_sell_pct=float(live_sell_pct),
        would_change=False,
        reason="classifier unavailable",
        classifier_prob_actionable_sell=classifier_prob_actionable_sell,
        gate_style=gate_style,
        threshold=float(threshold),
    )
    if classifier_prob_actionable_sell is None:
        return default

    actionable_health = _actionable_health_pass(
        mean_predicted=mean_predicted,
        mean_ic=mean_ic,
        aggregate_oos_r2=aggregate_oos_r2,
    )
    overlay_mode = live_mode
    overlay_sell_pct = float(live_sell_pct)
    reason = "classifier neutral"

    if gate_style == "veto_regression_sell":
        if live_mode == "ACTIONABLE" and classifier_prob_actionable_sell < threshold:
            overlay_mode = "DEFER-TO-TAX-DEFAULT"
            overlay_sell_pct = 0.50
            reason = "classifier vetoed weak regression sell"
        elif classifier_prob_actionable_sell >= threshold and actionable_health:
            reason = "classifier confirmed actionable sell"
        else:
            reason = "no regression sell to veto"
    else:
        if classifier_prob_actionable_sell >= threshold and actionable_health:
            overlay_mode = "ACTIONABLE"
            if consensus == "UNDERPERFORM":
                overlay_sell_pct = max(float(live_sell_pct), 0.75)
            reason = "classifier granted permission to deviate"
        elif classifier_prob_actionable_sell < threshold:
            reason = "classifier below deviation threshold"
        else:
            reason = "regression diagnostics too weak"

    would_change = (
        overlay_mode != live_mode
        or abs(overlay_sell_pct - float(live_sell_pct)) > 1e-12
    )
    return ShadowGateOverlay(
        variant=variant,
        recommendation_mode=overlay_mode,
        recommended_sell_pct=overlay_sell_pct,
        would_change=would_change,
        reason=reason,
        classifier_prob_actionable_sell=classifier_prob_actionable_sell,
        gate_style=gate_style,
        threshold=float(threshold),
    )


def build_decision_overlay_frame(
    *,
    live_mode: str,
    live_sell_pct: float,
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    aggregate_oos_r2: float | None,
    classifier_prob_actionable_sell: float | None,
    variant: str,
    gate_style: str,
    threshold: float,
) -> tuple[pd.DataFrame, ShadowGateOverlay]:
    """Return the live/shadow overlay frame plus the shadow overlay payload."""
    shadow = compute_shadow_gate_overlay(
        live_mode=live_mode,
        live_sell_pct=live_sell_pct,
        consensus=consensus,
        mean_predicted=mean_predicted,
        mean_ic=mean_ic,
        aggregate_oos_r2=aggregate_oos_r2,
        classifier_prob_actionable_sell=classifier_prob_actionable_sell,
        variant=variant,
        gate_style=gate_style,
        threshold=threshold,
    )
    live_row = {
        "variant": "live",
        "recommendation_mode": live_mode,
        "recommended_sell_pct": float(live_sell_pct),
        "would_change": False,
        "reason": "live production path",
        "classifier_prob_actionable_sell": classifier_prob_actionable_sell,
    }
    shadow_row = {
        "variant": "shadow_gate",
        "recommendation_mode": shadow.recommendation_mode,
        "recommended_sell_pct": shadow.recommended_sell_pct,
        "would_change": shadow.would_change,
        "reason": shadow.reason,
        "classifier_prob_actionable_sell": shadow.classifier_prob_actionable_sell,
    }
    return pd.DataFrame([live_row, shadow_row]), shadow
