"""Shared helpers for the v118-v121 prospective shadow-monitoring phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.v37_utils import RESULTS_DIR
from src.research.v102_utils import (
    build_overlay_alignment_frame,
    permission_overlay_hold_fraction,
    veto_overlay_hold_fraction,
)


@dataclass(frozen=True)
class SelectedShadowCandidate:
    """Resolved shadow candidate selected by the constrained v113 ranking."""

    variant: str
    gate_style: str
    threshold: float
    promotion_eligible: bool


def load_selected_shadow_candidate(
    results_path: Path | None = None,
) -> SelectedShadowCandidate:
    """Load the top v113 candidate or fall back to the default shadow overlay."""
    if results_path is None:
        results_path = RESULTS_DIR / "v113_constrained_candidate_selection_results.csv"
    if not results_path.exists():
        return SelectedShadowCandidate(
            variant="gemini_veto_0.50",
            gate_style="veto_regression_sell",
            threshold=0.50,
            promotion_eligible=False,
        )
    results_df = pd.read_csv(results_path)
    if results_df.empty:
        return SelectedShadowCandidate(
            variant="gemini_veto_0.50",
            gate_style="veto_regression_sell",
            threshold=0.50,
            promotion_eligible=False,
        )
    selected = results_df.iloc[0]
    return SelectedShadowCandidate(
        variant=str(selected["variant"]),
        gate_style=str(selected["gate_style"]),
        threshold=float(selected["threshold"]),
        promotion_eligible=bool(selected["promotion_eligible"]),
    )


def recommendation_mode_from_hold_fraction(hold_fraction: float) -> str:
    """Map hold-fraction output to a recommendation-mode proxy for replay."""
    if np.isclose(hold_fraction, 0.5):
        return "DEFER-TO-TAX-DEFAULT"
    return "ACTIONABLE"


def disagreement_label(live_hold: float, shadow_hold: float) -> str:
    """Return a readable label for the live-vs-shadow policy difference."""
    if np.isclose(live_hold, shadow_hold):
        return "no_change"
    if live_hold < 0.5 <= shadow_hold:
        return "veto_sell_to_defer"
    if np.isclose(live_hold, 0.5) and shadow_hold < 0.5:
        return "permission_to_sell"
    if live_hold > 0.5 >= shadow_hold:
        return "hold_to_defer"
    if np.isclose(live_hold, 0.5) and shadow_hold > 0.5:
        return "permission_to_hold"
    return "other_change"


def _max_consecutive_true(values: pd.Series) -> int:
    """Return the longest run of consecutive True values."""
    max_run = 0
    run = 0
    for value in values.fillna(False).astype(bool):
        if value:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def build_prospective_shadow_replay() -> pd.DataFrame:
    """Build a month-by-month replay of the selected shadow candidate vs live."""
    selected = load_selected_shadow_candidate()
    aligned, baseline_hold = build_overlay_alignment_frame()
    if selected.gate_style == "veto_regression_sell":
        shadow_hold = veto_overlay_hold_fraction(
            aligned,
            baseline_hold,
            threshold=selected.threshold,
        )
    else:
        shadow_hold = permission_overlay_hold_fraction(
            aligned,
            threshold=selected.threshold,
        )

    detail = aligned.copy()
    detail = detail.sort_index().rename_axis("date").reset_index()
    detail["live_hold_fraction"] = baseline_hold.reindex(detail["date"]).to_numpy(dtype=float)
    detail["shadow_hold_fraction"] = shadow_hold.reindex(detail["date"]).to_numpy(dtype=float)
    detail["live_sell_pct"] = 1.0 - detail["live_hold_fraction"]
    detail["shadow_sell_pct"] = 1.0 - detail["shadow_hold_fraction"]
    detail["live_recommendation_mode"] = detail["live_hold_fraction"].apply(
        recommendation_mode_from_hold_fraction
    )
    detail["shadow_recommendation_mode"] = detail["shadow_hold_fraction"].apply(
        recommendation_mode_from_hold_fraction
    )
    detail["would_change"] = ~np.isclose(
        detail["live_hold_fraction"],
        detail["shadow_hold_fraction"],
    )
    detail["disagreement_label"] = [
        disagreement_label(live_hold, shadow_hold)
        for live_hold, shadow_hold in zip(
            detail["live_hold_fraction"],
            detail["shadow_hold_fraction"],
            strict=False,
        )
    ]
    detail["live_policy_return"] = detail["live_hold_fraction"] * detail["realized"]
    detail["shadow_policy_return"] = detail["shadow_hold_fraction"] * detail["realized"]
    detail["shadow_minus_live"] = (
        detail["shadow_policy_return"] - detail["live_policy_return"]
    )
    detail["shadow_beat_live"] = detail["shadow_minus_live"] > 0.0
    detail["monitor_month_number"] = np.arange(1, len(detail) + 1)
    detail["cumulative_shadow_minus_live"] = detail["shadow_minus_live"].cumsum()
    detail["expanding_shadow_minus_live_mean"] = detail["shadow_minus_live"].expanding().mean()
    detail["expanding_agreement_rate"] = (~detail["would_change"]).expanding().mean()
    detail["cumulative_disagreement_months"] = detail["would_change"].cumsum()
    detail["selected_variant"] = selected.variant
    detail["selected_gate_style"] = selected.gate_style
    detail["selected_threshold"] = selected.threshold
    detail["selected_promotion_eligible"] = selected.promotion_eligible
    return detail


def summarize_disagreement_scorecard(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize all-month and disagreement-month prospective replay behavior."""
    disagreements = detail_df.loc[detail_df["would_change"]].copy()
    trailing_12 = detail_df.tail(12)
    row = {
        "selected_variant": str(detail_df["selected_variant"].iloc[0]),
        "gate_style": str(detail_df["selected_gate_style"].iloc[0]),
        "threshold": float(detail_df["selected_threshold"].iloc[0]),
        "review_months": int(len(detail_df)),
        "agreement_rate": float((~detail_df["would_change"]).mean()),
        "disagreement_months": int(detail_df["would_change"].sum()),
        "shadow_win_rate_all": float(detail_df["shadow_beat_live"].mean()),
        "shadow_win_rate_disagreement": (
            float(disagreements["shadow_beat_live"].mean())
            if not disagreements.empty
            else float("nan")
        ),
        "mean_shadow_minus_live_all": float(detail_df["shadow_minus_live"].mean()),
        "mean_shadow_minus_live_disagreement": (
            float(disagreements["shadow_minus_live"].mean())
            if not disagreements.empty
            else float("nan")
        ),
        "cumulative_shadow_minus_live_all": float(detail_df["shadow_minus_live"].sum()),
        "cumulative_shadow_minus_live_disagreement": (
            float(disagreements["shadow_minus_live"].sum())
            if not disagreements.empty
            else float("nan")
        ),
        "vetoed_sell_months": int(
            (detail_df["disagreement_label"] == "veto_sell_to_defer").sum()
        ),
        "max_consecutive_disagreements": int(
            _max_consecutive_true(detail_df["would_change"])
        ),
        "last_12m_mean_shadow_minus_live": float(trailing_12["shadow_minus_live"].mean()),
        "last_12m_cumulative_shadow_minus_live": float(
            trailing_12["shadow_minus_live"].sum()
        ),
    }
    return pd.DataFrame([row])


def summarize_prospective_gate_assessment(
    scorecard_df: pd.DataFrame,
    monitoring_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert the replay scorecard into a promotion-style next-step decision."""
    row = scorecard_df.iloc[0]
    monitoring = monitoring_summary_df.iloc[0]
    agreement_pass = float(row["agreement_rate"]) >= 0.90
    uplift_pass = float(row["cumulative_shadow_minus_live_all"]) >= 0.0
    disagreement_pass = (
        float(row["cumulative_shadow_minus_live_disagreement"]) >= 0.0
        if pd.notna(row["cumulative_shadow_minus_live_disagreement"])
        else False
    )
    churn_pass = int(row["max_consecutive_disagreements"]) <= 3
    matured_n = int(monitoring.get("matured_n", 0))
    has_live_matured_monitoring = matured_n > 0

    if agreement_pass and uplift_pass and disagreement_pass:
        decision = (
            "consider_limited_gate_promotion"
            if has_live_matured_monitoring and churn_pass
            else "advance_to_real_time_shadow_monitoring"
        )
    else:
        decision = "continue_offline_shadow_research"

    result = {
        "selected_variant": str(row["selected_variant"]),
        "gate_style": str(row["gate_style"]),
        "threshold": float(row["threshold"]),
        "agreement_pass": agreement_pass,
        "uplift_pass": uplift_pass,
        "disagreement_pass": disagreement_pass,
        "churn_pass": churn_pass,
        "matured_live_monitoring_n": matured_n,
        "decision": decision,
    }
    return pd.DataFrame([result])
