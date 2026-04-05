"""Helpers for the v12 shadow-baseline study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.research.evaluation import summarize_predictions
from src.research.policy_metrics import hold_fraction_from_policy


@dataclass(frozen=True)
class SnapshotSummary:
    """Compact summary of one live or shadow monthly snapshot."""

    label: str
    as_of: date
    candidate_name: str
    policy_name: str
    consensus: str
    confidence_tier: str
    recommendation_mode: str
    sell_pct: float
    mean_predicted: float
    mean_ic: float
    mean_hit_rate: float
    aggregate_oos_r2: float
    aggregate_nw_ic: float
    calibrated_prob_outperform: float | None = None


def previous_business_day(value: date) -> date:
    """Move a date backward to the prior weekday when needed."""
    current = value
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def recent_monthly_review_dates(end_as_of: date, months: int) -> list[date]:
    """Return the most recent business-month-end dates up to ``end_as_of``."""
    if months <= 0:
        return []
    period_end = pd.Period(end_as_of, freq="M")
    if period_end.to_timestamp("M").date() > end_as_of:
        period_end -= 1
    dates: list[date] = []
    for offset in range(months - 1, -1, -1):
        ts = (period_end - offset).to_timestamp("M").date()
        dates.append(previous_business_day(ts))
    return dates


def signal_from_prediction(predicted_return: float, threshold: float = 0.03) -> str:
    """Map a predicted relative return into OUT/UNDER/NEUTRAL labels."""
    if predicted_return > threshold:
        return "OUTPERFORM"
    if predicted_return < -threshold:
        return "UNDERPERFORM"
    return "NEUTRAL"


def confidence_from_hit_rate(hit_rate: float) -> str:
    """Map per-benchmark hit rate into a simple confidence tier."""
    if hit_rate >= 0.60:
        return "HIGH"
    if hit_rate >= 0.55:
        return "MODERATE"
    return "LOW"


def aggregate_health_from_prediction_frames(
    prediction_frames: list[pd.DataFrame],
    target_horizon_months: int,
) -> dict[str, float] | None:
    """Aggregate pooled OOS health from benchmark-level prediction frames."""
    if not prediction_frames:
        return None
    combined = pd.concat(prediction_frames, ignore_index=True)
    if combined.empty:
        return None
    pred_col = combined["y_hat"].rename("y_hat")
    true_col = combined["y_true"].rename("y_true")
    summary = summarize_predictions(
        pred_col,
        true_col,
        target_horizon_months=target_horizon_months,
    )
    return {
        "oos_r2": summary.oos_r2,
        "nw_ic": summary.nw_ic,
        "agg_hit": summary.hit_rate,
        "n_obs": float(summary.n_obs),
    }


def sell_pct_from_policy(
    mean_predicted: float,
    policy_name: str,
) -> float:
    """Map the selected policy to a current sell percentage."""
    hold_fraction = float(
        hold_fraction_from_policy(
            pd.Series([mean_predicted], index=[0], name="y_hat"),
            policy_name,
        ).iloc[0]
    )
    return 1.0 - hold_fraction


def render_existing_holdings_lines(existing_holdings: list[dict[str, Any]]) -> list[str]:
    """Render compact existing-holdings guidance for a memo."""
    lines = ["## Existing Holdings Guidance", ""]
    if not existing_holdings:
        return lines + ["- Existing-lot guidance unavailable.", ""]
    for row in existing_holdings[:5]:
        lines.append(
            "- "
            f"{row['tax_bucket']}: {row['vest_date']} @ ${row['cost_basis_per_share']:.2f} "
            f"({row['shares']:.2f} share(s)). {row['rationale']}"
        )
    lines.append("")
    return lines


def render_redeploy_bucket_lines(redeploy_buckets: list[dict[str, Any]]) -> list[str]:
    """Render diversification-first redeploy guidance."""
    lines = ["## Redeploy Guidance", ""]
    if not redeploy_buckets:
        return lines + ["- No redeploy buckets were selected.", ""]
    title_map = {
        "broad_us_equity": "Broad US Equity",
        "international_equity": "International Equity",
        "fixed_income": "Fixed Income",
        "real_assets": "Real Assets",
        "sector_context": "Sector Context",
    }
    for row in redeploy_buckets:
        bucket_name = title_map.get(str(row["bucket"]), str(row["bucket"]).replace("_", " ").title())
        lines.append(
            "- "
            f"{bucket_name}: {row['example_funds']}. {row['note']}"
        )
    lines.append("")
    return lines


def build_shadow_comparison_lines(
    live_summary: SnapshotSummary,
    shadow_summary: SnapshotSummary,
    next_vest_date: date,
    next_vest_type: str,
    existing_holdings: list[dict[str, Any]],
    redeploy_buckets: list[dict[str, Any]],
) -> list[str]:
    """Build a side-by-side shadow-promotion memo."""
    lines = [
        f"# V12 Shadow Comparison - {shadow_summary.as_of.isoformat()}",
        "",
        "## Recommendation",
        "",
        f"- Next vest: **{next_vest_date.isoformat()}** (`{next_vest_type}`)",
        f"- Live production: **{live_summary.recommendation_mode}**, sell **{live_summary.sell_pct:.0%}**.",
        f"- Shadow baseline: **{shadow_summary.recommendation_mode}**, sell **{shadow_summary.sell_pct:.0%}**.",
        "- Interpretation: v12 is testing whether a simpler baseline policy yields clearer and steadier guidance than the live model stack.",
        "",
        "## Live Production Snapshot",
        "",
        f"- Candidate: `{live_summary.candidate_name}`",
        f"- Policy: `{live_summary.policy_name}`",
        f"- Signal: **{live_summary.consensus}** ({live_summary.confidence_tier} confidence)",
        f"- Predicted 6M relative return: `{live_summary.mean_predicted:+.2%}`",
        f"- Mean IC / hit rate: `{live_summary.mean_ic:.4f}` / `{live_summary.mean_hit_rate:.1%}`",
        f"- Aggregate OOS R^2 / NW IC: `{live_summary.aggregate_oos_r2:.2%}` / `{live_summary.aggregate_nw_ic:.4f}`",
    ]
    if live_summary.calibrated_prob_outperform is not None:
        lines.append(
            f"- Calibrated P(outperform): `{live_summary.calibrated_prob_outperform:.1%}`"
        )
    lines += [
        "",
        "## Shadow Baseline Snapshot",
        "",
        f"- Candidate: `{shadow_summary.candidate_name}`",
        f"- Policy: `{shadow_summary.policy_name}`",
        f"- Signal: **{shadow_summary.consensus}** ({shadow_summary.confidence_tier} confidence)",
        f"- Predicted 6M relative return: `{shadow_summary.mean_predicted:+.2%}`",
        f"- Mean IC / hit rate: `{shadow_summary.mean_ic:.4f}` / `{shadow_summary.mean_hit_rate:.1%}`",
        f"- Aggregate OOS R^2 / NW IC: `{shadow_summary.aggregate_oos_r2:.2%}` / `{shadow_summary.aggregate_nw_ic:.4f}`",
        "- Shadow policy goal: simplify the decision rule, keep concentration-reduction front and center, and avoid over-reading noisy model fit.",
        "",
    ]
    lines += render_existing_holdings_lines(existing_holdings)
    lines += render_redeploy_bucket_lines(redeploy_buckets)
    lines += [
        "## Why This Matters",
        "",
        "- Funds highly correlated with PGR stay contextual only; they are not preferred destinations for sold exposure.",
        "- The shadow memo is useful even if the live and shadow sell percentages match, because the redeploy and tax-bucket guidance is more explicit.",
        "",
    ]
    return lines


def build_shadow_check_lines(
    live_summary: SnapshotSummary,
    shadow_summary: SnapshotSummary,
    *,
    active_path: str = "live",
) -> list[str]:
    """Render a compact shadow-baseline comparison for production reports."""
    same_action = abs(live_summary.sell_pct - shadow_summary.sell_pct) < 1e-9
    uses_promoted_cross_check = live_summary.candidate_name == "ensemble_ridge_gbt_v18"
    if active_path == "shadow":
        comparison_line = (
            "The promoted simpler diversification-first layer and the visible cross-check still land on the same vest action."
            if same_action
            else "The promoted simpler diversification-first layer disagrees with the visible cross-check, so treat the cross-check as a diagnostic rather than the active instruction."
        )
        note_line = (
            "v22 keeps the simpler diversification-first baseline as the active recommendation layer while promoting `ensemble_ridge_gbt_v18` as the visible cross-check."
            if uses_promoted_cross_check
            else "v13.1 promotes the simpler diversification-first baseline as the active recommendation layer while retaining the live model stack as a cross-check."
        )
    else:
        comparison_line = (
            "The simpler diversification-first baseline independently lands on the same vest action."
            if same_action
            else "The simpler diversification-first baseline would lead to a different vest action, so treat the current output as less settled."
        )
        note_line = (
            "v13 keeps the live model stack in place, but uses this simpler baseline as a recommendation-layer cross-check because it was steadier in the v12 shadow study."
        )
    return [
        "## Simple-Baseline Cross-Check",
        "",
        "| Path | Candidate | Policy | Signal | Recommendation Mode | Sell % | Predicted 6M Return | Aggregate OOS R^2 |",
        "|------|-----------|--------|--------|---------------------|--------|---------------------|------------------|",
        f"| Visible cross-check | `{live_summary.candidate_name}` | `{live_summary.policy_name}` | {live_summary.consensus} | **{live_summary.recommendation_mode}** | **{live_summary.sell_pct:.0%}** | {live_summary.mean_predicted:+.2%} | {live_summary.aggregate_oos_r2:.2%} |",
        f"| Simpler baseline | `{shadow_summary.candidate_name}` | `{shadow_summary.policy_name}` | {shadow_summary.consensus} | **{shadow_summary.recommendation_mode}** | **{shadow_summary.sell_pct:.0%}** | {shadow_summary.mean_predicted:+.2%} | {shadow_summary.aggregate_oos_r2:.2%} |",
        "",
        f"> {comparison_line}",
        f"> {note_line}",
        "",
    ]


def build_existing_holdings_markdown_lines(existing_holdings: list[dict[str, Any]]) -> list[str]:
    """Render markdown guidance for already-held PGR lots."""
    lines = ["## Existing Holdings Guidance", ""]
    if not existing_holdings:
        return lines + ["- Existing-lot guidance unavailable.", ""]
    for row in existing_holdings[:6]:
        lines.append(
            "- "
            f"{row['tax_bucket']}: {row['vest_date']} @ ${row['cost_basis_per_share']:.2f} "
            f"({row['shares']:.2f} share(s)). {row['rationale']}"
        )
    lines.append("")
    return lines


def build_redeploy_markdown_lines(redeploy_buckets: list[dict[str, Any]]) -> list[str]:
    """Render markdown guidance for diversification-first redeploy buckets."""
    lines = ["## Redeploy Guidance", ""]
    if not redeploy_buckets:
        return lines + ["- Redeploy guidance unavailable.", ""]
    title_map = {
        "broad_us_equity": "Broad US Equity",
        "international_equity": "International Equity",
        "fixed_income": "Fixed Income",
        "real_assets": "Real Assets",
        "sector_context": "Sector Context",
    }
    for row in redeploy_buckets:
        bucket_name = title_map.get(str(row["bucket"]), str(row["bucket"]).replace("_", " ").title())
        lines.append(
            "- "
            f"{bucket_name}: {row['example_funds']}. {row['note']}"
        )
    lines.append("")
    return lines
