"""Shared recommendation-mode and decision-section rendering helpers."""

from __future__ import annotations

from datetime import date
from typing import Any

import config


def sell_pct_from_consensus(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
) -> float:
    """Map consensus signal + IC to a sell percentage."""
    if mean_ic < 0.05:
        return 0.50
    if consensus == "OUTPERFORM":
        if mean_predicted > 0.15:
            return 0.25
        if mean_predicted > 0.05:
            return 0.50
        return 0.75
    if consensus == "UNDERPERFORM":
        return 1.00
    return 0.50


def determine_recommendation_mode(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    aggregate_health: dict | None,
    representative_cpcv: Any | None,
) -> dict[str, str | float]:
    """Downgrade weak-model months into monitoring or tax-default modes."""
    cpcv_verdict = representative_cpcv.stability_verdict if representative_cpcv is not None else "UNKNOWN"
    oos_r2 = aggregate_health["oos_r2"] if aggregate_health is not None else float("nan")

    if (
        aggregate_health is not None
        and oos_r2 >= config.DIAG_MIN_OOS_R2
        and mean_ic >= config.DIAG_MIN_IC
        and mean_hr >= config.DIAG_MIN_HIT_RATE
        and cpcv_verdict not in {"FAIL"}
    ):
        return {
            "mode": "actionable",
            "label": "ACTIONABLE",
            "sell_pct": sell_pct_from_consensus(consensus, mean_predicted, mean_ic),
            "summary": "Model quality is strong enough for the signal to influence the vest decision.",
            "action_note": "Prediction-led adjustment is allowed because aggregate model health is above threshold.",
        }

    if (
        aggregate_health is None
        or oos_r2 < 0.0
        or mean_ic < 0.03
        or mean_hr < 0.52
        or cpcv_verdict == "FAIL"
    ):
        return {
            "mode": "defer-to-tax-default",
            "label": "DEFER-TO-TAX-DEFAULT",
            "sell_pct": 0.50,
            "summary": "Model quality is too weak to justify a prediction-led vesting action.",
            "action_note": "Use the default diversification and tax-discipline rule rather than the point forecast.",
        }

    return {
        "mode": "monitoring-only",
        "label": "MONITORING-ONLY",
        "sell_pct": 0.50,
        "summary": "The signal is directionally interesting, but not trustworthy enough to override the default vesting rule.",
        "action_note": "Treat this as monitoring evidence only until the aggregate diagnostics strengthen.",
    }


def build_executive_summary_lines(
    as_of: date,
    consensus: str,
    confidence_tier: str,
    mean_predicted: float,
    sell_pct: float,
    recommendation_mode: dict[str, str | float],
    aggregate_health: dict | None,
    previous_summary: dict | None,
    next_vest_summary: dict | None,
) -> list[str]:
    """Build a concise decision memo at the top of recommendation.md."""
    del as_of
    quality_sentence = str(recommendation_mode["summary"])
    if previous_summary is None:
        change_line = "First tracked monthly memo on the refreshed v8 baseline."
    else:
        change_line = (
            f"Previous logged month ({previous_summary['as_of']}) was "
            f"{previous_summary['consensus']} at {previous_summary['predicted']} "
            f"with mean IC {previous_summary['mean_ic']}."
        )

    next_vest_line = "Next vest guidance unavailable because the lot file or latest PGR price is missing."
    if next_vest_summary is not None:
        next_vest_line = (
            f"Next vest is {next_vest_summary['vest_date']} ({next_vest_summary['rsu_type']}). "
            f"Default action today: sell {sell_pct:.0%} at vest unless model quality improves."
        )

    change_trigger = (
        "A more aggressive recommendation would require aggregate OOS R^2 >= 2%, "
        "mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check."
    )
    if recommendation_mode["mode"] == "actionable":
        change_trigger = (
            "This view would weaken if aggregate IC, hit rate, OOS R^2, or representative CPCV "
            "drops back below the current quality thresholds."
        )

    health_line = "Aggregate health unavailable."
    if aggregate_health is not None:
        health_line = (
            f"Aggregate health: OOS R^2 {aggregate_health['oos_r2']:.2%}, "
            f"IC {aggregate_health['nw_ic']:.4f}, hit rate {aggregate_health['agg_hit']:.1%}."
        )

    if consensus == "OUTPERFORM" and mean_predicted >= 0:
        model_view_line = (
            f"PGR is projected to outperform the benchmark set by {mean_predicted:+.2%} over the next 6 months. "
            f"Recommendation mode remains {recommendation_mode['label']}."
        )
    elif consensus == "UNDERPERFORM" and mean_predicted <= 0:
        model_view_line = (
            f"PGR is projected to lag the benchmark set by {mean_predicted:+.2%} over the next 6 months. "
            f"Recommendation mode remains {recommendation_mode['label']}."
        )
    else:
        model_view_line = (
            f"Consensus signal is {consensus}, but the average relative-return forecast is {mean_predicted:+.2%} "
            f"across benchmarks over the next 6 months. Recommendation mode remains {recommendation_mode['label']}."
        )

    return [
        "## Executive Summary",
        "",
        f"- What changed since last month: {change_line}",
        f"- Current model view: {model_view_line}",
        f"- How trustworthy it is: {quality_sentence} {health_line}",
        f"- What to do at the next vest: {next_vest_line}",
        f"- What would change the recommendation: {change_trigger}",
        "",
        "---",
        "",
    ]


def build_vest_decision_lines(
    next_vest_summary: dict | None,
    recommendation_mode: dict[str, str | float],
    sell_pct: float,
) -> list[str]:
    """Render the next-vest recommendation and provisional scenario table."""
    if next_vest_summary is None:
        return []

    scenario_result = next_vest_summary["scenario"]
    winner_label = (
        "Provisional scenario winner"
        if recommendation_mode["mode"] == "actionable"
        else "Tax-engine scenario ranking (informational only)"
    )
    scenario_note = (
        "The tax engine's highest-utility scenario aligns with the current point forecast."
        if recommendation_mode["mode"] == "actionable"
        else "Because recommendation mode is not ACTIONABLE, do not treat the tax-engine ranking below as a standalone trading instruction."
    )

    scenario_title = (
        "Tax timing scenarios (action-supporting)"
        if recommendation_mode["mode"] == "actionable"
        else "Tax timing scenarios (informational)"
    )

    lines = [
        "## Next Vest Decision",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Recommendation mode | **{recommendation_mode['label']}** |",
        f"| Next vest date | {next_vest_summary['vest_date']} |",
        f"| RSU type | {next_vest_summary['rsu_type']} |",
        f"| Current PGR price | ${next_vest_summary['current_price']:.2f} |",
        f"| Current in-scope shares | {next_vest_summary['shares']:.2f} |",
        f"| Average cost basis used | ${next_vest_summary['avg_basis']:.2f} |",
        f"| Suggested default vest action | Sell {sell_pct:.0%} of the vesting tranche |",
        "",
        f"> {recommendation_mode['action_note']}",
        "> The scenario table below is provisional and uses the current lot file as a proxy for the next vesting decision.",
        "",
        f"### {scenario_title}",
        "",
        "| Scenario | Timing | Tax Rate | Predicted Return | Probability | Use when |",
        "|----------|--------|----------|------------------|-------------|----------|",
    ]

    scenario_labels = {
        "SELL_NOW_STCG": "Sell at vest (STCG)",
        "HOLD_TO_LTCG": "Hold to LTCG date",
        "HOLD_FOR_LOSS": "Hold for downside / loss case",
    }
    use_when_labels = {
        "SELL_NOW_STCG": "Use the default diversification / tax-discipline rule or when the model edge is weak.",
        "HOLD_TO_LTCG": "Use only when the edge is strong enough to justify waiting for lower long-term tax treatment.",
        "HOLD_FOR_LOSS": "Use only when you are intentionally waiting for a downside or tax-loss outcome.",
    }
    for scenario in scenario_result.scenarios:
        lines.append(
            f"| {scenario_labels.get(scenario.label, scenario.label)} | {scenario.sell_date} | {scenario.tax_rate:.0%} | "
            f"{scenario.predicted_return:+.2%} | {scenario.probability:.1%} | "
            f"{use_when_labels.get(scenario.label, 'Informational only.')} |"
        )

    lines += [
        "",
        f"> {winner_label}: **{scenario_result.recommended_scenario}**.",
        f"> {scenario_note}",
        f"> STCG/LTCG breakeven from the tax engine: {scenario_result.stcg_ltcg_breakeven:.2%}.",
        "",
        "---",
        "",
    ]
    return lines
