"""Shared recommendation-mode and decision-section rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import TYPE_CHECKING, Any

import config

if TYPE_CHECKING:
    from src.tax.monte_carlo import MonteCarloTaxAnalysis


def sell_pct_from_consensus(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
) -> float:
    """Map the consensus signal to the ACTIONABLE sell percentage.

    Review 2026-09-25, F20: a bullish (OUTPERFORM) consensus never sells more
    than the 50 % default. The old mapping sold 75 % when an OUTPERFORM
    forecast was at most 5 %; replayed on the realised-only OOS record
    (``src.models.live_policy_backtest``, 186 dates to 2026-02) that bucket
    cost 2.4 pp per decision on 22 dates, and the whole mapping lost 0.11 pp
    per decision to always selling 50 %. A missing or non-finite IC counts as
    weak (fail closed).
    """
    if not math.isfinite(mean_ic) or mean_ic < 0.05:
        return 0.50
    if consensus == "OUTPERFORM":
        if mean_predicted > 0.15:
            return 0.25
        return 0.50
    if consensus == "UNDERPERFORM":
        return 1.00
    return 0.50


GATE_PASS = "PASS"
GATE_MARGINAL = "MARGINAL"
GATE_FAIL = "FAIL"


@dataclass(frozen=True)
class QualityGate:
    """One recommendation-mode gate: its value, status and threshold text."""

    name: str
    value: float | None
    status: str
    current: str
    threshold: str
    meaning: str


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _threshold_status(value: float | None, good: float, fail_below: float) -> str:
    if value is None:
        return GATE_FAIL
    if value >= good:
        return GATE_PASS
    if value < fail_below:
        return GATE_FAIL
    return GATE_MARGINAL


def cpcv_available(representative_cpcv: Any | None) -> bool:
    """True when the CPCV diagnostic ran and produced paths (verdict known)."""
    if representative_cpcv is None:
        return False
    verdict = getattr(representative_cpcv, "stability_verdict", "UNKNOWN")
    return str(verdict) not in {"UNKNOWN", ""}


def evaluate_quality_gates(
    mean_ic: float,
    aggregate_health: dict | None,
    representative_cpcv: Any | None,
) -> list[QualityGate]:
    """Evaluate the recommendation-mode gates (review 2026-09-25, WP7).

    - OOS R^2 against the prevailing mean of realised targets (F04): PASS at
      >= ``DIAG_MIN_OOS_R2``, FAIL below 0.
    - Equal-weight mean IC across benchmarks (F13): PASS at >= ``DIAG_MIN_IC``,
      FAIL below 0.03. Callers pass the equal-weight mean; quality weights
      come from the same OOS record and inflate it.
    - Directional skill (F13): one-sided Pesaran-Timmermann p-value,
      Driscoll-Kraay by date. PASS below ``DIAG_MAX_DIRECTIONAL_PVALUE``,
      FAIL at or above ``DIAG_MARGINAL_DIRECTIONAL_PVALUE`` or when undefined
      (e.g. a predictor that always calls the same sign). A raw hit rate is
      not gated: a 68 % base rate clears 55 % without skill.
    - CPCV completeness (F02, F20): CPCV is a combinatorial K-fold, so its
      verdict is diagnostic only and never gates. The run fails closed when
      the diagnostic is missing or UNKNOWN.

    A missing input fails its gate (fail closed).
    """
    health = aggregate_health or {}
    oos_r2 = _finite(health.get("oos_r2")) if aggregate_health is not None else None
    ic = _finite(mean_ic)
    pt_p = _finite(health.get("pt_p_value")) if aggregate_health is not None else None
    hit = _finite(health.get("agg_hit"))
    base = _finite(health.get("constant_rule_hit_rate"))

    if pt_p is None:
        pt_status = GATE_FAIL
    elif pt_p < config.DIAG_MAX_DIRECTIONAL_PVALUE:
        pt_status = GATE_PASS
    elif pt_p < config.DIAG_MARGINAL_DIRECTIONAL_PVALUE:
        pt_status = GATE_MARGINAL
    else:
        pt_status = GATE_FAIL
    if hit is not None and base is not None:
        pt_current = (
            f"p={pt_p:.3f}; hit {hit:.1%} vs base rate {base:.1%}"
            if pt_p is not None
            else f"undefined; hit {hit:.1%} vs base rate {base:.1%}"
        )
    else:
        pt_current = f"p={pt_p:.3f}" if pt_p is not None else "n/a"

    cpcv_ok = cpcv_available(representative_cpcv)
    cpcv_verdict = (
        str(getattr(representative_cpcv, "stability_verdict", "UNKNOWN"))
        if representative_cpcv is not None
        else "missing"
    )
    return [
        QualityGate(
            name="oos_r2",
            value=oos_r2,
            status=_threshold_status(oos_r2, config.DIAG_MIN_OOS_R2, 0.0),
            current=f"{oos_r2:.2%}" if oos_r2 is not None else "n/a",
            threshold=f">= {config.DIAG_MIN_OOS_R2:.2%}",
            meaning="Beats the prevailing mean of the targets realised by each forecast date.",
        ),
        QualityGate(
            name="mean_ic",
            value=ic,
            status=_threshold_status(ic, config.DIAG_MIN_IC, 0.03),
            current=f"{ic:.4f}" if ic is not None else "n/a",
            threshold=f">= {config.DIAG_MIN_IC:.4f}",
            meaning="Equal-weight mean of the per-benchmark OOS rank ICs.",
        ),
        QualityGate(
            name="directional_skill",
            value=pt_p,
            status=pt_status,
            current=pt_current,
            threshold=f"PT p < {config.DIAG_MAX_DIRECTIONAL_PVALUE:.2f}",
            meaning="Up/down calls beat chance given the base rate (Pesaran-Timmermann, clustered by date).",
        ),
        QualityGate(
            name="cpcv_completed",
            value=None,
            status=GATE_PASS if cpcv_ok else GATE_FAIL,
            current=f"ran ({cpcv_verdict}, diagnostic only)" if cpcv_ok else f"{cpcv_verdict}",
            threshold="diagnostic ran",
            meaning="Fail-closed completeness check; the CPCV verdict itself does not gate (K-fold).",
        ),
    ]


def determine_recommendation_mode(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    aggregate_health: dict | None,
    representative_cpcv: Any | None,
) -> dict[str, str | float]:
    """Downgrade weak-model months into monitoring or tax-default modes.

    ``mean_ic`` is the gated IC: the equal-weight mean of the per-benchmark
    OOS ICs. ``mean_hr`` is reported only; directional skill is gated on the
    Pesaran-Timmermann result in ``aggregate_health`` (``pt_p_value``). See
    ``evaluate_quality_gates``. ACTIONABLE needs every gate to PASS; any FAIL
    gives DEFER-TO-TAX-DEFAULT; otherwise MONITORING-ONLY. The ACTIONABLE sell
    percentage comes from ``sell_pct_from_consensus``.
    """
    del mean_hr  # reported elsewhere; not a gate since review 2026-09-25 (F13)
    gates = evaluate_quality_gates(mean_ic, aggregate_health, representative_cpcv)
    statuses = {gate.name: gate.status for gate in gates}

    if all(status == GATE_PASS for status in statuses.values()):
        return {
            "mode": "actionable",
            "label": "ACTIONABLE",
            "sell_pct": sell_pct_from_consensus(consensus, mean_predicted, mean_ic),
            "summary": "Model quality is strong enough for the signal to influence the vest decision.",
            "action_note": "Prediction-led adjustment is allowed because aggregate model health is above threshold.",
        }

    if any(status == GATE_FAIL for status in statuses.values()):
        if statuses["cpcv_completed"] == GATE_FAIL and all(
            status != GATE_FAIL for name, status in statuses.items() if name != "cpcv_completed"
        ):
            summary = (
                "The validation run is incomplete (the CPCV diagnostic did not run), "
                "so the signal cannot justify a prediction-led vesting action."
            )
        else:
            summary = "Model quality is too weak to justify a prediction-led vesting action."
        return {
            "mode": "defer-to-tax-default",
            "label": "DEFER-TO-TAX-DEFAULT",
            "sell_pct": 0.50,
            "summary": summary,
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
        f"A more aggressive recommendation would require aggregate OOS R^2 >= "
        f"{config.DIAG_MIN_OOS_R2:.0%} against the prevailing-mean benchmark, "
        f"equal-weight mean IC >= {config.DIAG_MIN_IC:.2f}, significant directional skill "
        f"(Pesaran-Timmermann p < {config.DIAG_MAX_DIRECTIONAL_PVALUE:.2f}), and a completed "
        "CPCV diagnostic."
    )
    if recommendation_mode["mode"] == "actionable":
        change_trigger = (
            "This view would weaken if the equal-weight IC, directional skill, or OOS R^2 "
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
        "The tax engine ranks scenarios by expected after-tax proceeds under the assumed PGR price return."
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
        "| Scenario | Timing | Tax Rate | Assumed PGR Return | Probability | Use when |",
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
        (
            f"> Holding to the LTCG date beats selling at STCG unless PGR's own price return is below "
            f"{scenario_result.stcg_ltcg_breakeven:+.2%} (breakeven for the average basis used)."
        ),
        (
            f"> Assumed PGR price return: {config.TAX_SCENARIO_PGR_ANNUAL_RETURN:+.1%}/yr "
            "(`TAX_SCENARIO_PGR_ANNUAL_RETURN`). The model's forecast is relative to the benchmarks, "
            "not a PGR price forecast, so it does not drive these scenarios."
        ),
        "",
    ]

    mc_analysis = next_vest_summary.get("mc_analysis")
    lines += _build_mc_sensitivity_lines(mc_analysis)

    lines += ["---", ""]
    return lines


def _build_mc_sensitivity_lines(mc_analysis: "MonteCarloTaxAnalysis | None") -> list[str]:
    """Render the Monte Carlo tax-sensitivity section (v35, Tier 4.5)."""
    if mc_analysis is None:
        return []

    mc = mc_analysis.hold_ltcg
    sell_now = mc_analysis.sell_now_net
    beat_pct = mc.prob_beats_sell_now * 100.0
    gain_pct = mc.prob_positive_gain * 100.0

    return [
        "### Monte Carlo Tax Sensitivity (HOLD_TO_LTCG vs. Sell Now)",
        "",
        f"> **{mc.n_paths:,} GBM paths** | drift {mc.annual_drift:+.1%}/yr | vol {mc.annual_vol:.1%}/yr | "
        f"horizon {mc_analysis.horizon_days} days",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sell-now reference (STCG net) | ${sell_now:,.0f} |",
        f"| HOLD_TO_LTCG — P10 net | ${mc.net_proceeds_p10:,.0f} |",
        f"| HOLD_TO_LTCG — median net | ${mc.net_proceeds_p50:,.0f} |",
        f"| HOLD_TO_LTCG — mean net | ${mc.net_proceeds_mean:,.0f} |",
        f"| HOLD_TO_LTCG — P90 net | ${mc.net_proceeds_p90:,.0f} |",
        f"| P(HOLD_TO_LTCG beats Sell Now) | {beat_pct:.1f}% |",
        f"| P(terminal price > cost basis) | {gain_pct:.1f}% |",
        "",
        (
            f"> At {mc.annual_vol:.1%} annualised volatility and a {mc.annual_drift:+.1%}/yr drift, "
            f"{beat_pct:.1f}% of simulated paths produce higher after-tax net proceeds from holding "
            f"to LTCG eligibility than from selling immediately at STCG rates."
        ),
        "",
    ]


def build_data_freshness_lines(freshness_report: dict[str, Any] | None) -> list[str]:
    """Render data freshness checks for recommendation.md."""
    if freshness_report is None:
        return []

    status_note = (
        "All monitored feeds are within freshness thresholds for this run."
        if freshness_report.get("overall_status") == "OK"
        else "Some upstream data is stale or missing. Treat this run with extra caution until the feeds refresh."
    )

    status_labels = {
        "OK": "OK",
        "STALE": "STALE",
        "MISSING": "MISSING",
    }

    lines = [
        "## Data Freshness",
        "",
        f"> {status_note}",
        "",
        "| Feed | Latest Date | Age | Limit | Status |",
        "|------|-------------|-----|-------|--------|",
    ]

    for row in freshness_report.get("checks", []):
        latest_date = row.get("latest_label") or row.get("latest_date") or "missing"
        if row.get("age_label"):
            age_text = str(row["age_label"])
        elif row.get("age_days") is not None:
            age_text = f"{row['age_days']} days"
        else:
            age_text = "n/a"
        limit_text = row.get("limit_label") or f"{row['max_age_days']} days"
        lines.append(
            f"| {row['feed']} | {latest_date} | {age_text} | "
            f"{limit_text} | **{status_labels.get(row['status'], row['status'])}** |"
        )

    if freshness_report.get("warnings"):
        lines += [
            "",
            "Warnings:",
        ]
        for warning in freshness_report["warnings"]:
            lines.append(f"- {warning}")

    lines += [
        "",
        "---",
        "",
    ]
    return lines
