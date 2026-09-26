"""Renders ``recommendation.md``, the monthly sell/hold report.
Functions in other ``pgr_vds.decision`` modules are called through the module
(``health.compute_aggregate_health(...)``), so a test patches a function
once, in the module that defines it.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from pgr_vds.decision import artifacts, rendering, tax_lots
from src.models.calibration import CalibrationResult
from src.models.drift_monitor import ModelDriftSummary
from src.models.live_policy_backtest import LIVE_MAPPING_POLICY
from src.models.policy_metrics import (
    FIXED_POLICIES,
    SIGNAL_POLICIES,
    PolicySummary,
)
from src.models.wfo_engine import CPCVResult
from src.portfolio.black_litterman import BLDiagnostics
from src.portfolio.redeploy_portfolio import render_redeploy_portfolio_markdown_lines
from src.reporting.confidence import benchmark_role_for_ticker, build_confidence_snapshot
from src.reporting.decision_rendering import (
    build_data_freshness_lines as render_data_freshness_lines,
)
from src.reporting.monthly_summary import (
    build_actionability_label,
    build_decision_headline,
    build_hold_vs_sell_label,
)
from src.reporting.snapshot_summary import (
    SnapshotSummary,
    build_existing_holdings_markdown_lines,
    build_redeploy_markdown_lines,
    build_shadow_check_lines,
)


def write_recommendation_md(
    out_dir: Path,
    as_of: date,
    run_date: date,
    conn,
    signals: pd.DataFrame,
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    sell_pct: float,
    dry_run: bool,
    mean_prob_outperform: float = 0.5,
    composite_confidence_tier: str = "LOW",
    cal_result: CalibrationResult | None = None,
    aggregate_health: dict | None = None,
    recommendation_mode: dict[str, str | float] | None = None,
    live_summary: SnapshotSummary | None = None,
    shadow_summary: SnapshotSummary | None = None,
    consensus_shadow_df: pd.DataFrame | None = None,
    existing_holdings: list[dict[str, object]] | None = None,
    redeploy_buckets: list[dict[str, object]] | None = None,
    redeploy_portfolio: dict[str, object] | None = None,
    recommendation_layer_label: str | None = None,
    representative_cpcv: CPCVResult | None = None,
    freshness_report: dict[str, object] | None = None,
    model_drift_summary: ModelDriftSummary | None = None,
    policy_summary: dict[str, PolicySummary] | None = None,
    bl_diagnostics: BLDiagnostics | None = None,
    visible_cross_check: bool = False,
    classification_shadow_summary: dict[str, Any] | None = None,
    shadow_gate_overlay: dict[str, Any] | None = None,
) -> None:
    """Write the human-readable recommendation report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recommendation.md"

    has_confidence = "prob_outperform" in signals.columns
    has_calibrated = (
        "calibrated_prob_outperform" in signals.columns
        and not signals.empty
        and cal_result is not None
        and cal_result.method != "uncalibrated"
    )
    mean_cal_prob = (
        float(signals["calibrated_prob_outperform"].mean())
        if has_calibrated
        else mean_prob_outperform
    )
    if recommendation_mode is None:
        recommendation_mode = {
            "mode": "monitoring-only",
            "label": "MONITORING-ONLY",
            "summary": "Recommendation mode defaulted because aggregate health was unavailable.",
            "action_note": "Use the default diversification rule.",
        }
    previous_summary = artifacts.load_previous_decision_summary(as_of)
    next_vest_summary = tax_lots.build_provisional_vest_scenario(conn, as_of, mean_predicted, mean_cal_prob)
    confidence_snapshot = build_confidence_snapshot(
        mean_ic=mean_ic,
        mean_hr=mean_hr,
        aggregate_health=aggregate_health,
        representative_cpcv=representative_cpcv,
    )
    hold_vs_sell_label = build_hold_vs_sell_label(sell_pct)
    actionability_label = build_actionability_label(str(recommendation_mode["label"]))
    decision_headline = build_decision_headline(
        str(recommendation_mode["label"]),
        sell_pct,
    )

    cross_check_agreement = "n/a"
    if consensus_shadow_df is not None and not consensus_shadow_df.empty:
        live_rows = consensus_shadow_df[consensus_shadow_df["is_live_path"]]
        shadow_rows = consensus_shadow_df[~consensus_shadow_df["is_live_path"]]
        if not live_rows.empty and not shadow_rows.empty:
            live_row = live_rows.iloc[0]
            shadow_row = shadow_rows.iloc[0]
            same_mode = str(live_row["recommendation_mode"]) == str(
                shadow_row["recommendation_mode"]
            )
            same_sell = abs(
                float(live_row["recommended_sell_pct"])
                - float(shadow_row["recommended_sell_pct"])
            ) <= 1e-9
            cross_check_agreement = "Aligned" if same_mode and same_sell else "Mixed"

    classifier_agreement = (
        str(classification_shadow_summary.get("agreement_label"))
        if isinstance(classification_shadow_summary, dict)
        and classification_shadow_summary.get("agreement_label") is not None
        else "n/a"
    )
    decision_surface_lines = [
        "## Decision At A Glance",
        "",
        f"- Hold vs Sell: **{hold_vs_sell_label}**",
        f"- Is this month actionable? **{actionability_label}**",
        f"- Top-line decision: **{decision_headline}**",
    ]
    if (
        isinstance(classification_shadow_summary, dict)
        and classification_shadow_summary.get("enabled")
        and classification_shadow_summary.get("probability_actionable_sell_label") is not None
    ):
        decision_surface_lines.append(
            "- Shadow classifier probability: "
            f"**{classification_shadow_summary.get('probability_actionable_sell_label')}** "
            f"({classification_shadow_summary.get('confidence_tier', 'n/a')})"
        )
        investable_label = classification_shadow_summary.get("probability_investable_pool_label")
        investable_stance = classification_shadow_summary.get("stance_investable_pool", "n/a")
        if investable_label is not None:
            decision_surface_lines.append(
                f"- **Portfolio-aligned P(Actionable Sell):** {investable_label} "
                f"[{investable_stance}] _(investable pool, fixed weights)_"
            )
        path_b_label = classification_shadow_summary.get("probability_path_b_temp_scaled_label")
        path_b_stance = classification_shadow_summary.get("stance_path_b", "n/a")
        if path_b_label is not None:
            decision_surface_lines.append(
                f"- **Path B P(Actionable Sell):** {path_b_label} "
                f"[{path_b_stance}] _(composite portfolio target, temp-scaled)_"
            )
    decision_surface_lines += [
        "",
        "## Agreement Panel",
        "",
        f"- Live recommendation: **{recommendation_mode['label']} / sell {sell_pct:.0%}**",
        f"- Consensus cross-check: **{cross_check_agreement}**",
        f"- Classifier shadow: **{classifier_agreement}**",
    ]
    if isinstance(shadow_gate_overlay, dict):
        decision_surface_lines.append(
            "- Shadow gate overlay: "
            f"**{shadow_gate_overlay.get('recommendation_mode', 'n/a')} / sell "
            f"{float(shadow_gate_overlay.get('recommended_sell_pct', sell_pct)):.0%}** "
            f"({'would change the live path' if shadow_gate_overlay.get('would_change') else 'no live change'})"
        )
    if cross_check_agreement == "Mixed" or classifier_agreement == "Mixed":
        decision_surface_lines.append(
            "- This month has meaningful internal disagreement, so treat the live output as lower-confidence than a fully aligned month."
        )

    lines = [
        f"# PGR Monthly Decision Report — {as_of.strftime('%B %Y')}",
        "",
        f"**As-of Date:** {as_of}  ",
        f"**Run Date:** {run_date}  ",
        f"**Model Version:** {rendering.MODEL_VERSION_LABEL}  ",
        f"**Recommendation Layer:** {recommendation_layer_label or config.RECOMMENDATION_LAYER_MODE}  ",
        "",
        "---",
        "",
        *rendering.build_executive_summary_lines(
            as_of,
            consensus,
            composite_confidence_tier,
            mean_predicted,
            sell_pct,
            recommendation_mode,
            aggregate_health,
            previous_summary,
            next_vest_summary,
        ),
        *render_data_freshness_lines(freshness_report),
        *decision_surface_lines,
        "",
        "---",
        "",
        "## Consensus Signal",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Signal | **{consensus} ({composite_confidence_tier} CONFIDENCE)** |",
        f"| Recommendation Mode | **{recommendation_mode['label']}** |",
        f"| Recommended Sell % | **{sell_pct:.0%}** |",
        f"| Predicted 6M Relative Return | {mean_predicted:+.2%} |",
    ]
    if has_calibrated:
        lines.append(f"| P(Outperform, calibrated) | {mean_cal_prob:.1%} |")

    # Conformal CI row — median CI bounds across benchmarks (robust to outlier widths)
    has_ci = "ci_lower" in signals.columns and not signals.empty
    if has_ci:
        valid_ci = signals[["ci_lower", "ci_upper"]].dropna()
        if not valid_ci.empty:
            ci_lo_med = float(valid_ci["ci_lower"].median())
            ci_hi_med = float(valid_ci["ci_upper"].median())
            lines.append(
                f"| {config.CONFORMAL_COVERAGE:.0%} Prediction Interval (median) "
                f"| {ci_lo_med:+.2%} to {ci_hi_med:+.2%} |"
            )

    directional_row = "| Directional skill (hit rate vs base rate) | n/a |"
    if aggregate_health is not None:
        pt_p = float(aggregate_health.get("pt_p_value", float("nan")))
        hit = float(aggregate_health.get("agg_hit", float("nan")))
        base = float(aggregate_health.get("constant_rule_hit_rate", float("nan")))
        pt_label = f"p = {pt_p:.3f}" if math.isfinite(pt_p) else "undefined"
        base_label = f"{hit:.1%} vs {base:.1%}; " if math.isfinite(hit) and math.isfinite(base) else ""
        directional_row = (
            "| Directional skill (hit rate vs base rate) | "
            f"{base_label}Pesaran–Timmermann {pt_label} |"
        )
    lines += [
        f"| Mean IC (equal-weight across benchmarks) | {mean_ic:.4f} |",
        f"| Mean Hit Rate (equal-weight) | {mean_hr:.1%} |",
        directional_row,
        (
            f"| Aggregate OOS R^2 (vs prevailing mean) | {aggregate_health['oos_r2']:.2%} |"
            if aggregate_health is not None
            else "| Aggregate OOS R^2 (vs prevailing mean) | n/a |"
        ),
        "",
        "> **Note:** The sell % recommendation is used only at actual vesting events",
        "> (January and July).  Monthly reports are monitoring tools, not trade signals.",
    ]
    shrinkage_values = (
        signals["shrinkage_alpha"].dropna()
        if "shrinkage_alpha" in signals.columns and not signals.empty
        else pd.Series(dtype=float)
    )
    if not shrinkage_values.empty:
        lines += [
            ">",
            f"> **Shrinkage:** the ensemble forecast is scaled by α = {float(shrinkage_values.iloc[0]):.2f}, "
            "the v38 grid value with the lowest squared error over the realised OOS record, re-chosen "
            "every month. Before review 2026-09-25 it was fixed at 0.50, a value chosen on the same "
            "history it was then scored on.",
        ]

    # Calibration status note
    if cal_result is None or cal_result.method == "uncalibrated":
        lines += [
            ">",
            "> **Calibration:** Phase 1 — too few realised OOS observations; P(outperform) is 50%.",
            f"> Platt scaling activates at n ≥ {config.CALIBRATION_MIN_OBS_PLATT} OOS observations.",
        ]
    else:
        method_label = "Platt scaling" if cal_result.method == "platt" else "Platt → Isotonic"
        lines += [
            ">",
            f"> **Calibration:** Phase 2 — {method_label} per benchmark.  "
            f"Prequential ECE = {cal_result.ece:.1%} "
            f"[95% CI: {cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}] over "
            f"{cal_result.n_obs:,} OOS benchmark-months, each scored by a calibrator fitted only on "
            "outcomes known at the time.",
        ]

    if (
        visible_cross_check
        and consensus_shadow_df is not None
        and not consensus_shadow_df.empty
    ):
        lines += [
            "",
            "---",
            "",
            "## Consensus Shadow Evaluation",
            "",
            "> The live path now uses the quality-weighted cross-benchmark consensus.",
            "> The equal-weight consensus is retained here as a production cross-check.",
            "",
            "| Variant | Consensus | Mean Pred. Return | Mean IC | Mean Hit Rate | P(Outperform) | Mode | Sell % | Top Weight |",
            "|---------|-----------|-------------------|---------|---------------|---------------|------|--------|------------|",
        ]
        for row in consensus_shadow_df.to_dict("records"):
            if bool(row.get("is_live_path")):
                variant_label = (
                    "Live quality-weighted"
                    if str(row.get("variant")) == "quality_weighted"
                    else "Live equal-weight"
                )
            else:
                variant_label = (
                    "Shadow equal-weight"
                    if str(row.get("variant")) == "equal_weight"
                    else "Shadow quality-weighted"
                )
            top_weight = (
                f"{row.get('top_benchmark', 'n/a')} ({float(row.get('top_benchmark_weight', 0.0)):.1%})"
                if row.get("top_benchmark")
                else "n/a"
            )
            lines.append(
                f"| {variant_label} | {row['consensus']} ({row['confidence_tier']}) | "
                f"{float(row['mean_predicted_return']):+.2%} | {float(row['mean_ic']):.4f} | "
                f"{float(row['mean_hit_rate']):.1%} | {float(row['mean_prob_outperform']):.1%} | "
                f"{row['recommendation_mode']} | {float(row['recommended_sell_pct']):.0%} | "
                f"{top_weight} |"
            )

    if (
        isinstance(classification_shadow_summary, dict)
        and classification_shadow_summary.get("enabled")
    ):
        lines += [
            "",
            "---",
            "",
            "## Classification Confidence Check",
            "",
            "> Shadow-only interpretation layer from the v87-v96 classifier research.",
            "> It does not change the live recommendation or sell percentage.",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Target | {classification_shadow_summary.get('target_label', 'n/a')} |",
            "| Construction | Separate benchmark logistic + quality-weighted aggregate |",
            f"| P(Actionable Sell) | {classification_shadow_summary.get('probability_actionable_sell_label', 'n/a')} |",
            f"| Confidence Tier | {classification_shadow_summary.get('confidence_tier', 'n/a')} |",
            f"| Classifier Stance | {classification_shadow_summary.get('stance', 'n/a')} |",
            f"| Portfolio-aligned P(Actionable Sell) | "
            f"{classification_shadow_summary.get('probability_investable_pool_label', 'n/a')} "
            f"[{classification_shadow_summary.get('stance_investable_pool', 'n/a')}] |",
            f"| Investable Pool Confidence Tier | "
            f"{classification_shadow_summary.get('confidence_tier_investable_pool', 'n/a')} |",
            f"| Path B P(Actionable Sell) | "
            f"{classification_shadow_summary.get('probability_path_b_temp_scaled_label', 'n/a')} "
            f"[{classification_shadow_summary.get('stance_path_b', 'n/a')}] |",
            f"| Path B Confidence Tier | "
            f"{classification_shadow_summary.get('confidence_tier_path_b', 'n/a')} |",
            f"| Agreement with Live Recommendation | {classification_shadow_summary.get('agreement_label', 'n/a')} |",
            f"| Interpretation | {classification_shadow_summary.get('interpretation', 'n/a')} |",
        ]

    lines += [
        "",
        "---",
        "",
        "## Confidence Snapshot",
        "",
        f"- {confidence_snapshot['summary']}",
        "",
        "| Check | Current | Threshold | Status | Meaning |",
        "|-------|---------|-----------|--------|---------|",
    ]
    for row in confidence_snapshot["rows"]:
        lines.append(
            f"| {row['check']} | {row['current']} | {row['threshold']} | **{row['status']}** | {row['meaning']} |"
        )

    if model_drift_summary is not None:
        drift_status = (
            f"Warning: rolling IC has been below {config.DIAG_MIN_IC:.2f} for "
            f"{model_drift_summary.ic_below_threshold_streak} consecutive monthly snapshots."
            if model_drift_summary.drift_flag
            else "Stable: no sustained rolling-IC drift alert is active."
        )
        lines += [
            "",
            "---",
            "",
            "## Model Health",
            "",
            f"- Latest tracked month: **{model_drift_summary.as_of_month}**",
            f"- Rolling {model_drift_summary.window_months}M IC: **{model_drift_summary.rolling_ic:.4f}**",
            f"- Rolling {model_drift_summary.window_months}M Hit Rate: **{model_drift_summary.rolling_hit_rate:.1%}**",
            f"- Rolling {model_drift_summary.window_months}M ECE: **{model_drift_summary.rolling_ece:.1%}**",
            f"- IC breach streak: **{model_drift_summary.ic_below_threshold_streak}** month(s)",
            f"- Status: **{drift_status}**",
        ]

    # v32.2 + v32.3 — Decision Policy Backtest and Heuristic Comparison
    if policy_summary is not None:
        signal_policies_present = [p for p in SIGNAL_POLICIES if p in policy_summary]
        fixed_policies_present = [p for p in FIXED_POLICIES if p in policy_summary]
        model_policy = policy_summary.get("sign_hold_vs_sell")  # primary signal policy

        _policy_labels: dict[str, str] = {
            "always_sell_100": "Sell 100% (always)",
            "always_sell_50": "Sell 50% (always)",
            "always_hold_100": "Hold 100% (always)",
            "sign_hold_vs_sell": "Model: sign (hold if pred > 0)",
            "tiered_25_50_100": "Model: tiered 25/50/100",
            "neutral_band_2pct": "Model: neutral band ±2%",
            "neutral_band_3pct": "Model: neutral band ±3%",
            LIVE_MAPPING_POLICY: "**Live ACTIONABLE mapping** (consensus, per date)",
        }

        lines += [
            "",
            "---",
            "",
            "## Decision Policy Backtest",
            "",
            "> OOS performance of each decision policy applied to all historical "
            "model predictions.  "
            "\"Mean Return\" is the portfolio-weighted realized relative return per "
            "vesting event.  \"Cumulative\" is the sum across all events.  "
            "\"Capture Ratio\" is the fraction of oracle (always hold when positive) "
            "gains captured.  N = number of OOS events.  The live ACTIONABLE "
            "mapping is the policy used when every gate passes; it is replayed on "
            "the realised-only record and scored once per date (the others once "
            "per benchmark and date).",
            "",
            "### Fixed Heuristic Baselines",
            "",
            "| Policy | N | Mean Return | Cumulative | Capture Ratio |",
            "|--------|---|-------------|------------|---------------|",
        ]
        for p in fixed_policies_present:
            s = policy_summary[p]
            lines.append(
                f"| {_policy_labels.get(p, p)} | {s.n_obs} "
                f"| {s.mean_policy_return:+.2%} "
                f"| {s.cumulative_policy_return:+.2%} "
                f"| {s.capture_ratio:.1%} |"
            )

        lines += [
            "",
            "### Model-Driven Policies vs. Heuristics",
            "",
            "| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |",
            "|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|",
        ]
        model_policies_present = signal_policies_present + (
            [LIVE_MAPPING_POLICY] if LIVE_MAPPING_POLICY in policy_summary else []
        )
        for p in model_policies_present:
            s = policy_summary[p]
            up_sell = f"{s.uplift_vs_sell_all:+.2%}" if not np.isnan(s.uplift_vs_sell_all) else "n/a"
            up_hold = f"{s.uplift_vs_hold_all:+.2%}" if not np.isnan(s.uplift_vs_hold_all) else "n/a"
            up_50 = f"{s.uplift_vs_sell_50:+.2%}" if not np.isnan(s.uplift_vs_sell_50) else "n/a"
            capture = f"{s.capture_ratio:.1%}" if not np.isnan(s.capture_ratio) else "n/a"
            lines.append(
                f"| {_policy_labels.get(p, p)} | {s.n_obs} "
                f"| {s.mean_policy_return:+.2%} "
                f"| {s.cumulative_policy_return:+.2%} "
                f"| {up_sell} | {up_hold} | {up_50} | {capture} |"
            )
        lines += [""]

    # v34.0 — Portfolio Optimizer Status (Black-Litterman diagnostic shadow run)
    lines += [
        "",
        "---",
        "",
        "## Portfolio Optimizer Status",
        "",
    ]
    if bl_diagnostics is None:
        lines += ["> Black-Litterman optimizer: not run (insufficient data or error during build).", ""]
    elif bl_diagnostics.fallback_used:
        reason_str = bl_diagnostics.fallback_reason or "unknown"
        lines += [
            f"> ⚠️ **Optimizer fallback active** — Black-Litterman optimization could not converge "
            f"(`{reason_str}`).  Portfolio weights fall back to equal-weight allocation.  "
            f"This does not affect the primary recommendation; it is a diagnostic indicator.",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |",
            f"| Status | ⚠️ Fallback — {reason_str} |",
            f"| Active benchmarks | {bl_diagnostics.n_active_tickers} |",
            f"| View tickers incorporated | {bl_diagnostics.n_view_tickers} |",
            "",
        ]
    else:
        lines += [
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |",
            f"| Status | ✅ Converged |",
            f"| Active benchmarks | {bl_diagnostics.n_active_tickers} |",
            f"| View tickers incorporated | {bl_diagnostics.n_view_tickers} |",
            "",
        ]

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    n_outperform = (signals["signal"] == "OUTPERFORM").sum() if not signals.empty else 0
    n_total = len(signals)
    outperform_frac = f"{n_outperform}/{n_total} ({n_outperform / n_total:.0%})" if n_total else "0/0"

    if recommendation_mode["mode"] == "defer-to-tax-default":
        lines += [
            f"The point forecast leans {consensus.lower()}, and {outperform_frac} benchmarks favour outperformance, "
            f"but the broader quality gate is failing.",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.",
        ]
    elif recommendation_mode["mode"] == "monitoring-only":
        lines += [
            f"The point forecast leans {consensus.lower()}, and {outperform_frac} benchmarks favour outperformance, "
            "but the signal should be treated as monitoring information rather than an execution-grade edge.",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** unless future monthly runs improve the quality gate.",
        ]
    elif consensus == "OUTPERFORM":
        lines += [
            f"The ensemble has **{composite_confidence_tier.lower()} conviction** that PGR "
            f"will outperform a diversified ETF portfolio over the next 6 months.  "
            f"{outperform_frac} benchmarks favour outperformance.",
            "",
            f"Recommended action at next vesting event: **HOLD {1 - sell_pct:.0%}** of vesting RSUs.",
        ]
    elif consensus == "UNDERPERFORM":
        lines += [
            f"The ensemble predicts PGR will underperform the benchmark portfolio over the next "
            f"6 months ({composite_confidence_tier.lower()} conviction).  "
            f"Only {outperform_frac} benchmarks favour outperformance.",
            "",
            f"Recommended action at next vesting event: **SELL {sell_pct:.0%}** of vesting RSUs and diversify.",
        ]
    else:
        lines += [
            f"Model signal is weak (mean IC below threshold or mixed directional signals).  "
            f"{outperform_frac} benchmarks favour outperformance.",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** for risk management.",
        ]

    # Recommendation-layer support sections
    lines += [
        "",
        "---",
        "",
    ]

    lines += rendering.build_vest_decision_lines(next_vest_summary, recommendation_mode, sell_pct)
    if existing_holdings:
        lines += build_existing_holdings_markdown_lines(existing_holdings)
    if redeploy_buckets:
        lines += build_redeploy_markdown_lines(redeploy_buckets)
    if redeploy_portfolio:
        lines += render_redeploy_portfolio_markdown_lines(redeploy_portfolio)
    if (
        config.RECOMMENDATION_LAYER_MODE in {"live_with_shadow", "shadow_promoted"}
        and live_summary is not None
        and shadow_summary is not None
    ):
        lines += build_shadow_check_lines(
            live_summary,
            shadow_summary,
            active_path="shadow" if config.RECOMMENDATION_LAYER_MODE == "shadow_promoted" else "live",
        )

    # Per-benchmark table — include confidence columns when available
    lines += [
        "## Per-Benchmark Signals",
        "",
        "- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.",
        "- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.",
        "",
    ]

    show_cal_col = has_calibrated and "calibrated_prob_outperform" in signals.columns
    show_ci_col = "ci_lower" in signals.columns and not signals.empty

    if has_confidence and show_cal_col and show_ci_col:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(cal) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|------------|--------|",
        ]
    elif has_confidence and show_cal_col:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | IC | Hit Rate | P(cal) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----|----------|--------|------------|--------|",
        ]
    elif has_confidence and show_ci_col:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(Outperform) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----------|----------|----|----------|---------------|------------|--------|",
        ]
    elif has_confidence:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | IC | Hit Rate | P(Outperform) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----|----------|---------------|------------|--------|",
        ]
    else:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | IC | Hit Rate | Signal |",
            "|-----------|----------------|-------------|----------------|----|----------|--------|",
        ]

    if not signals.empty:
        for ticker, row in signals.iterrows():
            desc = rendering.ETF_DESCRIPTIONS.get(str(ticker), str(ticker))
            role = benchmark_role_for_ticker(str(ticker))["role"]
            pred = f"{row['predicted_relative_return']:+.2%}" if not pd.isna(row.get("predicted_relative_return")) else "n/a"
            ic_val = f"{row['ic']:.4f}" if not pd.isna(row.get("ic")) else "n/a"
            hr_val = f"{row['hit_rate']:.1%}" if not pd.isna(row.get("hit_rate")) else "n/a"
            sig = row.get("signal", "N/A")
            ci_lo_str = f"{row['ci_lower']:+.2%}" if show_ci_col and not pd.isna(row.get("ci_lower")) else "n/a"
            ci_hi_str = f"{row['ci_upper']:+.2%}" if show_ci_col and not pd.isna(row.get("ci_upper")) else "n/a"
            if has_confidence and show_cal_col and show_ci_col:
                prob_cal = f"{row['calibrated_prob_outperform']:.1%}" if not pd.isna(row.get("calibrated_prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ci_lo_str} | {ci_hi_str} | {ic_val} | {hr_val} | {prob_cal} | {tier} | {sig} |")
            elif has_confidence and show_cal_col:
                prob_cal = f"{row['calibrated_prob_outperform']:.1%}" if not pd.isna(row.get("calibrated_prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ic_val} | {hr_val} | {prob_cal} | {tier} | {sig} |")
            elif has_confidence and show_ci_col:
                prob = f"{row['prob_outperform']:.1%}" if not pd.isna(row.get("prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ci_lo_str} | {ci_hi_str} | {ic_val} | {hr_val} | {prob} | {tier} | {sig} |")
            elif has_confidence:
                prob = f"{row['prob_outperform']:.1%}" if not pd.isna(row.get("prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ic_val} | {hr_val} | {prob} | {tier} | {sig} |")
            else:
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ic_val} | {hr_val} | {sig} |")

    # v7.3 — Tax Context section
    lines += tax_lots.build_tax_context_lines(mean_predicted, mean_cal_prob, as_of=as_of)

    lines += [
        "",
        "---",
        "",
        f"*Generated by `{rendering.REPORT_GENERATED_BY}`*{'  [DRY RUN]' if dry_run else ''}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {path}")
