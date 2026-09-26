"""Structured monthly summary payload writer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import config


def build_hold_vs_sell_label(sell_pct: float) -> str:
    """Return the top-line hold vs sell instruction."""
    hold_pct = max(0.0, 1.0 - float(sell_pct))
    return f"Hold {hold_pct:.0%} / Sell {float(sell_pct):.0%} of the next vest tranche"


def build_actionability_label(recommendation_mode: str) -> str:
    """Return a user-facing actionability label from the recommendation mode."""
    if str(recommendation_mode).upper() == "ACTIONABLE":
        return "Yes — this month is actionable."
    if str(recommendation_mode).upper() == "MONITORING-ONLY":
        return "Not yet — monitor, but keep the default vest rule."
    return "No — follow the default tax/diversification rule."


def build_decision_headline(
    recommendation_mode: str,
    sell_pct: float,
) -> str:
    """Return a compact decision headline for report surfaces."""
    return (
        f"{build_hold_vs_sell_label(sell_pct)}. "
        f"{build_actionability_label(recommendation_mode)}"
    )


def _format_pct(value: float | None, decimals: int = 2) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.{decimals}f}%"


def _format_signed_pct(value: float | None, decimals: int = 2) -> str | None:
    if value is None:
        return None
    return f"{value * 100:+.{decimals}f}%"


def _format_number(value: float | None, decimals: int = 4) -> str | None:
    if value is None:
        return None
    return f"{value:.{decimals}f}"


def _build_cross_check_summary(
    consensus_shadow_df: pd.DataFrame | None,
    *,
    visible_in_primary_surfaces: bool,
) -> dict[str, Any]:
    """Summarize the shadow comparison without requiring markdown parsing."""
    summary: dict[str, Any] = {
        "visible_in_primary_surfaces": visible_in_primary_surfaces,
        "artifact_retained": True,
        "retired_reason": (
            None
            if visible_in_primary_surfaces
            else (
                "Equal-weight consensus remained stable during the post-promotion "
                "stabilization window, so the comparison now stays diagnostic-only."
            )
        ),
        "mode_agreement": None,
        "sell_pct_agreement": None,
        "consensus_agreement": None,
        "live_variant": None,
        "shadow_variant": None,
        "live_recommendation_mode": None,
        "shadow_recommendation_mode": None,
        "live_sell_pct": None,
        "shadow_sell_pct": None,
    }
    if consensus_shadow_df is None or consensus_shadow_df.empty:
        return summary

    live_rows = consensus_shadow_df[consensus_shadow_df["is_live_path"]]
    shadow_rows = consensus_shadow_df[~consensus_shadow_df["is_live_path"]]
    if live_rows.empty or shadow_rows.empty:
        return summary

    live = live_rows.iloc[0]
    shadow = shadow_rows.iloc[0]
    live_mode = str(live["recommendation_mode"])
    shadow_mode = str(shadow["recommendation_mode"])
    live_sell_pct = float(live["recommended_sell_pct"])
    shadow_sell_pct = float(shadow["recommended_sell_pct"])
    live_consensus = str(live["consensus"])
    shadow_consensus = str(shadow["consensus"])

    summary.update(
        {
            "mode_agreement": live_mode == shadow_mode,
            "sell_pct_agreement": abs(live_sell_pct - shadow_sell_pct) <= 1e-9,
            "consensus_agreement": live_consensus == shadow_consensus,
            "live_variant": str(live["variant"]),
            "shadow_variant": str(shadow["variant"]),
            "live_recommendation_mode": live_mode,
            "shadow_recommendation_mode": shadow_mode,
            "live_sell_pct": live_sell_pct,
            "shadow_sell_pct": shadow_sell_pct,
        }
    )
    return summary


def build_monthly_summary_payload(
    *,
    as_of_date: str,
    run_date: str,
    recommendation_layer_label: str,
    consensus: str,
    confidence_tier: str,
    recommendation_mode: str,
    sell_pct: float,
    mean_predicted: float,
    mean_ic: float,
    mean_hit_rate: float,
    mean_prob_outperform: float | None,
    calibrated_prob_outperform: float | None,
    aggregate_oos_r2: float | None,
    aggregate_nw_ic: float | None,
    warnings: list[str],
    signals: pd.DataFrame,
    benchmark_quality_df: pd.DataFrame | None,
    consensus_shadow_df: pd.DataFrame | None,
    visible_cross_check: bool,
    classification_shadow_summary: dict[str, Any] | None = None,
    shadow_gate_overlay: dict[str, Any] | None = None,
    classification_shadow_variants: list[dict[str, Any]] | None = None,
    decision_overlay_variants: list[dict[str, Any]] | None = None,
    model_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable monthly summary payload.

    ``model_health`` (review 2026-09-25, WP7) carries the gate statuses and
    the realised-only health metrics behind the recommendation mode; see
    ``build_model_health_payload``.
    """
    benchmark_count = int(len(signals)) if not signals.empty else 0
    quality_count = int(len(benchmark_quality_df)) if benchmark_quality_df is not None else 0
    cross_check = _build_cross_check_summary(
        consensus_shadow_df,
        visible_in_primary_surfaces=visible_cross_check,
    )
    return {
        "schema_version": 3,
        "as_of_date": as_of_date,
        "run_date": run_date,
        "artifacts": {
            "recommendation_md": "recommendation.md",
            "diagnostic_md": "diagnostic.md",
            "signals_csv": "signals.csv",
            "benchmark_quality_csv": "benchmark_quality.csv",
            "consensus_shadow_csv": "consensus_shadow.csv",
            "classification_shadow_csv": "classification_shadow.csv",
            "decision_overlays_csv": "decision_overlays.csv",
            "dashboard_html": "dashboard.html",
            "monthly_summary_json": "monthly_summary.json",
            "run_manifest_json": "run_manifest.json",
        },
        "recommendation_layer": {
            "label": recommendation_layer_label,
            "visible_cross_check": visible_cross_check,
        },
        "recommendation": {
            "decision_headline": build_decision_headline(recommendation_mode, sell_pct),
            "hold_vs_sell_label": build_hold_vs_sell_label(sell_pct),
            "actionability_label": build_actionability_label(recommendation_mode),
            "signal": consensus,
            "confidence_tier": confidence_tier,
            "signal_label": f"{consensus} ({confidence_tier} CONFIDENCE)",
            "recommendation_mode": recommendation_mode,
            "recommended_sell_pct": sell_pct,
            "recommended_sell_pct_label": _format_pct(sell_pct, decimals=0),
            "predicted_6m_relative_return": mean_predicted,
            "predicted_6m_relative_return_label": _format_signed_pct(mean_predicted),
            "prob_outperform_raw": mean_prob_outperform,
            "prob_outperform_raw_label": _format_pct(mean_prob_outperform, decimals=1),
            "prob_outperform_calibrated": calibrated_prob_outperform,
            "prob_outperform_calibrated_label": _format_pct(
                calibrated_prob_outperform,
                decimals=1,
            ),
            "mean_ic": mean_ic,
            "mean_ic_label": _format_number(mean_ic),
            "mean_hit_rate": mean_hit_rate,
            "mean_hit_rate_label": _format_pct(mean_hit_rate, decimals=1),
            "aggregate_oos_r2": aggregate_oos_r2,
            "aggregate_oos_r2_label": _format_signed_pct(aggregate_oos_r2),
            "aggregate_nw_ic": aggregate_nw_ic,
            "aggregate_nw_ic_label": _format_number(aggregate_nw_ic),
        },
        "coverage": {
            "benchmark_count": benchmark_count,
            "benchmark_quality_count": quality_count,
        },
        "warnings": warnings,
        "cross_check": cross_check,
        "classification_shadow": classification_shadow_summary,
        "shadow_gate_overlay": shadow_gate_overlay,
        "classification_shadow_variants": classification_shadow_variants or [],
        "decision_overlay_variants": decision_overlay_variants or [],
        "model_health": model_health or {},
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_model_health_payload(
    *,
    mean_ic: float,
    aggregate_health: dict[str, Any] | None,
    representative_cpcv: Any | None,
    cal_result: Any | None,
    conformal_trailing_coverage: float | None,
    shrinkage_alpha: float | None,
    quality_weighted_ic: float | None = None,
) -> dict[str, Any]:
    """Machine-readable gates and health metrics (review 2026-09-25, WP7)."""
    from src.reporting.decision_rendering import evaluate_quality_gates

    gates = evaluate_quality_gates(mean_ic, aggregate_health, representative_cpcv)
    health = aggregate_health or {}
    cpcv: dict[str, Any] = {"available": representative_cpcv is not None}
    if representative_cpcv is not None:
        cpcv.update(
            {
                "verdict": str(getattr(representative_cpcv, "stability_verdict", "UNKNOWN")),
                "positive_paths": int(getattr(representative_cpcv, "n_positive_paths", 0)),
                "n_paths": int(getattr(representative_cpcv, "n_paths", 0)),
                "mean_path_ic": _finite_or_none(getattr(representative_cpcv, "mean_ic", None)),
                "gates_recommendation": False,
            }
        )
    calibration: dict[str, Any] = {}
    if cal_result is not None:
        calibration = {
            "method": str(getattr(cal_result, "method", "")),
            "prequential_ece": _finite_or_none(getattr(cal_result, "ece", None)),
            "ece_ci_lower": _finite_or_none(getattr(cal_result, "ece_ci_lower", None)),
            "ece_ci_upper": _finite_or_none(getattr(cal_result, "ece_ci_upper", None)),
            "n_obs": int(getattr(cal_result, "n_obs", 0)),
        }
    return {
        "metrics_version": config.MODEL_HEALTH_METRICS_VERSION,
        "gates": [
            {
                "name": gate.name,
                "status": gate.status,
                "value": gate.value,
                "current": gate.current,
                "threshold": gate.threshold,
            }
            for gate in gates
        ],
        "equal_weight_mean_ic": _finite_or_none(mean_ic),
        "quality_weighted_mean_ic": _finite_or_none(quality_weighted_ic),
        "aggregate_oos_r2": _finite_or_none(health.get("oos_r2")),
        "pooled_ic": _finite_or_none(health.get("nw_ic")),
        "pooled_ic_p_value": _finite_or_none(health.get("nw_pval")),
        "pooled_ic_p_value_method": health.get("ic_p_value_method"),
        "hit_rate": _finite_or_none(health.get("agg_hit")),
        "base_rate": _finite_or_none(health.get("base_rate")),
        "constant_rule_hit_rate": _finite_or_none(health.get("constant_rule_hit_rate")),
        "hit_rate_excess": _finite_or_none(health.get("hit_rate_excess")),
        "pesaran_timmermann_p_value": _finite_or_none(health.get("pt_p_value")),
        "clark_west_p_value": _finite_or_none(health.get("cw_p_value")),
        "shrinkage_alpha": _finite_or_none(shrinkage_alpha),
        "conformal_trailing_coverage": _finite_or_none(conformal_trailing_coverage),
        "calibration": calibration,
        "cpcv": cpcv,
    }


def write_monthly_summary(out_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the monthly summary JSON artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "monthly_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
