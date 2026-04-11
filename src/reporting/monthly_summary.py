"""Structured monthly summary payload writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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
) -> dict[str, Any]:
    """Build the machine-readable monthly summary payload."""
    benchmark_count = int(len(signals)) if not signals.empty else 0
    quality_count = int(len(benchmark_quality_df)) if benchmark_quality_df is not None else 0
    cross_check = _build_cross_check_summary(
        consensus_shadow_df,
        visible_in_primary_surfaces=visible_cross_check,
    )
    return {
        "schema_version": 2,
        "as_of_date": as_of_date,
        "run_date": run_date,
        "artifacts": {
            "recommendation_md": "recommendation.md",
            "diagnostic_md": "diagnostic.md",
            "signals_csv": "signals.csv",
            "benchmark_quality_csv": "benchmark_quality.csv",
            "consensus_shadow_csv": "consensus_shadow.csv",
            "dashboard_html": "dashboard.html",
            "monthly_summary_json": "monthly_summary.json",
            "run_manifest_json": "run_manifest.json",
        },
        "recommendation_layer": {
            "label": recommendation_layer_label,
            "visible_cross_check": visible_cross_check,
        },
        "recommendation": {
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
    }


def write_monthly_summary(out_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the monthly summary JSON artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "monthly_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
