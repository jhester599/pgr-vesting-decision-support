"""Research-only x16 indicator packaging helpers."""

from __future__ import annotations

from typing import Any


def build_indicator_package(
    *,
    x14_summary: dict[str, Any],
    x15_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build a research-only indicator package from x14/x15 artifacts."""
    candidate = x14_summary["indicator_candidate"]
    x15_rows = x15_summary.get("ranked_rows", [])
    six_month_x15 = [
        row for row in x15_rows
        if int(row.get("horizon_months", -1)) == 6
    ]
    pb_anchor_policy = (
        "retain_no_change_pb"
        if six_month_x15 and six_month_x15[0]["model_name"] == "no_change_pb_overlay"
        else "review_pb_anchor"
    )
    return {
        "indicator_name": "adjusted_structural_bvps_pb_6m",
        "status": x14_summary["recommendation"]["status"],
        "signal_family": candidate["signal_family"],
        "horizon_months": int(candidate["horizon_months"]),
        "model_name": candidate["model_name"],
        "pb_anchor_policy": pb_anchor_policy,
        "production_changes": False,
        "shadow_changes": False,
        "display_fields": [
            "forecast_horizon_months",
            "future_price_estimate",
            "implied_upside_pct",
            "current_pb_anchor",
            "bvps_path_model",
            "research_confidence",
            "caveat_text",
        ],
        "caveats": [
            "research_only",
            "no_shadow_wiring",
            "pb_anchor_is_no_change",
            "x15_overlay_did_not_improve_anchor",
        ],
    }


def build_peer_review_prompt(
    *,
    indicator_package: dict[str, Any],
    x14_summary: dict[str, Any],
    x15_summary: dict[str, Any],
) -> str:
    """Build a reusable deep-research prompt for external review."""
    x15_six_month = next(
        (
            row for row in x15_summary["ranked_rows"]
            if int(row.get("horizon_months", -1)) == 6 and int(row.get("rank", 1)) == 1
        ),
        x15_summary["ranked_rows"][0],
    )
    return "\n".join(
        [
            "# PGR x-series peer review prompt",
            "",
            "Please review the following research-only indicator candidate and",
            "challenge it aggressively.",
            "",
            f"- Candidate indicator: `{indicator_package['indicator_name']}`",
            f"- Signal family: `{indicator_package['signal_family']}`",
            f"- Horizon: `{indicator_package['horizon_months']}m`",
            f"- Current x14 candidate model: `{x14_summary['indicator_candidate']['model_name']}`",
            f"- Current P/B anchor policy after x15: `{indicator_package['pb_anchor_policy']}`",
            f"- x15 best 6m row overall: `{x15_six_month['model_name']}`",
            "",
            "Please address:",
            "1. Whether the structural BVPS x no-change-P/B framing is a",
            "   sensible research-only indicator under this sample regime.",
            "2. What missing P/B features or transformations might plausibly",
            "   improve the multiple leg without opening a large overfitting",
            "   surface.",
            "3. Whether there are cleaner insurer-specific capital or valuation",
            "   anchors that should replace or complement no-change P/B.",
            "4. What failure modes would make this indicator misleading in a",
            "   future monthly report/dashboard.",
            "5. What bounded next-step experiments should be run before any",
            "   production or shadow discussion.",
        ]
    )
