"""Research-only x24 indicator contract helpers."""

from __future__ import annotations

from typing import Any


def build_x24_indicator_contract(
    *,
    x16_package: dict[str, Any],
    x23_package: dict[str, Any],
    x8_summary: dict[str, Any],
) -> dict[str, Any]:
    """Bundle the surviving x-series indicators into one research-only contract."""
    structural_status = str(x16_package.get("status", "continue_research"))
    dividend_status = str(
        x23_package.get("recommendation", {}).get("status", "continue_research")
    )
    shadow_status = str(
        x8_summary.get("shadow_readiness", {}).get("status", "not_ready")
    )
    caveats = [
        "research_only",
        "no_production_wiring",
        "no_shadow_wiring",
    ]
    if shadow_status != "candidate_for_shadow_plan":
        caveats.append("shadow_not_ready")
    dividend_lane = x23_package.get("dividend_lane_shape", {})
    return {
        "version": "x24",
        "artifact_classification": "research",
        "bundle_name": "x_series_research_indicator_bundle",
        "status": "research_indicator_bundle_candidate",
        "production_changes": False,
        "shadow_changes": False,
        "caveats": caveats,
        "structural_signal": {
            "indicator_name": x16_package.get("indicator_name"),
            "signal_family": x16_package.get("signal_family"),
            "horizon_months": x16_package.get("horizon_months"),
            "model_name": x16_package.get("model_name"),
            "status": structural_status,
            "pb_anchor_policy": x16_package.get("pb_anchor_policy"),
        },
        "dividend_signal": {
            "indicator_name": "special_dividend_size_watch",
            "target_scale": dividend_lane.get("best_size_target_scale"),
            "model_name": dividend_lane.get("best_size_model_name"),
            "feature_set": dividend_lane.get("best_size_feature_set"),
            "occurrence_status": dividend_lane.get("occurrence_status"),
            "policy_change_date": dividend_lane.get("policy_change_date"),
            "status": dividend_status,
            "timing": "November month-end annual snapshot",
        },
        "report_surface": {
            "section_name": "x_series_research_watch",
            "display_fields": [
                "structural_indicator_name",
                "structural_horizon_months",
                "structural_model_name",
                "dividend_indicator_name",
                "dividend_target_scale",
                "dividend_timing",
                "bundle_status",
                "caveat_text",
            ],
        },
        "decision_context": {
            "shadow_readiness_status": shadow_status,
            "structural_source": "x16",
            "dividend_source": "x23",
        },
    }


def build_x24_peer_review_prompt(
    *,
    contract: dict[str, Any],
    x8_summary: dict[str, Any],
    x11_summary: dict[str, Any],
    x23_package: dict[str, Any],
) -> str:
    """Draft a holistic peer-review prompt for the bundled x-series indicators."""
    return f"""# x24 X-Series Bundle Peer Review Prompt

Please review the current research-only x-series indicator bundle for PGR.

Current bundle:
- Structural signal: `{contract['structural_signal']['indicator_name']}`
- Structural model: `{contract['structural_signal']['model_name']}`
- Dividend signal: `{contract['dividend_signal']['indicator_name']}`
- Dividend target scale: `{contract['dividend_signal']['target_scale']}`
- Dividend occurrence status: `{contract['dividend_signal']['occurrence_status']}`

Existing synthesis context:
- x8 shadow readiness: `{x8_summary.get('shadow_readiness', {}).get('status')}`
- x11 recommendation: `{x11_summary.get('recommendation', {}).get('status')}`
- x23 recommendation: `{x23_package.get('recommendation', {}).get('status')}`

Please challenge:
1. whether the structural 6m path and the annual dividend-size watch belong in the same future monthly research surface
2. whether the structural side still needs a stronger P/B leg before any practical packaging
3. whether the dividend-size watch should remain annual-only or be turned into a monthly tracked state variable
4. what evidence would justify promotion from research-only bundle to a reporting-only monthly/dashboard candidate

Constraints:
- strict temporal validation only
- no K-Fold CV
- no production wiring
- prefer low-complexity, small-sample-safe models
"""
