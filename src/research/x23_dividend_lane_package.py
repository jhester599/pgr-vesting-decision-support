"""Research-only x23 helpers for packaging the dividend lane."""

from __future__ import annotations

from typing import Any


def build_x23_recommendation(
    *,
    overlap_positive_rate: float,
    x22_best_row: dict[str, Any],
) -> dict[str, Any]:
    """Build a conservative x23 dividend-lane recommendation."""
    occurrence_status = (
        "underidentified_post_policy"
        if overlap_positive_rate in (0.0, 1.0)
        else "partially_identifiable"
    )
    if (
        occurrence_status == "underidentified_post_policy"
        and x22_best_row.get("feature_set") != "baseline_only"
        and x22_best_row.get("target_scale") == "to_current_bvps"
    ):
        status = "research_size_indicator_candidate"
        rationale = (
            "Occurrence remains underidentified, but current-BVPS-normalized "
            "size has beaten raw-dollar and baseline challengers."
        )
    else:
        status = "continue_research"
        rationale = (
            "The dividend lane is not yet strong enough to package beyond "
            "research diagnostics."
        )
    return {
        "status": status,
        "occurrence_status": occurrence_status,
        "production_changes": False,
        "shadow_changes": False,
        "rationale": rationale,
    }


def build_x23_summary(payloads: dict[str, Any]) -> dict[str, Any]:
    """Build the x23 dividend-lane package payload."""
    x20 = payloads["x20"]
    x21 = payloads["x21"]
    x22 = payloads["x22"]
    overlap_positive_rate = float(
        x20.get("overlap_comparison", {}).get("overlap_positive_rate", 0.0)
    )
    x22_best_row = x22.get("ranked_rows", [{}])[0]
    recommendation = build_x23_recommendation(
        overlap_positive_rate=overlap_positive_rate,
        x22_best_row=x22_best_row,
    )
    return {
        "version": "x23",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "dividend_lane_shape": {
            "policy_change_date": x20.get("policy_change_date"),
            "occurrence_status": recommendation["occurrence_status"],
            "best_size_target_scale": x22_best_row.get("target_scale"),
            "best_size_model_name": x22_best_row.get("model_name"),
            "best_size_feature_set": x22_best_row.get("feature_set"),
            "x21_best_target_scale": x21.get("ranked_rows", [{}])[0].get("target_scale"),
        },
        "recommendation": recommendation,
        "decision_questions": [
            {
                "question": "Is occurrence ready for practical use post-policy?",
                "criterion": "Overlap years must not be one-class.",
                "answer": recommendation["occurrence_status"],
            },
            {
                "question": "What dividend target scale survived x22?",
                "criterion": "Must beat raw dollars and baseline challengers on dollar MAE.",
                "answer": str(x22_best_row.get("target_scale")),
            },
            {
                "question": "How should the broader x-series use this lane next?",
                "criterion": "Only package a research-size signal unless occurrence becomes identifiable.",
                "answer": recommendation["status"],
            },
        ],
    }
