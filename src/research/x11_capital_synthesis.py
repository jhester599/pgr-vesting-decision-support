"""Research-only x11 synthesis helpers for BVPS and capital lanes."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _leaders_by_horizon(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["rank"] = pd.to_numeric(frame.get("rank", 1), errors="coerce").fillna(1)
    return frame.sort_values(
        ["horizon_months", "rank"],
        ascending=[True, True],
        kind="mergesort",
    ).groupby("horizon_months", sort=True).head(1)


def compare_bvps_leaders(
    x4_rows: list[dict[str, Any]],
    x9_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare x9 BVPS leaders against x4 by horizon."""
    x4 = _leaders_by_horizon(x4_rows)
    x9 = _leaders_by_horizon(x9_rows)
    if x4.empty or x9.empty:
        return []
    merged = x4.merge(
        x9,
        on="horizon_months",
        how="inner",
        suffixes=("_x4", "_x9"),
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        delta = float(row["future_bvps_mae_x9"] - row["future_bvps_mae_x4"])
        feature_block = row.get("feature_block_x9", row.get("feature_block"))
        rows.append(
            {
                "horizon_months": int(row["horizon_months"]),
                "x4_model_name": row["model_name_x4"],
                "x9_model_name": row["model_name_x9"],
                "x9_feature_block": feature_block,
                "x4_future_bvps_mae": float(row["future_bvps_mae_x4"]),
                "x9_future_bvps_mae": float(row["future_bvps_mae_x9"]),
                "future_bvps_mae_delta": round(delta, 12),
                "x9_beats_x4": bool(delta < 0.0),
            }
        )
    return rows


def compare_dividend_leaders(
    x6_rows: list[dict[str, Any]],
    x10_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare x10 special-dividend leader against x6."""
    if not x6_rows or not x10_rows:
        return {"x10_beats_x6": False, "confidence": "low"}
    x6_best = x6_rows[0]
    x10_best = x10_rows[0]
    delta = float(
        x10_best["expected_value_mae"] - x6_best["expected_value_mae"]
    )
    n_obs = int(x10_best.get("n_obs", 0))
    confidence = "low" if n_obs < 30 else "moderate"
    return {
        "x6_model_name": x6_best["model_name"],
        "x10_model_name": x10_best["model_name"],
        "x10_feature_set": x10_best.get("feature_set"),
        "x6_expected_value_mae": float(x6_best["expected_value_mae"]),
        "x10_expected_value_mae": float(x10_best["expected_value_mae"]),
        "expected_value_mae_delta": round(delta, 12),
        "x10_beats_x6": bool(delta < 0.0),
        "n_obs": n_obs,
        "confidence": confidence,
    }


def build_x11_recommendation(
    *,
    x9_beating_horizons: int,
    x10_beats_x6: bool,
    dividend_confidence: str,
) -> dict[str, Any]:
    """Build conservative x11 research recommendation."""
    if (
        x9_beating_horizons >= 3
        and x10_beats_x6
        and dividend_confidence != "low"
    ):
        status = "candidate_for_shadow_plan"
        rationale = (
            "BVPS bridge improves most horizons and dividend confidence is "
            "no longer low"
        )
    else:
        status = "continue_research"
        rationale = (
            "BVPS bridge evidence is horizon-specific and annual dividend "
            "confidence remains constrained"
        )
    return {
        "status": status,
        "production_changes": False,
        "shadow_changes": False,
        "rationale": rationale,
    }


def build_x11_summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build x11 synthesis payload."""
    bvps_comparison = compare_bvps_leaders(
        payloads["x4"].get("ranked_rows", []),
        payloads["x9"].get("ranked_rows", []),
    )
    dividend_comparison = compare_dividend_leaders(
        payloads["x6"].get("ranked_rows", []),
        payloads["x10"].get("ranked_rows", []),
    )
    x9_beating_horizons = sum(
        1 for row in bvps_comparison if row["x9_beats_x4"]
    )
    recommendation = build_x11_recommendation(
        x9_beating_horizons=x9_beating_horizons,
        x10_beats_x6=bool(dividend_comparison.get("x10_beats_x6", False)),
        dividend_confidence=str(dividend_comparison.get("confidence", "low")),
    )
    return {
        "version": "x11",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "bvps_comparison": bvps_comparison,
        "dividend_comparison": dividend_comparison,
        "recommendation": recommendation,
        "decision_questions": [
            {
                "question": "Did x9 improve BVPS enough to supersede x4?",
                "criterion": "x9 must beat x4 on future BVPS MAE in most horizons.",
                "answer": f"x9 beat x4 in {x9_beating_horizons} horizons.",
            },
            {
                "question": "Did x9 capital features help special dividends?",
                "criterion": "x10 must beat x6 on expected-value MAE.",
                "answer": (
                    "yes"
                    if dividend_comparison.get("x10_beats_x6", False)
                    else "no"
                ),
            },
            {
                "question": "Is this ready for shadow wiring?",
                "criterion": "Monthly and annual evidence must both be robust.",
                "answer": recommendation["status"],
            },
        ],
    }
