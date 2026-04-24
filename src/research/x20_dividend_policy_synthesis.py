"""Research-only x20 synthesis helpers for dividend-policy rebuild work."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _detail_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    frame["actual_excess"] = pd.to_numeric(frame["actual_excess"], errors="coerce")
    frame["expected_value_prediction"] = pd.to_numeric(
        frame["expected_value_prediction"],
        errors="coerce",
    )
    return frame.dropna(subset=["snapshot_date", "actual_excess", "expected_value_prediction"])


def _leader_on_dates(
    rows: list[dict[str, Any]],
    snapshot_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    frame = _detail_frame(rows)
    if frame.empty or snapshot_dates.empty:
        return {}
    overlap = frame[frame["snapshot_date"].isin(snapshot_dates)].copy()
    if overlap.empty:
        return {}
    overlap["abs_error"] = np.abs(
        overlap["expected_value_prediction"] - overlap["actual_excess"]
    )
    overlap["is_positive"] = (overlap["actual_excess"] > 0.0).astype(float)
    summary = (
        overlap.groupby(["feature_set", "model_name"], dropna=False)
        .agg(
            overlap_ev_mae=("abs_error", "mean"),
            overlap_positive_rate=("is_positive", "mean"),
            overlap_n_obs=("snapshot_date", "size"),
        )
        .reset_index()
    )
    best = summary.sort_values(
        ["overlap_ev_mae", "feature_set", "model_name"],
        ascending=[True, True, True],
        kind="mergesort",
    ).iloc[0]
    return {
        "feature_set": best["feature_set"],
        "model_name": best["model_name"],
        "overlap_ev_mae": float(best["overlap_ev_mae"]),
        "overlap_positive_rate": float(best["overlap_positive_rate"]),
        "overlap_n_obs": int(best["overlap_n_obs"]),
    }


def compare_overlap_leaders(
    x10_detail_rows: list[dict[str, Any]],
    x19_detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the best x10 and x19 rows on shared post-policy test years."""
    x10 = _detail_frame(x10_detail_rows)
    x19 = _detail_frame(x19_detail_rows)
    if x10.empty or x19.empty:
        return {"x19_beats_x10_overlap": False, "overlap_n_obs": 0}
    overlap_dates = pd.DatetimeIndex(
        sorted(set(x10["snapshot_date"]).intersection(set(x19["snapshot_date"])))
    )
    if overlap_dates.empty:
        return {"x19_beats_x10_overlap": False, "overlap_n_obs": 0}
    x10_best = _leader_on_dates(x10_detail_rows, overlap_dates)
    x19_best = _leader_on_dates(x19_detail_rows, overlap_dates)
    if not x10_best or not x19_best:
        return {"x19_beats_x10_overlap": False, "overlap_n_obs": 0}
    delta = float(x19_best["overlap_ev_mae"] - x10_best["overlap_ev_mae"])
    return {
        "x10_feature_set": x10_best["feature_set"],
        "x10_model_name": x10_best["model_name"],
        "x10_overlap_ev_mae": float(x10_best["overlap_ev_mae"]),
        "x19_feature_set": x19_best["feature_set"],
        "x19_model_name": x19_best["model_name"],
        "x19_overlap_ev_mae": float(x19_best["overlap_ev_mae"]),
        "overlap_positive_rate": float(x19_best["overlap_positive_rate"]),
        "overlap_n_obs": int(x19_best["overlap_n_obs"]),
        "ev_mae_delta": round(delta, 12),
        "x19_beats_x10_overlap": bool(delta < 0.0),
    }


def compare_sample_scope(
    *,
    x10_summary_rows: list[dict[str, Any]],
    x19_summary_rows: list[dict[str, Any]],
    post_policy_snapshot_count: int,
) -> dict[str, Any]:
    """Summarize sample-size and confidence constraints."""
    x10_n_obs = int(x10_summary_rows[0].get("n_obs", 0)) if x10_summary_rows else 0
    x19_n_obs = int(x19_summary_rows[0].get("n_obs", 0)) if x19_summary_rows else 0
    confidence = "low" if x19_n_obs < 8 else "moderate"
    return {
        "x10_oos_n_obs": x10_n_obs,
        "x19_oos_n_obs": x19_n_obs,
        "post_policy_snapshot_count": int(post_policy_snapshot_count),
        "confidence": confidence,
    }


def build_x20_recommendation(
    *,
    x19_beats_x10_overlap: bool,
    overlap_n_obs: int,
    overlap_positive_rate: float,
) -> dict[str, Any]:
    """Build a conservative x20 recommendation."""
    if overlap_n_obs < 3:
        status = "continue_research"
        rationale = "Post-policy overlap is too small to upgrade the dividend lane."
    elif overlap_positive_rate in (0.0, 1.0):
        status = "continue_research_size_only"
        rationale = (
            "Post-policy overlap is one-class on occurrence, so only size-target "
            "experiments are currently identifiable."
        )
    elif x19_beats_x10_overlap:
        status = "continue_research_post_policy"
        rationale = "Cleaner post-policy labels helped on overlapping years."
    else:
        status = "continue_research"
        rationale = "The rebuilt lane has not yet cleared the overlap benchmark."
    return {
        "status": status,
        "production_changes": False,
        "shadow_changes": False,
        "rationale": rationale,
    }


def build_x20_summary(payloads: dict[str, Any]) -> dict[str, Any]:
    """Build the x20 synthesis payload."""
    x18_summary = payloads.get("x18", {})
    x10_summary = payloads.get("x10", {})
    x19_summary = payloads.get("x19", {})
    overlap = compare_overlap_leaders(
        x10_summary.get("detail_rows", []),
        x19_summary.get("detail_rows", []),
    )
    sample_scope = compare_sample_scope(
        x10_summary_rows=x10_summary.get("ranked_rows", []),
        x19_summary_rows=x19_summary.get("ranked_rows", []),
        post_policy_snapshot_count=int(
            x18_summary.get("post_policy_snapshot_count", 0)
        ),
    )
    recommendation = build_x20_recommendation(
        x19_beats_x10_overlap=bool(overlap.get("x19_beats_x10_overlap", False)),
        overlap_n_obs=int(overlap.get("overlap_n_obs", 0)),
        overlap_positive_rate=float(overlap.get("overlap_positive_rate", 0.0)),
    )
    return {
        "version": "x20",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "policy_change_date": x18_summary.get("policy_change_date"),
        "sample_scope": sample_scope,
        "overlap_comparison": overlap,
        "recommendation": recommendation,
        "decision_questions": [
            {
                "question": "Did the post-policy rebuild beat x10 on overlapping years?",
                "criterion": "x19 must lower overlap expected-value MAE.",
                "answer": "yes" if overlap.get("x19_beats_x10_overlap") else "no",
            },
            {
                "question": "Is occurrence currently identifiable post-policy?",
                "criterion": "Overlap years must contain both positive and zero outcomes.",
                "answer": (
                    "no"
                    if float(overlap.get("overlap_positive_rate", 0.0)) in (0.0, 1.0)
                    else "yes"
                ),
            },
            {
                "question": "What should x21 prioritize?",
                "criterion": "If occurrence is one-class, move to size-target scaling.",
                "answer": recommendation["status"],
            },
        ],
    }
