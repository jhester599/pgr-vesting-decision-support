"""Research-only x14 indicator synthesis helpers."""

from __future__ import annotations

from typing import Any


def select_indicator_candidate(
    *,
    evidence_rows: list[dict[str, Any]],
    x13_comparison: list[dict[str, Any]],
    x8_status: str,
    x11_status: str,
) -> dict[str, Any]:
    """Select one bounded indicator candidate from x13 evidence."""
    if x8_status != "not_ready" or x11_status != "continue_research":
        return {
            "status": "no_candidate",
            "reason": (
                "Broader x-series gating did not preserve the research-only "
                "status needed for an indicator candidate."
            ),
        }
    evidence_by_horizon = {
        int(row["horizon_months"]): row
        for row in evidence_rows
    }
    improving = [
        row for row in x13_comparison
        if bool(row.get("adjusted_beats_raw", False))
        and bool(
            evidence_by_horizon.get(int(row["horizon_months"]), {}).get(
                "x5_uses_no_change_pb",
                False,
            )
        )
    ]
    if not improving:
        return {
            "status": "no_candidate",
            "reason": (
                "No horizon cleared the x13 adjusted-support gate under the "
                "current x5 structural anchor."
            ),
        }
    improving = sorted(
        improving,
        key=lambda row: (row.get("mae_delta", 0.0), -row.get("horizon_months", 0)),
    )
    best = improving[0]
    return {
        "status": "candidate",
        "horizon_months": int(best["horizon_months"]),
        "model_name": best["adjusted_model_name"],
        "signal_family": "adjusted_structural_bvps_pb",
        "x11_status": x11_status,
        "rationale": (
            "Adjusted decomposition improved the raw path at this horizon, "
            "with x12 providing supportive but non-independent target-audit "
            "context while the broader x-series remained research-only."
        ),
    }


def build_x14_recommendation(
    *,
    candidate_status: str,
    x8_status: str,
    x11_status: str,
) -> dict[str, Any]:
    """Build research-only x14 recommendation."""
    if (
        candidate_status == "candidate"
        and x8_status == "not_ready"
        and x11_status == "continue_research"
    ):
        status = "research_indicator_candidate"
        rationale = (
            "One bounded indicator candidate is worth carrying into a later "
            "monthly-report/dashboard discussion, but not into production yet."
        )
    else:
        status = "no_indicator_candidate"
        rationale = "Evidence is not coherent enough to nominate an indicator."
    return {
        "status": status,
        "x11_status": x11_status,
        "production_changes": False,
        "shadow_changes": False,
        "rationale": rationale,
    }


def summarize_horizon_evidence(
    *,
    x3_rows: list[dict[str, Any]],
    x5_rows: list[dict[str, Any]],
    x12_rows: list[dict[str, Any]],
    x13_comparison: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize same-horizon evidence used to gate x14 candidates."""

    def _top_row(
        rows: list[dict[str, Any]],
        horizon_months: int,
        *,
        target_variant: str | None = None,
        target_kind: str | None = None,
    ) -> dict[str, Any] | None:
        candidates = [
            row for row in rows
            if int(row.get("horizon_months", -1)) == horizon_months
            and (target_variant is None or row.get("target_variant") == target_variant)
            and (target_kind is None or row.get("target_kind") == target_kind)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda row: int(row.get("rank", 999999)))

    horizons = sorted(
        {
            int(row["horizon_months"])
            for row in x13_comparison
        }
    )
    evidence: list[dict[str, Any]] = []
    for horizon in horizons:
        x3_return_top = _top_row(x3_rows, horizon, target_kind="return")
        x3_log_return_top = _top_row(x3_rows, horizon, target_kind="log_return")
        x5_top = _top_row(x5_rows, horizon)
        x12_raw = _top_row(x12_rows, horizon, target_variant="raw")
        x12_adjusted = _top_row(x12_rows, horizon, target_variant="adjusted")
        x13_row = next(
            row for row in x13_comparison
            if int(row["horizon_months"]) == horizon
        )
        x12_adjusted_beats_raw = (
            x12_raw is not None
            and x12_adjusted is not None
            and float(x12_adjusted["future_bvps_mae"]) < float(x12_raw["future_bvps_mae"])
        )
        evidence.append(
            {
                "horizon_months": horizon,
                "x3_return_leader_model_name": None if x3_return_top is None else x3_return_top["model_name"],
                "x3_return_leader_beats_no_change": None if x3_return_top is None else bool(
                    x3_return_top.get("beats_no_change", False)
                ),
                "x3_log_return_leader_model_name": None if x3_log_return_top is None else x3_log_return_top["model_name"],
                "x3_log_return_leader_beats_no_change": None if x3_log_return_top is None else bool(
                    x3_log_return_top.get("beats_no_change", False)
                ),
                "x5_leader_model_name": None if x5_top is None else x5_top["model_name"],
                "x5_uses_no_change_pb": None if x5_top is None else bool(
                    x5_top.get("pb_model_name") == "no_change_pb"
                ),
                "x12_adjusted_beats_raw": x12_adjusted_beats_raw,
                "x12_supports_adjusted_target_family": x12_adjusted_beats_raw,
                "x12_raw_best_model_name": None if x12_raw is None else x12_raw["model_name"],
                "x12_adjusted_best_model_name": None if x12_adjusted is None else x12_adjusted["model_name"],
                "x13_adjusted_beats_raw": bool(x13_row["adjusted_beats_raw"]),
                "x13_raw_model_name": x13_row["raw_model_name"],
                "x13_adjusted_model_name": x13_row["adjusted_model_name"],
                "x13_mae_delta": float(x13_row["mae_delta"]),
            }
        )
    return evidence
