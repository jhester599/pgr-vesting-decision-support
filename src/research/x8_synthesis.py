"""Research-only x8 synthesis utilities for x-series artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records."""
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def extract_horizon_leaders(
    rows: list[dict[str, Any]],
    *,
    lane: str,
    primary_metric: str,
) -> list[dict[str, Any]]:
    """Select the top-ranked row for each horizon in a lane."""
    frame = pd.DataFrame(rows)
    if frame.empty or "horizon_months" not in frame.columns:
        return []
    frame["rank"] = pd.to_numeric(frame.get("rank", 1), errors="coerce").fillna(1)
    frame = frame.sort_values(
        ["horizon_months", "rank"],
        ascending=[True, True],
        kind="mergesort",
    )
    leaders = frame.groupby("horizon_months", sort=True).head(1).copy()
    leaders["lane"] = lane
    leaders["primary_metric"] = primary_metric
    leaders["primary_metric_value"] = leaders.get(primary_metric)
    preferred = [
        "lane",
        "horizon_months",
        "model_name",
        "variant",
        "rank",
        "primary_metric",
        "primary_metric_value",
        "n_obs",
        "directional_hit_rate",
        "balanced_accuracy",
        "brier_score",
        "implied_price_mae",
        "future_bvps_mae",
        "expected_value_mae",
        "beats_base_rate",
        "beats_no_change",
        "beats_no_change_bvps",
    ]
    present = [column for column in preferred if column in leaders.columns]
    return json_records(leaders[present])


def count_gate_successes(
    rows: list[dict[str, Any]],
    *,
    gate_column: str,
) -> dict[str, Any]:
    """Count true gate observations and distinct horizons with observations."""
    frame = pd.DataFrame(rows)
    if frame.empty or gate_column not in frame.columns:
        return {
            "gate_column": gate_column,
            "true_count": 0,
            "true_horizon_count": 0,
            "observed_row_count": 0,
            "observed_horizon_count": 0,
        }
    observed = frame[frame[gate_column].notna()].copy()
    horizon_count = (
        int(observed["horizon_months"].nunique())
        if "horizon_months" in observed.columns
        else 0
    )
    true_rows = observed[observed[gate_column].astype(bool)]
    true_horizon_count = (
        int(true_rows["horizon_months"].nunique())
        if "horizon_months" in true_rows.columns
        else 0
    )
    return {
        "gate_column": gate_column,
        "true_count": int(len(true_rows)),
        "true_horizon_count": true_horizon_count,
        "observed_row_count": int(len(observed)),
        "observed_horizon_count": horizon_count,
    }


def build_shadow_readiness(
    *,
    classification_gate_horizons: int,
    direct_return_gate_horizons: int,
    decomposition_path_consistent: bool,
    special_dividend_n_obs: int,
) -> dict[str, Any]:
    """Return the conservative x8 shadow-readiness decision."""
    blocking_reasons: list[str] = []
    if classification_gate_horizons < 3:
        blocking_reasons.append(
            "classification evidence is mixed and horizon-specific"
        )
    if direct_return_gate_horizons < 2:
        blocking_reasons.append("direct-return benchmarks remain baseline-heavy")
    if not decomposition_path_consistent:
        blocking_reasons.append("decomposition still depends on no-change P/B")
    if special_dividend_n_obs < 30:
        blocking_reasons.append("special-dividend sample is very small")
    status = "not_ready" if blocking_reasons else "candidate_for_shadow_plan"
    rationale = "; ".join(blocking_reasons) if blocking_reasons else (
        "all major x-series gates cleared enough for a separate shadow plan"
    )
    return {
        "status": status,
        "production_changes": False,
        "shadow_changes": False,
        "rationale": rationale,
    }


def build_x8_summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the x8 summary payload from x-series summary JSON payloads."""
    x2_rows = payloads["x2"].get("ranked_rows", [])
    x3_rows = payloads["x3"].get("ranked_rows", [])
    x4_rows = payloads["x4"].get("ranked_rows", [])
    x5_rows = payloads["x5"].get("ranked_rows", [])
    x6_rows = payloads["x6"].get("ranked_rows", [])
    x7_rows = payloads["x7"].get("ranked_rows", [])

    x7_gate_horizons = max(
        (int(row.get("cleared_horizon_count", 0)) for row in x7_rows),
        default=0,
    )
    x3_gate_count = count_gate_successes(
        x3_rows,
        gate_column="beats_no_change",
    )
    x3_gate_horizons = x3_gate_count["true_horizon_count"]
    best_x6_n_obs = int(x6_rows[0].get("n_obs", 0)) if x6_rows else 0
    readiness = build_shadow_readiness(
        classification_gate_horizons=x7_gate_horizons,
        direct_return_gate_horizons=x3_gate_horizons,
        decomposition_path_consistent=False,
        special_dividend_n_obs=best_x6_n_obs,
    )

    return {
        "version": "x8",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "horizon_leaders": (
            extract_horizon_leaders(
                x2_rows,
                lane="x2_absolute_classification",
                primary_metric="balanced_accuracy",
            )
            + extract_horizon_leaders(
                x3_rows,
                lane="x3_direct_return",
                primary_metric="implied_price_mae",
            )
            + extract_horizon_leaders(
                x4_rows,
                lane="x4_bvps_forecasting",
                primary_metric="future_bvps_mae",
            )
            + extract_horizon_leaders(
                x5_rows,
                lane="x5_bvps_pb_decomposition",
                primary_metric="implied_price_mae",
            )
        ),
        "gate_counts": {
            "x2": count_gate_successes(x2_rows, gate_column="beats_base_rate"),
            "x3": x3_gate_count,
            "x4": count_gate_successes(
                x4_rows,
                gate_column="beats_no_change_bvps",
            ),
            "x7_best_cleared_horizon_count": x7_gate_horizons,
        },
        "special_dividend_leader": x6_rows[0] if x6_rows else None,
        "ta_leader": x7_rows[0] if x7_rows else None,
        "shadow_readiness": readiness,
        "recommendations": [
            {
                "lane": "absolute_direction",
                "recommendation": (
                    "continue targeted x7-style replacement experiments for "
                    "3m and 6m; do not promote broad TA feature expansion"
                ),
            },
            {
                "lane": "direct_return",
                "recommendation": (
                    "keep as benchmark; only the 12m drift path clearly "
                    "clears the current no-change gate"
                ),
            },
            {
                "lane": "bvps_pb_decomposition",
                "recommendation": (
                    "advance BVPS modeling, but treat P/B no-change as the "
                    "current structural anchor until P/B models improve"
                ),
            },
            {
                "lane": "special_dividend",
                "recommendation": (
                    "retain annual sidecar status; historical occurrence "
                    "rate plus ridge size is best but sample confidence is low"
                ),
            },
        ],
    }
