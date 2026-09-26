"""Helpers for interpretation and confidence pass on model predictions."""

from __future__ import annotations

from typing import Any

import math

from src.portfolio.redeploy_portfolio import v27_benchmark_pruning_review


def benchmark_role_map() -> dict[str, dict[str, str]]:
    """Return presentation labels for benchmark tickers."""
    role_labels = {
        "keep_for_redeploy": "Buy candidate",
        "optional_substitute": "Optional substitute",
        "contextual_only": "Context only",
        "not_preferred_for_redeploy": "Forecast only",
    }
    role_map: dict[str, dict[str, str]] = {}
    for row in v27_benchmark_pruning_review():
        status = row["status"]
        role_map[row["benchmark"]] = {
            "role": role_labels.get(status, "Forecast only"),
            "status": status,
            "reason": row["reason"],
        }
    return role_map


def benchmark_role_for_ticker(ticker: str) -> dict[str, str]:
    """Return one benchmark's presentation role."""
    return benchmark_role_map().get(
        ticker,
        {
            "role": "Forecast only",
            "status": "unknown",
            "reason": "Used for forecasting context only.",
        },
    )


def build_confidence_snapshot(
    *,
    mean_ic: float,
    mean_hr: float,
    aggregate_health: dict[str, Any] | None,
    representative_cpcv: Any | None,
) -> dict[str, Any]:
    """Build a compact gate-style confidence snapshot.

    The rows are the recommendation-mode gates themselves
    (``evaluate_quality_gates``), so the table always agrees with the mode.
    Two rows are shown but not gated: the plain mean hit rate and the CPCV
    verdict (review 2026-09-25: a hit rate below the base rate is not skill,
    and CPCV is a K-fold).
    """
    from src.reporting.decision_rendering import evaluate_quality_gates

    labels = {
        "oos_r2": "Aggregate OOS R^2",
        "mean_ic": "Mean IC (equal-weight)",
        "directional_skill": "Directional skill",
        "cpcv_completed": "CPCV diagnostic ran",
    }
    gates = evaluate_quality_gates(mean_ic, aggregate_health, representative_cpcv)
    rows = [
        {
            "check": labels.get(gate.name, gate.name),
            "current": gate.current,
            "threshold": gate.threshold,
            "status": gate.status,
            "meaning": gate.meaning,
        }
        for gate in gates
    ]

    pass_count = sum(row["status"] == "PASS" for row in rows)
    fail_count = sum(row["status"] == "FAIL" for row in rows)
    marginal_count = sum(row["status"] == "MARGINAL" for row in rows)

    if pass_count == len(rows):
        summary = "All core quality gates pass, so the signal is eligible to influence the vest decision."
    elif fail_count:
        summary = (
            f"{pass_count}/{len(rows)} core gates pass and {fail_count} fail, so the quality gate "
            "is too weak for a prediction-led vest action."
        )
    else:
        summary = (
            f"{pass_count}/{len(rows)} core gates pass ({marginal_count} marginal). The signal is usable "
            "for monitoring, but not strong enough to fully trust as an execution-grade edge."
        )

    cpcv_verdict = (
        str(getattr(representative_cpcv, "stability_verdict", "UNKNOWN"))
        if representative_cpcv is not None
        else "missing"
    )
    hr_value = float(mean_hr) if mean_hr is not None and math.isfinite(float(mean_hr)) else float("nan")
    diagnostic_rows = [
        {
            "check": "Mean hit rate",
            "current": f"{hr_value:.1%}" if not math.isnan(hr_value) else "n/a",
            "threshold": "not gated",
            "status": "INFO",
            "meaning": "Directional accuracy versus zero; compare with the base rate above.",
        },
        {
            "check": "CPCV verdict",
            "current": cpcv_verdict,
            "threshold": "not gated",
            "status": "INFO",
            "meaning": "Stability across purged combinatorial paths (a K-fold, so diagnostic only).",
        },
    ]

    return {
        "rows": rows + diagnostic_rows,
        "gate_rows": rows,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unknown_count": 0,
        "summary": summary,
    }


__all__ = [
    "benchmark_role_for_ticker",
    "benchmark_role_map",
    "build_confidence_snapshot",
]
