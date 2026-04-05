"""Helpers for the v29 interpretation and confidence pass."""

from __future__ import annotations

from typing import Any

import math

import config
from src.research.v27 import v27_benchmark_pruning_review


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


def _status_from_threshold(value: float, threshold: float, higher_is_better: bool = True) -> str:
    if math.isnan(value):
        return "UNKNOWN"
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def build_confidence_snapshot(
    *,
    mean_ic: float,
    mean_hr: float,
    aggregate_health: dict[str, Any] | None,
    representative_cpcv: Any | None,
) -> dict[str, Any]:
    """Build a compact gate-style confidence snapshot."""
    aggregate_oos_r2 = float(aggregate_health["oos_r2"]) if aggregate_health is not None else float("nan")
    cpcv_verdict = representative_cpcv.stability_verdict if representative_cpcv is not None else "UNKNOWN"

    rows = [
        {
            "check": "Mean IC",
            "current": f"{mean_ic:.4f}",
            "threshold": f">= {config.DIAG_MIN_IC:.4f}",
            "status": _status_from_threshold(mean_ic, config.DIAG_MIN_IC),
            "meaning": "Cross-benchmark ranking signal.",
        },
        {
            "check": "Mean hit rate",
            "current": f"{mean_hr:.1%}",
            "threshold": f">= {config.DIAG_MIN_HIT_RATE:.1%}",
            "status": _status_from_threshold(mean_hr, config.DIAG_MIN_HIT_RATE),
            "meaning": "Directional accuracy versus zero.",
        },
        {
            "check": "Aggregate OOS R^2",
            "current": f"{aggregate_oos_r2:.2%}" if not math.isnan(aggregate_oos_r2) else "n/a",
            "threshold": f">= {config.DIAG_MIN_OOS_R2:.2%}",
            "status": _status_from_threshold(aggregate_oos_r2, config.DIAG_MIN_OOS_R2),
            "meaning": "Calibration / fit versus a naive benchmark.",
        },
        {
            "check": "Representative CPCV",
            "current": str(cpcv_verdict),
            "threshold": "not FAIL",
            "status": "PASS" if cpcv_verdict not in {"FAIL", "UNKNOWN"} else ("FAIL" if cpcv_verdict == "FAIL" else "UNKNOWN"),
            "meaning": "Stability across purged cross-validation paths.",
        },
    ]

    pass_count = sum(row["status"] == "PASS" for row in rows)
    fail_count = sum(row["status"] == "FAIL" for row in rows)
    unknown_count = sum(row["status"] == "UNKNOWN" for row in rows)

    if pass_count == len(rows):
        summary = "All core quality gates pass, so the signal is eligible to influence the vest decision."
    elif fail_count >= 2:
        summary = (
            f"{pass_count}/{len(rows)} core gates pass. The signal may still be directionally interesting, "
            "but the quality gate remains too weak for a prediction-led vest action."
        )
    else:
        summary = (
            f"{pass_count}/{len(rows)} core gates pass. The signal is usable for monitoring, "
            "but not strong enough to fully trust as an execution-grade edge."
        )
    if unknown_count:
        summary += " Some checks are unavailable and should be treated as unresolved rather than implicitly passing."

    return {
        "rows": rows,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unknown_count": unknown_count,
        "summary": summary,
    }


__all__ = [
    "benchmark_role_for_ticker",
    "benchmark_role_map",
    "build_confidence_snapshot",
]
