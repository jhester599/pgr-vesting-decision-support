"""Frozen v201 regression safeguards; no secondary route to selection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def candidate_disposition(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Missing safeguards fail; numeric epsilon only handles float roundoff."""
    epsilon = 1e-12

    def finite(mapping: Mapping[str, Any], key: str) -> bool:
        value = mapping.get(key)
        return isinstance(value, (int, float)) and math.isfinite(value)

    def no_worse(key: str, increasing: bool) -> bool:
        if not finite(candidate, key) or not finite(control, key):
            return False
        gap = float(candidate[key]) - float(control[key])
        return gap >= -epsilon if increasing else gap <= epsilon

    gates = {
        "delta_r2": finite(comparison, "delta_r2")
        and comparison["delta_r2"] >= 0.010 - epsilon,
        "holm_primary": finite(comparison, "holm38_p")
        and comparison["holm38_p"] < 0.05,
        "ic": finite(candidate, "equal_weight_ic")
        and finite(control, "equal_weight_ic")
        and candidate["equal_weight_ic"]
        >= control["equal_weight_ic"] - 0.01 - epsilon,
        "hit_rate": no_worse("hit_rate", True),
        "directional_skill": no_worse("directional_skill", True),
        "brier": no_worse("brier", False),
        "log_loss": no_worse("log_loss", False),
        "ece": no_worse("ece", False),
        "coverage_80": finite(candidate, "coverage_80")
        and finite(control, "coverage_80")
        and abs(candidate["coverage_80"] - 0.8)
        <= abs(control["coverage_80"] - 0.8) + epsilon,
    }
    return {
        "passes": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "delta_r2": comparison.get("delta_r2"),
        "disposition": "eligible research finalist"
        if all(gates.values())
        else "research only; no finalist",
    }


def nominate_finalist(
    order: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Highest primary delta among passers; ties retain preregistered order."""
    passed = [name for name in order if results[name]["passes"]]
    return (
        max(passed, key=lambda name: results[name]["delta_r2"])
        if passed
        else None
    )
