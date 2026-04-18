"""Helpers for the Firth logistic shadow reporting lane (v159)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIRTH_SHADOW_VARIANT_NAME: str = "firth_shadow_v159"
FIRTH_SHADOW_VARIANT_LABEL: str = "Firth Logistic Shadow"
FIRTH_BENCHMARKS: frozenset[str] = frozenset({"VMBS", "BND"})


def load_firth_candidate() -> dict[str, Any]:
    """Load the v154 Firth logistic research candidate for shadow reporting."""
    candidate_path = (
        PROJECT_ROOT / "results" / "research" / "v154_firth_candidate.json"
    )
    return json.loads(candidate_path.read_text(encoding="utf-8"))


def build_firth_shadow_payload(
    *,
    probability_actionable_sell: float | None,
    probability_actionable_sell_label: str | None = None,
    confidence_tier: str | None,
    stance: str | None,
    probability_investable_pool_label: str | None = None,
    probability_path_b_temp_scaled_label: str | None = None,
) -> dict[str, Any]:
    """Build the Firth logistic reporting-only shadow payload for monthly artifacts."""
    candidate = load_firth_candidate()
    firth_winners: list[str] = candidate.get("firth_winners", [])
    winner_deltas: dict[str, float] = {}
    for row in candidate.get("rows", []):
        bm = row.get("benchmark", "")
        delta = row.get("delta_ba_covered")
        if (
            bm in firth_winners
            and isinstance(delta, (int, float))
            and not math.isnan(float(delta))
        ):
            winner_deltas[bm] = float(delta)
    return {
        "variant": FIRTH_SHADOW_VARIANT_NAME,
        "label": FIRTH_SHADOW_VARIANT_LABEL,
        "reporting_only": True,
        "probability_actionable_sell": probability_actionable_sell,
        "probability_actionable_sell_label": probability_actionable_sell_label,
        "confidence_tier": confidence_tier,
        "stance": stance,
        "probability_investable_pool_label": probability_investable_pool_label,
        "probability_path_b_temp_scaled_label": probability_path_b_temp_scaled_label,
        "firth_benchmarks": sorted(FIRTH_BENCHMARKS),
        "firth_winners": firth_winners,
        "firth_winner_deltas": winner_deltas,
        "firth_recommendation": candidate.get("recommendation"),
    }
