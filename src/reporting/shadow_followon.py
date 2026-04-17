"""Helpers for the autoresearch follow-on side-by-side shadow lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLLOWON_VARIANT_NAME = "autoresearch_followon_v150"
FOLLOWON_VARIANT_LABEL = "Autoresearch Follow-On"


def _read_text(path: Path) -> str:
    """Read one small UTF-8 candidate file."""
    return path.read_text(encoding="utf-8").strip()


def load_followon_candidate_bundle() -> dict[str, Any]:
    """Load the surviving v139-v150 research candidates for shadow reporting."""
    research_dir = PROJECT_ROOT / "results" / "research"
    return {
        "v141_blend_weight": float(_read_text(research_dir / "v141_blend_weight_candidate.txt")),
        "v143_corr_prune": float(_read_text(research_dir / "v143_corr_prune_candidate.txt")),
        "v144_conformal": json.loads(_read_text(research_dir / "v144_conformal_candidate.json")),
        "v149_kelly": json.loads(_read_text(research_dir / "v149_kelly_candidate.json")),
        "v150_neutral_band": float(_read_text(research_dir / "v150_neutral_band_candidate.txt")),
    }


def build_followon_shadow_payload(
    *,
    probability_actionable_sell: float | None,
    confidence_tier: str | None,
    stance: str | None,
) -> dict[str, Any]:
    """Build the compact side-by-side follow-on shadow payload."""
    return {
        "variant": FOLLOWON_VARIANT_NAME,
        "label": FOLLOWON_VARIANT_LABEL,
        "reporting_only": True,
        "probability_actionable_sell": probability_actionable_sell,
        "confidence_tier": confidence_tier,
        "stance": stance,
        "candidate_sources": load_followon_candidate_bundle(),
    }
