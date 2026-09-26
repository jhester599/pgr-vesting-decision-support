"""Helpers for the autoresearch follow-on side-by-side shadow lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLLOWON_VARIANT_NAME = "autoresearch_followon_v150"
FOLLOWON_VARIANT_LABEL = "Autoresearch Follow-On"

# Candidate files written by the v141-v150 studies (research/studies/, review
# 2026-09-25 phase 3).
_STUDIES_DIR = PROJECT_ROOT / "research" / "studies"


def _study_output(folder: str, filename: str) -> Path:
    """Return the path of one committed study output."""
    return _STUDIES_DIR / folder / "outputs" / filename


FOLLOWON_CANDIDATE_PATHS: dict[str, Path] = {
    "v141_blend_weight": _study_output("v141_blend_eval", "v141_blend_weight_candidate.txt"),
    "v143_corr_prune": _study_output("v143_corr_prune_eval", "v143_corr_prune_candidate.txt"),
    "v144_conformal": _study_output("v144_conformal_eval", "v144_conformal_candidate.json"),
    "v149_kelly": _study_output("v149_kelly_eval", "v149_kelly_candidate.json"),
    "v150_neutral_band": _study_output(
        "v150_neutral_band_eval", "v150_neutral_band_candidate.txt"
    ),
}


def _read_text(path: Path) -> str:
    """Read one small UTF-8 candidate file."""
    return path.read_text(encoding="utf-8").strip()


def load_followon_candidate_bundle() -> dict[str, Any]:
    """Load the surviving v139-v150 research candidates for shadow reporting."""
    return {
        "v141_blend_weight": float(_read_text(FOLLOWON_CANDIDATE_PATHS["v141_blend_weight"])),
        "v143_corr_prune": float(_read_text(FOLLOWON_CANDIDATE_PATHS["v143_corr_prune"])),
        "v144_conformal": json.loads(_read_text(FOLLOWON_CANDIDATE_PATHS["v144_conformal"])),
        "v149_kelly": json.loads(_read_text(FOLLOWON_CANDIDATE_PATHS["v149_kelly"])),
        "v150_neutral_band": float(_read_text(FOLLOWON_CANDIDATE_PATHS["v150_neutral_band"])),
    }


def build_followon_shadow_payload(
    *,
    probability_actionable_sell: float | None,
    probability_actionable_sell_label: str | None = None,
    confidence_tier: str | None,
    stance: str | None,
    probability_investable_pool_label: str | None = None,
    probability_path_b_temp_scaled_label: str | None = None,
) -> dict[str, Any]:
    """Build the compact side-by-side follow-on shadow payload."""
    return {
        "variant": FOLLOWON_VARIANT_NAME,
        "label": FOLLOWON_VARIANT_LABEL,
        "reporting_only": True,
        "probability_actionable_sell": probability_actionable_sell,
        "probability_actionable_sell_label": probability_actionable_sell_label,
        "confidence_tier": confidence_tier,
        "stance": stance,
        "probability_investable_pool_label": probability_investable_pool_label,
        "probability_path_b_temp_scaled_label": probability_path_b_temp_scaled_label,
        "candidate_sources": load_followon_candidate_bundle(),
    }


def build_followon_decision_overlay_payload(
    baseline_overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a reporting-only decision-layer comparison payload."""
    payload = dict(baseline_overlay or {})
    payload["variant"] = FOLLOWON_VARIANT_NAME
    payload["label"] = FOLLOWON_VARIANT_LABEL
    payload["reporting_only"] = True
    payload["candidate_sources"] = load_followon_candidate_bundle()
    return payload
