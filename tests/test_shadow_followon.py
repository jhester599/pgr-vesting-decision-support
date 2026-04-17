from __future__ import annotations

from src.reporting.shadow_followon import (
    FOLLOWON_VARIANT_NAME,
    build_followon_shadow_payload,
    load_followon_candidate_bundle,
)


def test_load_followon_candidate_bundle_reads_v139_v150_winners() -> None:
    bundle = load_followon_candidate_bundle()

    assert bundle["v141_blend_weight"] == 0.60
    assert bundle["v143_corr_prune"] == 0.80
    assert bundle["v144_conformal"] == {"coverage": 0.75, "aci_gamma": 0.03}
    assert bundle["v149_kelly"] == {"fraction": 0.50, "cap": 0.25}
    assert bundle["v150_neutral_band"] == 0.015


def test_build_followon_shadow_payload_is_named_and_reporting_only() -> None:
    payload = build_followon_shadow_payload(
        probability_actionable_sell=0.56,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )

    assert payload["variant"] == FOLLOWON_VARIANT_NAME
    assert payload["label"] == "Autoresearch Follow-On"
    assert payload["reporting_only"] is True
    assert payload["candidate_sources"]["v149_kelly"]["fraction"] == 0.50


def test_build_followon_shadow_payload_preserves_live_recommendation_fields() -> None:
    payload = build_followon_shadow_payload(
        probability_actionable_sell=0.56,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )

    assert payload["reporting_only"] is True
    assert "recommended_sell_pct" not in payload
