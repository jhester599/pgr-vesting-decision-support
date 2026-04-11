from __future__ import annotations

from src.models.classification_shadow import (
    agreement_with_live_recommendation,
    classification_confidence_tier,
    classification_interpretation,
    classification_stance,
)


def test_classification_confidence_tier_thresholds() -> None:
    assert classification_confidence_tier(0.75) == "HIGH"
    assert classification_confidence_tier(0.62) == "MODERATE"
    assert classification_confidence_tier(0.50) == "LOW"
    assert classification_confidence_tier(0.28) == "HIGH"


def test_classification_stance_thresholds() -> None:
    assert classification_stance(0.74) == "ACTIONABLE-SELL"
    assert classification_stance(0.28) == "NON-ACTIONABLE"
    assert classification_stance(0.52) == "NEUTRAL"


def test_agreement_with_live_recommendation() -> None:
    assert agreement_with_live_recommendation("ACTIONABLE-SELL", "ACTIONABLE") is True
    assert agreement_with_live_recommendation("NON-ACTIONABLE", "DEFER-TO-TAX-DEFAULT") is True
    assert agreement_with_live_recommendation("ACTIONABLE-SELL", "DEFER-TO-TAX-DEFAULT") is False


def test_classification_interpretation_mentions_probability() -> None:
    interpretation = classification_interpretation(0.28, "NON-ACTIONABLE", "HIGH")
    assert "28.0%" in interpretation
    assert "hold/defer" in interpretation
