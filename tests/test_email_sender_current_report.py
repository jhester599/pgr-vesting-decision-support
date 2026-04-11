from __future__ import annotations

from src.reporting.email_sender import build_email_message


_CURRENT_STYLE_BODY = """\
# PGR Monthly Decision - April 2026

**As-of Date:** 2026-04-11  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

## Executive Summary

- What changed since last month: Previous logged month was OUTPERFORM.

---

## Consensus Signal

| Field | Value |
|-------|-------|
| Signal | **NEUTRAL (LOW CONFIDENCE)** |
| Recommendation Mode | **DEFER-TO-TAX-DEFAULT** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | -2.34% |
| P(Outperform, calibrated) | 61.9% |
| Mean IC (across benchmarks) | 0.1744 |
| Mean Hit Rate | 66.8% |
| Aggregate OOS R^2 | -2.75% |

---

## Consensus Shadow Evaluation

> The live path now uses the quality-weighted cross-benchmark consensus.

| Variant | Consensus | Mean Pred. Return | Mean IC | Mean Hit Rate | P(Outperform) | Mode | Sell % | Top Weight |
|---------|-----------|-------------------|---------|---------------|---------------|------|--------|------------|
| Shadow equal-weight | NEUTRAL (LOW) | -2.45% | 0.1703 | 66.3% | 50.0% | DEFER-TO-TAX-DEFAULT | 50% | VOO (12.5%) |
| Live quality-weighted | NEUTRAL (LOW) | -2.34% | 0.1744 | 66.8% | 50.0% | DEFER-TO-TAX-DEFAULT | 50% | BND (13.4%) |

## Confidence Snapshot

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Mean IC | 0.1744 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
"""


def test_current_consensus_shadow_section_is_rendered_in_plaintext() -> None:
    msg = build_email_message(
        _CURRENT_STYLE_BODY,
        "from@example.com",
        "to@example.com",
        "April 2026",
    )
    plain_body = msg.get_payload()[0].get_payload(decode=True).decode()
    assert "Consensus cross-check:" in plain_body
    assert "Live quality-weighted" in plain_body
    assert "Shadow equal-weight" in plain_body


def test_current_consensus_shadow_section_is_rendered_in_html() -> None:
    msg = build_email_message(
        _CURRENT_STYLE_BODY,
        "from@example.com",
        "to@example.com",
        "April 2026",
    )
    html_body = msg.get_payload()[1].get_payload(decode=True).decode()
    assert "Consensus cross-check" in html_body
    assert "Live quality-weighted" in html_body
    assert "Shadow equal-weight" in html_body


def test_dashboard_snapshot_link_can_be_rendered_in_email() -> None:
    msg = build_email_message(
        _CURRENT_STYLE_BODY,
        "from@example.com",
        "to@example.com",
        "April 2026",
        dashboard_snapshot_label="results/monthly_decisions/2026-04/dashboard.html",
        dashboard_snapshot_url="https://github.com/example/repo/blob/master/results/monthly_decisions/2026-04/dashboard.html",
    )
    plain_body = msg.get_payload()[0].get_payload(decode=True).decode()
    html_body = msg.get_payload()[1].get_payload(decode=True).decode()
    assert "Dashboard snapshot:" in plain_body
    assert "dashboard.html" in plain_body
    assert "Dashboard snapshot" in html_body
    assert "Open dashboard snapshot" in html_body


def test_structured_summary_payload_can_drive_email_top_level_fields() -> None:
    msg = build_email_message(
        "# Minimal body",
        "from@example.com",
        "to@example.com",
        "April 2026",
        summary_payload={
            "recommendation": {
                "signal_label": "NEUTRAL (LOW CONFIDENCE)",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "predicted_6m_relative_return_label": "-2.34%",
                "mean_ic_label": "0.1744",
                "mean_hit_rate_label": "66.8%",
                "prob_outperform_calibrated_label": "61.9%",
                "aggregate_oos_r2_label": "-2.75%",
            },
            "recommendation_layer": {
                "label": "Live production recommendation layer (quality-weighted consensus)",
            },
        },
    )
    plain_body = msg.get_payload()[0].get_payload(decode=True).decode()
    html_body = msg.get_payload()[1].get_payload(decode=True).decode()
    assert "NEUTRAL (LOW CONFIDENCE)" in msg["Subject"]
    assert "Recommendation mode: DEFER-TO-TAX-DEFAULT" in plain_body
    assert "Predicted 6M relative return: -2.34%" in plain_body
    assert "Aggregate OOS R^2" in html_body


def test_structured_summary_payload_renders_classification_shadow_section() -> None:
    msg = build_email_message(
        "# Minimal body",
        "from@example.com",
        "to@example.com",
        "April 2026",
        summary_payload={
            "recommendation": {
                "signal_label": "NEUTRAL (LOW CONFIDENCE)",
                "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
                "predicted_6m_relative_return_label": "-2.34%",
            },
            "classification_shadow": {
                "enabled": True,
                "probability_actionable_sell_label": "28.4%",
                "confidence_tier": "HIGH",
                "stance": "NON-ACTIONABLE",
                "agreement_label": "Aligned",
                "interpretation": "Supports a hold/defer interpretation.",
            },
        },
    )
    plain_body = msg.get_payload()[0].get_payload(decode=True).decode()
    html_body = msg.get_payload()[1].get_payload(decode=True).decode()
    assert "Classification confidence check:" in plain_body
    assert "P(actionable sell): 28.4%" in plain_body
    assert "Classifier stance: NON-ACTIONABLE" in plain_body
    assert "Classification confidence check" in html_body
    assert "28.4%" in html_body
