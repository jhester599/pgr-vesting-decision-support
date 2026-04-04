from __future__ import annotations

from datetime import date

from src.research.v12 import (
    SnapshotSummary,
    build_shadow_comparison_lines,
    recent_monthly_review_dates,
    signal_from_prediction,
)


def test_recent_monthly_review_dates_returns_business_month_ends() -> None:
    dates = recent_monthly_review_dates(date(2026, 4, 4), months=3)
    assert dates == [
        date(2026, 1, 30),
        date(2026, 2, 27),
        date(2026, 3, 31),
    ]


def test_signal_from_prediction_uses_neutral_band() -> None:
    assert signal_from_prediction(0.05) == "OUTPERFORM"
    assert signal_from_prediction(-0.05) == "UNDERPERFORM"
    assert signal_from_prediction(0.01) == "NEUTRAL"


def test_build_shadow_comparison_lines_includes_redeploy_and_lot_guidance() -> None:
    live = SnapshotSummary(
        label="live",
        as_of=date(2026, 4, 4),
        candidate_name="production_4_model_ensemble",
        policy_name="current_production_mapping",
        consensus="OUTPERFORM",
        confidence_tier="MODERATE",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        sell_pct=0.5,
        mean_predicted=0.04,
        mean_ic=0.10,
        mean_hit_rate=0.56,
        aggregate_oos_r2=-1.0,
        aggregate_nw_ic=0.11,
        calibrated_prob_outperform=0.67,
    )
    shadow = SnapshotSummary(
        label="shadow",
        as_of=date(2026, 4, 4),
        candidate_name="baseline_historical_mean",
        policy_name="neutral_band_3pct",
        consensus="OUTPERFORM",
        confidence_tier="LOW",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        sell_pct=0.5,
        mean_predicted=0.03,
        mean_ic=-0.14,
        mean_hit_rate=0.62,
        aggregate_oos_r2=-0.20,
        aggregate_nw_ic=-0.05,
    )
    lines = build_shadow_comparison_lines(
        live_summary=live,
        shadow_summary=shadow,
        next_vest_date=date(2026, 7, 17),
        next_vest_type="performance",
        existing_holdings=[
            {
                "tax_bucket": "LOSS",
                "vest_date": date(2025, 1, 21),
                "cost_basis_per_share": 240.0,
                "shares": 1.0,
                "rationale": "Trim loss lots first.",
            }
        ],
        redeploy_buckets=[
            {
                "bucket": "fixed_income",
                "example_funds": "BND, VMBS",
                "note": "Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.",
            }
        ],
    )
    rendered = "\n".join(lines)
    assert "Live Production Snapshot" in rendered
    assert "Shadow Baseline Snapshot" in rendered
    assert "Existing Holdings Guidance" in rendered
    assert "Redeploy Guidance" in rendered
    assert "BND, VMBS" in rendered
