from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from scripts import monthly_decision
from src.research.v12 import SnapshotSummary


def test_recommendation_report_includes_v13_sections(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(monthly_decision, "_load_previous_decision_summary", lambda as_of: None)
    monkeypatch.setattr(
        monthly_decision,
        "_build_provisional_vest_scenario",
        lambda conn, as_of, mean_predicted, prob_outperform: {
            "vest_date": date(2026, 7, 17),
            "rsu_type": "performance",
            "current_price": 198.84,
            "avg_basis": 133.38,
            "shares": 8.0,
            "scenario": type(
                "ScenarioResult",
                (),
                {
                    "scenarios": [],
                    "recommended_scenario": "SELL_NOW_STCG",
                    "stcg_ltcg_breakeven": 0.2125,
                },
            )(),
        },
    )

    signals = pd.DataFrame(
        [
            {
                "benchmark": "VTI",
                "predicted_relative_return": 0.05,
                "ic": 0.10,
                "hit_rate": 0.56,
                "signal": "OUTPERFORM",
                "prob_outperform": 0.62,
                "calibrated_prob_outperform": 0.64,
                "confidence_tier": "MODERATE",
                "ci_lower": -0.10,
                "ci_upper": 0.20,
            }
        ]
    ).set_index("benchmark")

    live_summary = SnapshotSummary(
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
        aggregate_oos_r2=-1.18,
        aggregate_nw_ic=0.13,
        calibrated_prob_outperform=0.66,
    )
    shadow_summary = SnapshotSummary(
        label="shadow",
        as_of=date(2026, 4, 4),
        candidate_name="baseline_historical_mean",
        policy_name="neutral_band_3pct",
        consensus="OUTPERFORM",
        confidence_tier="LOW",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        sell_pct=0.5,
        mean_predicted=0.05,
        mean_ic=-0.14,
        mean_hit_rate=0.62,
        aggregate_oos_r2=-0.16,
        aggregate_nw_ic=-0.01,
    )

    monthly_decision._write_recommendation_md(  # noqa: SLF001
        out_dir=tmp_path,
        as_of=date(2026, 4, 4),
        run_date=date(2026, 4, 4),
        conn=None,
        signals=signals,
        consensus="OUTPERFORM",
        mean_predicted=0.04,
        mean_ic=0.10,
        mean_hr=0.56,
        sell_pct=0.5,
        dry_run=True,
        mean_prob_outperform=0.62,
        composite_confidence_tier="MODERATE",
        cal_result=None,
        aggregate_health={"oos_r2": -1.18, "nw_ic": 0.13, "agg_hit": 0.56},
        recommendation_mode={
            "mode": "defer-to-tax-default",
            "label": "DEFER-TO-TAX-DEFAULT",
            "sell_pct": 0.5,
            "summary": "Weak model quality.",
            "action_note": "Use the default diversification rule.",
        },
        live_summary=live_summary,
        shadow_summary=shadow_summary,
        existing_holdings=[
            {
                "vest_date": date(2025, 1, 21),
                "shares": 1.0,
                "cost_basis_per_share": 240.0,
                "tax_bucket": "LOSS",
                "unrealized_gain": -40.0,
                "unrealized_return": -0.17,
                "rationale": "Trim loss lots first when reducing concentration.",
            }
        ],
        redeploy_buckets=[
            {
                "bucket": "fixed_income",
                "example_funds": "BND, VMBS",
                "mean_diversification_score": 0.6,
                "note": "Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.",
            }
        ],
        recommendation_layer_label="Live production recommendation layer + v13 simpler-baseline cross-check",
    )

    content = (tmp_path / "recommendation.md").read_text(encoding="utf-8")
    assert "Recommendation Layer" in content
    assert "## Existing Holdings Guidance" in content
    assert "## Redeploy Guidance" in content
    assert "## Simple-Baseline Cross-Check" in content
