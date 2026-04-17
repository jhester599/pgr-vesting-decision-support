from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.dashboard_snapshot import write_dashboard_snapshot


def test_write_dashboard_snapshot_creates_static_html(tmp_path: Path) -> None:
    path = write_dashboard_snapshot(
        tmp_path,
        as_of_date="2026-04-11",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        consensus="NEUTRAL",
        sell_pct=0.5,
        mean_predicted=-0.0234,
        mean_ic=0.1744,
        mean_hit_rate=0.668,
        aggregate_oos_r2=-0.0275,
        recommendation_layer_label="Live production recommendation layer (quality-weighted consensus)",
        warnings=["Aggregate OOS R^2 below threshold."],
        signals=pd.DataFrame(
            {
                "benchmark": ["VOO"],
                "signal": ["NEUTRAL"],
                "predicted_relative_return": [-0.02],
                "ic": [0.17],
                "hit_rate": [0.66],
                "confidence_tier": ["LOW"],
                "calibrated_prob_outperform": [0.62],
            }
        ),
        benchmark_quality_df=pd.DataFrame(
            {
                "benchmark": ["VOO"],
                "oos_r2": [-0.03],
                "nw_ic": [0.17],
                "hit_rate": [0.66],
                "cw_t_stat": [3.4],
                "cw_p_value": [0.01],
            }
        ),
        consensus_shadow_df=pd.DataFrame(
            {
                "variant": ["quality_weighted", "equal_weight"],
                "consensus": ["NEUTRAL", "NEUTRAL"],
                "mean_predicted_return": [-0.0234, -0.0220],
                "mean_ic": [0.1744, 0.1680],
                "mean_hit_rate": [0.668, 0.660],
                "recommendation_mode": ["DEFER-TO-TAX-DEFAULT", "DEFER-TO-TAX-DEFAULT"],
                "recommended_sell_pct": [0.5, 0.5],
                "top_benchmark": ["BND", "VOO"],
                "top_benchmark_weight": [0.134, 0.125],
                "is_live_path": [True, False],
            }
        ),
        classification_shadow_summary={
            "enabled": True,
            "probability_actionable_sell_label": "28.4%",
            "confidence_tier": "HIGH",
            "stance": "NON-ACTIONABLE",
            "agreement_label": "Aligned",
            "interpretation": "Supports a hold/defer interpretation.",
        },
        shadow_gate_overlay={
            "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
            "recommended_sell_pct": 0.5,
            "would_change": False,
        },
        classification_shadow_variants=[
            {
                "variant": "baseline_shadow",
                "label": "Current Shadow",
                "stance": "NON-ACTIONABLE",
                "probability_actionable_sell_label": "28.4%",
            },
            {
                "variant": "autoresearch_followon_v150",
                "label": "Autoresearch Follow-On",
                "stance": "NEUTRAL",
                "probability_actionable_sell_label": "31.2%",
            },
        ],
    )

    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "PGR Monthly Snapshot" in html
    assert "Benchmark Quality" in html
    assert "Classification Confidence Check" in html
    assert "Shadow Variant Comparison" in html
    assert "Current Shadow" in html
    assert "Autoresearch Follow-On" in html
    assert "Agreement Panel" in html
    assert "28.4%" in html
    assert "Per-Benchmark Signals" in html
    assert "monthly_summary.json" in html
