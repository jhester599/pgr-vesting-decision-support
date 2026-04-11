from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.reporting.monthly_summary import (
    build_monthly_summary_payload,
    write_monthly_summary,
)


def test_monthly_summary_payload_and_writer(tmp_path: Path) -> None:
    payload = build_monthly_summary_payload(
        as_of_date="2026-04-11",
        run_date="2026-04-11",
        recommendation_layer_label="Live production recommendation layer (quality-weighted consensus)",
        consensus="NEUTRAL",
        confidence_tier="LOW",
        recommendation_mode="DEFER-TO-TAX-DEFAULT",
        sell_pct=0.5,
        mean_predicted=-0.0234,
        mean_ic=0.1744,
        mean_hit_rate=0.668,
        mean_prob_outperform=0.5,
        calibrated_prob_outperform=0.619,
        aggregate_oos_r2=-0.0275,
        aggregate_nw_ic=0.1744,
        warnings=["Aggregate OOS R^2 below threshold."],
        signals=pd.DataFrame({"benchmark": ["VOO", "BND"]}),
        benchmark_quality_df=pd.DataFrame({"benchmark": ["VOO", "BND"]}),
        consensus_shadow_df=pd.DataFrame(
            {
                "variant": ["quality_weighted", "equal_weight"],
                "consensus": ["NEUTRAL", "NEUTRAL"],
                "recommendation_mode": ["DEFER-TO-TAX-DEFAULT", "DEFER-TO-TAX-DEFAULT"],
                "recommended_sell_pct": [0.5, 0.5],
                "is_live_path": [True, False],
            }
        ),
        visible_cross_check=False,
    )

    path = write_monthly_summary(tmp_path, payload)

    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["recommendation"]["signal_label"] == "NEUTRAL (LOW CONFIDENCE)"
    assert written["cross_check"]["visible_in_primary_surfaces"] is False
    assert written["cross_check"]["mode_agreement"] is True
