from __future__ import annotations

import json

from pathlib import Path

from dashboard.data import load_latest_run_bundle, parse_aggregate_health, parse_recommendation_summary


def test_load_latest_run_bundle_reads_new_monthly_artifacts(tmp_path: Path) -> None:
    latest_dir = tmp_path / "2026-04"
    latest_dir.mkdir()

    (latest_dir / "run_manifest.json").write_text(json.dumps({"as_of_date": "2026-04-11"}), encoding="utf-8")
    (latest_dir / "signals.csv").write_text("benchmark,signal\nVOO,NEUTRAL\n", encoding="utf-8")
    (latest_dir / "recommendation.md").write_text("# sample", encoding="utf-8")
    (latest_dir / "monthly_summary.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-04-11",
                "recommendation": {"signal_label": "NEUTRAL (LOW CONFIDENCE)"},
            }
        ),
        encoding="utf-8",
    )
    (latest_dir / "benchmark_quality.csv").write_text(
        "benchmark,oos_r2,nw_ic,hit_rate,cw_p_value\nVOO,-0.02,0.15,0.6,0.04\n",
        encoding="utf-8",
    )
    (latest_dir / "consensus_shadow.csv").write_text(
        "variant,consensus,recommendation_mode\nquality_weighted,NEUTRAL,DEFER-TO-TAX-DEFAULT\n",
        encoding="utf-8",
    )
    (latest_dir / "classification_shadow.csv").write_text(
        "benchmark,classifier_prob_actionable_sell\nVOO,0.28\n",
        encoding="utf-8",
    )
    (latest_dir / "decision_overlays.csv").write_text(
        "variant,recommendation_mode,recommended_sell_pct\nshadow_gate,DEFER-TO-TAX-DEFAULT,0.5\n",
        encoding="utf-8",
    )

    bundle = load_latest_run_bundle(tmp_path)

    assert bundle["latest_dir"] == latest_dir
    assert bundle["manifest"]["as_of_date"] == "2026-04-11"
    assert not bundle["signals"].empty
    assert not bundle["benchmark_quality"].empty
    assert not bundle["consensus_shadow"].empty
    assert not bundle["classification_shadow"].empty
    assert not bundle["decision_overlays"].empty
    assert bundle["summary"]["recommendation"]["signal_label"] == "NEUTRAL (LOW CONFIDENCE)"


def test_parse_aggregate_health_handles_current_recommendation_text() -> None:
    text = (
        "- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. "
        "Aggregate health: OOS R^2 -2.75%, IC 0.2006, hit rate 68.1%."
    )
    parsed = parse_aggregate_health(text)
    assert parsed == {"oos_r2": -2.75, "ic": 0.2006, "hit_rate": 68.1}


def test_parse_recommendation_summary_reads_current_fields() -> None:
    text = """\
| Signal | **NEUTRAL (LOW CONFIDENCE)** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | -2.34% |
> **Calibration:** Phase 2 — Platt scaling active (n=1,188 OOS obs).  ECE = 1.1% [95% CI: 1.4%–6.0%].
"""
    parsed = parse_recommendation_summary(text)
    assert parsed["signal"] == "NEUTRAL (LOW CONFIDENCE)"
    assert parsed["sell_pct"] == "50%"
    assert parsed["predicted_return"] == "-2.34%"
    assert parsed["calibration_ece"] == "1.1%"
