from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from scripts import monthly_decision
from src.database import db_client
from src.models.calibration import CalibrationResult


def test_monthly_decision_main_writes_core_artifacts_with_stubbed_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Run the monthly workflow end-to-end with heavy modeling steps stubbed."""
    db_path = tmp_path / "test_monthly.db"
    out_dir = tmp_path / "results" / "2026-04"

    signals = pd.DataFrame(
        {
            "predicted_relative_return": [-0.042, -0.018],
            "ic": [0.081, 0.074],
            "hit_rate": [0.58, 0.56],
            "signal": ["UNDERPERFORM", "UNDERPERFORM"],
            "prob_outperform": [0.39, 0.44],
            "confidence_tier": ["MODERATE", "MODERATE"],
            "calibrated_prob_outperform": [0.41, 0.46],
            "ci_lower": [-0.11, -0.08],
            "ci_upper": [0.02, 0.03],
            "ci_width": [0.13, 0.11],
            "ci_empirical_coverage": [0.82, 0.79],
            "ci_n_calibration": [24, 24],
            "ci_trailing_empirical_coverage": [0.65, 0.60],
            "ci_trailing_coverage_gap": [-0.15, -0.20],
            "ci_trailing_n": [12, 12],
        },
        index=pd.Index(["VOO", "BND"], name="benchmark"),
    )

    cal_result = CalibrationResult(
        n_obs=48,
        method="platt",
        ece=0.041,
        ece_ci_lower=0.020,
        ece_ci_upper=0.073,
    )

    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    monkeypatch.setattr(config, "RECOMMENDATION_LAYER_MODE", "live_only")
    monkeypatch.setattr(monthly_decision, "_output_dir", lambda as_of: out_dir)
    monkeypatch.setattr(monthly_decision, "_already_ran", lambda as_of: False)
    monkeypatch.setattr(monthly_decision, "_fetch_fred_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        monthly_decision,
        "_generate_signals",
        lambda *args, **kwargs: (
            signals.copy(),
            {"VOO": object(), "BND": object()},
            {"obs_feature_report": None, "representative_cpcv": None},
        ),
    )
    monkeypatch.setattr(
        monthly_decision,
        "_calibrate_signals",
        lambda signals, ensemble_results, target_horizon_months=6: (
            signals.copy(),
            cal_result,
            pd.Series([0.41, 0.46], dtype=float).to_numpy(),
            pd.Series([1, 0], dtype=int).to_numpy(),
        ),
    )
    monkeypatch.setattr(
        monthly_decision,
        "_compute_conformal_intervals",
        lambda signals, ensemble_results: signals.copy(),
    )
    monkeypatch.setattr(
        monthly_decision,
        "_consensus_signal",
        lambda signals: ("UNDERPERFORM", -0.03, 0.0775, 0.57, 0.415, "MODERATE"),
    )
    monkeypatch.setattr(
        monthly_decision,
        "_compute_aggregate_health",
        lambda *args, **kwargs: {
            "oos_r2": 0.031,
            "nw_ic": 0.081,
            "agg_hit": 0.57,
            "cw_t_stat": 2.15,
            "cw_p_value": 0.04,
            "benchmark_quality_df": pd.DataFrame(
                {
                    "benchmark": ["VOO", "BND"],
                    "n_obs": [24, 24],
                    "oos_r2": [0.04, 0.02],
                    "nw_ic": [0.08, 0.07],
                    "nw_p_value": [0.03, 0.04],
                    "hit_rate": [0.58, 0.56],
                    "cw_t_stat": [2.20, 2.05],
                    "cw_p_value": [0.03, 0.04],
                    "cw_mean_adjusted_differential": [0.01, 0.01],
                    "r2_flag": ["✅", "✅"],
                    "ic_flag": ["✅", "✅"],
                    "hr_flag": ["✅", "✅"],
                }
            ),
        },
    )
    monkeypatch.setattr(monthly_decision, "_build_provisional_vest_scenario", lambda *args, **kwargs: None)
    monkeypatch.setattr(monthly_decision, "_load_previous_decision_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(monthly_decision, "_append_decision_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(monthly_decision, "_plot_calibration_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(monthly_decision, "_build_existing_holdings_guidance", lambda *args, **kwargs: [])
    monkeypatch.setattr(monthly_decision, "_build_redeploy_guidance", lambda *args, **kwargs: [])
    monkeypatch.setattr(monthly_decision, "_build_redeploy_portfolio", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        monthly_decision,
        "build_classification_shadow_summary",
        lambda *args, **kwargs: (
            type(
                "ShadowSummaryStub",
                (),
                {
                    "to_payload": lambda self: {
                        "enabled": True,
                        "target_label": "actionable_sell_3pct",
                        "feature_anchor_date": "2026-03-31",
                        "probability_actionable_sell": 0.284,
                        "probability_actionable_sell_label": "28.4%",
                        "confidence_tier": "HIGH",
                        "stance": "NON-ACTIONABLE",
                        "agreement_label": "Aligned",
                        "interpretation": "Supports a hold/defer interpretation.",
                    }
                },
            )(),
            pd.DataFrame(
                {
                    "benchmark": ["VOO", "BND"],
                    "classifier_prob_actionable_sell": [0.31, 0.26],
                    "classifier_shadow_tier": ["HIGH", "HIGH"],
                    "classifier_weight": [0.5, 0.5],
                    "classifier_weighted_contribution": [0.155, 0.130],
                }
            ),
        ),
    )
    monkeypatch.setattr(
        monthly_decision.db_client,
        "warn_if_db_behind",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        monthly_decision.db_client,
        "check_data_freshness",
        lambda conn, reference_date: {
            "reference_date": reference_date.isoformat(),
            "overall_status": "OK",
            "checks": [
                {
                    "feed": "Daily prices",
                    "latest_date": "2026-04-05",
                    "age_days": 0,
                    "max_age_days": 10,
                    "status": "OK",
                }
            ],
            "warnings": [],
        },
    )

    def _write_stub_diagnostic(
        out_dir: Path,
        as_of: date,
        ensemble_results,
        target_horizon_months: int = 6,
        cal_result: CalibrationResult | None = None,
        signals: pd.DataFrame | None = None,
        obs_feature_report=None,
        representative_cpcv=None,
        conformal_coverage_summary=None,
        importance_stability=None,
        vif_series=None,
        benchmark_quality_df=None,
        shadow_gate_overlay=None,
        classifier_monitoring_summary=None,
    ) -> None:
        del as_of, ensemble_results, target_horizon_months, cal_result, signals
        del obs_feature_report, representative_cpcv, conformal_coverage_summary
        del importance_stability, vif_series, benchmark_quality_df
        del shadow_gate_overlay, classifier_monitoring_summary
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diagnostic.md").write_text("# Diagnostic Stub\n", encoding="utf-8")

    monkeypatch.setattr(monthly_decision, "_write_diagnostic_report", _write_stub_diagnostic)

    monthly_decision.main(as_of_date_str="2026-04-05", dry_run=True, skip_fred=True)

    recommendation_path = out_dir / "recommendation.md"
    signals_path = out_dir / "signals.csv"
    diagnostic_path = out_dir / "diagnostic.md"
    benchmark_quality_path = out_dir / "benchmark_quality.csv"
    consensus_shadow_path = out_dir / "consensus_shadow.csv"
    classification_shadow_path = out_dir / "classification_shadow.csv"
    decision_overlays_path = out_dir / "decision_overlays.csv"
    dashboard_path = out_dir / "dashboard.html"
    monthly_summary_path = out_dir / "monthly_summary.json"
    manifest_path = out_dir / "run_manifest.json"

    assert recommendation_path.exists()
    assert signals_path.exists()
    assert diagnostic_path.exists()
    assert benchmark_quality_path.exists()
    assert consensus_shadow_path.exists()
    assert classification_shadow_path.exists()
    assert decision_overlays_path.exists()
    assert dashboard_path.exists()
    assert monthly_summary_path.exists()
    assert manifest_path.exists()

    recommendation_text = recommendation_path.read_text(encoding="utf-8")
    assert "PGR Monthly Decision Report" in recommendation_text
    assert "## Data Freshness" in recommendation_text
    assert "## Consensus Signal" in recommendation_text
    assert "## Decision At A Glance" in recommendation_text
    assert "## Classification Confidence Check" in recommendation_text
    assert "## Model Health" in recommendation_text
    assert "Rolling 12M IC" in recommendation_text
    assert "UNDERPERFORM" in recommendation_text
    assert "28.4%" in recommendation_text

    signals_df = pd.read_csv(signals_path)
    assert {"benchmark", "predicted_relative_return", "signal"}.issubset(signals_df.columns)
    assert {"classifier_prob_actionable_sell", "classifier_shadow_tier"}.issubset(
        signals_df.columns
    )
    assert len(signals_df) == 2
    quality_df = pd.read_csv(benchmark_quality_path)
    assert {"benchmark", "cw_t_stat", "cw_p_value"}.issubset(quality_df.columns)
    assert len(quality_df) == 2
    consensus_shadow_df = pd.read_csv(consensus_shadow_path)
    assert {"variant", "recommendation_mode", "recommended_sell_pct"}.issubset(
        consensus_shadow_df.columns
    )
    assert len(consensus_shadow_df) == 2
    classification_shadow_df = pd.read_csv(classification_shadow_path)
    assert {
        "benchmark",
        "classifier_raw_prob_actionable_sell",
        "classifier_prob_actionable_sell",
    }.issubset(classification_shadow_df.columns)
    decision_overlays_df = pd.read_csv(decision_overlays_path)
    assert {"variant", "recommendation_mode", "recommended_sell_pct"}.issubset(
        decision_overlays_df.columns
    )
    assert len(decision_overlays_df) == 2
    monthly_summary = json.loads(monthly_summary_path.read_text(encoding="utf-8"))
    assert monthly_summary["recommendation"]["signal_label"] == "UNDERPERFORM (LOW CONFIDENCE)"
    assert monthly_summary["schema_version"] == 3
    assert monthly_summary["cross_check"]["visible_in_primary_surfaces"] is False
    assert monthly_summary["classification_shadow"]["probability_actionable_sell_label"] == "28.4%"
    assert monthly_summary["shadow_gate_overlay"]["variant"] in {
        "gemini_veto_0.50",
        "permission_overlay",
        "live",
    }

    dashboard_html = dashboard_path.read_text(encoding="utf-8")
    assert "Classification Confidence Check" in dashboard_html
    assert "Agreement Panel" in dashboard_html

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow_name"] == "monthly_decision"
    assert any(output.endswith("recommendation.md") for output in manifest["outputs"])
    assert any(output.endswith("benchmark_quality.csv") for output in manifest["outputs"])
    assert any(output.endswith("consensus_shadow.csv") for output in manifest["outputs"])
    assert any(output.endswith("classification_shadow.csv") for output in manifest["outputs"])
    assert any(output.endswith("decision_overlays.csv") for output in manifest["outputs"])
    assert any(output.endswith("dashboard.html") for output in manifest["outputs"])
    assert any(output.endswith("monthly_summary.json") for output in manifest["outputs"])
    assert any(
        "Trailing conformal coverage deviates materially from nominal" in warning
        for warning in manifest["warnings"]
    )

    conn = db_client.get_connection(str(db_path))
    perf_log = db_client.get_model_performance_log(conn)
    conn.close()
    assert len(perf_log) == 1
    assert perf_log["aggregate_nw_ic"].iloc[0] == 0.081
    assert perf_log["conformal_trailing_empirical_coverage"].iloc[0] == 0.625
