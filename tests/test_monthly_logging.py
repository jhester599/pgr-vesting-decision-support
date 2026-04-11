"""Tests for structured logging in monthly_decision.py."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import config
from scripts import monthly_decision
from src.models.calibration import CalibrationResult
from src.research.v12 import SnapshotSummary


@patch("src.ingestion.fred_loader.fetch_all_fred_macro", side_effect=RuntimeError("boom"))
@patch("src.ingestion.fred_loader.upsert_fred_to_db")
def test_fetch_fred_step_logs_exception_context(
    mock_upsert,
    mock_fetch,
    monkeypatch,
    caplog,
) -> None:
    del mock_fetch
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

    with caplog.at_level(logging.ERROR):
        monthly_decision._fetch_fred_step(MagicMock(), dry_run=False, skip_fred=False)

    assert "[FRED] Fetch failed. Continuing with cached data." in caplog.text
    assert "RuntimeError: boom" in caplog.text
    mock_upsert.assert_not_called()


def test_main_logs_cross_check_fallback_and_completes(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
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
    monkeypatch.setattr(config, "RECOMMENDATION_LAYER_MODE", "live_with_shadow")
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
            "cw_t_stat": 2.10,
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
        "_build_shadow_baseline_summary",
        lambda *args, **kwargs: (
            SnapshotSummary(
                label="shadow",
                as_of=pd.Timestamp("2026-04-05").date(),
                candidate_name="shadow",
                policy_name="shadow_policy",
                consensus="UNDERPERFORM",
                confidence_tier="MODERATE",
                recommendation_mode="Reduce",
                sell_pct=0.15,
                mean_predicted=-0.02,
                mean_ic=0.07,
                mean_hit_rate=0.56,
                aggregate_oos_r2=0.03,
                aggregate_nw_ic=0.08,
                calibrated_prob_outperform=0.44,
            ),
            None,
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
            "checks": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        monthly_decision,
        "build_promoted_cross_check_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cross-check boom")),
    )

    def _write_stub_diagnostic(
        out_dir: Path,
        as_of,
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
    ) -> None:
        del as_of, ensemble_results, target_horizon_months, cal_result, signals
        del obs_feature_report, representative_cpcv, conformal_coverage_summary
        del importance_stability, vif_series, benchmark_quality_df
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diagnostic.md").write_text("# Diagnostic Stub\n", encoding="utf-8")

    monkeypatch.setattr(monthly_decision, "_write_diagnostic_report", _write_stub_diagnostic)

    with caplog.at_level(logging.ERROR):
        monthly_decision.main(as_of_date_str="2026-04-05", dry_run=True, skip_fred=True)

    assert "Promoted v22 cross-check build failed" in caplog.text
    assert "RuntimeError: cross-check boom" in caplog.text
    assert (out_dir / "recommendation.md").exists()
    assert (out_dir / "consensus_shadow.csv").exists()
    assert (out_dir / "dashboard.html").exists()
    assert (out_dir / "monthly_summary.json").exists()
