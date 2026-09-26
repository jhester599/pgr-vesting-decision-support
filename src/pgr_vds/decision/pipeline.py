"""The monthly decision run: one function, ``main``, called by ``cli/monthly_decision.py``.

Runs on or after the 20th of each month (first business day on or before
the 20th) via GitHub Actions and writes a sell/hold recommendation from the
latest data and the multi-benchmark WFO ensemble. It is a monitoring and
signal-tracking tool: sell/hold decisions are executed only at vesting dates
(January and July).

Steps, each in its own module:

1. ``schedule`` resolves the as-of date and the recommendation-layer mode;
   ``artifacts.already_ran`` makes a repeat run a no-op.
2. ``refresh`` refreshes the FRED macro series (skipped in dry runs).
3. ``signal_generation`` builds the as-of features, trains the ensemble,
   calibrates P(outperform), adds conformal intervals and resolves the live
   consensus.
4. ``health`` computes the realised-only OOS health, the recommendation-mode
   gate, the model-health snapshot, the policy backtest and the manifest
   warnings.
5. ``tax_lots`` and ``portfolio`` add the tax context, holdings guidance,
   redeploy guidance and the Black-Litterman diagnostic.
6. ``recommendation_report`` and ``diagnostic_report`` render the reports;
   ``artifacts`` writes the CSVs, the decision log, the shadow ledgers and
   the run manifest.

Output per run (``artifacts/monthly_decisions/YYYY-MM/``; dry runs write the
gitignored ``results/dry_run/monthly_decisions/YYYY-MM/`` instead):
``recommendation.md``, ``diagnostic.md``, ``signals.csv``,
``benchmark_quality.csv``, ``consensus_shadow.csv``,
``classification_shadow.csv``, ``decision_overlays.csv``, ``dashboard.html``,
``monthly_summary.json``, ``plots/calibration_curve.png`` and
``run_manifest.json``. A production run also appends one row to
``decision_log.md`` and to the shadow ledgers.

Functions in other ``pgr_vds.decision`` modules are called through the
module (``signal_generation.generate_signals(...)``), so a test patches a
function once, in the module that defines it.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import Any

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import config
from pgr_vds.decision import (
    artifacts,
    diagnostic_report,
    health,
    portfolio,
    recommendation_report,
    refresh,
    schedule,
    signal_generation,
    tax_lots,
)
from src.database import db_client
from src.logging_config import configure_logging
from src.models.calibration import CalibrationResult
from src.models.classification_gate_overlay import (
    build_decision_overlay_frame,
    resolve_overlay_policy_variant,
)
from src.models.classification_monitoring import summarize_matured_classifier_history
from src.models.classification_shadow import (
    build_classification_shadow_summary,
    build_ta_replacement_shadow_variants,
)
from src.models.policy_metrics import PolicySummary
from src.reporting.classification_artifacts import (
    write_classification_shadow_csv,
    write_decision_overlays_csv,
)
from src.reporting.cross_check import build_promoted_cross_check_summary
from src.reporting.dashboard_snapshot import write_dashboard_snapshot
from src.reporting.monthly_summary import (
    build_model_health_payload,
    build_monthly_summary_payload,
    write_monthly_summary,
)
from src.reporting.shadow_followon import (
    FOLLOWON_VARIANT_NAME,
    build_followon_decision_overlay_payload,
    build_followon_shadow_payload,
)
from src.reporting.snapshot_summary import SnapshotSummary

logger = logging.getLogger(__name__)

# The six per-benchmark classifier columns copied into ``signals``.
_CLASSIFIER_SIGNAL_COLUMNS: tuple[str, ...] = (
    "classifier_raw_prob_actionable_sell",
    "classifier_prob_actionable_sell",
    "classifier_history_obs",
    "classifier_shadow_tier",
    "classifier_weight",
    "classifier_weighted_contribution",
)


def _build_classification_shadow(
    conn,
    as_of: date,
    signals: pd.DataFrame,
    active_recommendation_mode: dict[str, str | float],
    aggregate_health: dict | None,
) -> tuple[dict[str, Any] | None, list[dict[str, object]], pd.DataFrame]:
    """Build the classifier shadow and its variants.

    Adds the per-benchmark classifier columns to ``signals`` in place.
    Returns ``(summary, variants, artifact_df)``: the baseline shadow payload
    (``None`` when it could not be built), the baseline, follow-on and TA
    replacement variant payloads, and the rows of
    ``classification_shadow.csv``.
    """
    classification_shadow_summary: dict[str, Any] | None = None
    classification_shadow_df = pd.DataFrame()
    try:
        shadow_summary_obj, classification_shadow_df = build_classification_shadow_summary(
            conn,
            as_of,
            live_recommendation_mode=str(active_recommendation_mode["label"]),
            benchmark_quality_df=(
                aggregate_health.get("benchmark_quality_df")
                if aggregate_health is not None
                else None
            ),
            live_sell_pct=float(active_recommendation_mode["sell_pct"]),
        )
        classification_shadow_summary = shadow_summary_obj.to_payload()
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[Classifier shadow] Could not build monthly shadow classifier summary. Error=%r",
            exc,
        )
        classification_shadow_summary = None
        classification_shadow_df = pd.DataFrame()

    if not classification_shadow_df.empty:
        detail_index = classification_shadow_df.set_index("benchmark")
        for column in _CLASSIFIER_SIGNAL_COLUMNS:
            if column in detail_index.columns:
                signals[column] = detail_index[column]

    classification_shadow_variants: list[dict[str, object]] = []
    classification_shadow_artifact_df = classification_shadow_df.copy()
    if not classification_shadow_artifact_df.empty:
        classification_shadow_artifact_df["variant"] = "baseline_shadow"

    if isinstance(classification_shadow_summary, dict) and classification_shadow_summary.get("enabled"):
        baseline_variant = {
            "variant": "baseline_shadow",
            "label": "Current Shadow",
            **classification_shadow_summary,
        }
        followon_variant = build_followon_shadow_payload(
            probability_actionable_sell=classification_shadow_summary.get("probability_actionable_sell"),
            probability_actionable_sell_label=classification_shadow_summary.get(
                "probability_actionable_sell_label"
            ),
            confidence_tier=str(classification_shadow_summary.get("confidence_tier"))
            if classification_shadow_summary.get("confidence_tier") is not None
            else None,
            stance=str(classification_shadow_summary.get("stance"))
            if classification_shadow_summary.get("stance") is not None
            else None,
            probability_investable_pool_label=classification_shadow_summary.get(
                "probability_investable_pool_label"
            ),
            probability_path_b_temp_scaled_label=classification_shadow_summary.get(
                "probability_path_b_temp_scaled_label"
            ),
        )
        classification_shadow_variants = [baseline_variant, followon_variant]
        if not classification_shadow_artifact_df.empty:
            followon_detail_df = classification_shadow_artifact_df.copy()
            followon_detail_df["variant"] = FOLLOWON_VARIANT_NAME
            classification_shadow_artifact_df = pd.concat(
                [classification_shadow_artifact_df, followon_detail_df],
                ignore_index=True,
            )
        try:
            ta_detail_df, ta_variant_payloads = build_ta_replacement_shadow_variants(
                conn,
                as_of,
                baseline_detail_df=classification_shadow_df,
            )
            if ta_variant_payloads:
                classification_shadow_variants.extend(ta_variant_payloads)
            if not ta_detail_df.empty:
                classification_shadow_artifact_df = pd.concat(
                    [classification_shadow_artifact_df, ta_detail_df],
                    ignore_index=True,
                    sort=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[TA shadow] Could not build TA replacement shadow variants. Error=%r",
                exc,
            )
    return (
        classification_shadow_summary,
        classification_shadow_variants,
        classification_shadow_artifact_df,
    )


def _build_decision_overlays(
    classification_shadow_summary: dict[str, Any] | None,
    active_recommendation_mode: dict[str, str | float],
    sell_pct: float,
    consensus: str,
    mean_pred: float,
    mean_ic: float,
    aggregate_health: dict | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, list[dict[str, object]]]:
    """Build the classifier shadow-gate overlay.

    Returns ``(overlay_df, shadow_gate_overlay, variants)``: the rows of
    ``decision_overlays.csv``, the baseline overlay payload (``None`` without
    a classifier probability) and the baseline and follow-on payloads.
    """
    overlay_df = pd.DataFrame()
    shadow_gate_overlay = None
    decision_overlay_variants: list[dict[str, object]] = []
    overlay_variant, overlay_gate_style, overlay_threshold = resolve_overlay_policy_variant()
    if (
        isinstance(classification_shadow_summary, dict)
        and classification_shadow_summary.get("probability_actionable_sell") is not None
    ):
        classifier_prob = float(classification_shadow_summary["probability_actionable_sell"])
        overlay_df, overlay_obj = build_decision_overlay_frame(
            live_mode=str(active_recommendation_mode["label"]),
            live_sell_pct=float(sell_pct),
            consensus=consensus,
            mean_predicted=float(mean_pred),
            mean_ic=float(mean_ic),
            aggregate_oos_r2=(
                float(aggregate_health["oos_r2"]) if aggregate_health is not None else None
            ),
            classifier_prob_actionable_sell=classifier_prob,
            variant=overlay_variant,
            gate_style=overlay_gate_style,
            threshold=overlay_threshold,
        )
        shadow_gate_overlay = overlay_obj.to_payload()
        decision_overlay_variants = [
            {"variant": "baseline_shadow", "label": "Current Shadow", **shadow_gate_overlay},
            build_followon_decision_overlay_payload(shadow_gate_overlay),
        ]
        if not overlay_df.empty:
            baseline_overlay_df = overlay_df.copy()
            baseline_overlay_df["variant"] = "baseline_shadow"
            followon_overlay_df = overlay_df.copy()
            followon_overlay_df["variant"] = FOLLOWON_VARIANT_NAME
            overlay_df = pd.concat(
                [baseline_overlay_df, followon_overlay_df],
                ignore_index=True,
            )
    return overlay_df, shadow_gate_overlay, decision_overlay_variants


def _print_console_summary(
    *,
    signals: pd.DataFrame,
    consensus: str,
    confidence_tier: str,
    mean_pred: float,
    mean_cal: float | None,
    mean_ic: float,
    mean_hr: float,
    aggregate_health: dict | None,
    active_recommendation_mode: dict[str, str | float],
    sell_pct: float,
    consensus_shadow_df: pd.DataFrame | None,
    live_summary: SnapshotSummary,
    shadow_summary: SnapshotSummary | None,
    classification_shadow_summary: dict[str, Any] | None,
    shadow_gate_overlay: dict[str, Any] | None,
) -> None:
    """Print the run's headline numbers to stdout (the Actions log)."""
    print(f"\n  Consensus signal: {consensus} ({confidence_tier} CONFIDENCE)")
    print(f"  Predicted 6M relative return: {mean_pred:+.2%}")
    if "calibrated_prob_outperform" in signals.columns and not signals.empty:
        print(f"  P(outperform, calibrated): {mean_cal:.1%}")
    print(f"  Mean IC (equal-weight): {mean_ic:.4f}  |  Mean hit rate: {mean_hr:.1%}")
    if aggregate_health is not None:
        print(f"  Aggregate OOS R^2 (vs prevailing mean): {aggregate_health['oos_r2']:.2%}")
        print(
            "  Directional skill: hit "
            f"{float(aggregate_health.get('agg_hit', float('nan'))):.1%} vs base rate "
            f"{float(aggregate_health.get('constant_rule_hit_rate', float('nan'))):.1%}; "
            f"Pesaran-Timmermann p={float(aggregate_health.get('pt_p_value', float('nan'))):.3f}"
        )
    print(f"  Recommendation mode: {active_recommendation_mode['label']}")
    print(f"  Sell %: {sell_pct:.0%}")
    if consensus_shadow_df is not None and not consensus_shadow_df.empty:
        shadow_row = consensus_shadow_df[~consensus_shadow_df["is_live_path"]]
        if not shadow_row.empty:
            row = shadow_row.iloc[0]
            print(
                "  Shadow cross-check: "
                f"{row['recommendation_mode']} / sell {float(row['recommended_sell_pct']):.0%} / "
                f"{float(row['mean_predicted_return']):+.2%}"
            )
    if shadow_summary is not None:
        print(
            "  Visible cross-check: "
            f"{live_summary.candidate_name} / {live_summary.recommendation_mode} / "
            f"sell {live_summary.sell_pct:.0%} / {live_summary.mean_predicted:+.2%}"
        )
        print(
            "  Simpler baseline: "
            f"{shadow_summary.recommendation_mode} / sell {shadow_summary.sell_pct:.0%} / "
            f"{shadow_summary.mean_predicted:+.2%}"
        )
    if isinstance(classification_shadow_summary, dict) and classification_shadow_summary.get("enabled"):
        print(
            "  Classification shadow: "
            f"P(actionable sell) {classification_shadow_summary.get('probability_actionable_sell_label', 'n/a')} / "
            f"{classification_shadow_summary.get('confidence_tier', 'n/a')} / "
            f"{classification_shadow_summary.get('agreement_label', 'n/a')}"
        )
    if isinstance(shadow_gate_overlay, dict):
        print(
            "  Shadow gate overlay: "
            f"{shadow_gate_overlay.get('recommendation_mode', 'n/a')} / sell "
            f"{float(shadow_gate_overlay.get('recommended_sell_pct', sell_pct)):.0%} / "
            f"{shadow_gate_overlay.get('reason', 'n/a')}"
        )


def main(
    as_of_date_str: str | None = None,
    dry_run: bool = False,
    skip_fred: bool = False,
) -> None:
    """Run the monthly decision for one as-of date (see the module docstring)."""
    configure_logging()
    layer_mode = schedule.validate_layer_mode(config.RECOMMENDATION_LAYER_MODE)
    as_of = schedule.resolve_as_of_date(as_of_date_str)
    run_date = date.today()

    logger.info("%sPGR Monthly Decision - as-of %s", "[DRY RUN] " if dry_run else "", as_of)
    logger.info("Run date: %s", run_date)

    # Idempotency: skip if this month's report already exists
    if artifacts.already_ran(as_of) and not dry_run:
        logger.info("Report for %s already exists. Skipping.", as_of.strftime("%Y-%m"))
        artifacts.write_step_output("generated", "false")
        return

    if dry_run:
        # Read-only: no migrations and no writes. Any write attempt raises.
        conn = db_client.get_connection(config.DB_PATH, read_only=True)
    else:
        conn = db_client.get_connection(config.DB_PATH)
        db_client.initialize_schema(conn)
    db_client.warn_if_db_behind(conn, context="monthly_decision")

    # Step 1: Refresh FRED data
    refresh.fetch_fred_step(conn, dry_run=dry_run, skip_fred=skip_fred)

    # Checked after the FRED refresh so the report describes the data used.
    freshness_report = db_client.check_data_freshness(conn, run_date)
    for message in freshness_report["warnings"]:
        logger.warning("[data-freshness] %s", message)

    # Step 2: Generate signals (lean Ridge + GBT ensemble per benchmark)
    logger.info("Generating ensemble signals (as-of %s)...", as_of)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", category=ConvergenceWarning)
        signals, ensemble_results, diagnostics = signal_generation.generate_signals(
            conn, as_of, target_horizon_months=6
        )

    convergence_warnings = [
        w for w in captured_warnings if issubclass(w.category, ConvergenceWarning)
    ]
    if convergence_warnings:
        logger.warning(
            "[Modeling] %s convergence warnings were suppressed during WFO fitting. "
            "Results completed; consider stronger regularisation or leaner feature sets if this count grows.",
            len(convergence_warnings),
        )

    prequential_panel = diagnostics.get("prequential_panel")

    # Step 2.5: Calibrate P(outperform) per benchmark (Platt); prequential ECE
    logger.info("Calibrating probabilities...")
    cal_result: CalibrationResult
    signals, cal_result, cal_probs, cal_outcomes = signal_generation.calibrate_signals(
        signals, ensemble_results, target_horizon_months=6, panel=prequential_panel
    )
    logger.info(
        "Calibration: %s (n=%s prequential OOS obs, ECE=%s)",
        cal_result.method,
        f"{cal_result.n_obs:,}",
        f"{cal_result.ece:.1%}",
    )

    # Step 2.7: Compute conformal prediction intervals (ACI / split conformal)
    logger.info("Computing conformal prediction intervals...")
    signals = signal_generation.compute_conformal_intervals(signals, ensemble_results, panel=prequential_panel)
    conformal_coverage_summary = signal_generation.summarize_conformal_coverage(signals)
    if "ci_lower" in signals.columns and not signals.empty:
        valid_ci = signals[["ci_lower", "ci_upper"]].dropna()
        if not valid_ci.empty:
            ci_lo_med = float(valid_ci["ci_lower"].median())
            ci_hi_med = float(valid_ci["ci_upper"].median())
            logger.info(
                "%s CI (median across benchmarks): %s to %s",
                f"{config.CONFORMAL_COVERAGE:.0%}",
                f"{ci_lo_med:+.2%}",
                f"{ci_hi_med:+.2%}",
            )

    aggregate_health = health.compute_aggregate_health(
        ensemble_results, target_horizon_months=6, panel=prequential_panel
    )
    representative_cpcv = diagnostics.get("representative_cpcv")
    # Step 3: Compute consensus
    (
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        mean_prob,
        confidence_tier,
    ), consensus_shadow_df = signal_generation.resolve_live_consensus(
        signals,
        aggregate_health,
        representative_cpcv,
    )
    recommendation_mode = health.determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        representative_cpcv,
    )
    sell_pct = float(recommendation_mode["sell_pct"])
    mean_cal = (
        float(signals["calibrated_prob_outperform"].mean())
        if "calibrated_prob_outperform" in signals.columns and not signals.empty
        else None
    )
    fallback_live_summary = SnapshotSummary(
        label="live",
        as_of=as_of,
        candidate_name="production_quality_weighted_consensus",
        policy_name="quality_weighted_consensus_v74",
        consensus=consensus,
        confidence_tier=confidence_tier,
        recommendation_mode=str(recommendation_mode["label"]),
        sell_pct=sell_pct,
        mean_predicted=mean_pred,
        mean_ic=mean_ic,
        mean_hit_rate=mean_hr,
        aggregate_oos_r2=float(aggregate_health["oos_r2"]) if aggregate_health is not None else float("nan"),
        aggregate_nw_ic=float(aggregate_health["nw_ic"]) if aggregate_health is not None else float("nan"),
        calibrated_prob_outperform=mean_cal,
    )
    live_summary = fallback_live_summary
    if layer_mode in {"live_with_shadow", "shadow_promoted"}:
        try:
            promoted_summary = build_promoted_cross_check_summary(
                conn,
                as_of,
                target_horizon_months=6,
            )
            if promoted_summary is not None:
                live_summary = promoted_summary
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[Cross-check] Promoted v22 cross-check build failed; falling back to the current production ensemble snapshot. Error=%r",
                exc,
            )
    shadow_summary = None
    if layer_mode in {"live_with_shadow", "shadow_promoted"}:
        shadow_summary, _ = signal_generation.build_shadow_baseline_summary(
            conn,
            as_of,
            target_horizon_months=6,
        )
    active_recommendation_mode = recommendation_mode
    visible_cross_check = False
    recommendation_layer_label = "Live production recommendation layer (quality-weighted consensus)"
    if layer_mode == "live_with_shadow":
        recommendation_layer_label = (
            "Live production recommendation layer (quality-weighted consensus); "
            "equal-weight comparison retained in diagnostic artifacts"
        )
    elif layer_mode == "shadow_promoted" and shadow_summary is not None:
        active_recommendation_mode = health.mode_payload_from_summary(shadow_summary)
        sell_pct = float(active_recommendation_mode["sell_pct"])
        recommendation_layer_label = (
            "v13.1 promoted simpler diversification-first recommendation layer; "
            "quality-weighted comparison retained in diagnostic artifacts"
        )
    model_drift_summary = health.record_model_health_snapshot(
        conn,
        as_of,
        aggregate_health,
        cal_result,
        conformal_coverage_summary,
        dry_run=dry_run,
    )
    existing_holdings = tax_lots.build_existing_holdings_guidance(conn, as_of)
    redeploy_buckets = portfolio.build_redeploy_guidance(conn)
    redeploy_portfolio = portfolio.build_redeploy_portfolio(
        conn,
        signals,
        active_recommendation_mode,
    )
    (
        classification_shadow_summary,
        classification_shadow_variants,
        classification_shadow_artifact_df,
    ) = _build_classification_shadow(
        conn, as_of, signals, active_recommendation_mode, aggregate_health
    )
    overlay_df, shadow_gate_overlay, decision_overlay_variants = _build_decision_overlays(
        classification_shadow_summary,
        active_recommendation_mode,
        sell_pct,
        consensus,
        mean_pred,
        mean_ic,
        aggregate_health,
    )

    _print_console_summary(
        signals=signals,
        consensus=consensus,
        confidence_tier=confidence_tier,
        mean_pred=mean_pred,
        mean_cal=mean_cal,
        mean_ic=mean_ic,
        mean_hr=mean_hr,
        aggregate_health=aggregate_health,
        active_recommendation_mode=active_recommendation_mode,
        sell_pct=sell_pct,
        consensus_shadow_df=consensus_shadow_df,
        live_summary=live_summary,
        shadow_summary=shadow_summary,
        classification_shadow_summary=classification_shadow_summary,
        shadow_gate_overlay=shadow_gate_overlay,
    )

    # Step 4: Write outputs
    # v32.2 + v32.3 — compute policy backtest summary for recommendation.md
    policy_summary: dict[str, PolicySummary] | None = None
    try:
        policy_summary = health.compute_policy_summary(ensemble_results, panel=prequential_panel)
    except Exception:
        logger.warning(
            "Could not compute policy summary; Decision Policy Backtest section will be omitted",
            exc_info=True,
        )

    bl_diagnostics = portfolio.build_bl_diagnostics(conn, ensemble_results, as_of)

    out_dir = artifacts.dry_run_output_dir(as_of) if dry_run else artifacts.output_dir(as_of)
    if dry_run:
        logger.info("[DRY RUN] Writing artifacts to %s (production files untouched).", out_dir)
    classification_shadow_path = write_classification_shadow_csv(
        out_dir,
        classification_shadow_artifact_df,
    )
    print(f"  Wrote {classification_shadow_path}")
    decision_overlay_path = write_decision_overlays_csv(out_dir, overlay_df)
    print(f"  Wrote {decision_overlay_path}")

    # Shadow ledgers live next to the production month folders.
    history_df = artifacts.update_shadow_ledgers(
        conn,
        as_of=as_of,
        run_date=run_date,
        history_base_dir=artifacts.output_dir(as_of).parent,
        dry_run=dry_run,
        classification_shadow_summary=classification_shadow_summary,
        classification_shadow_variants=classification_shadow_variants,
        live_recommendation_mode=str(active_recommendation_mode["label"]),
        live_sell_pct=float(sell_pct),
        shadow_gate_overlay=shadow_gate_overlay,
    )
    classifier_monitoring_summary = summarize_matured_classifier_history(history_df).to_payload()

    recommendation_report.write_recommendation_md(
        out_dir, as_of, run_date, conn, signals,
        consensus, mean_pred, mean_ic, mean_hr, sell_pct, dry_run,
        mean_prob_outperform=mean_prob,
        composite_confidence_tier=confidence_tier,
        cal_result=cal_result,
        aggregate_health=aggregate_health,
        recommendation_mode=active_recommendation_mode,
        live_summary=live_summary,
        shadow_summary=shadow_summary,
        consensus_shadow_df=consensus_shadow_df,
        existing_holdings=existing_holdings,
        redeploy_buckets=redeploy_buckets,
        redeploy_portfolio=redeploy_portfolio,
        recommendation_layer_label=recommendation_layer_label,
        representative_cpcv=representative_cpcv,
        freshness_report=freshness_report,
        model_drift_summary=model_drift_summary,
        policy_summary=policy_summary,
        bl_diagnostics=bl_diagnostics,
        visible_cross_check=visible_cross_check,
        classification_shadow_summary=classification_shadow_summary,
        shadow_gate_overlay=shadow_gate_overlay,
    )
    artifacts.write_signals_csv(out_dir, signals)
    if dry_run:
        logger.info("[DRY RUN] Not appending to decision_log.md.")
    else:
        artifacts.append_decision_log(
            as_of, run_date, consensus, sell_pct, mean_pred, mean_ic, mean_hr, dry_run,
        )

    # Step 5: Write diagnostic OOS evaluation report
    importance_stability = health.compute_importance_stability(ensemble_results)

    print("\nWriting diagnostic report...")
    diagnostic_report.write_diagnostic_report(
        out_dir, as_of, ensemble_results,
        target_horizon_months=6, cal_result=cal_result,
        signals=signals,
        obs_feature_report=diagnostics.get("obs_feature_report"),
        representative_cpcv=representative_cpcv,
        conformal_coverage_summary=conformal_coverage_summary,
        importance_stability=importance_stability,
        vif_series=diagnostics.get("vif_series"),
        benchmark_quality_df=(
            aggregate_health.get("benchmark_quality_df")
            if aggregate_health is not None
            else None
        ),
        shadow_gate_overlay=shadow_gate_overlay,
        classifier_monitoring_summary=classifier_monitoring_summary,
        aggregate_health=aggregate_health,
        shrinkage_alpha=diagnostics.get("shrinkage_alpha"),
    )
    artifacts.write_benchmark_quality_csv(
        out_dir,
        aggregate_health.get("benchmark_quality_df") if aggregate_health is not None else None,
    )
    artifacts.write_consensus_shadow_csv(out_dir, consensus_shadow_df)
    # Step 5.1 (P2.7): Calibration reliability diagram
    diagnostic_report.plot_calibration_curve(out_dir, cal_probs, cal_outcomes, cal_result)

    snapshot = db_client.get_operational_snapshot(conn)

    # v35.1: evaluate retrain trigger and record to audit log
    health.evaluate_and_record_retrain_trigger(conn, model_drift_summary, dry_run=dry_run)

    nan_live_features = list(diagnostics.get("nan_live_features") or [])
    manifest_warnings = health.build_manifest_warnings(
        freshness_report=freshness_report,
        aggregate_health=aggregate_health,
        representative_cpcv=representative_cpcv,
        obs_feature_report=diagnostics.get("obs_feature_report"),
        conformal_coverage_summary=conformal_coverage_summary,
        nan_live_features=nan_live_features,
        model_drift_summary=model_drift_summary,
        shadow_summary=shadow_summary,
        live_summary=live_summary,
        consensus_shadow_df=consensus_shadow_df,
        recommendation_mode=recommendation_mode,
        shadow_gate_overlay=shadow_gate_overlay,
    )

    write_dashboard_snapshot(
        out_dir,
        as_of_date=as_of.isoformat(),
        recommendation_mode=str(active_recommendation_mode["label"]),
        consensus=consensus,
        sell_pct=float(sell_pct),
        mean_predicted=float(mean_pred),
        mean_ic=float(mean_ic),
        mean_hit_rate=float(mean_hr),
        aggregate_oos_r2=(
            float(aggregate_health["oos_r2"]) if aggregate_health is not None else None
        ),
        recommendation_layer_label=recommendation_layer_label,
        warnings=list(manifest_warnings),
        signals=signals.reset_index(),
        benchmark_quality_df=(
            aggregate_health.get("benchmark_quality_df") if aggregate_health is not None else None
        ),
        consensus_shadow_df=consensus_shadow_df,
        classification_shadow_summary=classification_shadow_summary,
        shadow_gate_overlay=shadow_gate_overlay,
        classification_shadow_variants=classification_shadow_variants,
    )
    monthly_summary = build_monthly_summary_payload(
        as_of_date=as_of.isoformat(),
        run_date=run_date.isoformat(),
        recommendation_layer_label=recommendation_layer_label,
        consensus=consensus,
        confidence_tier=confidence_tier,
        recommendation_mode=str(active_recommendation_mode["label"]),
        sell_pct=float(sell_pct),
        mean_predicted=float(mean_pred),
        mean_ic=float(mean_ic),
        mean_hit_rate=float(mean_hr),
        # The raw BayesianRidge probability was retired (it was a constant
        # 0.5); only the calibrated probability is reported (F21).
        mean_prob_outperform=None,
        calibrated_prob_outperform=mean_cal,
        aggregate_oos_r2=(
            float(aggregate_health["oos_r2"]) if aggregate_health is not None else None
        ),
        aggregate_nw_ic=(
            float(aggregate_health["nw_ic"]) if aggregate_health is not None else None
        ),
        warnings=list(manifest_warnings),
        signals=signals.reset_index(),
        benchmark_quality_df=(
            aggregate_health.get("benchmark_quality_df") if aggregate_health is not None else None
        ),
        consensus_shadow_df=consensus_shadow_df,
        visible_cross_check=visible_cross_check,
        classification_shadow_summary=classification_shadow_summary,
        shadow_gate_overlay=shadow_gate_overlay,
        classification_shadow_variants=classification_shadow_variants,
        decision_overlay_variants=decision_overlay_variants,
        model_health=build_model_health_payload(
            mean_ic=float(mean_ic),
            aggregate_health=aggregate_health,
            representative_cpcv=representative_cpcv,
            cal_result=cal_result,
            conformal_trailing_coverage=(
                conformal_coverage_summary.trailing_empirical_coverage
                if conformal_coverage_summary is not None
                else None
            ),
            shrinkage_alpha=diagnostics.get("shrinkage_alpha"),
            quality_weighted_ic=signal_generation.live_variant_mean_ic(consensus_shadow_df),
        ),
    )
    monthly_summary_path = write_monthly_summary(out_dir, monthly_summary)
    print(f"  Wrote {monthly_summary_path}")

    artifacts.write_monthly_run_manifest(
        out_dir,
        snapshot=snapshot,
        as_of=as_of,
        manifest_warnings=manifest_warnings,
        dry_run=dry_run,
        nan_live_features=nan_live_features,
    )

    conn.close()
    logger.info("Done. Results written to %s/", out_dir)
    # The workflow commits the charts and sends the email only when a new
    # production report was generated (review F26).
    artifacts.write_step_output("generated", "false" if dry_run else "true")
