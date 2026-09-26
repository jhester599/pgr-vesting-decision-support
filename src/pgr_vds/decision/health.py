"""Model health and gating for the monthly decision run.

Realised-only (prequential) OOS diagnostics, the recommendation-mode gate,
the model-health snapshot and drift/retrain audit, the decision-policy
backtest, and the warnings recorded in ``run_manifest.json``.
Functions in other ``pgr_vds.decision`` modules are called through the module
(``health.compute_aggregate_health(...)``), so a test patches a function
once, in the module that defines it.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd

import config
from src.database import db_client
from src.models.calibration import CalibrationResult
from src.models.conformal import ConformalCoverageBacktest
from src.models.drift_monitor import ModelDriftSummary, summarize_latest_model_drift
from src.models.evaluation import (
    FeatureImportanceStability,
    compute_feature_importance_stability,
)
from src.models.forecast_diagnostics import (
    IC_P_VALUE_METHOD,
    summarize_panel_diagnostics,
)
from src.models.live_policy_backtest import (
    LIVE_MAPPING_POLICY,
    evaluate_live_mapping,
    historical_live_decisions,
)
from src.models.policy_metrics import (
    FIXED_POLICIES,
    SIGNAL_POLICIES,
    PolicySummary,
    evaluate_policy_series,
)
from src.models.prequential import build_prequential_panel
from src.models.retrain_trigger import RetainTriggerResult, evaluate_retrain_trigger
from src.models.wfo_engine import CPCVResult, WFOResult
from src.reporting.decision_rendering import (
    determine_recommendation_mode as render_determine_recommendation_mode,
)
from src.reporting.decision_rendering import (
    sell_pct_from_consensus,
)
from src.reporting.snapshot_summary import SnapshotSummary

logger = logging.getLogger(__name__)


def prequential_panel(
    ensemble_results: dict,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the realised-only OOS panel, building it when not supplied."""
    if panel is not None:
        return panel
    return build_prequential_panel(ensemble_results)


def benchmark_rows(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One benchmark's panel rows in date order."""
    if panel.empty:
        return panel
    rows = panel[panel["benchmark"] == str(ticker)]
    return rows.sort_values("date", kind="mergesort")


def flag(value: float, good: float, marginal: float, higher_is_better: bool = True) -> str:
    """Return ✅ / ⚠️ / ❌ based on value vs. thresholds."""
    if higher_is_better:
        if value >= good:
            return "✅"
        if value >= marginal:
            return "⚠️"
        return "❌"
    # lower is better (e.g. MAE)
    if value <= good:
        return "✅"
    if value <= marginal:
        return "⚠️"
    return "❌"


def directional_flag(p_value: float) -> str:
    """✅ / ⚠️ / ❌ for a one-sided Pesaran-Timmermann p-value."""
    if p_value is None or not np.isfinite(p_value):
        return "❌"
    if p_value < config.DIAG_MAX_DIRECTIONAL_PVALUE:
        return "✅"
    if p_value < config.DIAG_MARGINAL_DIRECTIONAL_PVALUE:
        return "⚠️"
    return "❌"


def build_benchmark_quality_frame(
    ensemble_results: dict,
    target_horizon_months: int = 6,
    panel: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Build pooled and per-benchmark OOS diagnostics from the prequential panel.

    Predictions use realised-only ensemble weights and shrinkage; OOS R^2 is
    scored against each benchmark's prevailing mean of realised targets,
    training history included (review 2026-09-25, F04 and F13).
    """
    panel = prequential_panel(ensemble_results, panel)
    if panel.empty:
        return (
            pd.Series(dtype=float, name="y_hat"),
            pd.Series(dtype=float, name="y_true"),
            pd.DataFrame(),
        )

    _, per_benchmark = summarize_panel_diagnostics(panel, target_horizon_months=target_horizon_months)
    counts = panel.groupby("benchmark")["y_true"].count()
    per_benchmark = per_benchmark[
        per_benchmark["benchmark"].map(counts).fillna(0) >= 2
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    for summary in per_benchmark.to_dict("records"):
        rows.append(
            {
                "benchmark": str(summary["benchmark"]),
                "n_obs": int(summary["n_obs"]),
                "oos_r2": float(summary["oos_r2"]),
                "nw_ic": float(summary["nw_ic"]),
                "nw_p_value": float(summary["nw_p_value"]),
                "hit_rate": float(summary["hit_rate"]),
                "base_rate": float(summary["base_rate"]),
                "hit_rate_excess": float(summary["hit_rate_excess"]),
                "pt_p_value": float(summary["pt_p_value"]),
                "cw_t_stat": float(summary["cw_t_stat"]),
                "cw_p_value": float(summary["cw_p_value"]),
                "cw_mean_adjusted_differential": float(
                    summary["cw_mean_adjusted_differential"]
                ),
                "r2_flag": flag(float(summary["oos_r2"]), config.DIAG_MIN_OOS_R2, 0.005),
                "ic_flag": flag(float(summary["nw_ic"]), config.DIAG_MIN_IC, 0.03),
                "hr_flag": directional_flag(float(summary["pt_p_value"])),
            }
        )

    dates = pd.DatetimeIndex(panel["date"])
    agg_predicted = pd.Series(panel["y_hat"].to_numpy(dtype=float), index=dates, name="y_hat")
    agg_realized = pd.Series(panel["y_true"].to_numpy(dtype=float), index=dates, name="y_true")

    benchmark_quality_df = pd.DataFrame(rows)
    if not benchmark_quality_df.empty:
        benchmark_quality_df = benchmark_quality_df.sort_values(
            by=["cw_t_stat", "nw_ic", "oos_r2"],
            ascending=False,
        ).reset_index(drop=True)

    return agg_predicted, agg_realized, benchmark_quality_df


def compute_aggregate_health(
    ensemble_results: dict,
    target_horizon_months: int = 6,
    panel: pd.DataFrame | None = None,
) -> dict | None:
    """Compute the aggregate health metrics shared by recommendation and diagnostics.

    All metrics come from the prequential panel (review 2026-09-25, WP7):
    pooled OOS R^2 against each benchmark's prevailing mean; pooled IC with a
    Driscoll-Kraay p-value clustered by date (``nw_pval``); hit rate against
    the base rate with the Pesaran-Timmermann test the gate uses.
    """
    panel = prequential_panel(ensemble_results, panel)
    agg_predicted, agg_realized, benchmark_quality_df = build_benchmark_quality_frame(
        ensemble_results=ensemble_results,
        target_horizon_months=target_horizon_months,
        panel=panel,
    )
    if len(agg_realized) < 4:
        return None

    summary, _ = summarize_panel_diagnostics(panel, target_horizon_months=target_horizon_months)
    nw_lags = target_horizon_months - 1
    oos_r2 = float(summary["oos_r2"])
    nw_ic = float(summary["nw_ic"])
    nw_pval = float(summary["nw_p_value"])
    agg_hit = float(summary["hit_rate"])
    pt_p_value = float(summary["pt_p_value"])

    return {
        "n_agg": int(summary["n_obs"]),
        "n_dates": int(summary["n_dates"]),
        "nw_lags": nw_lags,
        "oos_r2": oos_r2,
        "nw_ic": nw_ic,
        "nw_pval": nw_pval,
        "ic_p_value_method": IC_P_VALUE_METHOD,
        "agg_hit": agg_hit,
        "n_calls": int(summary["n_calls"]),
        "base_rate": float(summary["base_rate"]),
        "constant_rule_hit_rate": float(summary["constant_rule_hit_rate"]),
        "hit_rate_excess": float(summary["hit_rate_excess"]),
        "pt_stat": float(summary["pt_stat"]),
        "pt_p_value": pt_p_value,
        "cw_t_stat": float(summary["cw_t_stat"]),
        "cw_p_value": float(summary["cw_p_value"]),
        "cw_mean_adjusted_differential": float(summary["cw_mean_adjusted_differential"]),
        "r2_flag": flag(oos_r2, config.DIAG_MIN_OOS_R2, 0.005),
        "ic_flag": flag(nw_ic, config.DIAG_MIN_IC, 0.03),
        "hr_flag": directional_flag(pt_p_value),
        "per_benchmark_rows": benchmark_quality_df.to_dict("records"),
        "benchmark_quality_df": benchmark_quality_df,
        "agg_predicted": agg_predicted,
        "agg_score": pd.Series(
            panel["z"].to_numpy(dtype=float), index=pd.DatetimeIndex(panel["date"]), name="z"
        ),
        "agg_realized": agg_realized,
        "panel": panel,
        "shrinkage_alpha_last": float(panel["alpha"].iloc[-1]) if not panel.empty else float("nan"),
    }


def determine_recommendation_mode(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    aggregate_health: dict | None,
    representative_cpcv: CPCVResult | None,
) -> dict[str, str | float]:
    """Compatibility wrapper around the extracted decision-rendering helper."""
    return render_determine_recommendation_mode(
        consensus=consensus,
        mean_predicted=mean_predicted,
        mean_ic=mean_ic,
        mean_hr=mean_hr,
        aggregate_health=aggregate_health,
        representative_cpcv=representative_cpcv,
    )


def mode_payload_from_summary(summary: SnapshotSummary) -> dict[str, str | float]:
    """Convert a snapshot label back into the report's recommendation-mode payload."""
    label = summary.recommendation_mode
    if label == "ACTIONABLE":
        return {
            "mode": "actionable",
            "label": label,
            "sell_pct": summary.sell_pct,
            "summary": "The simpler diversification-first baseline is active and strong enough to influence the vest decision.",
            "action_note": "Use the simpler diversification-first baseline as the active recommendation layer for this run.",
        }
    if label == "DEFER-TO-TAX-DEFAULT":
        return {
            "mode": "defer-to-tax-default",
            "label": label,
            "sell_pct": summary.sell_pct,
            "summary": "The simpler diversification-first baseline is active, but still points back to the default diversification and tax-discipline rule.",
            "action_note": "Use the simpler diversification-first baseline, which still defers to the default vesting rule.",
        }
    return {
        "mode": "monitoring-only",
        "label": label,
        "sell_pct": summary.sell_pct,
        "summary": "The simpler diversification-first baseline is active, but only supports monitoring rather than a prediction-led change.",
        "action_note": "Use the simpler diversification-first baseline as monitoring evidence only.",
    }


def record_model_health_snapshot(
    conn,
    as_of: date,
    aggregate_health: dict | None,
    cal_result: CalibrationResult | None,
    conformal_coverage_summary: ConformalCoverageBacktest | None,
    dry_run: bool = False,
) -> ModelDriftSummary | None:
    """Persist the monthly model-health snapshot and return the latest drift summary.

    Every stored value is look-ahead-free (review 2026-09-25, WP7): OOS R^2
    against the prevailing mean of realised targets, prequential ECE and
    trailing conformal coverage. Rows are tagged with
    ``config.MODEL_HEALTH_METRICS_VERSION``, and the drift summary only uses
    rows of that version.

    With ``dry_run`` the snapshot is merged into the stored history in memory
    only, so the drift summary matches a real run without writing to the DB.
    """
    if aggregate_health is None or cal_result is None:
        return None

    month_end = (
        pd.Timestamp(as_of).to_period("M").to_timestamp("M").date().isoformat()
    )
    records = [
        {
            "month_end": month_end,
            "metrics_version": config.MODEL_HEALTH_METRICS_VERSION,
            "aggregate_oos_r2": float(aggregate_health["oos_r2"]),
            "aggregate_nw_ic": float(aggregate_health["nw_ic"]),
            "aggregate_hit_rate": float(aggregate_health["agg_hit"]),
            "ece": float(cal_result.ece),
            "ece_ci_lower": float(cal_result.ece_ci_lower),
            "ece_ci_upper": float(cal_result.ece_ci_upper),
            "conformal_target_coverage": (
                float(conformal_coverage_summary.target_coverage)
                if conformal_coverage_summary is not None
                else None
            ),
            "conformal_empirical_coverage": (
                float(conformal_coverage_summary.empirical_coverage)
                if conformal_coverage_summary is not None
                else None
            ),
            "conformal_trailing_empirical_coverage": (
                float(conformal_coverage_summary.trailing_empirical_coverage)
                if conformal_coverage_summary is not None
                else None
            ),
            "conformal_trailing_coverage_gap": (
                float(conformal_coverage_summary.trailing_coverage_gap)
                if conformal_coverage_summary is not None
                else None
            ),
        }
    ]
    if dry_run:
        logger.info("[DRY RUN] Not writing the model-health snapshot to the DB.")
        history = db_client.get_model_performance_log(conn)
        snapshot_df = pd.DataFrame(records)
        snapshot_df["month_end"] = pd.to_datetime(snapshot_df["month_end"])
        snapshot_df = snapshot_df.set_index("month_end")
        if not history.empty:
            history = history.loc[history.index != snapshot_df.index[0]]
            history = pd.concat([history, snapshot_df]).sort_index()
        else:
            history = snapshot_df
        history.index.name = "month_end"
    else:
        db_client.upsert_model_performance_log(conn, records)
        history = db_client.get_model_performance_log(conn)
    if history.empty:
        return None
    # A back-dated run must not see snapshots logged after its own month.
    history = history.loc[history.index <= pd.Timestamp(month_end)]
    if history.empty:
        return None
    return summarize_latest_model_drift(history.reset_index())


def evaluate_and_record_retrain_trigger(
    conn,
    drift_summary: ModelDriftSummary | None,
    dry_run: bool = False,
) -> RetainTriggerResult | None:
    """
    Evaluate the retrain trigger against the current drift state and persist
    the result to model_retrain_log for governance/audit (skipped in dry runs).

    Returns the RetainTriggerResult, or None if the DB table is unavailable.
    """
    try:
        last_date = db_client.get_last_retrain_trigger_date(conn)
        result = evaluate_retrain_trigger(
            drift_summary=drift_summary,
            last_trigger_date=last_date,
        )
        if dry_run:
            logger.info("[DRY RUN] Not recording the retrain-trigger evaluation.")
        else:
            db_client.record_retrain_event(
                conn,
                triggered_at=result.evaluated_at,
                breach_streak=result.breach_streak,
                triggered=result.triggered,
                cooldown_active=result.cooldown_active,
                last_trigger_date=result.last_trigger_date,
                notes=result.notes,
            )
        if result.triggered:
            logger.warning(
                "Retrain trigger fired: %s (streak=%d, last=%s)",
                result.notes,
                result.breach_streak,
                result.last_trigger_date or "never",
            )
        else:
            logger.info("Retrain trigger evaluated: %s", result.notes)
        return result
    except Exception:
        logger.debug("Retrain trigger evaluation skipped", exc_info=True)
        return None


def compute_policy_summary(
    ensemble_results: dict,
    panel: pd.DataFrame | None = None,
) -> dict[str, PolicySummary] | None:
    """Compute OOS decision-policy summaries from the monthly ensemble.

    Aggregates OOS predictions and realized relative returns from the deployed
    inverse-variance ensemble across all benchmarks, then evaluates every fixed
    and signal-driven policy defined in ``src.models.policy_metrics``.

    Returns a dict of ``{policy_name: PolicySummary}`` or ``None`` if fewer
    than 4 OOS observations are available.
    """
    predicted, realized, _ = build_benchmark_quality_frame(ensemble_results, panel=panel)
    if len(realized) < 4:
        return None

    summaries: dict[str, PolicySummary] = {}
    for policy in list(FIXED_POLICIES) + list(SIGNAL_POLICIES):
        try:
            summaries[policy] = evaluate_policy_series(
                predicted=predicted,
                realized_relative_return=realized,
                policy_name=policy,
            )
        except Exception:
            logger.warning(
                "compute_policy_summary: failed to evaluate policy '%s'; skipping",
                policy,
                exc_info=True,
            )

    # The live ACTIONABLE mapping itself (review 2026-09-25, F20): the live
    # consensus and sell-% function replayed at every OOS date, scored on the
    # equal-weight mean relative return of that date.
    try:
        live_panel = prequential_panel(ensemble_results, panel)
        decisions = historical_live_decisions(live_panel, sell_pct_from_consensus)
        if not decisions.empty:
            summaries[LIVE_MAPPING_POLICY] = evaluate_live_mapping(decisions)
    except Exception:
        logger.warning(
            "compute_policy_summary: failed to backtest the live mapping; skipping",
            exc_info=True,
        )

    return summaries if summaries else None


def compute_importance_stability(
    ensemble_results: dict,
) -> FeatureImportanceStability | None:
    """Feature-importance stability of the primary model (v32.0), or ``None``.

    The primary model is the ``elasticnet`` member (else the first member) of
    the ``config.V13_SHADOW_BASELINE_STRATEGY`` or VTI ensemble, else of the
    first benchmark's ensemble.
    """
    importance_stability: FeatureImportanceStability | None = None
    try:
        primary_wfo: WFOResult | None = None
        for _etf in (config.V13_SHADOW_BASELINE_STRATEGY, "VTI"):
            ens = ensemble_results.get(_etf)
            if ens is not None:
                primary_wfo = ens.model_results.get(
                    "elasticnet",
                    next(iter(ens.model_results.values()), None),
                )
                if primary_wfo is not None:
                    break
        if primary_wfo is None and ensemble_results:
            first_ens = next(iter(ensemble_results.values()))
            primary_wfo = next(iter(first_ens.model_results.values()), None)
        if primary_wfo is not None:
            importance_stability = compute_feature_importance_stability(primary_wfo)
    except Exception:
        logger.warning(
            "Could not compute feature importance stability; skipping section",
            exc_info=True,
        )
    return importance_stability


def build_manifest_warnings(
    *,
    freshness_report: dict,
    aggregate_health: dict | None,
    representative_cpcv: CPCVResult | None,
    obs_feature_report: dict | None,
    conformal_coverage_summary: ConformalCoverageBacktest | None,
    nan_live_features: list[str],
    model_drift_summary: ModelDriftSummary | None,
    shadow_summary: SnapshotSummary | None,
    live_summary: SnapshotSummary,
    consensus_shadow_df: pd.DataFrame | None,
    recommendation_mode: dict[str, str | float],
    shadow_gate_overlay: dict | None,
) -> list[str]:
    """Warnings recorded in ``run_manifest.json``, the dashboard and the summary.

    ``recommendation_mode`` is the live (quality-weighted) mode before any
    shadow promotion; the consensus cross-check is compared with it.
    """
    manifest_warnings: list[str] = []
    manifest_warnings.extend(freshness_report["warnings"])
    if aggregate_health is not None and aggregate_health["oos_r2"] < config.DIAG_MIN_OOS_R2:
        manifest_warnings.append(
            f"Aggregate OOS R^2 below threshold: {aggregate_health['oos_r2']:.2%} < {config.DIAG_MIN_OOS_R2:.2%}."
        )
    if representative_cpcv is None or representative_cpcv.stability_verdict == "UNKNOWN":
        manifest_warnings.append(
            "Representative CPCV diagnostic did not produce paths; ACTIONABLE is withheld (fail closed)."
        )
    elif representative_cpcv.stability_verdict == "FAIL":
        manifest_warnings.append(
            "Representative CPCV verdict is FAIL (diagnostic only; it does not gate the recommendation)."
        )
    if obs_feature_report is not None:
        obs_report = obs_feature_report
        if obs_report.get("verdict") != "OK":
            manifest_warnings.append(
                f"Observation-to-feature report is {obs_report.get('verdict')} "
                f"(ratio={obs_report.get('ratio', float('nan')):.2f})."
            )
    if (
        conformal_coverage_summary is not None
        and not math.isnan(conformal_coverage_summary.trailing_coverage_gap)
        and abs(conformal_coverage_summary.trailing_coverage_gap) > 0.10
    ):
        manifest_warnings.append(
            "Trailing conformal coverage deviates materially from nominal: "
            f"{conformal_coverage_summary.trailing_empirical_coverage:.1%} "
            f"vs {conformal_coverage_summary.target_coverage:.0%}."
        )
    if nan_live_features:
        manifest_warnings.append(
            f"{len(nan_live_features)} live-model feature(s) are NaN in the decision row "
            f"and were median-imputed: {', '.join(nan_live_features)}."
        )
    if model_drift_summary is not None and model_drift_summary.drift_flag:
        manifest_warnings.append(
            "Rolling model IC drift alert active: "
            f"{model_drift_summary.ic_below_threshold_streak} consecutive monthly "
            f"snapshots below {config.DIAG_MIN_IC:.2f}."
        )

    if shadow_summary is not None:
        if shadow_summary.recommendation_mode != live_summary.recommendation_mode:
            manifest_warnings.append(
                "The v13 simpler-baseline cross-check disagrees with the live recommendation mode."
            )
        elif abs(shadow_summary.sell_pct - live_summary.sell_pct) > 1e-9:
            manifest_warnings.append(
                "The v13 simpler-baseline cross-check suggests a different sell percentage."
            )
    if consensus_shadow_df is not None and not consensus_shadow_df.empty:
        shadow_rows = consensus_shadow_df[~consensus_shadow_df["is_live_path"]]
        if not shadow_rows.empty:
            shadow_row = shadow_rows.iloc[0]
            if str(shadow_row["recommendation_mode"]) != str(recommendation_mode["label"]):
                manifest_warnings.append(
                    "The consensus cross-check disagrees with the live recommendation mode."
                )
            elif abs(float(shadow_row["recommended_sell_pct"]) - float(recommendation_mode["sell_pct"])) > 1e-9:
                manifest_warnings.append(
                    "The consensus cross-check suggests a different sell percentage."
                )
    if isinstance(shadow_gate_overlay, dict) and shadow_gate_overlay.get("would_change"):
        manifest_warnings.append(
            "The classifier shadow gate would change the live recommendation output."
        )
    return manifest_warnings
