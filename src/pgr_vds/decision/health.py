"""Model health and gating for the monthly decision run.

Realised-only (prequential) OOS diagnostics, the readiness contract
(``wfo_completed`` and ``data_ready``, pre-v200 remediation R3), the
recommendation-mode gate, the model-health snapshot and drift/retrain audit,
the decision-policy backtest, and the warnings recorded in
``run_manifest.json``.
Functions in other ``pgr_vds.decision`` modules are called through the module
(``health.compute_aggregate_health(...)``), so a test patches a function
once, in the module that defines it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date
from typing import Any

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
from src.models.wfo_engine import WFOResult
from src.processing.feature_engineering import get_feature_columns
from src.processing.total_return import forward_window_end
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
) -> dict[str, Any]:
    """Compatibility wrapper around the extracted decision-rendering helper."""
    return render_determine_recommendation_mode(
        consensus=consensus,
        mean_predicted=mean_predicted,
        mean_ic=mean_ic,
        mean_hr=mean_hr,
        aggregate_health=aggregate_health,
    )


# ---------------------------------------------------------------------------
# Readiness contract (pre-v200 remediation R3)
# ---------------------------------------------------------------------------

READINESS_KEYS: tuple[str, ...] = (
    "wfo_completed",
    "wfo_required_pairs",
    "wfo_failed_pairs",
    "wfo_optional_excluded",
    "data_ready",
    "missing_live_features",
    "stale_required_feeds",
    "decision_row_date",
    "readiness_basis",
    "readiness_note",
    "as_of",
    "run_date",
)

BACKDATED_READINESS_NOTE: str = (
    "Back-dated run: readiness is reconstructed from the values stored now and dated "
    "(or filed) on or before the as-of date. The DB does not record when each value was "
    "fetched, so a value repaired later can pass here although the original decision did "
    "not have it; this is not evidence of what that decision saw."
)


def readiness_basis(as_of: date, run_date: date) -> str:
    """``live`` when the run is in the as-of month, else a reconstruction."""
    if (as_of.year, as_of.month) == (run_date.year, run_date.month):
        return "live"
    return "backdated_reconstruction"


def required_live_features(frame: pd.DataFrame | None = None) -> list[str]:
    """Features the live ensemble models are fed, in model order, deduplicated.

    Each ``config.ENSEMBLE_MODELS`` model's ``config.MODEL_FEATURE_OVERRIDES``
    list is required in full: a configured feature absent from the frame
    counts as missing rather than silently dropping out of the model. A model
    without an override uses every feature column of ``frame``.
    """
    features: list[str] = []
    for model_type in config.ENSEMBLE_MODELS:
        columns = list(config.MODEL_FEATURE_OVERRIDES.get(model_type, []))
        if not columns and frame is not None:
            columns = get_feature_columns(frame)
        for column in columns:
            if column not in features:
                features.append(column)
    return features


def find_nonfinite_live_features(live_row: pd.DataFrame) -> list[str]:
    """Live-model features that are NaN, infinite or absent in the decision row.

    Checked before the model median-imputes its inputs, so a stale upstream
    series cannot silently become a training-median constant (F07).
    """
    if live_row is None or live_row.empty:
        return required_live_features()
    row = live_row.iloc[-1]
    missing: list[str] = []
    for column in required_live_features(live_row):
        value = row.get(column, np.nan)
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite and column not in missing:
            missing.append(column)
    return missing


def _pair_problem(
    wfo: WFOResult,
    as_of_ts: pd.Timestamp,
    horizon: int,
    total_gap: int,
) -> str | None:
    """The first audit failure of one model/benchmark WFO result, or None."""
    if not wfo.folds:
        return "no folds"
    previous_test_end: pd.Timestamp | None = None
    for fold in wfo.folds:
        dates = pd.DatetimeIndex(fold._test_dates)
        y_hat = np.asarray(fold.y_hat, dtype=float)
        y_true = np.asarray(fold.y_true, dtype=float)
        if fold.n_test <= 0 or len(y_hat) == 0 or len(dates) != len(y_hat):
            return f"empty fold {fold.fold_idx}"
        if len(y_true) != len(y_hat) or not (np.isfinite(y_hat).all() and np.isfinite(y_true).all()):
            return f"non-finite OOS prediction or outcome in fold {fold.fold_idx}"
        if not dates.is_monotonic_increasing:
            return f"test dates out of order in fold {fold.fold_idx}"
        test_start = pd.Timestamp(fold.test_start)
        train_end = pd.Timestamp(fold.train_end)
        if previous_test_end is not None and test_start <= previous_test_end:
            return f"test fold {fold.fold_idx} is not after the previous fold"
        previous_test_end = pd.Timestamp(fold.test_end)
        if train_end >= test_start:
            return f"training ends on or after the test start in fold {fold.fold_idx}"
        gap_months = (test_start.year - train_end.year) * 12 + test_start.month - train_end.month - 1
        if gap_months < total_gap:
            return (
                f"gap of {gap_months} months before fold {fold.fold_idx} is below the "
                f"protocol's {total_gap}"
            )
        if forward_window_end(train_end, horizon) > test_start:
            return f"training label not realised before the test start in fold {fold.fold_idx}"
        if dates.max() > as_of_ts:
            return f"test date after the as-of date in fold {fold.fold_idx}"
        if forward_window_end(dates.max(), horizon) > as_of_ts:
            return f"outcome not realised by the as-of date in fold {fold.fold_idx}"
    return None


def assess_wfo_completion(
    ensemble_results: dict,
    signals: pd.DataFrame | None,
    as_of: date,
    target_horizon_months: int = 6,
    required_benchmarks: Sequence[str] | None = None,
    required_models: Sequence[str] | None = None,
    optional_benchmarks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Whether the walk-forward validation the decision needs actually completed.

    Required pairs are ``required_benchmarks`` (default
    ``config.PRIMARY_FORECAST_UNIVERSE`` minus ``config.WFO_OPTIONAL_BENCHMARKS``)
    times ``required_models`` (default ``config.ENSEMBLE_MODELS``). A pair is
    complete when its WFO result exists and every fold:

    - is non-empty, with finite predictions and outcomes;
    - tests after the previous fold (chronological, non-overlapping);
    - trains on rows that end at least ``horizon + purge buffer`` months
      before its test start (the production gap) and whose last label window
      ends by the test start;
    - tests only on rows whose outcome was realised by ``as_of``.

    Each required benchmark also needs a finite live forecast in ``signals``
    (reported as the ``<benchmark>/ensemble`` pair). One successful model is
    not completion; an optional benchmark without a complete result is listed
    in ``wfo_optional_excluded``.
    """
    if required_models is None:
        required_models = list(config.ENSEMBLE_MODELS)
    if optional_benchmarks is None:
        optional_benchmarks = list(config.WFO_OPTIONAL_BENCHMARKS)
    if required_benchmarks is None:
        required_benchmarks = [
            b for b in config.PRIMARY_FORECAST_UNIVERSE if b not in set(optional_benchmarks)
        ]
    purge = config.WFO_PURGE_BUFFER_6M if target_horizon_months <= 6 else config.WFO_PURGE_BUFFER_12M
    total_gap = target_horizon_months + purge
    as_of_ts = pd.Timestamp(as_of)

    def _problems(benchmark: str) -> list[dict[str, str]]:
        problems: list[dict[str, str]] = []
        ensemble = (ensemble_results or {}).get(benchmark)
        model_results = getattr(ensemble, "model_results", None)
        for model in required_models:
            reason: str | None
            if ensemble is None:
                reason = "missing benchmark result"
            elif not isinstance(model_results, dict) or model not in model_results:
                reason = "missing model result"
            else:
                reason = _pair_problem(model_results[model], as_of_ts, target_horizon_months, total_gap)
            if reason is not None:
                problems.append({"benchmark": benchmark, "model": model, "reason": reason})
        live = None
        if signals is not None and not signals.empty and benchmark in signals.index:
            if "predicted_relative_return" in signals.columns:
                live = signals.at[benchmark, "predicted_relative_return"]
        try:
            live_ok = live is not None and math.isfinite(float(live))
        except (TypeError, ValueError):
            live_ok = False
        if not live_ok:
            problems.append(
                {"benchmark": benchmark, "model": "ensemble", "reason": "no finite live forecast"}
            )
        return problems

    failed: list[dict[str, str]] = []
    for benchmark in required_benchmarks:
        failed.extend(_problems(benchmark))
    optional_excluded = [b for b in optional_benchmarks if _problems(b)]
    return {
        "wfo_completed": bool(required_benchmarks) and not failed,
        "wfo_required_pairs": [f"{b}/{m}" for b in required_benchmarks for m in required_models],
        "wfo_failed_pairs": failed,
        "wfo_optional_excluded": optional_excluded,
    }


def assess_data_readiness(
    live_row: pd.DataFrame | None,
    feed_readiness: dict[str, Any] | None,
    as_of: date,
    run_date: date,
) -> dict[str, Any]:
    """Whether the decision's required inputs are finite and fresh at ``as_of``.

    - ``missing_live_features``: required live features that are NaN,
      infinite or absent in the decision row, before imputation.
    - ``stale_required_feeds``: the feeds ``feed_readiness`` reports as not
      OK (``db_client.check_required_feed_readiness``), plus the decision row
      itself when it is not the as-of date's decision month. A missing feed
      report is unknown and so not ready.

    ``data_ready`` is True only when both lists are empty.
    """
    missing = find_nonfinite_live_features(live_row) if live_row is not None else required_live_features()
    stale: list[str] = []
    row_date: str | None = None
    year, month = db_client.decision_month(as_of)
    needed = f"{year:04d}-{month:02d}"
    if live_row is None or live_row.empty:
        stale.append(f"Feature row (none on or before {as_of.isoformat()}; needs {needed})")
    else:
        row_ts = pd.Timestamp(live_row.index[-1])
        row_date = row_ts.date().isoformat()
        if (row_ts.year, row_ts.month) != (year, month):
            stale.append(f"Feature row {row_date} (needs {needed})")
    if feed_readiness is None or not isinstance(feed_readiness.get("stale_required_feeds"), list):
        stale.append("Feed freshness unknown")
    else:
        stale.extend(str(feed) for feed in feed_readiness["stale_required_feeds"])
    basis = readiness_basis(as_of, run_date)
    return {
        "data_ready": not missing and not stale,
        "missing_live_features": missing,
        "stale_required_feeds": stale,
        "decision_row_date": row_date,
        "readiness_basis": basis,
        "readiness_note": BACKDATED_READINESS_NOTE if basis != "live" else "",
        "as_of": as_of.isoformat(),
        "run_date": run_date.isoformat(),
    }


def build_readiness(
    conn,
    *,
    as_of: date,
    run_date: date,
    ensemble_results: dict,
    signals: pd.DataFrame | None,
    diagnostics: dict[str, Any] | None,
    target_horizon_months: int = 6,
) -> dict[str, Any]:
    """The full readiness contract for one decision, evaluated at ``as_of``.

    A failure to evaluate a part leaves it unknown, which fails closed.
    """
    diagnostics = diagnostics or {}
    try:
        feed_readiness: dict[str, Any] | None = db_client.check_required_feed_readiness(conn, as_of)
    except Exception:  # noqa: BLE001
        logger.exception("[Readiness] Could not check the required feeds; data_ready fails closed.")
        feed_readiness = None
    readiness = assess_data_readiness(
        diagnostics.get("live_feature_row"),
        feed_readiness,
        as_of=as_of,
        run_date=run_date,
    )
    readiness.update(
        assess_wfo_completion(
            ensemble_results,
            signals,
            as_of=as_of,
            target_horizon_months=target_horizon_months,
        )
    )
    readiness["feed_report"] = feed_readiness
    return readiness


def attach_readiness(aggregate_health: dict | None, readiness: dict[str, Any] | None) -> dict | None:
    """Copy of ``aggregate_health`` carrying the readiness fields the gates read.

    ``None`` stays ``None`` (too few OOS rows: every gate fails anyway); a
    missing readiness field stays missing, which fails its gate.
    """
    if aggregate_health is None:
        return None
    merged = dict(aggregate_health)
    for key in READINESS_KEYS:
        if readiness is not None and key in readiness:
            merged[key] = readiness[key]
    return merged


def mode_payload_from_summary(summary: SnapshotSummary) -> dict[str, Any]:
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
    obs_feature_report: dict | None,
    conformal_coverage_summary: ConformalCoverageBacktest | None,
    nan_live_features: list[str],
    model_drift_summary: ModelDriftSummary | None,
    shadow_summary: SnapshotSummary | None,
    live_summary: SnapshotSummary,
    consensus_shadow_df: pd.DataFrame | None,
    recommendation_mode: dict[str, Any],
    shadow_gate_overlay: dict | None,
    readiness: dict[str, Any] | None = None,
) -> list[str]:
    """Warnings recorded in ``run_manifest.json``, the dashboard and the summary.

    ``recommendation_mode`` is the live (quality-weighted) mode before any
    shadow promotion; the consensus cross-check is compared with it. A
    non-ACTIONABLE mode adds one warning naming its reasons.
    """
    manifest_warnings: list[str] = []
    manifest_warnings.extend(freshness_report["warnings"])
    if aggregate_health is not None and aggregate_health["oos_r2"] < config.DIAG_MIN_OOS_R2:
        manifest_warnings.append(
            f"Aggregate OOS R^2 below threshold: {aggregate_health['oos_r2']:.2%} < {config.DIAG_MIN_OOS_R2:.2%}."
        )
    if readiness is None or readiness.get("wfo_completed") is not True:
        manifest_warnings.append(
            "Walk-forward validation is incomplete; ACTIONABLE is withheld (fail closed)."
        )
    if readiness is None or readiness.get("data_ready") is not True:
        manifest_warnings.append(
            "Required inputs are not ready at the as-of date; ACTIONABLE is withheld (fail closed)."
        )
    if readiness is not None and readiness.get("readiness_basis") == "backdated_reconstruction":
        manifest_warnings.append(BACKDATED_READINESS_NOTE)
    reasons = recommendation_mode.get("deferral_reasons") or []
    if recommendation_mode.get("mode") != "actionable" and reasons:
        manifest_warnings.append(
            f"Recommendation is {recommendation_mode.get('label')}: {'; '.join(str(r) for r in reasons)}."
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
            f"{len(nan_live_features)} live-model feature(s) are NaN or infinite in the decision "
            f"row (median-imputed for the forecast; data_ready fails): {', '.join(nan_live_features)}."
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
