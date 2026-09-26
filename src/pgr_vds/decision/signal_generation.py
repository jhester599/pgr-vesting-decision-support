"""Feature and signal generation for the monthly decision run.

Builds the as-of feature matrix, trains the production Ridge+GBT ensemble per
benchmark (``src.models.multi_benchmark_wfo``, walk-forward with purge and
embargo), then calibrates P(outperform), adds conformal intervals and
resolves the live consensus. The v13 simpler-baseline cross-check is built
here too.
Functions in other ``pgr_vds.decision`` modules are called through the module
(``health.compute_aggregate_health(...)``), so a test patches a function
once, in the module that defines it.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

import config
from pgr_vds.decision import health
from src.models.calibration import (
    CalibrationResult,
    block_bootstrap_ece_ci,
    calibrate_prediction,
    compute_ece,
    confidence_tier_from_probability,
    fit_calibration_model,
    prequential_platt_probabilities,
)
from src.models.conformal import (
    ConformalCoverageBacktest,
    backtest_conformal_coverage,
    conformal_interval_from_ensemble,
)
from src.models.consensus_shadow import build_shadow_consensus_table
from src.models.evaluation import (
    evaluate_baseline_strategy,
    reconstruct_baseline_predictions,
)
from src.models.multi_benchmark_wfo import (
    get_ensemble_signals,
    run_ensemble_benchmarks,
)
from src.models.prequential import build_prequential_panel, live_shrinkage_alpha
from src.models.wfo_engine import CPCVResult, run_cpcv
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    compute_obs_feature_ratio,
    compute_vif,
    get_feature_columns,
    get_model_feature_columns,
    get_X_y_relative,
    truncate_relative_target_for_asof,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.reporting.snapshot_summary import (
    SnapshotSummary,
    aggregate_health_from_prediction_frames,
    confidence_from_hit_rate,
    honest_prediction_frame,
    sell_pct_from_policy,
    signal_from_prediction,
)

logger = logging.getLogger(__name__)


def generate_signals(
    conn,
    as_of: date,
    target_horizon_months: int = 6,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build feature matrix (sliced to as_of), train ensemble WFO models, return signals.

    Uses the production Ridge+GBT ensemble (v11.0) on the PRIMARY_FORECAST_UNIVERSE
    with v18 lean feature sets, combined by inverse-variance weights and scaled
    by the prequential shrinkage alpha.

    Returns:
        (signals, ensemble_results, diagnostics) where signals is a DataFrame indexed by
        benchmark with columns predicted_relative_return, raw_ensemble_prediction,
        shrinkage_alpha, ic, hit_rate, signal, prob_outperform, confidence_tier;
        ensemble_results is the dict returned by ``run_ensemble_benchmarks``
        (ETF ticker → EnsembleWFOResult); diagnostics carries the representative
        CPCV, the prequential OOS panel and the live shrinkage alpha.
    """
    as_of_ts = pd.Timestamp(as_of)

    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    feature_cols = get_feature_columns(df_full)
    X_full = df_full[feature_cols]

    # Strict temporal cutoff: only data available on or before as_of
    X_event = X_full.loc[X_full.index <= as_of_ts]
    if X_event.empty:
        return pd.DataFrame(), {}, {}

    X_current = X_event.iloc[[-1]]
    nan_live_features = find_nan_live_features(X_current)
    if nan_live_features:
        logger.warning(
            "[Live features] %s live-model feature(s) are NaN in the decision row "
            "(%s) and will be median-imputed from the training window: %s",
            len(nan_live_features),
            X_current.index[-1].date(),
            ", ".join(nan_live_features),
        )

    # v32.1 — compute VIF for the feature matrix (safe; falls back to empty Series)
    vif_series: pd.Series
    try:
        vif_series = compute_vif(X_event, feature_cols=feature_cols)
    except Exception:
        logger.warning("VIF computation failed; skipping multicollinearity diagnostics", exc_info=True)
        vif_series = pd.Series(dtype=float)

    # v11.0: compute obs/feature ratio against the primary (ridge) feature set
    # so the diagnostic reflects the model actually being run, not the full matrix.
    primary_ridge_cols = [c for c in config.MODEL_FEATURE_OVERRIDES.get("ridge", []) if c in X_event.columns]
    X_primary_for_ratio = X_event[primary_ridge_cols] if primary_ridge_cols else X_event

    diagnostics: dict[str, object] = {
        "obs_feature_report": compute_obs_feature_ratio(X_primary_for_ratio, warn=False),
        "representative_cpcv": None,
        "vif_series": vif_series,
        "nan_live_features": nan_live_features,
    }

    # Load relative return matrix for the primary forecast universe only.
    # ETF_BENCHMARK_UNIVERSE still governs data ingestion; PRIMARY_FORECAST_UNIVERSE
    # governs which benchmarks the production ensemble trains on.
    rel_matrix_cols = {}
    for etf in config.PRIMARY_FORECAST_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, etf, target_horizon_months)
        if not rel_series.empty:
            rel_series = truncate_relative_target_for_asof(
                rel_series,
                as_of=as_of_ts,
                horizon_months=target_horizon_months,
            )
            rel_matrix_cols[etf] = rel_series
    if not rel_matrix_cols:
        return pd.DataFrame(), {}, diagnostics

    rel_matrix = pd.DataFrame(rel_matrix_cols)

    # v11.0: representative CPCV uses VOO + ridge (core benchmark of the primary
    # universe). Diagnostic only (review 2026-09-25, F02): CPCV is a
    # combinatorial K-fold, so its verdict never gates the recommendation; a
    # run where it fails to produce paths fails closed (F20).
    if "VOO" in rel_matrix.columns:
        try:
            rel_series_voo = rel_matrix["VOO"].rename(f"VOO_{target_horizon_months}m")
            X_voo_all, y_voo = get_X_y_relative(X_event, rel_series_voo, drop_na_target=True)
            ridge_cols = [c for c in config.MODEL_FEATURE_OVERRIDES.get("ridge", []) if c in X_voo_all.columns]
            X_voo = X_voo_all[ridge_cols] if ridge_cols else X_voo_all
            diagnostics["representative_cpcv"] = run_cpcv(
                X_voo,
                y_voo,
                model_type="ridge",
                target_horizon_months=target_horizon_months,
                benchmark="VOO",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[CPCV] Representative CPCV run failed; continuing without CPCV diagnostic. Error=%r",
                exc,
            )

    # Train lean Ridge+GBT ensemble per primary benchmark with v18 feature sets.
    ensemble_results = run_ensemble_benchmarks(
        X_event,
        rel_matrix,
        target_horizon_months=target_horizon_months,
        model_feature_overrides=config.MODEL_FEATURE_OVERRIDES,
    )

    # Realised-only reconstruction of every benchmark's OOS record (F04, F13):
    # prequential ensemble weights and shrinkage, and the prevailing-mean naive.
    prequential_panel = build_prequential_panel(ensemble_results)
    shrinkage_alpha = live_shrinkage_alpha(prequential_panel)
    diagnostics["prequential_panel"] = prequential_panel
    diagnostics["shrinkage_alpha"] = shrinkage_alpha
    logger.info(
        "[Shrinkage] Prequential alpha %.3f from %s realised OOS rows.",
        shrinkage_alpha,
        f"{len(prequential_panel):,}",
    )

    # Generate ensemble signals; calibrated probabilities and tiers come later.
    signals = get_ensemble_signals(
        X_full=X_event,
        relative_return_matrix=rel_matrix,
        ensemble_results=ensemble_results,
        X_current=X_current,
        shrinkage_alpha=shrinkage_alpha,
    )

    # Normalize column names for downstream consumers (consensus, report writer)
    signals = signals.rename(columns={
        "point_prediction": "predicted_relative_return",
        "mean_ic":          "ic",
        "mean_hit_rate":    "hit_rate",
    })

    return signals, ensemble_results, diagnostics


def find_nan_live_features(X_current: pd.DataFrame) -> list[str]:
    """Return live-model features that are NaN in the decision row.

    The live ensemble median-imputes NaN inputs from its training window, so a
    stale upstream series (e.g. an unrefreshed FRED series) would otherwise
    silently turn a feature into a constant. Callers log and flag these.
    Features are the columns each ``config.ENSEMBLE_MODELS`` model is fed.
    """
    if X_current.empty:
        return []
    row = X_current.iloc[-1]
    missing: list[str] = []
    for model_type in config.ENSEMBLE_MODELS:
        for col in get_model_feature_columns(X_current, model_type=model_type):
            if col not in missing and pd.isna(row[col]):
                missing.append(col)
    return missing


def live_calibration_score(signals: pd.DataFrame, ticker: str) -> float:
    """Score the calibrators see live: the ensemble before shrinkage."""
    if "raw_ensemble_prediction" in signals.columns and not pd.isna(
        signals.at[ticker, "raw_ensemble_prediction"]
    ):
        return float(signals.at[ticker, "raw_ensemble_prediction"])
    return float(signals.at[ticker, "predicted_relative_return"])


def calibrate_signals(
    signals: pd.DataFrame,
    ensemble_results: dict,
    target_horizon_months: int = 6,
    panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, CalibrationResult, np.ndarray, np.ndarray]:
    """
    Calibrate per-benchmark P(outperform) and report a prequential ECE.

    Live: one Platt (logistic) model per benchmark, fitted on that benchmark's
    realised OOS history of ensemble scores ``z`` (before shrinkage) against
    1{relative return > 0}, is applied to the current score. Per-benchmark
    calibration keeps cross-benchmark discrimination; isotonic stays disabled
    until each benchmark has ~500 OOS rows.

    Reported ECE (review 2026-09-25, F13): every historical OOS month is scored
    by the calibrator the monthly run would have had then, fitted only on
    rows whose 6-month target had been realised. The ECE and its date-block
    bootstrap CI are computed on those pairs; the old ECE was measured on the
    calibrator's own training rows.

    The calibrated probability also sets ``prob_outperform`` and, read in the
    direction of each benchmark's signal, ``confidence_tier`` (F21).

    Returns:
        ``(signals, CalibrationResult, probs, outcomes)`` where ``probs`` and
        ``outcomes`` are the pooled prequential pairs (reliability diagram).
    """
    uncalibrated = CalibrationResult(
        n_obs=0, method="uncalibrated", ece=0.0,
        ece_ci_lower=0.0, ece_ci_upper=1.0,
    )
    empty = np.array([], dtype=float)
    if signals.empty or "predicted_relative_return" not in signals.columns:
        return signals, uncalibrated, empty, np.array([], dtype=int)

    panel = health.prequential_panel(ensemble_results, panel)
    signals = signals.copy()
    calibrated_probs: list[float] = []
    tiers: list[str] = []
    pooled_probs: list[np.ndarray] = []
    pooled_outcomes: list[np.ndarray] = []
    pooled_dates: list[pd.DatetimeIndex] = []
    methods_used: list[str] = []

    for ticker in signals.index:
        rows = health.benchmark_rows(panel, str(ticker))
        signal = str(signals.at[ticker, "signal"]) if "signal" in signals.columns else "NEUTRAL"
        if rows.empty:
            calibrated_probs.append(0.5)
            tiers.append(confidence_tier_from_probability(0.5, signal))
            continue

        scores = rows["z"].to_numpy(dtype=float)
        outcomes = (rows["y_true"].to_numpy(dtype=float) > 0).astype(int)
        dates = pd.DatetimeIndex(rows["date"])

        # Live calibrator: every OOS row is realised by the as-of date.
        bm_model, bm_result = fit_calibration_model(
            scores,
            outcomes,
            min_obs_platt=config.CALIBRATION_MIN_OBS_PLATT,
            min_obs_isotonic=10_000,   # effectively disables isotonic
            n_bins=config.CALIBRATION_N_BINS,
            block_len=target_horizon_months,
            n_bootstrap=0,
        )
        cal_prob = calibrate_prediction(bm_model, live_calibration_score(signals, str(ticker)))
        calibrated_probs.append(cal_prob)
        tiers.append(confidence_tier_from_probability(cal_prob, signal))
        methods_used.append(bm_result.method)

        # Reported calibration: realised-only calibrators, one per OOS month.
        probs = prequential_platt_probabilities(
            scores,
            outcomes,
            dates,
            horizon_months=target_horizon_months,
            min_obs=config.CALIBRATION_MIN_OBS_PLATT,
        )
        scored = np.isfinite(probs)
        pooled_probs.append(probs[scored])
        pooled_outcomes.append(outcomes[scored])
        pooled_dates.append(dates[scored])

    signals["calibrated_prob_outperform"] = calibrated_probs
    signals["prob_outperform"] = calibrated_probs
    signals["confidence_tier"] = tiers

    probs_arr = np.concatenate(pooled_probs) if pooled_probs else empty
    outcomes_arr = (
        np.concatenate(pooled_outcomes).astype(int) if pooled_outcomes else np.array([], dtype=int)
    )
    if len(probs_arr) >= 4:
        dates_arr = pd.DatetimeIndex(np.concatenate([d.to_numpy() for d in pooled_dates]))
        agg_ece = compute_ece(probs_arr, outcomes_arr, n_bins=config.CALIBRATION_N_BINS)
        ci_lo, ci_hi = block_bootstrap_ece_ci(
            probs_arr, outcomes_arr,
            n_bins=config.CALIBRATION_N_BINS,
            block_len=target_horizon_months,
            n_bootstrap=config.CALIBRATION_BOOTSTRAP_REPS,
            dates=dates_arr,
        )
        dominant_method = "platt" if "platt" in methods_used else "uncalibrated"
        result = CalibrationResult(
            n_obs=int(len(probs_arr)),
            method=dominant_method,
            ece=agg_ece,
            ece_ci_lower=ci_lo,
            ece_ci_upper=ci_hi,
        )
    else:
        result = uncalibrated
        probs_arr = empty
        outcomes_arr = np.array([], dtype=int)

    return signals, result, probs_arr, outcomes_arr


def compute_conformal_intervals(
    signals: pd.DataFrame,
    ensemble_results: dict,
    panel: pd.DataFrame | None = None,
    target_horizon_months: int = 6,
) -> pd.DataFrame:
    """
    Compute per-benchmark conformal prediction intervals for the current predictions.

    The live interval is calibrated on every realised OOS residual of the
    (prequential) ensemble prediction. Uses ACI by default, falling back to
    split conformal when fewer than 4 residuals are available.

    Reported coverage is trailing and prequential (review 2026-09-25, F13):
    each of the last 12 OOS points is scored with an interval calibrated only
    on residuals whose 6-month target had been realised by that point. The
    in-sample share of calibration residuals inside the interval is no longer
    reported.

    Coverage, method, and gamma are read from config constants:
      CONFORMAL_COVERAGE  (default 0.80 = 80% CI)
      CONFORMAL_METHOD    ("aci" or "split")
      CONFORMAL_ACI_GAMMA (default 0.05)

    Adds the columns ci_lower, ci_upper, ci_width, ci_n_calibration,
    ci_trailing_empirical_coverage, ci_trailing_coverage_gap, ci_trailing_n.
    """
    if signals.empty:
        return signals

    pred_col = "predicted_relative_return"
    if pred_col not in signals.columns:
        return signals

    panel = health.prequential_panel(ensemble_results, panel)
    signals = signals.copy()
    columns: dict[str, list[float | int]] = {
        "ci_lower": [],
        "ci_upper": [],
        "ci_width": [],
        "ci_n_calibration": [],
        "ci_trailing_empirical_coverage": [],
        "ci_trailing_coverage_gap": [],
        "ci_trailing_n": [],
    }

    for ticker in signals.index:
        rows = health.benchmark_rows(panel, str(ticker))
        if rows.empty:
            for key in columns:
                columns[key].append(0 if key in {"ci_n_calibration", "ci_trailing_n"} else float("nan"))
            continue

        y_hat_oos = rows["y_hat"].to_numpy(dtype=float)
        y_true_oos = rows["y_true"].to_numpy(dtype=float)
        conf_result = conformal_interval_from_ensemble(
            y_hat_current=float(signals.at[ticker, pred_col]),
            y_hat_oos=y_hat_oos,
            y_true_oos=y_true_oos,
            coverage=config.CONFORMAL_COVERAGE,
            method=config.CONFORMAL_METHOD,
            gamma=config.CONFORMAL_ACI_GAMMA,
        )
        coverage_backtest = backtest_conformal_coverage(
            y_hat_oos=y_hat_oos,
            y_true_oos=y_true_oos,
            coverage=config.CONFORMAL_COVERAGE,
            method=config.CONFORMAL_METHOD,
            gamma=config.CONFORMAL_ACI_GAMMA,
            trailing_window=12,
            dates=pd.DatetimeIndex(rows["date"]),
            horizon_months=target_horizon_months,
        )
        columns["ci_lower"].append(conf_result.lower)
        columns["ci_upper"].append(conf_result.upper)
        columns["ci_width"].append(conf_result.width)
        columns["ci_n_calibration"].append(conf_result.n_calibration)
        columns["ci_trailing_empirical_coverage"].append(coverage_backtest.trailing_empirical_coverage)
        columns["ci_trailing_coverage_gap"].append(coverage_backtest.trailing_coverage_gap)
        columns["ci_trailing_n"].append(coverage_backtest.trailing_n)

    for key, values in columns.items():
        signals[key] = values
    return signals


def summarize_conformal_coverage(signals: pd.DataFrame | None) -> ConformalCoverageBacktest | None:
    """Aggregate recent conformal coverage diagnostics across benchmarks."""
    if (
        signals is None
        or signals.empty
        or "ci_trailing_empirical_coverage" not in signals.columns
        or "ci_trailing_n" not in signals.columns
    ):
        return None

    valid_rows = signals[signals["ci_trailing_n"].fillna(0) > 0].dropna(
        subset=["ci_trailing_empirical_coverage"]
    )
    if valid_rows.empty:
        return None

    target = config.CONFORMAL_COVERAGE
    trailing_empirical = float(valid_rows["ci_trailing_empirical_coverage"].mean())
    trailing_n = int(valid_rows["ci_trailing_n"].sum())
    return ConformalCoverageBacktest(
        n_evaluated=trailing_n,
        empirical_coverage=trailing_empirical,
        target_coverage=target,
        coverage_gap=trailing_empirical - target,
        trailing_n=trailing_n,
        trailing_empirical_coverage=trailing_empirical,
        trailing_coverage_gap=trailing_empirical - target,
        method=config.CONFORMAL_METHOD,
    )


def consensus_signal(
    signals: pd.DataFrame,
) -> tuple[str, float, float, float, float, str]:
    """
    Derive a consensus signal from per-benchmark signals.

    Returns:
        (consensus_signal, mean_predicted_return, mean_ic, mean_hit_rate,
         mean_prob_outperform, composite_confidence_tier)
    """
    if signals.empty:
        return "NEUTRAL", 0.0, 0.0, 0.0, 0.5, "LOW"

    mean_pred = float(signals["predicted_relative_return"].mean())
    mean_ic = float(signals["ic"].mean())
    mean_hr = float(signals["hit_rate"].mean())

    # Ensemble confidence columns (present when ensemble path is used)
    if "prob_outperform" in signals.columns:
        mean_prob = float(signals["prob_outperform"].mean())
    else:
        mean_prob = 0.5

    outperform_count = (signals["signal"] == "OUTPERFORM").sum()
    underperform_count = (signals["signal"] == "UNDERPERFORM").sum()

    total = len(signals)
    if outperform_count > total / 2:
        consensus = "OUTPERFORM"
    elif underperform_count > total / 2:
        consensus = "UNDERPERFORM"
    else:
        consensus = "NEUTRAL"

    # Calibrated P(outperform) read in the direction of the consensus (F21).
    confidence_tier = confidence_tier_from_probability(mean_prob, consensus)

    return consensus, mean_pred, mean_ic, mean_hr, mean_prob, confidence_tier


def build_v74_shadow_consensus(
    signals: pd.DataFrame,
    aggregate_health: dict | None,
    representative_cpcv: CPCVResult | None = None,
) -> pd.DataFrame | None:
    """Build the live-vs-shadow consensus comparison table.

    Each variant keeps its own weighted direction and mean prediction, but the
    recommendation mode of every variant is gated on the equal-weight mean IC
    (review 2026-09-25, F13): the quality weights are fitted on the same OOS
    record, so a quality-weighted IC is inflated.
    """
    if signals.empty or aggregate_health is None:
        return None
    gate_ic = equal_weight_mean(signals, "ic")

    benchmark_quality_df = aggregate_health.get("benchmark_quality_df")
    if benchmark_quality_df is None:
        return None

    shadow_df = build_shadow_consensus_table(
        signals=signals,
        benchmark_quality_df=benchmark_quality_df,
        score_col=config.V74_SHADOW_CONSENSUS_SCORE_COL,
        lambda_mix=config.V74_SHADOW_CONSENSUS_LAMBDA_MIX,
    )
    if shadow_df.empty:
        return None

    enriched_rows: list[dict[str, object]] = []
    live_variant = (
        "quality_weighted"
        if config.CONSENSUS_WEIGHTING_MODE == "quality_weighted"
        else "equal_weight"
    )
    for row in shadow_df.to_dict("records"):
        recommendation_mode = health.determine_recommendation_mode(
            str(row["consensus"]),
            float(row["mean_predicted_return"]),
            gate_ic,
            float(row["mean_hit_rate"]),
            aggregate_health,
            representative_cpcv,
        )
        enriched = dict(row)
        enriched["recommendation_mode"] = str(recommendation_mode["label"])
        enriched["recommended_sell_pct"] = float(recommendation_mode["sell_pct"])
        enriched["is_live_path"] = bool(row["variant"] == live_variant)
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def live_variant_mean_ic(consensus_shadow_df: pd.DataFrame | None) -> float | None:
    """Mean IC of the live (quality-weighted) consensus variant, for reporting."""
    if consensus_shadow_df is None or consensus_shadow_df.empty:
        return None
    live_rows = consensus_shadow_df[consensus_shadow_df["is_live_path"]]
    if live_rows.empty:
        return None
    return float(live_rows.iloc[0]["mean_ic"])


def equal_weight_mean(signals: pd.DataFrame, column: str) -> float:
    """Equal-weight mean of one per-benchmark column (NaN when absent)."""
    if signals.empty or column not in signals.columns:
        return float("nan")
    return float(signals[column].astype(float).mean())


def resolve_live_consensus(
    signals: pd.DataFrame,
    aggregate_health: dict | None,
    representative_cpcv: CPCVResult | None = None,
) -> tuple[tuple[str, float, float, float, float, str], pd.DataFrame | None]:
    """Resolve the promoted live consensus tuple and the comparison table.

    The tuple is (consensus, mean_predicted, mean_ic, mean_hit_rate,
    mean_prob_outperform, confidence_tier). Direction, prediction, probability
    and tier come from the live (quality-weighted) variant; ``mean_ic`` and
    ``mean_hit_rate`` are equal-weight, because the IC is gated and quality
    weights inflate it (review 2026-09-25, F13).
    """
    default = consensus_signal(signals)
    consensus_shadow_df = build_v74_shadow_consensus(
        signals,
        aggregate_health,
        representative_cpcv,
    )
    if consensus_shadow_df is None or consensus_shadow_df.empty:
        return default, None

    live_row = consensus_shadow_df[consensus_shadow_df["is_live_path"]]
    if live_row.empty:
        return default, consensus_shadow_df

    row = live_row.iloc[0]
    return (
        (
            str(row["consensus"]),
            float(row["mean_predicted_return"]),
            equal_weight_mean(signals, "ic"),
            equal_weight_mean(signals, "hit_rate"),
            float(row["mean_prob_outperform"]),
            str(row["confidence_tier"]),
        ),
        consensus_shadow_df,
    )


def current_baseline_prediction(y_aligned: pd.Series) -> float:
    """Current-point forecast for the historical-mean baseline."""
    window = min(len(y_aligned), config.WFO_TRAIN_WINDOW_MONTHS)
    return float(y_aligned.iloc[-window:].mean())


def build_shadow_baseline_summary(
    conn,
    as_of: date,
    target_horizon_months: int = 6,
) -> tuple[SnapshotSummary | None, pd.DataFrame]:
    """Build the v13 simpler-baseline recommendation-layer cross-check."""
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    X_event = df_full.loc[df_full.index <= pd.Timestamp(as_of)]
    if X_event.empty:
        return None, pd.DataFrame()

    signal_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for benchmark in config.V13_SHADOW_FORECAST_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, benchmark, target_horizon_months)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(X_event, rel_series, drop_na_target=True)
        except ValueError:
            continue
        if X_aligned.empty or y_aligned.empty:
            continue

        metrics = evaluate_baseline_strategy(
            X_aligned,
            y_aligned,
            strategy=config.V13_SHADOW_BASELINE_STRATEGY,
            target_horizon_months=target_horizon_months,
        )
        pred_series, realized = reconstruct_baseline_predictions(
            X_aligned,
            y_aligned,
            strategy=config.V13_SHADOW_BASELINE_STRATEGY,
            target_horizon_months=target_horizon_months,
        )
        current_pred = current_baseline_prediction(y_aligned)
        signal_rows.append(
            {
                "benchmark": benchmark,
                "predicted_relative_return": current_pred,
                "ic": float(metrics["ic"]),
                "hit_rate": float(metrics["hit_rate"]),
                "signal": signal_from_prediction(current_pred),
                "confidence_tier": confidence_from_hit_rate(float(metrics["hit_rate"])),
            }
        )
        prediction_frames.append(
            honest_prediction_frame(
                benchmark,
                pred_series,
                realized,
                target_history=y_aligned,
                target_horizon_months=target_horizon_months,
            )
        )

    if not signal_rows:
        return None, pd.DataFrame()

    shadow_signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(prediction_frames, target_horizon_months)
    consensus, mean_pred, mean_ic, mean_hr, _, confidence_tier = consensus_signal(shadow_signals)
    recommendation_mode = health.determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        representative_cpcv=None,
    )
    if recommendation_mode["mode"] == "actionable":
        sell_pct = sell_pct_from_policy(mean_pred, config.V13_SHADOW_BASELINE_POLICY)
    else:
        sell_pct = float(recommendation_mode["sell_pct"])

    return (
        SnapshotSummary(
            label="shadow",
            as_of=as_of,
            candidate_name=f"baseline_{config.V13_SHADOW_BASELINE_STRATEGY}",
            policy_name=config.V13_SHADOW_BASELINE_POLICY,
            consensus=consensus,
            confidence_tier=confidence_tier,
            recommendation_mode=str(recommendation_mode["label"]),
            sell_pct=sell_pct,
            mean_predicted=mean_pred,
            mean_ic=mean_ic,
            mean_hit_rate=mean_hr,
            aggregate_oos_r2=float(aggregate_health["oos_r2"]) if aggregate_health is not None else float("nan"),
            aggregate_nw_ic=float(aggregate_health["nw_ic"]) if aggregate_health is not None else float("nan"),
        ),
        shadow_signals,
    )
