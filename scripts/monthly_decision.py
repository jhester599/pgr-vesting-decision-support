"""
Monthly decision report generator for PGR RSU vesting decisions.

Runs on or after the 20th of each month (first business day ≥ 20th) via
GitHub Actions.  Generates a structured sell/hold recommendation based on
the latest available market data and the v3.0 multi-benchmark WFO model.

The monthly decision is a monitoring and signal-tracking tool — it does NOT
replace the biannual vesting-event recommendation.  Sell/hold decisions are
only executed at actual vesting dates (January and July).  Monthly runs
answer the question: "is the model reliably predicting PGR relative returns,
or did we get lucky on 20 coin flips?"

Output per run (in results/monthly_decisions/YYYY-MM/):
  recommendation.md      — Human-readable sell/hold report with signal details
  signals.csv            — Per-benchmark: ticker, IC, hit_rate, predicted_return, signal
  backtest_summary.csv   — Monthly stability stats at this point in time
  plots/                 — IC time series and regime breakdown charts

The decision_log.md in results/monthly_decisions/ is appended with one
summary row per run.

Usage:
    python scripts/monthly_decision.py [--as-of YYYY-MM-DD] [--dry-run] [--skip-fred]

Options:
    --as-of YYYY-MM-DD  Override the as-of date (default: today).
                        Use for back-dated runs and testing.
    --dry-run           Generate outputs but do not update the DB or
                        commit to git.  Useful for local testing.
    --skip-fred         Skip the FRED data fetch step.  Useful when
                        FRED_API_KEY is not set or during testing.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import config
from src.database import db_client
from src.models.multi_benchmark_wfo import (
    get_ensemble_signals,
    run_ensemble_benchmarks,
)
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    compute_obs_feature_ratio,
    compute_vif,
    get_feature_columns,
    get_X_y_relative,
)
from src.processing.multi_total_return import load_relative_return_matrix
import numpy as np
from src.models.evaluation import (
    FeatureImportanceStability,
    compute_feature_importance_stability,
    evaluate_baseline_strategy,
    reconstruct_ensemble_oos_predictions,
    reconstruct_baseline_predictions,
)
from src.models.forecast_diagnostics import summarize_prediction_diagnostics
from src.models.consensus_shadow import build_shadow_consensus_table
from src.models.policy_metrics import (
    FIXED_POLICIES,
    SIGNAL_POLICIES,
    PolicySummary,
    evaluate_policy_series,
)
from src.portfolio.redeploy_buckets import (
    add_destination_roles,
    recommend_redeploy_buckets,
    summarize_existing_holdings_actions,
)
from src.reporting.snapshot_summary import (
    SnapshotSummary,
    aggregate_health_from_prediction_frames,
    build_existing_holdings_markdown_lines,
    build_redeploy_markdown_lines,
    build_shadow_check_lines,
    confidence_from_hit_rate,
    sell_pct_from_policy,
    signal_from_prediction,
)
from src.portfolio.diversification import score_benchmarks_against_pgr
from src.reporting.cross_check import build_promoted_cross_check_summary
from src.portfolio.redeploy_portfolio import (
    recommend_redeploy_portfolio,
    render_redeploy_portfolio_markdown_lines,
    v27_investable_redeploy_universe,
)
from src.reporting.confidence import benchmark_role_for_ticker, build_confidence_snapshot

from src.models.calibration import (
    CalibrationResult,
    calibrate_prediction,
    fit_calibration_model,
)
from src.models.conformal import (
    ConformalCoverageBacktest,
    ConformalResult,
    backtest_conformal_coverage,
    conformal_interval_from_ensemble,
)
from src.models.drift_monitor import ModelDriftSummary, summarize_latest_model_drift
from src.models.retrain_trigger import RetainTriggerResult, evaluate_retrain_trigger
from src.portfolio.black_litterman import BLDiagnostics, build_bl_weights
from src.models.wfo_engine import CPCVResult, WFOResult, run_cpcv
from src.reporting.backtest_report import (
    compute_newey_west_ic,
    compute_oos_r_squared,
    export_backtest_to_csv,
    generate_rolling_ic_series,
)
from src.reporting.decision_rendering import (
    build_data_freshness_lines as render_data_freshness_lines,
    build_executive_summary_lines as render_executive_summary_lines,
    build_vest_decision_lines as render_vest_decision_lines,
    determine_recommendation_mode as render_determine_recommendation_mode,
)
from src.logging_config import configure_logging
from src.reporting.run_manifest import build_run_manifest, write_run_manifest
from src.tax.capital_gains import compute_three_scenarios, load_position_lots
from src.tax.monte_carlo import (
    MonteCarloTaxAnalysis,
    estimate_annual_vol,
    run_monte_carlo_tax_analysis,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ETF short descriptions (shown in per-benchmark tables)
# Stripped of provider names ("Vanguard", "SPDR", "iShares") and generic
# suffixes ("Fund", "ETF") per reporting guidelines.
# ---------------------------------------------------------------------------

_ETF_DESCRIPTIONS: dict[str, str] = {
    "VTI":  "Total Stock Market",
    "VOO":  "S&P 500",
    "VGT":  "Information Technology",
    "VHT":  "Health Care",
    "VFH":  "Financials",
    "VIS":  "Industrials",
    "VDE":  "Energy",
    "VPU":  "Utilities",
    "KIE":  "S&P Insurance",
    "VXUS": "Total International Stock",
    "VEA":  "Developed Markets ex-US",
    "VWO":  "Emerging Markets",
    "VIG":  "Dividend Appreciation",
    "SCHD": "US Dividend Equity",
    "BND":  "Total Bond Market",
    "BNDX": "Total International Bond",
    "VCIT": "Intermediate-Term Corporate Bond",
    "VMBS": "Mortgage-Backed Securities",
    "VNQ":  "Real Estate",
    "GLD":  "Gold Shares",
    "DBC":  "DB Commodity Index",
}

_MODEL_VERSION_LABEL = (
    "v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, "
    "8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, "
    "v38 post-ensemble shrinkage alpha=0.50, "
    "C(8,2)=28 CPCV paths; ElasticNet+BayesianRidge retired after v18/v20 "
    "research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)"
)


# ---------------------------------------------------------------------------
# Business-day helpers
# ---------------------------------------------------------------------------

def _is_business_day(d: date) -> bool:
    """Return True if ``d`` is a weekday (Mon–Fri)."""
    return d.weekday() < 5


def _first_business_day_on_or_after(d: date) -> date:
    """Advance ``d`` to the next business day if it falls on a weekend."""
    while not _is_business_day(d):
        d += timedelta(days=1)
    return d


def _resolve_as_of_date(as_of_arg: str | None) -> date:
    """
    Determine the as-of date for this run.

    If ``as_of_arg`` is provided, parse it.  Otherwise use today.
    If the result falls on a weekend, advance to Monday.
    """
    if as_of_arg:
        return date.fromisoformat(as_of_arg)
    return _first_business_day_on_or_after(date.today())


def _output_dir(as_of: date) -> Path:
    """Return the output directory for the given month."""
    month_str = as_of.strftime("%Y-%m")
    return Path("results") / "monthly_decisions" / month_str


def _already_ran(as_of: date) -> bool:
    """Return True if a report already exists for this exact as-of date.

    Checks the run_manifest.json as_of_date field so that re-running with a
    different date in the same month (e.g. after a code update) overwrites the
    stale report rather than skipping.
    """
    out_dir = _output_dir(as_of)
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        import json as _json
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("as_of_date") == as_of.isoformat()
    except Exception:
        # Unreadable manifest — fall back to presence check
        return (out_dir / "recommendation.md").exists()


# ---------------------------------------------------------------------------
# FRED fetch step
# ---------------------------------------------------------------------------

def _fetch_fred_step(conn, dry_run: bool = False, skip_fred: bool = False) -> None:
    """Fetch the latest FRED macro series and upsert into the DB."""
    if skip_fred or dry_run:
        if skip_fred:
            logger.info("[FRED] Skipping FRED fetch (--skip-fred).")
        else:
            logger.info("[FRED] Dry run - skipping FRED HTTP calls.")
        return

    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    logger.info("[FRED] Fetching %s macro series...", len(config.FRED_SERIES_MACRO))
    try:
        df = fetch_all_fred_macro(config.FRED_SERIES_MACRO)
        n = upsert_fred_to_db(conn, df)
        logger.info("[FRED] %s rows upserted.", n)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[FRED] Fetch failed. Continuing with cached data. Error=%r",
            exc,
        )


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _generate_signals(
    conn,
    as_of: date,
    target_horizon_months: int = 6,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build feature matrix (sliced to as_of), train ensemble WFO models, return signals.

    Uses the production Ridge+GBT ensemble (v11.0) on the PRIMARY_FORECAST_UNIVERSE
    with v18 lean feature sets.  Inverse-variance weighting across models drives
    the ``confidence_tier`` and ``prob_outperform`` columns in the output.

    Returns:
        (signals, ensemble_results, diagnostics) where signals is a DataFrame indexed by benchmark
        with columns predicted_relative_return, ic, hit_rate, signal, prediction_std,
        prob_outperform, confidence_tier; and ensemble_results is the dict returned by
        ``run_ensemble_benchmarks`` (ETF ticker → EnsembleWFOResult).
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
    }

    # Load relative return matrix for the primary forecast universe only.
    # ETF_BENCHMARK_UNIVERSE still governs data ingestion; PRIMARY_FORECAST_UNIVERSE
    # governs which benchmarks the production ensemble trains on.
    rel_matrix_cols = {}
    for etf in config.PRIMARY_FORECAST_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, etf, target_horizon_months)
        if not rel_series.empty:
            rel_matrix_cols[etf] = rel_series
    if not rel_matrix_cols:
        return pd.DataFrame(), {}, diagnostics

    rel_matrix = pd.DataFrame(rel_matrix_cols)

    # v11.0: representative CPCV uses VOO + ridge (core benchmark of the primary
    # universe; matches the promoted model type replacing the former VTI+elasticnet).
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

    # Generate ensemble signals (includes prob_outperform and confidence_tier)
    signals = get_ensemble_signals(
        X_full=X_event,
        relative_return_matrix=rel_matrix,
        ensemble_results=ensemble_results,
        X_current=X_current,
    )

    # Normalize column names for downstream consumers (consensus, report writer)
    signals = signals.rename(columns={
        "point_prediction": "predicted_relative_return",
        "mean_ic":          "ic",
        "mean_hit_rate":    "hit_rate",
    })

    return signals, ensemble_results, diagnostics


# ---------------------------------------------------------------------------
# Consensus signal
# ---------------------------------------------------------------------------

def _reconstruct_ensemble_oos(
    ens_result,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the inverse-variance ensemble OOS predictions for one benchmark.

    Combines per-model WFO fold y_hat values using the same 1/MAE² weights
    that ``get_ensemble_signals()`` uses for live prediction, giving a faithful
    approximation of what the ensemble would have predicted at each historical
    OOS observation.

    Args:
        ens_result: EnsembleWFOResult for a single benchmark.

    Returns:
        ``(y_hat_ensemble, y_true)`` — aligned arrays across all folds.
    """
    y_hat, y_true = reconstruct_ensemble_oos_predictions(ens_result)
    return y_hat.to_numpy(dtype=float), y_true.to_numpy(dtype=float)


def _calibrate_signals(
    signals: pd.DataFrame,
    ensemble_results: dict,
    target_horizon_months: int = 6,
) -> tuple[pd.DataFrame, CalibrationResult]:
    """
    Calibrate per-benchmark P(outperform) using per-benchmark Platt scaling.

    Fits one Platt (logistic regression) model per ETF benchmark on that
    benchmark's own OOS fold history.  Each benchmark's current live
    ``predicted_relative_return`` is then passed through its own sigmoid to
    produce a calibrated probability.

    Using a single global calibration model would conflate 21 asset classes
    with very different return distributions (e.g., GLD vs BND vs VGT), causing
    the isotonic regression to return a single plateau value for all benchmarks.
    Per-benchmark calibration preserves cross-benchmark discrimination.

    Isotonic regression is intentionally disabled here.  With n=78–260 OOS
    observations per benchmark (as of 2026), the isotonic step function produces
    degenerate plateaus on out-of-sample inputs.  Isotonic will be re-evaluated
    when each benchmark accumulates ≥500 OOS observations (roughly 2028+).

    Adds a ``calibrated_prob_outperform`` column to the signals DataFrame.
    The raw ``prob_outperform`` column is preserved for diagnostic comparison.

    Args:
        signals:               Per-benchmark signals from ``_generate_signals()``.
        ensemble_results:      Dict of ETF ticker → EnsembleWFOResult.
        target_horizon_months: Prediction horizon (used as block_len for ECE CI).

    Returns:
        ``(updated_signals, CalibrationResult)`` where CalibrationResult is an
        aggregate summary (pooled ECE computed after per-benchmark calibration).
    """
    if signals.empty:
        return signals, CalibrationResult(
            n_obs=0, method="uncalibrated", ece=0.0,
            ece_ci_lower=0.0, ece_ci_upper=1.0,
        )

    pred_col = "predicted_relative_return"
    if pred_col not in signals.columns:
        return signals, CalibrationResult(
            n_obs=0, method="uncalibrated", ece=0.0,
            ece_ci_lower=0.0, ece_ci_upper=1.0,
        )

    signals = signals.copy()
    calibrated_probs: list[float] = []

    # For aggregate ECE reporting — pool calibrated probs and outcomes post-fit
    all_cal_probs: list[float] = []
    all_outcomes_pool: list[int] = []
    n_total = 0
    methods_used: list[str] = []

    for ticker in signals.index:
        ens_result = ensemble_results.get(str(ticker))
        if ens_result is None:
            calibrated_probs.append(0.5)
            continue

        y_hat_bm, y_true_bm = _reconstruct_ensemble_oos(ens_result)
        if len(y_hat_bm) == 0:
            calibrated_probs.append(0.5)
            continue

        # Platt-only per benchmark (isotonic disabled — see docstring)
        bm_model, bm_result = fit_calibration_model(
            y_hat_bm,
            (y_true_bm > 0).astype(int),
            min_obs_platt=config.CALIBRATION_MIN_OBS_PLATT,
            min_obs_isotonic=10_000,   # effectively disables isotonic
            n_bins=config.CALIBRATION_N_BINS,
            block_len=target_horizon_months,
            n_bootstrap=max(50, config.CALIBRATION_BOOTSTRAP_REPS // 10),
        )

        current_pred = float(signals.at[ticker, pred_col])
        cal_prob = calibrate_prediction(bm_model, current_pred)
        calibrated_probs.append(cal_prob)
        methods_used.append(bm_result.method)
        n_total += bm_result.n_obs

        # Collect calibrated training probs for aggregate ECE
        if bm_model is not None and len(y_hat_bm) >= config.CALIBRATION_MIN_OBS_PLATT:
            from src.models.calibration import calibrate_prediction as _cal
            train_cal_probs = [_cal(bm_model, float(y)) for y in y_hat_bm]
            all_cal_probs.extend(train_cal_probs)
            all_outcomes_pool.extend((y_true_bm > 0).astype(int).tolist())

    signals["calibrated_prob_outperform"] = calibrated_probs

    # ------------------------------------------------------------------
    # Aggregate CalibrationResult for reporting
    # ------------------------------------------------------------------
    from src.models.calibration import compute_ece, block_bootstrap_ece_ci

    if len(all_cal_probs) >= 4:
        agg_probs = np.array(all_cal_probs, dtype=float)
        agg_outcomes = np.array(all_outcomes_pool, dtype=int)
        agg_ece = compute_ece(agg_probs, agg_outcomes, n_bins=config.CALIBRATION_N_BINS)
        ci_lo, ci_hi = block_bootstrap_ece_ci(
            agg_probs, agg_outcomes,
            n_bins=config.CALIBRATION_N_BINS,
            block_len=target_horizon_months,
            n_bootstrap=config.CALIBRATION_BOOTSTRAP_REPS,
        )
        dominant_method = "platt" if "platt" in methods_used else "uncalibrated"
    else:
        agg_ece, ci_lo, ci_hi = 0.0, 0.0, 1.0
        dominant_method = "uncalibrated"

    # Expose pooled arrays for the reliability diagram (P2.7)
    cal_probs_arr = np.array(all_cal_probs, dtype=float) if len(all_cal_probs) >= 4 else np.array([], dtype=float)
    cal_outcomes_arr = np.array(all_outcomes_pool, dtype=int) if len(all_outcomes_pool) >= 4 else np.array([], dtype=int)

    return signals, CalibrationResult(
        n_obs=n_total,
        method=dominant_method,
        ece=agg_ece,
        ece_ci_lower=ci_lo,
        ece_ci_upper=ci_hi,
    ), cal_probs_arr, cal_outcomes_arr


def _compute_conformal_intervals(
    signals: pd.DataFrame,
    ensemble_results: dict,
) -> pd.DataFrame:
    """
    Compute per-benchmark conformal prediction intervals for the current ensemble predictions.

    Uses ACI (Adaptive Conformal Inference) by default, falling back to split conformal
    when insufficient calibration data is available (n < 4).

    Coverage, method, and gamma are read from config constants:
      CONFORMAL_COVERAGE  (default 0.80 = 80% CI)
      CONFORMAL_METHOD    ("aci" or "split")
      CONFORMAL_ACI_GAMMA (default 0.05)

    Adds the following columns to the signals DataFrame:
      ci_lower              Lower bound of the prediction interval.
      ci_upper              Upper bound of the prediction interval.
      ci_width              Total CI width (upper − lower).
      ci_empirical_coverage Fraction of calibration residuals inside the interval.
      ci_n_calibration      Number of calibration residuals used.

    Args:
        signals:          Per-benchmark signals from ``_generate_signals()``.
        ensemble_results: Dict of ETF ticker → EnsembleWFOResult.

    Returns:
        Updated signals DataFrame with CI columns added.
    """
    if signals.empty:
        return signals

    pred_col = "predicted_relative_return"
    if pred_col not in signals.columns:
        return signals

    signals = signals.copy()
    ci_lowers: list[float] = []
    ci_uppers: list[float] = []
    ci_widths: list[float] = []
    ci_empirical: list[float] = []
    ci_n_cal: list[int] = []
    ci_trailing_empirical: list[float] = []
    ci_trailing_gap: list[float] = []
    ci_trailing_n: list[int] = []

    for ticker in signals.index:
        ens_result = ensemble_results.get(str(ticker))
        y_hat_current = float(signals.at[ticker, pred_col])

        if ens_result is None:
            ci_lowers.append(float("nan"))
            ci_uppers.append(float("nan"))
            ci_widths.append(float("nan"))
            ci_empirical.append(float("nan"))
            ci_n_cal.append(0)
            ci_trailing_empirical.append(float("nan"))
            ci_trailing_gap.append(float("nan"))
            ci_trailing_n.append(0)
            continue

        y_hat_oos, y_true_oos = _reconstruct_ensemble_oos(ens_result)
        if len(y_hat_oos) < 1:
            ci_lowers.append(float("nan"))
            ci_uppers.append(float("nan"))
            ci_widths.append(float("nan"))
            ci_empirical.append(float("nan"))
            ci_n_cal.append(0)
            ci_trailing_empirical.append(float("nan"))
            ci_trailing_gap.append(float("nan"))
            ci_trailing_n.append(0)
            continue

        conf_result = conformal_interval_from_ensemble(
            y_hat_current=y_hat_current,
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
        )
        ci_lowers.append(conf_result.lower)
        ci_uppers.append(conf_result.upper)
        ci_widths.append(conf_result.width)
        ci_empirical.append(conf_result.empirical_coverage)
        ci_n_cal.append(conf_result.n_calibration)
        ci_trailing_empirical.append(coverage_backtest.trailing_empirical_coverage)
        ci_trailing_gap.append(coverage_backtest.trailing_coverage_gap)
        ci_trailing_n.append(coverage_backtest.trailing_n)

    signals["ci_lower"] = ci_lowers
    signals["ci_upper"] = ci_uppers
    signals["ci_width"] = ci_widths
    signals["ci_empirical_coverage"] = ci_empirical
    signals["ci_n_calibration"] = ci_n_cal
    signals["ci_trailing_empirical_coverage"] = ci_trailing_empirical
    signals["ci_trailing_coverage_gap"] = ci_trailing_gap
    signals["ci_trailing_n"] = ci_trailing_n

    return signals


def _summarize_conformal_coverage(signals: pd.DataFrame | None) -> ConformalCoverageBacktest | None:
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


def _record_model_health_snapshot(
    conn,
    as_of: date,
    aggregate_health: dict | None,
    cal_result: CalibrationResult | None,
    conformal_coverage_summary: ConformalCoverageBacktest | None,
) -> ModelDriftSummary | None:
    """Persist the monthly model-health snapshot and return the latest drift summary."""
    if aggregate_health is None or cal_result is None:
        return None

    month_end = (
        pd.Timestamp(as_of).to_period("M").to_timestamp("M").date().isoformat()
    )
    db_client.upsert_model_performance_log(
        conn,
        [
            {
                "month_end": month_end,
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
        ],
    )
    history = db_client.get_model_performance_log(conn)
    if history.empty:
        return None
    return summarize_latest_model_drift(history.reset_index())


def _evaluate_and_record_retrain_trigger(
    conn,
    drift_summary: ModelDriftSummary | None,
) -> RetainTriggerResult | None:
    """
    Evaluate the retrain trigger against the current drift state and persist
    the result to model_retrain_log for governance/audit.

    Returns the RetainTriggerResult, or None if the DB table is unavailable.
    """
    try:
        last_date = db_client.get_last_retrain_trigger_date(conn)
        result = evaluate_retrain_trigger(
            drift_summary=drift_summary,
            last_trigger_date=last_date,
        )
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


def _consensus_signal(
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

    # Composite tier mirrors get_confidence_tier thresholds
    if mean_prob >= 0.70 or mean_prob <= 0.30:
        confidence_tier = "HIGH"
    elif mean_prob >= 0.60 or mean_prob <= 0.40:
        confidence_tier = "MODERATE"
    else:
        confidence_tier = "LOW"

    return consensus, mean_pred, mean_ic, mean_hr, mean_prob, confidence_tier


def _build_v74_shadow_consensus(
    signals: pd.DataFrame,
    aggregate_health: dict | None,
    representative_cpcv: CPCVResult | None = None,
) -> pd.DataFrame | None:
    """Build the v74 quality-weighted shadow consensus comparison table."""
    if signals.empty or aggregate_health is None:
        return None

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
    for row in shadow_df.to_dict("records"):
        recommendation_mode = _determine_recommendation_mode(
            str(row["consensus"]),
            float(row["mean_predicted_return"]),
            float(row["mean_ic"]),
            float(row["mean_hit_rate"]),
            aggregate_health,
            representative_cpcv,
        )
        enriched = dict(row)
        enriched["recommendation_mode"] = str(recommendation_mode["label"])
        enriched["recommended_sell_pct"] = float(recommendation_mode["sell_pct"])
        enriched["is_live_path"] = bool(row["variant"] == "equal_weight")
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def _build_benchmark_quality_frame(
    ensemble_results: dict,
    target_horizon_months: int = 6,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Build ensemble-level pooled and per-benchmark OOS diagnostics."""
    predicted_parts: list[pd.Series] = []
    realized_parts: list[pd.Series] = []
    rows: list[dict[str, float | int | str]] = []

    for benchmark, ens_result in ensemble_results.items():
        y_hat, y_true = reconstruct_ensemble_oos_predictions(ens_result)
        if len(y_true) < 2:
            continue

        y_hat = y_hat.sort_index()
        y_true = y_true.sort_index()
        summary = summarize_prediction_diagnostics(
            y_hat,
            y_true,
            target_horizon_months=target_horizon_months,
        )
        predicted_parts.append(y_hat)
        realized_parts.append(y_true)
        rows.append(
            {
                "benchmark": benchmark,
                "n_obs": int(summary["n_obs"]),
                "oos_r2": float(summary["oos_r2"]),
                "nw_ic": float(summary["nw_ic"]),
                "nw_p_value": float(summary["nw_p_value"]),
                "hit_rate": float(summary["hit_rate"]),
                "cw_t_stat": float(summary["cw_t_stat"]),
                "cw_p_value": float(summary["cw_p_value"]),
                "cw_mean_adjusted_differential": float(
                    summary["cw_mean_adjusted_differential"]
                ),
                "r2_flag": _flag(float(summary["oos_r2"]), config.DIAG_MIN_OOS_R2, 0.005),
                "ic_flag": _flag(float(summary["nw_ic"]), config.DIAG_MIN_IC, 0.03),
                "hr_flag": _flag(float(summary["hit_rate"]), config.DIAG_MIN_HIT_RATE, 0.52),
            }
        )

    if predicted_parts:
        agg_predicted = pd.concat(predicted_parts).sort_index()
        agg_realized = pd.concat(realized_parts).sort_index()
        agg_predicted, agg_realized = agg_predicted.align(agg_realized, join="inner")
    else:
        agg_predicted = pd.Series(dtype=float, name="y_hat")
        agg_realized = pd.Series(dtype=float, name="y_true")

    benchmark_quality_df = pd.DataFrame(rows)
    if not benchmark_quality_df.empty:
        benchmark_quality_df = benchmark_quality_df.sort_values(
            by=["cw_t_stat", "nw_ic", "oos_r2"],
            ascending=False,
        ).reset_index(drop=True)

    return agg_predicted, agg_realized, benchmark_quality_df


def _compute_aggregate_health(
    ensemble_results: dict,
    target_horizon_months: int = 6,
) -> dict | None:
    """Compute the aggregate health metrics shared by recommendation and diagnostics."""
    agg_predicted, agg_realized, benchmark_quality_df = _build_benchmark_quality_frame(
        ensemble_results=ensemble_results,
        target_horizon_months=target_horizon_months,
    )
    if len(agg_realized) < 4:
        return None

    summary = summarize_prediction_diagnostics(
        agg_predicted,
        agg_realized,
        target_horizon_months=target_horizon_months,
    )
    nw_lags = target_horizon_months - 1
    oos_r2 = float(summary["oos_r2"])
    nw_ic = float(summary["nw_ic"])
    nw_pval = float(summary["nw_p_value"])
    agg_hit = float(summary["hit_rate"])

    return {
        "n_agg": int(summary["n_obs"]),
        "nw_lags": nw_lags,
        "oos_r2": oos_r2,
        "nw_ic": nw_ic,
        "nw_pval": nw_pval,
        "agg_hit": agg_hit,
        "cw_t_stat": float(summary["cw_t_stat"]),
        "cw_p_value": float(summary["cw_p_value"]),
        "cw_mean_adjusted_differential": float(summary["cw_mean_adjusted_differential"]),
        "r2_flag": _flag(oos_r2, config.DIAG_MIN_OOS_R2, 0.005),
        "ic_flag": _flag(nw_ic, config.DIAG_MIN_IC, 0.03),
        "hr_flag": _flag(agg_hit, config.DIAG_MIN_HIT_RATE, 0.52),
        "per_benchmark_rows": benchmark_quality_df.to_dict("records"),
        "benchmark_quality_df": benchmark_quality_df,
        "agg_predicted": agg_predicted,
        "agg_realized": agg_realized,
    }


def _get_next_vest_info(as_of: date) -> tuple[date, str]:
    """Return the next vest date and RSU type after the as-of date."""
    candidates = [
        (date(as_of.year, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY), "time"),
        (date(as_of.year, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY), "performance"),
        (date(as_of.year + 1, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY), "time"),
        (date(as_of.year + 1, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY), "performance"),
    ]
    future = sorted((d, rsu_type) for d, rsu_type in candidates if d >= as_of)
    return future[0]


def _load_previous_decision_summary(as_of: date) -> dict | None:
    """Load the most recent prior row from decision_log.md, if available."""
    path = Path("results/monthly_decisions/decision_log.md")
    if not path.exists():
        return None

    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        if "As-Of Date" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "as_of": parts[0],
                "run_date": parts[1],
                "consensus": parts[2],
                "sell_pct": parts[3],
                "predicted": parts[4],
                "mean_ic": parts[5],
                "mean_hr": parts[6],
                "notes": parts[7],
            }
        )

    prior_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            if date.fromisoformat(row["as_of"]) < as_of:
                prior_rows.append(row)
        except ValueError:
            continue
    if not prior_rows:
        return None
    prior_rows.sort(key=lambda row: row["as_of"])
    return prior_rows[-1]


def _determine_recommendation_mode(
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


def _sell_pct_from_consensus(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
) -> float:
    """Map consensus signal + IC to a sell-percentage recommendation (0.0–1.0)."""
    if mean_ic < 0.05:
        return 0.50  # weak signal → default diversification
    if consensus == "OUTPERFORM":
        if mean_predicted > 0.15:
            return 0.25  # high conviction → hold most
        if mean_predicted > 0.05:
            return 0.50
        return 0.75
    if consensus == "UNDERPERFORM":
        return 1.00   # model predicts underperformance → diversify fully
    return 0.50       # NEUTRAL


# ---------------------------------------------------------------------------
# v7.3 — Tax context section builder
# ---------------------------------------------------------------------------

def _build_tax_context_lines(
    predicted_6m_return: float,
    prob_outperform: float,
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
    as_of: date | None = None,
) -> list[str]:
    """Build the ## Tax Context section for recommendation.md.

    Shows the STCG/LTCG breakeven return and interprets the current model
    prediction against it.  No lot-specific data is needed; this section
    provides universal tax-timing context for any RSU holder.

    The key insight (v7.1 finding): PGR's typical model prediction of 1–7%
    is far below the ~21% breakeven required for an immediate STCG sale to
    beat waiting 366 days for LTCG treatment.  This section makes that
    comparison explicit every month.

    Args:
        predicted_6m_return: Ensemble mean 6M relative return prediction.
        prob_outperform:      Mean P(outperform) across benchmarks.
        stcg_rate:            STCG rate override (default: config.STCG_RATE).
        ltcg_rate:            LTCG rate override (default: config.LTCG_RATE).
        as_of:                Reporting as-of date for next-vest calculations.

    Returns:
        List of markdown lines (without leading blank line separator).
    """
    from src.tax.capital_gains import compute_stcg_ltcg_breakeven

    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    if as_of is None:
        as_of = date.today()

    breakeven = compute_stcg_ltcg_breakeven(stcg_rate, ltcg_rate)
    tax_differential = stcg_rate - ltcg_rate

    # Interpret predicted return vs breakeven
    pred_vs_breakeven_gap = predicted_6m_return - breakeven
    if predicted_6m_return >= breakeven:
        verdict = (
            f"⚠️ **Model prediction ({predicted_6m_return:+.1%}) EXCEEDS the "
            f"LTCG breakeven ({breakeven:.1%}).**  Immediate sale at STCG may be "
            f"warranted — verify with lot-specific analysis."
        )
    elif predicted_6m_return > 0:
        verdict = (
            f"✓ **Model prediction ({predicted_6m_return:+.1%}) is below the "
            f"LTCG breakeven ({breakeven:.1%}) by {abs(pred_vs_breakeven_gap):.1%}.**  "
            f"Holding RSUs for 366 days post-vest to qualify for LTCG treatment "
            f"is likely the higher after-tax outcome."
        )
    else:
        verdict = (
            f"⚠️ **Model predicts negative return ({predicted_6m_return:+.1%}).**  "
            f"Consider capital-loss harvesting scenario — a tax loss at {stcg_rate:.0%} "
            f"STCG rate can offset other gains.  See three-scenario analysis at vesting."
        )

    # Next vest dates from config
    this_year = as_of.year
    next_time_vest = date(this_year, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY)
    next_perf_vest = date(this_year, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY)
    if next_time_vest < as_of:
        next_time_vest = date(this_year + 1, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY)
    if next_perf_vest < as_of:
        next_perf_vest = date(this_year + 1, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY)

    lines = [
        "",
        "---",
        "",
        "## Tax Context",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| STCG Rate (federal) | {stcg_rate:.0%} |",
        f"| LTCG Rate (federal) | {ltcg_rate:.0%} |",
        f"| Tax-rate differential | {tax_differential:.0%} |",
        f"| **LTCG breakeven return** | **{breakeven:.2%}** |",
        f"| Current model prediction (6M) | {predicted_6m_return:+.2%} |",
        f"| P(outperform) | {prob_outperform:.1%} |",
        f"| Next time-based vest | {next_time_vest} |",
        f"| Next performance vest | {next_perf_vest} |",
        "",
        verdict,
        "",
        "> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum",
        "> return needed on RSUs held to LTCG eligibility (366 days post-vest) to",
        "> produce higher after-tax proceeds than selling immediately at STCG.",
        "> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.",
    ]

    return lines


def _build_provisional_vest_scenario(
    conn,
    as_of: date,
    mean_predicted: float,
    prob_outperform: float,
) -> dict | None:
    """Build a provisional three-scenario view for the next vest using current lots."""
    lots_path = Path("data/processed/position_lots.csv")
    if not lots_path.exists():
        return None

    lots = load_position_lots(str(lots_path))
    if not lots:
        return None

    prices = db_client.get_prices(conn, "PGR", end_date=str(as_of))
    if prices.empty:
        return None

    current_price = float(prices["close"].iloc[-1])
    total_shares = sum(lot.shares_remaining for lot in lots if lot.shares_remaining and lot.shares_remaining > 0)
    if total_shares <= 0:
        return None

    avg_basis = sum(
        lot.shares_remaining * lot.cost_basis_per_share
        for lot in lots
        if lot.shares_remaining and lot.shares_remaining > 0
    ) / total_shares
    vest_date, rsu_type = _get_next_vest_info(as_of)
    scenario = compute_three_scenarios(
        vest_date=vest_date,
        rsu_type=rsu_type,
        shares=total_shares,
        cost_basis_per_share=avg_basis,
        current_price=current_price,
        predicted_6m_return=mean_predicted,
        predicted_12m_return=mean_predicted * 2.0,
        prob_outperform_6m=prob_outperform,
        prob_outperform_12m=prob_outperform,
    )

    # v35: Monte Carlo tax-sensitivity analysis
    mc_analysis: MonteCarloTaxAnalysis | None = None
    try:
        close_prices = prices["close"].dropna().values
        if len(close_prices) >= 30:
            annual_vol = estimate_annual_vol(close_prices)
            annual_drift = mean_predicted * 2.0  # annualise 6M forecast
            mc_analysis = run_monte_carlo_tax_analysis(
                current_price=current_price,
                cost_basis_per_share=avg_basis,
                shares=total_shares,
                annual_vol=annual_vol,
                annual_drift=annual_drift,
            )
    except Exception:
        logger.debug("Monte Carlo tax analysis skipped", exc_info=True)

    return {
        "vest_date": vest_date,
        "rsu_type": rsu_type,
        "current_price": current_price,
        "avg_basis": avg_basis,
        "shares": total_shares,
        "scenario": scenario,
        "mc_analysis": mc_analysis,
    }


def _current_baseline_prediction(y_aligned: pd.Series) -> float:
    """Current-point forecast for the historical-mean baseline."""
    window = min(len(y_aligned), config.WFO_TRAIN_WINDOW_MONTHS)
    return float(y_aligned.iloc[-window:].mean())


def _build_shadow_baseline_summary(
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
        current_pred = _current_baseline_prediction(y_aligned)
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
            pd.DataFrame(
                {
                    "y_hat": pred_series.values,
                    "y_true": realized.values,
                }
            )
        )

    if not signal_rows:
        return None, pd.DataFrame()

    shadow_signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(prediction_frames, target_horizon_months)
    consensus, mean_pred, mean_ic, mean_hr, _, confidence_tier = _consensus_signal(shadow_signals)
    recommendation_mode = _determine_recommendation_mode(
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


def _build_existing_holdings_guidance(conn, as_of: date) -> list[dict[str, object]]:
    """Build tax-bucketed guidance for already-held PGR shares."""
    lots_path = Path("data/processed/position_lots.csv")
    if not lots_path.exists():
        return []
    lots = load_position_lots(str(lots_path))
    if not lots:
        return []
    prices = db_client.get_prices(conn, "PGR", end_date=str(as_of))
    if prices.empty:
        return []
    current_price = float(prices["close"].iloc[-1])
    return [
        {
            "vest_date": action.vest_date,
            "shares": action.shares,
            "cost_basis_per_share": action.cost_basis_per_share,
            "tax_bucket": action.tax_bucket,
            "unrealized_gain": action.unrealized_gain,
            "unrealized_return": action.unrealized_return,
            "rationale": action.rationale,
        }
        for action in summarize_existing_holdings_actions(
            lots,
            current_price=current_price,
            sell_date=as_of,
        )
    ]


def _build_redeploy_guidance(conn) -> list[dict[str, object]]:
    """Build diversification-first redeploy buckets for production messaging."""
    investable_universe = v27_investable_redeploy_universe()
    scoreboard = score_benchmarks_against_pgr(conn, investable_universe)
    if scoreboard.empty:
        return []
    scoreboard["composite_score"] = 0.0
    scoreboard = add_destination_roles(scoreboard)
    return recommend_redeploy_buckets(scoreboard, investable_universe)


def _load_etf_monthly_returns(
    conn,
    tickers: list[str],
    end_date: date,
) -> pd.DataFrame:
    """Load daily ETF prices, resample to month-end close, return pct_change matrix.

    Used by the Black-Litterman diagnostic call to build the covariance matrix.
    Tickers with no price data are silently excluded.

    Returns:
        DataFrame with tickers as columns and monthly returns as rows
        (DatetimeIndex).  May be empty if no data is available.
    """
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = db_client.get_prices(conn, ticker, end_date=end_date.isoformat())
        except Exception:  # noqa: BLE001
            logger.warning("Could not load prices for %s; excluding from BL covariance", ticker, exc_info=True)
            continue
        if df.empty:
            continue
        monthly = df["close"].resample("ME").last().dropna()
        frames[ticker] = monthly.pct_change().dropna()
    if not frames:
        return pd.DataFrame()
    merged = pd.DataFrame(frames)
    return merged.dropna(how="all")


def _build_redeploy_portfolio(
    conn,
    signals: pd.DataFrame,
    recommendation_mode: dict[str, str | float] | None,
) -> dict[str, object] | None:
    """Build the concrete monthly redeploy portfolio recommendation."""
    if signals.empty:
        return None
    investable_universe = v27_investable_redeploy_universe()
    scoreboard = score_benchmarks_against_pgr(conn, investable_universe)
    if scoreboard.empty:
        return None
    mode_label = (
        str(recommendation_mode.get("label", "DEFER-TO-TAX-DEFAULT"))
        if recommendation_mode is not None
        else "DEFER-TO-TAX-DEFAULT"
    )
    return recommend_redeploy_portfolio(
        signals=signals,
        diversification_scoreboard=scoreboard,
        recommendation_mode_label=mode_label,
    )


def _mode_payload_from_summary(summary: SnapshotSummary) -> dict[str, str | float]:
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


def _build_executive_summary_lines(
    as_of: date,
    consensus: str,
    confidence_tier: str,
    mean_predicted: float,
    sell_pct: float,
    recommendation_mode: dict[str, str | float],
    aggregate_health: dict | None,
    previous_summary: dict | None,
    next_vest_summary: dict | None,
) -> list[str]:
    """Compatibility wrapper around the extracted summary-rendering helper."""
    return render_executive_summary_lines(
        as_of=as_of,
        consensus=consensus,
        confidence_tier=confidence_tier,
        mean_predicted=mean_predicted,
        sell_pct=sell_pct,
        recommendation_mode=recommendation_mode,
        aggregate_health=aggregate_health,
        previous_summary=previous_summary,
        next_vest_summary=next_vest_summary,
    )


def _build_vest_decision_lines(
    next_vest_summary: dict | None,
    recommendation_mode: dict[str, str | float],
    sell_pct: float,
) -> list[str]:
    """Compatibility wrapper around the extracted vest-section helper."""
    return render_vest_decision_lines(
        next_vest_summary=next_vest_summary,
        recommendation_mode=recommendation_mode,
        sell_pct=sell_pct,
    )


# ---------------------------------------------------------------------------
# v32.2 — Vesting decision policy backtest helper (Tier 4.3)
# ---------------------------------------------------------------------------

def _compute_policy_summary(
    ensemble_results: dict,
) -> dict[str, PolicySummary] | None:
    """Compute OOS decision-policy summaries from the monthly ensemble.

    Aggregates OOS predictions and realized relative returns from the deployed
    inverse-variance ensemble across all benchmarks, then evaluates every fixed
    and signal-driven policy defined in ``src.research.policy_metrics``.

    Returns a dict of ``{policy_name: PolicySummary}`` or ``None`` if fewer
    than 4 OOS observations are available.
    """
    predicted, realized, _ = _build_benchmark_quality_frame(ensemble_results)
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
                "_compute_policy_summary: failed to evaluate policy '%s'; skipping",
                policy,
                exc_info=True,
            )

    return summaries if summaries else None


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_recommendation_md(
    out_dir: Path,
    as_of: date,
    run_date: date,
    conn,
    signals: pd.DataFrame,
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    sell_pct: float,
    dry_run: bool,
    mean_prob_outperform: float = 0.5,
    composite_confidence_tier: str = "LOW",
    cal_result: CalibrationResult | None = None,
    aggregate_health: dict | None = None,
    recommendation_mode: dict[str, str | float] | None = None,
    live_summary: SnapshotSummary | None = None,
    shadow_summary: SnapshotSummary | None = None,
    consensus_shadow_df: pd.DataFrame | None = None,
    existing_holdings: list[dict[str, object]] | None = None,
    redeploy_buckets: list[dict[str, object]] | None = None,
    redeploy_portfolio: dict[str, object] | None = None,
    recommendation_layer_label: str | None = None,
    representative_cpcv: CPCVResult | None = None,
    freshness_report: dict[str, object] | None = None,
    model_drift_summary: ModelDriftSummary | None = None,
    policy_summary: dict[str, PolicySummary] | None = None,
    bl_diagnostics: BLDiagnostics | None = None,
) -> None:
    """Write the human-readable recommendation report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recommendation.md"

    has_confidence = "prob_outperform" in signals.columns
    has_calibrated = (
        "calibrated_prob_outperform" in signals.columns
        and not signals.empty
        and cal_result is not None
        and cal_result.method != "uncalibrated"
    )
    mean_cal_prob = (
        float(signals["calibrated_prob_outperform"].mean())
        if has_calibrated
        else mean_prob_outperform
    )
    if recommendation_mode is None:
        recommendation_mode = {
            "mode": "monitoring-only",
            "label": "MONITORING-ONLY",
            "summary": "Recommendation mode defaulted because aggregate health was unavailable.",
            "action_note": "Use the default diversification rule.",
        }
    previous_summary = _load_previous_decision_summary(as_of)
    next_vest_summary = _build_provisional_vest_scenario(conn, as_of, mean_predicted, mean_cal_prob)
    confidence_snapshot = build_confidence_snapshot(
        mean_ic=mean_ic,
        mean_hr=mean_hr,
        aggregate_health=aggregate_health,
        representative_cpcv=representative_cpcv,
    )

    lines = [
        f"# PGR Monthly Decision Report — {as_of.strftime('%B %Y')}",
        "",
        f"**As-of Date:** {as_of}  ",
        f"**Run Date:** {run_date}  ",
        f"**Model Version:** {_MODEL_VERSION_LABEL}  ",
        f"**Recommendation Layer:** {recommendation_layer_label or config.RECOMMENDATION_LAYER_MODE}  ",
        "",
        "---",
        "",
        *_build_executive_summary_lines(
            as_of,
            consensus,
            composite_confidence_tier,
            mean_predicted,
            sell_pct,
            recommendation_mode,
            aggregate_health,
            previous_summary,
            next_vest_summary,
        ),
        *render_data_freshness_lines(freshness_report),
        "## Consensus Signal",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Signal | **{consensus} ({composite_confidence_tier} CONFIDENCE)** |",
        f"| Recommendation Mode | **{recommendation_mode['label']}** |",
        f"| Recommended Sell % | **{sell_pct:.0%}** |",
        f"| Predicted 6M Relative Return | {mean_predicted:+.2%} |",
    ]
    if has_confidence:
        lines.append(f"| P(Outperform, raw) | {mean_prob_outperform:.1%} |")
    if has_calibrated:
        lines.append(f"| P(Outperform, calibrated) | {mean_cal_prob:.1%} |")

    # Conformal CI row — median CI bounds across benchmarks (robust to outlier widths)
    has_ci = "ci_lower" in signals.columns and not signals.empty
    if has_ci:
        valid_ci = signals[["ci_lower", "ci_upper"]].dropna()
        if not valid_ci.empty:
            ci_lo_med = float(valid_ci["ci_lower"].median())
            ci_hi_med = float(valid_ci["ci_upper"].median())
            lines.append(
                f"| {config.CONFORMAL_COVERAGE:.0%} Prediction Interval (median) "
                f"| {ci_lo_med:+.2%} to {ci_hi_med:+.2%} |"
            )

    lines += [
        f"| Mean IC (across benchmarks) | {mean_ic:.4f} |",
        f"| Mean Hit Rate | {mean_hr:.1%} |",
        f"| Aggregate OOS R^2 | {aggregate_health['oos_r2']:.2%} |" if aggregate_health is not None else "| Aggregate OOS R^2 | n/a |",
        "",
        "> **Note:** The sell % recommendation is used only at actual vesting events",
        "> (January and July).  Monthly reports are monitoring tools, not trade signals.",
    ]

    # Calibration status note
    if cal_result is None or cal_result.method == "uncalibrated":
        lines += [
            ">",
            "> **Calibration:** Phase 1 — P(outperform) uses uncalibrated BayesianRidge posteriors.",
            f"> Platt scaling activates at n ≥ {config.CALIBRATION_MIN_OBS_PLATT} OOS observations.",
        ] 
    else:
        method_label = "Platt scaling" if cal_result.method == "platt" else "Platt → Isotonic"
        lines += [
            ">",
            f"> **Calibration:** Phase 2 — {method_label} active "
            f"(n={cal_result.n_obs:,} OOS obs).  "
            f"ECE = {cal_result.ece:.1%} "
            f"[95% CI: {cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}].",
        ]

    if consensus_shadow_df is not None and not consensus_shadow_df.empty:
        lines += [
            "",
            "---",
            "",
            "## Consensus Shadow Evaluation",
            "",
            "> v74 monitors a quality-weighted cross-benchmark consensus in shadow mode.",
            "> It is reported for promotion evaluation only and does not change the live recommendation above.",
            "",
            "| Variant | Consensus | Mean Pred. Return | Mean IC | Mean Hit Rate | P(Outperform) | Mode | Sell % | Top Weight |",
            "|---------|-----------|-------------------|---------|---------------|---------------|------|--------|------------|",
        ]
        for row in consensus_shadow_df.to_dict("records"):
            variant_label = (
                "Live equal-weight"
                if bool(row.get("is_live_path"))
                else "v74 quality-weighted"
            )
            top_weight = (
                f"{row.get('top_benchmark', 'n/a')} ({float(row.get('top_benchmark_weight', 0.0)):.1%})"
                if row.get("top_benchmark")
                else "n/a"
            )
            lines.append(
                f"| {variant_label} | {row['consensus']} ({row['confidence_tier']}) | "
                f"{float(row['mean_predicted_return']):+.2%} | {float(row['mean_ic']):.4f} | "
                f"{float(row['mean_hit_rate']):.1%} | {float(row['mean_prob_outperform']):.1%} | "
                f"{row['recommendation_mode']} | {float(row['recommended_sell_pct']):.0%} | "
                f"{top_weight} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Confidence Snapshot",
        "",
        f"- {confidence_snapshot['summary']}",
        "",
        "| Check | Current | Threshold | Status | Meaning |",
        "|-------|---------|-----------|--------|---------|",
    ]
    for row in confidence_snapshot["rows"]:
        lines.append(
            f"| {row['check']} | {row['current']} | {row['threshold']} | **{row['status']}** | {row['meaning']} |"
        )

    if model_drift_summary is not None:
        drift_status = (
            f"Warning: rolling IC has been below {config.DIAG_MIN_IC:.2f} for "
            f"{model_drift_summary.ic_below_threshold_streak} consecutive monthly snapshots."
            if model_drift_summary.drift_flag
            else "Stable: no sustained rolling-IC drift alert is active."
        )
        lines += [
            "",
            "---",
            "",
            "## Model Health",
            "",
            f"- Latest tracked month: **{model_drift_summary.as_of_month}**",
            f"- Rolling {model_drift_summary.window_months}M IC: **{model_drift_summary.rolling_ic:.4f}**",
            f"- Rolling {model_drift_summary.window_months}M Hit Rate: **{model_drift_summary.rolling_hit_rate:.1%}**",
            f"- Rolling {model_drift_summary.window_months}M ECE: **{model_drift_summary.rolling_ece:.1%}**",
            f"- IC breach streak: **{model_drift_summary.ic_below_threshold_streak}** month(s)",
            f"- Status: **{drift_status}**",
        ]

    # v32.2 + v32.3 — Decision Policy Backtest and Heuristic Comparison
    if policy_summary is not None:
        signal_policies_present = [p for p in SIGNAL_POLICIES if p in policy_summary]
        fixed_policies_present = [p for p in FIXED_POLICIES if p in policy_summary]
        model_policy = policy_summary.get("sign_hold_vs_sell")  # primary signal policy

        _policy_labels: dict[str, str] = {
            "always_sell_100": "Sell 100% (always)",
            "always_sell_50": "Sell 50% (always)",
            "always_hold_100": "Hold 100% (always)",
            "sign_hold_vs_sell": "Model: sign (hold if pred > 0)",
            "tiered_25_50_100": "Model: tiered 25/50/100",
            "neutral_band_2pct": "Model: neutral band ±2%",
            "neutral_band_3pct": "Model: neutral band ±3%",
        }

        lines += [
            "",
            "---",
            "",
            "## Decision Policy Backtest",
            "",
            "> OOS performance of each decision policy applied to all historical "
            "model predictions.  "
            "\"Mean Return\" is the portfolio-weighted realized relative return per "
            "vesting event.  \"Cumulative\" is the sum across all events.  "
            "\"Capture Ratio\" is the fraction of oracle (always hold when positive) "
            "gains captured.  N = number of OOS events.",
            "",
            "### Fixed Heuristic Baselines",
            "",
            "| Policy | N | Mean Return | Cumulative | Capture Ratio |",
            "|--------|---|-------------|------------|---------------|",
        ]
        for p in fixed_policies_present:
            s = policy_summary[p]
            lines.append(
                f"| {_policy_labels.get(p, p)} | {s.n_obs} "
                f"| {s.mean_policy_return:+.2%} "
                f"| {s.cumulative_policy_return:+.2%} "
                f"| {s.capture_ratio:.1%} |"
            )

        lines += [
            "",
            "### Model-Driven Policies vs. Heuristics",
            "",
            "| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |",
            "|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|",
        ]
        for p in signal_policies_present:
            s = policy_summary[p]
            up_sell = f"{s.uplift_vs_sell_all:+.2%}" if not np.isnan(s.uplift_vs_sell_all) else "n/a"
            up_hold = f"{s.uplift_vs_hold_all:+.2%}" if not np.isnan(s.uplift_vs_hold_all) else "n/a"
            up_50 = f"{s.uplift_vs_sell_50:+.2%}" if not np.isnan(s.uplift_vs_sell_50) else "n/a"
            capture = f"{s.capture_ratio:.1%}" if not np.isnan(s.capture_ratio) else "n/a"
            lines.append(
                f"| {_policy_labels.get(p, p)} | {s.n_obs} "
                f"| {s.mean_policy_return:+.2%} "
                f"| {s.cumulative_policy_return:+.2%} "
                f"| {up_sell} | {up_hold} | {up_50} | {capture} |"
            )
        lines += [""]

    # v34.0 — Portfolio Optimizer Status (Black-Litterman diagnostic shadow run)
    lines += [
        "",
        "---",
        "",
        "## Portfolio Optimizer Status",
        "",
    ]
    if bl_diagnostics is None:
        lines += ["> Black-Litterman optimizer: not run (insufficient data or error during build).", ""]
    elif bl_diagnostics.fallback_used:
        reason_str = bl_diagnostics.fallback_reason or "unknown"
        lines += [
            f"> ⚠️ **Optimizer fallback active** — Black-Litterman optimization could not converge "
            f"(`{reason_str}`).  Portfolio weights fall back to equal-weight allocation.  "
            f"This does not affect the primary recommendation; it is a diagnostic indicator.",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |",
            f"| Status | ⚠️ Fallback — {reason_str} |",
            f"| Active benchmarks | {bl_diagnostics.n_active_tickers} |",
            f"| View tickers incorporated | {bl_diagnostics.n_view_tickers} |",
            "",
        ]
    else:
        lines += [
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |",
            f"| Status | ✅ Converged |",
            f"| Active benchmarks | {bl_diagnostics.n_active_tickers} |",
            f"| View tickers incorporated | {bl_diagnostics.n_view_tickers} |",
            "",
        ]

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    n_outperform = (signals["signal"] == "OUTPERFORM").sum() if not signals.empty else 0
    n_total = len(signals)
    outperform_frac = f"{n_outperform}/{n_total} ({n_outperform / n_total:.0%})" if n_total else "0/0"

    if recommendation_mode["mode"] == "defer-to-tax-default":
        lines += [
            f"The point forecast leans {consensus.lower()}, and {outperform_frac} benchmarks favour outperformance, "
            f"but the broader quality gate is failing.",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.",
        ]
    elif recommendation_mode["mode"] == "monitoring-only":
        lines += [
            f"The point forecast leans {consensus.lower()}, and {outperform_frac} benchmarks favour outperformance, "
            "but the signal should be treated as monitoring information rather than an execution-grade edge.",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** unless future monthly runs improve the quality gate.",
        ]
    elif consensus == "OUTPERFORM":
        lines += [
            f"The ensemble has **{composite_confidence_tier.lower()} conviction** that PGR "
            f"will outperform a diversified ETF portfolio over the next 6 months.  "
            f"{outperform_frac} benchmarks favour outperformance.",
            "",
            f"Recommended action at next vesting event: **HOLD {1 - sell_pct:.0%}** of vesting RSUs.",
        ]
    elif consensus == "UNDERPERFORM":
        lines += [
            f"The ensemble predicts PGR will underperform the benchmark portfolio over the next "
            f"6 months ({composite_confidence_tier.lower()} conviction).  "
            f"Only {outperform_frac} benchmarks favour outperformance.",
            "",
            f"Recommended action at next vesting event: **SELL {sell_pct:.0%}** of vesting RSUs and diversify.",
        ]
    else:
        lines += [
            f"Model signal is weak (mean IC below threshold or mixed directional signals).  "
            f"{outperform_frac} benchmarks favour outperformance.",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** for risk management.",
        ]

    # Recommendation-layer support sections
    lines += [
        "",
        "---",
        "",
    ]

    lines += _build_vest_decision_lines(next_vest_summary, recommendation_mode, sell_pct)
    if existing_holdings:
        lines += build_existing_holdings_markdown_lines(existing_holdings)
    if redeploy_buckets:
        lines += build_redeploy_markdown_lines(redeploy_buckets)
    if redeploy_portfolio:
        lines += render_redeploy_portfolio_markdown_lines(redeploy_portfolio)
    if (
        config.RECOMMENDATION_LAYER_MODE in {"live_with_shadow", "shadow_promoted"}
        and live_summary is not None
        and shadow_summary is not None
    ):
        lines += build_shadow_check_lines(
            live_summary,
            shadow_summary,
            active_path="shadow" if config.RECOMMENDATION_LAYER_MODE == "shadow_promoted" else "live",
        )

    # Per-benchmark table — include confidence columns when available
    lines += [
        "## Per-Benchmark Signals",
        "",
        "- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.",
        "- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.",
        "",
    ]

    show_cal_col = has_calibrated and "calibrated_prob_outperform" in signals.columns
    show_ci_col = "ci_lower" in signals.columns and not signals.empty

    if has_confidence and show_cal_col and show_ci_col:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|",
        ]
    elif has_confidence and show_cal_col:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----|----------|--------|--------|------------|--------|",
        ]
    elif has_confidence and show_ci_col:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(Outperform) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----------|----------|----|----------|---------------|------------|--------|",
        ]
    elif has_confidence:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | IC | Hit Rate | P(Outperform) | Confidence | Signal |",
            "|-----------|----------------|-------------|----------------|----|----------|---------------|------------|--------|",
        ]
    else:
        lines += [
            "| Benchmark | Benchmark Role | Description | Predicted Return | IC | Hit Rate | Signal |",
            "|-----------|----------------|-------------|----------------|----|----------|--------|",
        ]

    if not signals.empty:
        for ticker, row in signals.iterrows():
            desc = _ETF_DESCRIPTIONS.get(str(ticker), str(ticker))
            role = benchmark_role_for_ticker(str(ticker))["role"]
            pred = f"{row['predicted_relative_return']:+.2%}" if not pd.isna(row.get("predicted_relative_return")) else "n/a"
            ic_val = f"{row['ic']:.4f}" if not pd.isna(row.get("ic")) else "n/a"
            hr_val = f"{row['hit_rate']:.1%}" if not pd.isna(row.get("hit_rate")) else "n/a"
            sig = row.get("signal", "N/A")
            ci_lo_str = f"{row['ci_lower']:+.2%}" if show_ci_col and not pd.isna(row.get("ci_lower")) else "n/a"
            ci_hi_str = f"{row['ci_upper']:+.2%}" if show_ci_col and not pd.isna(row.get("ci_upper")) else "n/a"
            if has_confidence and show_cal_col and show_ci_col:
                prob_raw = f"{row['prob_outperform']:.1%}" if not pd.isna(row.get("prob_outperform")) else "n/a"
                prob_cal = f"{row['calibrated_prob_outperform']:.1%}" if not pd.isna(row.get("calibrated_prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ci_lo_str} | {ci_hi_str} | {ic_val} | {hr_val} | {prob_raw} | {prob_cal} | {tier} | {sig} |")
            elif has_confidence and show_cal_col:
                prob_raw = f"{row['prob_outperform']:.1%}" if not pd.isna(row.get("prob_outperform")) else "n/a"
                prob_cal = f"{row['calibrated_prob_outperform']:.1%}" if not pd.isna(row.get("calibrated_prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ic_val} | {hr_val} | {prob_raw} | {prob_cal} | {tier} | {sig} |")
            elif has_confidence and show_ci_col:
                prob = f"{row['prob_outperform']:.1%}" if not pd.isna(row.get("prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ci_lo_str} | {ci_hi_str} | {ic_val} | {hr_val} | {prob} | {tier} | {sig} |")
            elif has_confidence:
                prob = f"{row['prob_outperform']:.1%}" if not pd.isna(row.get("prob_outperform")) else "n/a"
                tier = row.get("confidence_tier", "n/a")
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ic_val} | {hr_val} | {prob} | {tier} | {sig} |")
            else:
                lines.append(f"| {ticker} | {role} | {desc} | {pred} | {ic_val} | {hr_val} | {sig} |")

    # v7.3 — Tax Context section
    lines += _build_tax_context_lines(mean_predicted, mean_cal_prob, as_of=as_of)

    lines += [
        "",
        "---",
        "",
        f"*Generated by `scripts/monthly_decision.py`*{'  [DRY RUN]' if dry_run else ''}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {path}")


def _write_signals_csv(out_dir: Path, signals: pd.DataFrame) -> None:
    """Write per-benchmark signals to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "signals.csv"
    if signals.empty:
        pd.DataFrame(columns=[
            "benchmark", "predicted_relative_return", "ic", "hit_rate",
            "signal", "prob_outperform", "confidence_tier", "benchmark_role",
        ]).to_csv(path, index=False)
    else:
        export = signals.reset_index()
        export["benchmark_role"] = export["benchmark"].map(
            lambda ticker: benchmark_role_for_ticker(str(ticker))["role"]
        )
        export.to_csv(path, index=False)
    print(f"  Wrote {path}")


def _write_benchmark_quality_csv(
    out_dir: Path,
    benchmark_quality_df: pd.DataFrame | None,
) -> None:
    """Write ensemble OOS benchmark-quality diagnostics to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark_quality.csv"
    columns = [
        "benchmark",
        "n_obs",
        "oos_r2",
        "nw_ic",
        "nw_p_value",
        "hit_rate",
        "cw_t_stat",
        "cw_p_value",
        "cw_mean_adjusted_differential",
        "r2_flag",
        "ic_flag",
        "hr_flag",
    ]
    if benchmark_quality_df is None or benchmark_quality_df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        benchmark_quality_df.loc[:, columns].to_csv(path, index=False)
    print(f"  Wrote {path}")


def _write_consensus_shadow_csv(
    out_dir: Path,
    consensus_shadow_df: pd.DataFrame | None,
) -> None:
    """Write the v74 shadow consensus comparison table to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "consensus_shadow.csv"
    columns = [
        "variant",
        "n_benchmarks",
        "consensus",
        "mean_predicted_return",
        "mean_ic",
        "mean_hit_rate",
        "mean_prob_outperform",
        "confidence_tier",
        "weight_mode",
        "score_col",
        "lambda_mix",
        "top_benchmark",
        "top_benchmark_weight",
        "recommendation_mode",
        "recommended_sell_pct",
        "is_live_path",
    ]
    if consensus_shadow_df is None or consensus_shadow_df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        consensus_shadow_df.loc[:, columns].to_csv(path, index=False)
    print(f"  Wrote {path}")


def _append_decision_log(
    as_of: date,
    run_date: date,
    consensus: str,
    sell_pct: float,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    dry_run: bool,
    _log_path_override: Path | None = None,
) -> None:
    """Append one row to the persistent decision_log.md.

    Inserts after the last data row of the ## Log table, identified by the
    separator line immediately below the log table's header row.  This prevents
    orphaned rows appearing in the Column Definitions or other sections.

    v7.3 fix: replaced the previous "find last | line in entire file" approach
    which incorrectly anchored to rows in the Column Definitions table.

    Args:
        _log_path_override: If provided, write to this path instead of the
                            default results/monthly_decisions/decision_log.md.
                            Used in tests only.
    """
    log_path = _log_path_override or (
        Path("results") / "monthly_decisions" / "decision_log.md"
    )
    if not log_path.exists():
        return

    content = log_path.read_text(encoding="utf-8")

    new_row = (
        f"| {as_of} | {run_date} | {consensus} | {sell_pct:.0%} "
        f"| {mean_predicted:+.2%} | {mean_ic:.4f} | {mean_hr:.1%} "
        f"| {'[DRY RUN]' if dry_run else ''} |"
    )
    row_prefix = (
        f"| {as_of} | {run_date} | {consensus} | {sell_pct:.0%} "
        f"| {mean_predicted:+.2%} | {mean_ic:.4f} | {mean_hr:.1%} |"
    )
    if new_row in content or any(line.startswith(row_prefix) for line in content.splitlines()):
        print(f"  Decision log already contains this row; skipping append.")
        return

    # Replace the placeholder on first use.
    placeholder = "| *(first entry will appear here after the first automated run)* |"
    if placeholder in content:
        content = content.replace(placeholder, new_row, 1)
        log_path.write_text(content, encoding="utf-8")
        print(f"  Appended to {log_path}")
        return

    # Locate the log table by its fixed separator line.  The log table header
    # is the only table whose separator contains "Consensus Signal".  Find the
    # last data row within that table (before the next "---" section divider or
    # end of file) and insert after it.
    LOG_SEPARATOR = "|------------|----------|-----------------|"
    lines = content.splitlines()

    sep_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(LOG_SEPARATOR)),
        -1,
    )
    if sep_idx < 0:
        # Fallback: can't find the log table — append at end of file.
        content = content.rstrip("\n") + "\n" + new_row + "\n"
        log_path.write_text(content, encoding="utf-8")
        print(f"  Appended to {log_path} (fallback: no log table separator found)")
        return

    # Find the last log-table row: scan forward from sep_idx while lines start
    # with "| " and stop at the first "---" divider or non-table line.
    last_data_idx = sep_idx
    for i in range(sep_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("---") or (stripped and not stripped.startswith("|")):
            break
        if stripped.startswith("|"):
            last_data_idx = i

    lines.insert(last_data_idx + 1, new_row)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Appended to {log_path}")


# ---------------------------------------------------------------------------
# P2.7 — Calibration reliability diagram
# ---------------------------------------------------------------------------

def _plot_calibration_curve(
    out_dir: Path,
    cal_probs: np.ndarray,
    cal_outcomes: np.ndarray,
    cal_result: CalibrationResult,
    n_bins: int | None = None,
) -> Path | None:
    """Generate a reliability diagram (calibration curve) and save to disk.

    The reliability diagram plots predicted probability (x-axis) against the
    fraction of positive outcomes in each bin (y-axis).  A perfectly calibrated
    model lies on the diagonal.  Points above the diagonal are under-confident;
    points below are over-confident.

    The plot includes:
      - Binned calibration curve (blue circles, connected)
      - Diagonal perfect-calibration reference (dashed grey)
      - Histogram of predicted probabilities (bottom subpanel)
      - ECE annotation with 95% bootstrap CI

    Args:
        out_dir:      Output directory (YYYY-MM folder).
        cal_probs:    Array of pooled calibrated P(outperform) values.
        cal_outcomes: Array of corresponding binary outcomes (1 = outperform).
        cal_result:   CalibrationResult for ECE annotation.
        n_bins:       Number of probability bins (default: config.CALIBRATION_N_BINS).

    Returns:
        Path to the saved PNG, or ``None`` if insufficient data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [calibration plot] matplotlib not available — skipping plot.")
        return None

    if len(cal_probs) < 4 or cal_result.method == "uncalibrated":
        return None

    if n_bins is None:
        n_bins = getattr(config, "CALIBRATION_N_BINS", 10)

    # Compute reliability bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    fraction_pos: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(float(cal_probs[mask].mean()))
        fraction_pos.append(float(cal_outcomes[mask].mean()))
        bin_counts.append(int(mask.sum()))

    if len(bin_centers) < 2:
        return None

    # ---- Plot ----
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Reliability curve
    ax_main.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.2, label="Perfect calibration")
    ax_main.plot(bin_centers, fraction_pos, "o-", color="#1f77b4", linewidth=2,
                 markersize=7, label="Model calibration")

    # Annotate ECE
    ece_txt = (
        f"ECE = {cal_result.ece:.1%}  "
        f"[95% CI: {cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}]"
    )
    ax_main.text(
        0.04, 0.95, ece_txt,
        transform=ax_main.transAxes,
        fontsize=9, verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Mean predicted probability")
    ax_main.set_ylabel("Fraction of positives (actual)")
    ax_main.set_title(
        f"Calibration Reliability Diagram  "
        f"(n={cal_result.n_obs:,} obs, {cal_result.method})"
    )
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    ax_hist.hist(cal_probs, bins=n_bins, range=(0, 1), color="#1f77b4", alpha=0.6)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "calibration_curve.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  Wrote {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Diagnostic OOS evaluation report (v4.3.1)
# ---------------------------------------------------------------------------

def _flag(value: float, good: float, marginal: float, higher_is_better: bool = True) -> str:
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


def _write_diagnostic_report(
    out_dir: Path,
    as_of: date,
    ensemble_results: dict,
    target_horizon_months: int = 6,
    cal_result: CalibrationResult | None = None,
    signals: pd.DataFrame | None = None,
    obs_feature_report: dict | None = None,
    representative_cpcv: CPCVResult | None = None,
    conformal_coverage_summary: ConformalCoverageBacktest | None = None,
    importance_stability: FeatureImportanceStability | None = None,
    vif_series: pd.Series | None = None,
    benchmark_quality_df: pd.DataFrame | None = None,
) -> None:
    """
    Write diagnostic.md alongside recommendation.md.

    Aggregates OOS predictions and realized returns across all benchmarks from
    the deployed inverse-variance ensemble, then computes:

    - Campbell-Thompson OOS R² (model vs. naive historical mean)
    - Newey-West HAC-adjusted Spearman IC (accounts for overlapping return windows)
    - Clark-West adjusted predictive-value diagnostics
    - Per-benchmark health table from ensemble OOS predictions

    Args:
        out_dir:                Output directory (YYYY-MM folder).
        as_of:                  As-of date for the report header.
        ensemble_results:       Dict of ETF ticker → EnsembleWFOResult from
                                ``run_ensemble_benchmarks()``.
        target_horizon_months:  Forward return horizon used during training.
        obs_feature_report:     Output of ``compute_obs_feature_ratio()`` for the
                                feature matrix used in this monthly run.
        representative_cpcv:    Optional representative CPCV diagnostic
                                (currently VTI + elasticnet).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnostic.md"

    nw_lags = target_horizon_months - 1  # Newey-West overlap lags (5 for 6M)
    agg_predicted, agg_realized, computed_quality_df = _build_benchmark_quality_frame(
        ensemble_results=ensemble_results,
        target_horizon_months=target_horizon_months,
    )
    if benchmark_quality_df is None:
        benchmark_quality_df = computed_quality_df

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    if len(agg_realized) < 4:
        # Not enough data — write a minimal report
        lines = [
            f"# PGR Diagnostic Report — {as_of.strftime('%B %Y')}",
            "",
            "> ⚠️ Insufficient OOS observations for aggregate diagnostics "
            f"(need ≥ 4, got {len(agg_realized)}).",
            "",
            "*Generated by `scripts/monthly_decision.py`*",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote {path} (insufficient data)")
        return

    aggregate_summary = summarize_prediction_diagnostics(
        agg_predicted,
        agg_realized,
        target_horizon_months=target_horizon_months,
    )
    oos_r2 = float(aggregate_summary["oos_r2"])
    nw_ic = float(aggregate_summary["nw_ic"])
    nw_pval = float(aggregate_summary["nw_p_value"])
    agg_hit = float(aggregate_summary["hit_rate"])
    cw_t_stat = float(aggregate_summary["cw_t_stat"])
    cw_p_value = float(aggregate_summary["cw_p_value"])
    n_agg = int(aggregate_summary["n_obs"])

    r2_flag = _flag(oos_r2, config.DIAG_MIN_OOS_R2, 0.005)
    ic_flag_agg = _flag(nw_ic, config.DIAG_MIN_IC, 0.03)
    hr_flag_agg = _flag(agg_hit, config.DIAG_MIN_HIT_RATE, 0.52)
    sig_marker = "✅ p < 0.05" if (not np.isnan(nw_pval) and nw_pval < 0.05) else (
        "⚠️ p < 0.10" if (not np.isnan(nw_pval) and nw_pval < 0.10) else "❌ not sig."
    )
    cw_marker = "? p < 0.05" if (not np.isnan(cw_p_value) and cw_p_value < 0.05) else (
        "?? p < 0.10" if (not np.isnan(cw_p_value) and cw_p_value < 0.10) else "? not sig."
    )

    sig_marker = "\u2705 p < 0.05" if (not np.isnan(nw_pval) and nw_pval < 0.05) else (
        "\u26a0\ufe0f p < 0.10" if (not np.isnan(nw_pval) and nw_pval < 0.10) else "\u274c not sig."
    )
    cw_marker = "\u2705 p < 0.05" if (not np.isnan(cw_p_value) and cw_p_value < 0.05) else (
        "\u26a0\ufe0f p < 0.10" if (not np.isnan(cw_p_value) and cw_p_value < 0.10) else "\u274c not sig."
    )

    # ------------------------------------------------------------------
    # Build markdown
    # ------------------------------------------------------------------
    cpcv_value = "N/A"
    cpcv_status = "—"
    cpcv_threshold = f"≥ {config.DIAG_CPCV_MIN_POSITIVE_PATHS}/28"
    cpcv_note_lines = [
        "> **CPCV note (v5.0/v7.4):** Representative CPCV uses ElasticNet vs VTI",
        "> as a monthly stability check. Full 4-model × 20-benchmark CPCV remains",
        "> available on demand via `run_cpcv()` in `wfo_engine.py`.",
    ]
    if representative_cpcv is not None and representative_cpcv.n_paths > 0:
        cpcv_value = (
            f"{representative_cpcv.n_positive_paths}/{representative_cpcv.n_paths} "
            f"({representative_cpcv.positive_path_fraction:.1%})"
        )
        cpcv_status = {
            "GOOD": "✅",
            "MARGINAL": "⚠️",
            "FAIL": "❌",
            "UNKNOWN": "⚠️",
        }.get(representative_cpcv.stability_verdict, "⚠️")
        scaled_positive_threshold = math.ceil(
            config.DIAG_CPCV_MIN_POSITIVE_PATHS * representative_cpcv.n_paths / 28
        )
        cpcv_threshold = f"≥ {scaled_positive_threshold}/{representative_cpcv.n_paths}"
        cpcv_note_lines = [
            f"> **Representative CPCV:** benchmark={representative_cpcv.benchmark}, "
            f"model={representative_cpcv.model_type}, paths={representative_cpcv.n_paths}, "
            f"mean IC={representative_cpcv.mean_ic:.4f}, IC std={representative_cpcv.ic_std:.4f}.",
            f"> Stability verdict: {representative_cpcv.stability_verdict}. "
            f"Scaled monthly threshold: ≥ {scaled_positive_threshold}/{representative_cpcv.n_paths} "
            f"(maps from the full C(8,2) standard of ≥ {config.DIAG_CPCV_MIN_POSITIVE_PATHS}/28 positive paths).",
        ]

    obs_feature_lines: list[str] = []
    if obs_feature_report is not None:
        obs_feature_lines = [
            "",
            "## Feature Governance",
            "",
            "| Metric | Value | Status | Threshold (Good) |",
            "|--------|-------|--------|-----------------|",
            f"| Full obs/feature ratio | {obs_feature_report['ratio']:.2f} | "
            f"{ {'OK': '✅', 'WARNING': '⚠️', 'FAIL': '❌'}.get(obs_feature_report['verdict'], '⚠️') } | ≥ 4.0 |",
            f"| Per-fold obs/feature ratio | {obs_feature_report['per_fold_ratio']:.2f} | "
            f"{ {'OK': '✅', 'WARNING': '⚠️', 'FAIL': '❌'}.get(obs_feature_report['verdict'], '⚠️') } | ≥ 4.0 |",
            f"| Features in monthly run | {obs_feature_report['n_features']} | — | — |",
            f"| Fully populated observations | {obs_feature_report['n_obs']} | — | — |",
            "",
            f"> {obs_feature_report['message']}",
            "",
        ]

        # v32.0 — Feature importance stability subsection
        if importance_stability is not None:
            _stab_icon = {
                "STABLE": "✅",
                "MARGINAL": "⚠️",
                "UNSTABLE": "❌",
            }.get(importance_stability.verdict, "⚠️")
            obs_feature_lines += [
                "### Feature Importance Stability",
                "",
                "| Metric | Value | Status | Threshold (Good) |",
                "|--------|-------|--------|-----------------|",
                f"| Mean consecutive-fold Spearman ρ | "
                f"{importance_stability.stability_score:.4f} | "
                f"{_stab_icon} {importance_stability.verdict} | ≥ 0.70 |",
                f"| Folds included | {importance_stability.n_folds} | — | — |",
                "",
                "> Stability score measures mean pairwise Spearman rank-correlation "
                "between consecutive WFO fold importance rankings. "
                "A score < 0.40 indicates unstable feature rankings; "
                "model predictions may be driven by different features each period.",
                "",
                "**Top 10 features by mean WFO rank:**",
                "",
                "| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |",
                "|------|---------|-----------|----------|----------------|",
            ]
            top10 = importance_stability.per_feature.head(10)
            for feat, row in top10.iterrows():
                obs_feature_lines.append(
                    f"| {int(row['mean_rank']):.0f} | {feat} | "
                    f"{row['mean_rank']:.1f} | {row['rank_std']:.1f} | "
                    f"{row['mean_importance']:.4f} |"
                )
            obs_feature_lines += [""]

        # v32.1 — VIF multicollinearity subsection
        if vif_series is not None and not vif_series.empty:
            high_vif = vif_series[vif_series > config.VIF_HIGH_THRESHOLD]
            warn_vif = vif_series[
                (vif_series > config.VIF_WARN_THRESHOLD) &
                (vif_series <= config.VIF_HIGH_THRESHOLD)
            ]
            if not high_vif.empty:
                vif_overall = "❌ HIGH multicollinearity"
            elif not warn_vif.empty:
                vif_overall = "⚠️ MODERATE multicollinearity"
            else:
                vif_overall = "✅ LOW multicollinearity"
            obs_feature_lines += [
                "### Multicollinearity (VIF)",
                "",
                f"**Overall:** {vif_overall}  ",
                f"Features flagged high (VIF > {config.VIF_HIGH_THRESHOLD:.0f}): "
                f"**{len(high_vif)}**  ",
                f"Features flagged moderate (VIF {config.VIF_WARN_THRESHOLD:.0f}–"
                f"{config.VIF_HIGH_THRESHOLD:.0f}): **{len(warn_vif)}**  ",
                "",
                "> VIF measures how much variance in a feature is explained by the "
                "other features. VIF > 10 indicates severe multicollinearity and "
                "may cause unstable coefficient estimates.",
                "",
                "**All features by VIF (descending):**",
                "",
                "| Feature | VIF | Status |",
                "|---------|-----|--------|",
            ]
            for feat, vif_val in vif_series.items():
                if vif_val > config.VIF_HIGH_THRESHOLD:
                    flag = "❌ HIGH"
                elif vif_val > config.VIF_WARN_THRESHOLD:
                    flag = "⚠️ MODERATE"
                else:
                    flag = "✅ OK"
                obs_feature_lines.append(
                    f"| {feat} | {vif_val:.2f} | {flag} |"
                )
            obs_feature_lines += [""]

        obs_feature_lines += [
            "---",
            "",
        ]

    lines = [
        f"# PGR Diagnostic Report — {as_of.strftime('%B %Y')}",
        "",
        f"**As-of Date:** {as_of}  ",
        f"**Horizon:** {target_horizon_months}M  ",
        f"**OOS observations (aggregate):** {n_agg}  ",
        f"**Newey-West lags:** {nw_lags} (accounts for {target_horizon_months - 1}-month "
        "return-window overlap)  ",
        "",
        "---",
        "",
        "## Aggregate Model Health",
        "",
        "| Metric | Value | Status | Threshold (Good) |",
        "|--------|-------|--------|-----------------|",
        f"| OOS R² (Campbell-Thompson) | {oos_r2:.4f} ({oos_r2:.2%}) | {r2_flag} | ≥ 2.00% |",
        f"| IC (Newey-West HAC) | {nw_ic:.4f} | {ic_flag_agg} | ≥ 0.07 |",
        f"| IC significance | {nw_pval:.4f} | {sig_marker} | p < 0.05 |",
        f"| Clark-West t-stat | {cw_t_stat:.4f} | {cw_marker} | p < 0.05 |",
        f"| Clark-West p-value | {cw_p_value:.4f} | {cw_marker} | p < 0.05 |",
        f"| Hit Rate | {agg_hit:.1%} | {hr_flag_agg} | ≥ 55.0% |",
        f"| CPCV Positive Paths | {cpcv_value} | {cpcv_status} | {cpcv_threshold} |",
        "",
        *cpcv_note_lines,
        "",
        "---",
        "",
        *obs_feature_lines,
        "## Calibration Phase",
        "",
        "| Phase | Description | Status |",
        "|-------|-------------|--------|",
    ]

    # Determine which phase is active based on cal_result
    if cal_result is None or cal_result.method == "uncalibrated":
        phase1_status = "✅ Active"
        phase2_status = f"⏳ Activates at n ≥ {config.CALIBRATION_MIN_OBS_PLATT}"
        phase3_status = f"⏳ Activates at n ≥ {config.CALIBRATION_MIN_OBS_ISOTONIC}"
    elif cal_result.method == "platt":
        phase1_status = "⬛ Superseded"
        phase2_status = (
            f"✅ Active (n={cal_result.n_obs:,}  ECE={cal_result.ece:.1%} "
            f"[{cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}])"
        )
        phase3_status = f"⏳ Activates at n ≥ {config.CALIBRATION_MIN_OBS_ISOTONIC}"
    else:  # isotonic
        phase1_status = "⬛ Superseded"
        phase2_status = "⬛ Superseded by Phase 3"
        phase3_status = (
            f"✅ Active (n={cal_result.n_obs:,}  ECE={cal_result.ece:.1%} "
            f"[{cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}])"
        )

    lines += [
        f"| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | {phase1_status} |",
        f"| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | {phase2_status} |",
        f"| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | {phase3_status} |",
        "",
        "---",
        "",
        "## Conformal Prediction Intervals",
        "",
        f"**Method:** {config.CONFORMAL_METHOD.upper()} "
        f"({'Adaptive Conformal Inference — adjusts α_t for distribution shift' if config.CONFORMAL_METHOD == 'aci' else 'Split Conformal — finite-sample corrected quantile of absolute residuals'})  ",
        f"**Nominal Coverage:** {config.CONFORMAL_COVERAGE:.0%}  ",
        "",
    ]

    # Build per-benchmark coverage table from signals CI columns
    has_ci_data = (
        signals is not None
        and not signals.empty
        and "ci_empirical_coverage" in signals.columns
        and "ci_n_calibration" in signals.columns
    )
    if has_ci_data:
        valid_rows = signals[signals["ci_n_calibration"] > 0]
        if not valid_rows.empty:
            mean_emp_cov = float(valid_rows["ci_empirical_coverage"].mean())
            cov_flag = _flag(mean_emp_cov, config.CONFORMAL_COVERAGE, config.CONFORMAL_COVERAGE - 0.05)
            lines += [
                f"**Mean empirical coverage:** {mean_emp_cov:.1%} "
                f"(target ≥ {config.CONFORMAL_COVERAGE:.0%}) {cov_flag}  ",
                "",
            ]
            if conformal_coverage_summary is not None:
                trailing_cov_flag = _flag(
                    abs(conformal_coverage_summary.trailing_coverage_gap),
                    good=0.05,
                    marginal=0.10,
                    higher_is_better=False,
                )
                lines += [
                    f"**Mean trailing 12-point empirical coverage:** "
                    f"{conformal_coverage_summary.trailing_empirical_coverage:.1%} "
                    f"(gap {conformal_coverage_summary.trailing_coverage_gap:+.1%} vs nominal) "
                    f"{trailing_cov_flag}  ",
                    "",
                ]
            lines += [
                "| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |",
                "|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|",
            ]
            for ticker, row in signals.iterrows():
                if pd.isna(row.get("ci_lower")):
                    continue
                desc = _ETF_DESCRIPTIONS.get(str(ticker), str(ticker))
                pred_str = f"{row['predicted_relative_return']:+.2%}" if not pd.isna(row.get("predicted_relative_return")) else "n/a"
                ci_lo = f"{row['ci_lower']:+.2%}"
                ci_hi = f"{row['ci_upper']:+.2%}"
                ci_w = f"{row['ci_width']:.2%}"
                emp_cov = f"{row['ci_empirical_coverage']:.1%}"
                trailing_cov = (
                    f"{row['ci_trailing_empirical_coverage']:.1%}"
                    if not pd.isna(row.get("ci_trailing_empirical_coverage"))
                    else "n/a"
                )
                n_cal = int(row["ci_n_calibration"])
                emp_flag = "✅" if row["ci_empirical_coverage"] >= config.CONFORMAL_COVERAGE else "⚠️"
                lines.append(
                    f"| {ticker} | {desc} | {pred_str} | {ci_lo} | {ci_hi} | {ci_w} | "
                    f"{emp_cov} {emp_flag} | {trailing_cov} | {n_cal} |"
                )
        else:
            lines.append("> ⚠️ No benchmarks had sufficient calibration data for conformal intervals.")
    else:
        lines.append("> ⚠️ Conformal interval data not available (signals not passed to diagnostic).")

    lines += [
        "",
        "> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate",
        "> larger historical prediction errors.  ACI dynamically adjusts coverage when errors",
        "> cluster (distribution shift), providing stronger guarantees than static split conformal.",
        "",
        "---",
        "",
        "## Per-Benchmark Health",
        "",
        "| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |",
        "|-----------|-------------|-------|--------|-------|----------|------|------|",
    ]

    benchmark_rows = (
        benchmark_quality_df.to_dict("records")
        if benchmark_quality_df is not None and not benchmark_quality_df.empty
        else []
    )
    for row in benchmark_rows:
        desc = _ETF_DESCRIPTIONS.get(str(row["benchmark"]), str(row["benchmark"]))
        lines.append(
            f"| {row['benchmark']} | {desc} | {row['n_obs']} "
            f"| {row['oos_r2']:.2%} | {row['nw_ic']:.4f} "
            f"| {row['hit_rate']:.1%} | {row['cw_t_stat']:.4f} | {row['cw_p_value']:.4f} |"
        )

    # Summary counts
    n_ok_ic = sum(1 for r in benchmark_rows if r["ic_flag"] == "✅")
    n_warn_ic = sum(1 for r in benchmark_rows if r["ic_flag"] == "⚠️")
    n_fail_ic = sum(1 for r in benchmark_rows if r["ic_flag"] == "❌")
    n_ok_hr = sum(1 for r in benchmark_rows if r["hr_flag"] == "✅")
    n_ok_cw = sum(1 for r in benchmark_rows if float(r["cw_p_value"]) < 0.05)

    lines += [
        "",
        f"**IC summary:** {n_ok_ic} ✅  {n_warn_ic} ⚠️  {n_fail_ic} ❌  "
        f"(of {len(benchmark_rows)} benchmarks)  ",
        f"**Hit rate ✅:** {n_ok_hr}/{len(benchmark_rows)} benchmarks above 55% threshold  ",
        f"**Clark-West ✅:** {n_ok_cw}/{len(benchmark_rows)} benchmarks with p < 0.05  ",
        "",
        "---",
        "",
        "## Threshold Reference",
        "",
        "| Metric | Good | Marginal | Failing | Source |",
        "|--------|------|----------|---------|--------|",
        "| OOS R² | > 2% | 0.5–2% | < 0% | Campbell & Thompson (2008) |",
        "| Mean IC | > 0.07 | 0.03–0.07 | < 0.03 | Harvey et al. (2016) |",
        "| Clark-West | p < 0.05 | p < 0.10 | ≥ 0.10 | Clark & West (2007) |",
        "| Hit Rate | > 55% | 52–55% | < 52% | Industry consensus |",
        "| CPCV +paths | ≥ 19/28 | 14–18/28 | < 14/28 | López de Prado (2018) |",
        "| PBO | < 15% | 15–40% | > 40% | Bailey et al. (2014) |",
        "",
        "---",
        "",
        "*Generated by `scripts/monthly_decision.py`*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    as_of_date_str: str | None = None,
    dry_run: bool = False,
    skip_fred: bool = False,
) -> None:
    configure_logging()
    as_of = _resolve_as_of_date(as_of_date_str)
    run_date = date.today()
    layer_mode = config.RECOMMENDATION_LAYER_MODE
    if layer_mode not in config.RECOMMENDATION_LAYER_VALID_MODES:
        logger.warning(
            "[Recommendation Layer] Unknown mode '%s', defaulting to 'shadow_promoted'.",
            layer_mode,
        )
        layer_mode = "shadow_promoted"

    logger.info("%sPGR Monthly Decision - as-of %s", "[DRY RUN] " if dry_run else "", as_of)
    logger.info("Run date: %s", run_date)

    # Idempotency: skip if this month's report already exists
    if _already_ran(as_of) and not dry_run:
        logger.info("Report for %s already exists. Skipping.", as_of.strftime("%Y-%m"))
        return

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)
    db_client.warn_if_db_behind(conn, context="monthly_decision")
    freshness_report = db_client.check_data_freshness(conn, run_date)
    for message in freshness_report["warnings"]:
        logger.warning("[data-freshness] %s", message)

    # Step 1: Refresh FRED data
    _fetch_fred_step(conn, dry_run=dry_run, skip_fred=skip_fred)

    # Step 2: Generate signals (ensemble: ElasticNet + Ridge + BayesianRidge + GBT)
    logger.info("Generating ensemble signals (as-of %s)...", as_of)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", category=ConvergenceWarning)
        signals, ensemble_results, diagnostics = _generate_signals(
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

    # Step 2.5: Calibrate P(outperform) using Platt / isotonic on OOS fold history
    logger.info("Calibrating probabilities...")
    signals, cal_result, cal_probs, cal_outcomes = _calibrate_signals(
        signals, ensemble_results, target_horizon_months=6
    )
    logger.info(
        "Calibration: %s (n=%s OOS obs, ECE=%s)",
        cal_result.method,
        f"{cal_result.n_obs:,}",
        f"{cal_result.ece:.1%}",
    )

    # Step 2.7: Compute conformal prediction intervals (ACI / split conformal)
    logger.info("Computing conformal prediction intervals...")
    signals = _compute_conformal_intervals(signals, ensemble_results)
    conformal_coverage_summary = _summarize_conformal_coverage(signals)
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

    # Step 3: Compute consensus
    consensus, mean_pred, mean_ic, mean_hr, mean_prob, confidence_tier = _consensus_signal(signals)
    aggregate_health = _compute_aggregate_health(ensemble_results, target_horizon_months=6)
    recommendation_mode = _determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        diagnostics.get("representative_cpcv"),
    )
    consensus_shadow_df = _build_v74_shadow_consensus(
        signals,
        aggregate_health,
        diagnostics.get("representative_cpcv"),
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
        candidate_name="production_4_model_ensemble",
        policy_name="current_production_mapping",
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
        shadow_summary, _ = _build_shadow_baseline_summary(
            conn,
            as_of,
            target_horizon_months=6,
        )
    active_recommendation_mode = recommendation_mode
    recommendation_layer_label = "Live production recommendation layer"
    if layer_mode == "live_with_shadow":
        recommendation_layer_label = (
            "Live production recommendation layer + v22 visible cross-check + v13 simpler-baseline cross-check"
        )
    elif layer_mode == "shadow_promoted" and shadow_summary is not None:
        active_recommendation_mode = _mode_payload_from_summary(shadow_summary)
        sell_pct = float(active_recommendation_mode["sell_pct"])
        recommendation_layer_label = (
            "v13.1 promoted simpler diversification-first recommendation layer + "
            "v22 promoted visible cross-check"
        )
    model_drift_summary = _record_model_health_snapshot(
        conn,
        as_of,
        aggregate_health,
        cal_result,
        conformal_coverage_summary,
    )
    existing_holdings = _build_existing_holdings_guidance(conn, as_of)
    redeploy_buckets = _build_redeploy_guidance(conn)
    redeploy_portfolio = _build_redeploy_portfolio(
        conn,
        signals,
        active_recommendation_mode,
    )

    print(f"\n  Consensus signal: {consensus} ({confidence_tier} CONFIDENCE)")
    print(f"  Predicted 6M relative return: {mean_pred:+.2%}")
    print(f"  P(outperform, raw): {mean_prob:.1%}")
    if "calibrated_prob_outperform" in signals.columns and not signals.empty:
        print(f"  P(outperform, calibrated): {mean_cal:.1%}")
    print(f"  Mean IC: {mean_ic:.4f}  |  Mean hit rate: {mean_hr:.1%}")
    if aggregate_health is not None:
        print(f"  Aggregate OOS R^2: {aggregate_health['oos_r2']:.2%}")
    print(f"  Recommendation mode: {active_recommendation_mode['label']}")
    print(f"  Sell %: {sell_pct:.0%}")
    if consensus_shadow_df is not None and not consensus_shadow_df.empty:
        shadow_quality = consensus_shadow_df[
            consensus_shadow_df["variant"] == "quality_weighted"
        ]
        if not shadow_quality.empty:
            row = shadow_quality.iloc[0]
            print(
                "  v74 shadow consensus: "
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

    # Step 4: Write outputs
    # v32.2 + v32.3 — compute policy backtest summary for recommendation.md
    policy_summary: dict[str, PolicySummary] | None = None
    try:
        policy_summary = _compute_policy_summary(ensemble_results)
    except Exception:
        logger.warning(
            "Could not compute policy summary; Decision Policy Backtest section will be omitted",
            exc_info=True,
        )

    # v34.0 — Black-Litterman diagnostic shadow run (Tier 1.4)
    # Calls build_bl_weights as a read-only diagnostic: the result is surfaced
    # in the Portfolio Optimizer Status section of recommendation.md only.
    # It does NOT alter the primary recommendation or redeploy portfolio weights.
    bl_diagnostics: BLDiagnostics | None = None
    try:
        etf_returns_df = _load_etf_monthly_returns(conn, config.ETF_BENCHMARK_UNIVERSE, as_of)
        if not etf_returns_df.empty and len(etf_returns_df) >= 12:
            _, bl_diagnostics = build_bl_weights(  # type: ignore[misc]
                ensemble_results,
                etf_returns_df,
                return_diagnostics=True,
            )
        else:
            logger.info(
                "BL diagnostic skipped: ETF return matrix has only %d rows (need ≥12).",
                len(etf_returns_df),
            )
    except Exception:
        logger.warning("BL diagnostic shadow run failed; Portfolio Optimizer section will show 'not run'", exc_info=True)

    out_dir = _output_dir(as_of)
    _write_recommendation_md(
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
        representative_cpcv=diagnostics.get("representative_cpcv"),
        freshness_report=freshness_report,
        model_drift_summary=model_drift_summary,
        policy_summary=policy_summary,
        bl_diagnostics=bl_diagnostics,
    )
    _write_signals_csv(out_dir, signals)
    _append_decision_log(
        as_of, run_date, consensus, sell_pct, mean_pred, mean_ic, mean_hr, dry_run,
    )

    # Step 5: Write diagnostic OOS evaluation report
    # v32.0 — Compute feature importance stability from the primary model
    # (elasticnet vs VTI, or first available benchmark/model).
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

    print("\nWriting diagnostic report...")
    _write_diagnostic_report(
        out_dir, as_of, ensemble_results,
        target_horizon_months=6, cal_result=cal_result,
        signals=signals,
        obs_feature_report=diagnostics.get("obs_feature_report"),
        representative_cpcv=diagnostics.get("representative_cpcv"),
        conformal_coverage_summary=conformal_coverage_summary,
        importance_stability=importance_stability,
        vif_series=diagnostics.get("vif_series"),
        benchmark_quality_df=(
            aggregate_health.get("benchmark_quality_df")
            if aggregate_health is not None
            else None
        ),
    )
    _write_benchmark_quality_csv(
        out_dir,
        aggregate_health.get("benchmark_quality_df") if aggregate_health is not None else None,
    )
    _write_consensus_shadow_csv(out_dir, consensus_shadow_df)

    # Step 5.1 (P2.7): Calibration reliability diagram
    _plot_calibration_curve(out_dir, cal_probs, cal_outcomes, cal_result)

    snapshot = db_client.get_operational_snapshot(conn)
    manifest_warnings: list[str] = []
    manifest_warnings.extend(freshness_report["warnings"])
    if aggregate_health is not None and aggregate_health["oos_r2"] < config.DIAG_MIN_OOS_R2:
        manifest_warnings.append(
            f"Aggregate OOS R^2 below threshold: {aggregate_health['oos_r2']:.2%} < {config.DIAG_MIN_OOS_R2:.2%}."
        )
    if diagnostics.get("representative_cpcv") is not None:
        verdict = diagnostics["representative_cpcv"].stability_verdict
        if verdict == "FAIL":
            manifest_warnings.append("Representative CPCV verdict is FAIL.")
    if diagnostics.get("obs_feature_report") is not None:
        obs_report = diagnostics["obs_feature_report"]
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
    if model_drift_summary is not None and model_drift_summary.drift_flag:
        manifest_warnings.append(
            "Rolling model IC drift alert active: "
            f"{model_drift_summary.ic_below_threshold_streak} consecutive monthly "
            f"snapshots below {config.DIAG_MIN_IC:.2f}."
        )

    # v35.1: evaluate retrain trigger and record to audit log
    _evaluate_and_record_retrain_trigger(conn, model_drift_summary)

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
        shadow_quality = consensus_shadow_df[
            consensus_shadow_df["variant"] == "quality_weighted"
        ]
        if not shadow_quality.empty:
            shadow_row = shadow_quality.iloc[0]
            if str(shadow_row["recommendation_mode"]) != str(recommendation_mode["label"]):
                manifest_warnings.append(
                    "The v74 quality-weighted consensus shadow path disagrees with the live recommendation mode."
                )
            elif abs(float(shadow_row["recommended_sell_pct"]) - float(recommendation_mode["sell_pct"])) > 1e-9:
                manifest_warnings.append(
                    "The v74 quality-weighted consensus shadow path suggests a different sell percentage."
                )

    manifest = build_run_manifest(
        workflow_name="monthly_decision",
        script_name="scripts/monthly_decision.py",
        as_of_date=as_of,
        schema_version=snapshot["schema_version"],
        latest_dates=snapshot["latest_dates"],
        row_counts=snapshot["row_counts"],
        warnings=manifest_warnings,
        outputs=[
            str(out_dir / "recommendation.md"),
            str(out_dir / "diagnostic.md"),
            str(out_dir / "signals.csv"),
            str(out_dir / "benchmark_quality.csv"),
            str(out_dir / "consensus_shadow.csv"),
        ],
        artifact_classification="production",
        cwd=str(Path(__file__).parent.parent),
    )
    write_run_manifest(out_dir, manifest)

    conn.close()
    logger.info("Done. Results written to %s/", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR v3.0 monthly decision report generator."
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        help="Override the as-of date (default: today or next business day after 20th).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate outputs without making HTTP calls or committing to git.",
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip the FRED data fetch step.",
    )
    args = parser.parse_args()
    main(
        as_of_date_str=args.as_of,
        dry_run=args.dry_run,
        skip_fred=args.skip_fred,
    )


