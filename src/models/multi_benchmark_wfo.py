"""
Multi-benchmark Walk-Forward Optimization runner for v2/v3 relative return models.

Runs one independent WFO model per ETF benchmark, using the PGR-minus-ETF
relative return as the target variable.  Each model is entirely separate —
no regularization or feature weights are shared across benchmarks.

Primary entry points:
  - ``run_all_benchmarks``: trains 20 WFO models (one per ETF column in the
    relative return matrix) and returns a dict keyed by ETF ticker.
  - ``run_ensemble_benchmarks`` (v3.1): trains 3 models per benchmark
    (ElasticNet, Ridge, BayesianRidge) and returns EnsembleWFOResult per ETF.
  - ``get_current_signals``: given a completed set of WFO results, refits each
    model on the most recent data and generates a prediction for today.

Computational note:
  Each fold takes < 2 s on ~180 rows × 15 features.  20 models × ~8 folds ≈
  160 fits ≈ 60 s total — well within acceptable limits for a batch run.
  The ensemble runner is 3× slower (~3 min total) and is intended for
  monthly batch runs rather than interactive use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
import config

from src.models.wfo_engine import WFOResult, run_wfo, predict_current
from src.processing.feature_engineering import get_X_y_relative

logger = logging.getLogger(__name__)


def apply_prediction_shrinkage(
    prediction: float | np.ndarray,
    alpha: float | None = None,
) -> float | np.ndarray:
    """Apply the promoted v38 calibration shrinkage to ensemble predictions."""
    if alpha is None:
        alpha = config.ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA
    if isinstance(prediction, np.ndarray):
        return alpha * prediction
    return float(alpha * prediction)


# ---------------------------------------------------------------------------
# v3.1 Ensemble result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnsembleWFOResult:
    """
    Aggregated result from training ElasticNet + Ridge + BayesianRidge + GBT on
    a single ETF benchmark (v5.0: 4-model ensemble).

    The ``mean_ic`` and ``mean_hit_rate`` are equal-weight averages across the
    models' out-of-sample fold statistics.  Live predictions use inverse-
    variance weighting (1/MAE²) — see ``get_ensemble_signals()``.

    Attributes:
        benchmark:      ETF ticker.
        target_horizon: Forward return horizon in months.
        mean_ic:        Equal-weight mean IC across all models.
        mean_hit_rate:  Equal-weight mean directional hit rate.
        mean_mae:       Equal-weight mean absolute error.
        model_results:  Individual WFOResult per model type key.
    """
    benchmark: str
    target_horizon: int
    mean_ic: float
    mean_hit_rate: float
    mean_mae: float
    model_results: dict[str, WFOResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Multi-benchmark training
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    X: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    model_type: Literal["lasso", "ridge", "elasticnet"] = "elasticnet",
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
) -> dict[str, WFOResult]:
    """
    Train one WFO model per ETF benchmark.

    For each ETF column in ``relative_return_matrix``, aligns the feature
    matrix to the relative return series, drops NaN targets, and calls
    ``run_wfo()`` with the correct embargo for the target horizon.

    Args:
        X:                      Feature DataFrame from
                                ``build_feature_matrix_from_db()`` (or
                                ``build_feature_matrix()``).  Monthly
                                DatetimeIndex.
        relative_return_matrix: DataFrame with one column per ETF benchmark,
                                values = PGR return − ETF return.  Typically
                                the output of
                                ``build_relative_return_targets()`` or loaded
                                via ``load_relative_return_matrix()`` for each
                                ETF.  Index must overlap with ``X``.
        model_type:             ``"lasso"`` (default) or ``"ridge"``.
        target_horizon_months:  Forward return horizon in months (6 or 12).
                                Sets the WFO embargo gap.

    Returns:
        Dict mapping ETF ticker → WFOResult.  Benchmarks skipped due to
        insufficient overlapping data are absent from the dict (no empty
        WFOResult placeholders).

    Raises:
        ValueError: If ``relative_return_matrix`` is empty.
    """
    if relative_return_matrix.empty:
        raise ValueError("relative_return_matrix is empty — no benchmarks to train.")

    results: dict[str, WFOResult] = {}

    for etf in relative_return_matrix.columns:
        rel_series = relative_return_matrix[etf].rename(f"{etf}_{target_horizon_months}m")

        try:
            X_aligned, y_aligned = get_X_y_relative(
                X, rel_series, drop_na_target=True
            )
        except ValueError:
            # No overlapping data for this benchmark — skip silently.
            continue

        if len(y_aligned) == 0:
            continue

        try:
            wfo_result = run_wfo(
                X_aligned,
                y_aligned,
                model_type=model_type,
                target_horizon_months=target_horizon_months,
                benchmark=etf,
                purge_buffer=purge_buffer,
            )
        except ValueError:
            # Dataset too small for this benchmark — skip.
            continue

        results[etf] = wfo_result

    return results


# ---------------------------------------------------------------------------
# v3.1 Ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble_benchmarks(
    X: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
    model_feature_overrides: dict[str, list[str]] | None = None,
) -> dict[str, EnsembleWFOResult]:
    """
    Train the production ensemble (ENSEMBLE_MODELS) per ETF benchmark.

    For each ETF column in ``relative_return_matrix``, one independent WFO
    model per model type is trained.  Out-of-sample IC, hit rate, and MAE are
    averaged (equal weight) across models.  Live ensemble predictions use
    inverse-variance weighting (1/MAE²) — see ``get_ensemble_signals()``.

    Args:
        X:                      Feature DataFrame (monthly DatetimeIndex).
        relative_return_matrix: DataFrame of PGR-minus-ETF relative returns.
        target_horizon_months:  Forward return horizon in months (6 or 12).
        purge_buffer:           Extra months of purge buffer beyond horizon.
                                None uses config defaults.
        model_feature_overrides: Optional dict mapping model type → list of
                                feature column names.  When provided, each
                                model is trained only on its specified features
                                (columns not present in X are silently dropped).
                                None uses all columns of X for every model.

    Returns:
        Dict mapping ETF ticker → EnsembleWFOResult.
    """
    if relative_return_matrix.empty:
        raise ValueError("relative_return_matrix is empty.")

    ensemble_results: dict[str, EnsembleWFOResult] = {}
    model_types: list[str] = list(config.ENSEMBLE_MODELS)

    for etf in relative_return_matrix.columns:
        rel_series = relative_return_matrix[etf].rename(f"{etf}_{target_horizon_months}m")

        try:
            X_aligned, y_aligned = get_X_y_relative(X, rel_series, drop_na_target=True)
        except ValueError:
            continue

        if len(y_aligned) == 0:
            continue

        per_model: dict[str, WFOResult] = {}
        for mtype in model_types:
            # Apply model-specific feature selection when overrides are provided.
            if model_feature_overrides and mtype in model_feature_overrides:
                selected = [c for c in model_feature_overrides[mtype] if c in X_aligned.columns]
                X_for_wfo = X_aligned[selected] if selected else X_aligned
            else:
                X_for_wfo = X_aligned
            try:
                result = run_wfo(
                    X_for_wfo,
                    y_aligned,
                    model_type=cast(Literal["lasso", "ridge", "elasticnet", "bayesian_ridge", "gbt"], mtype),
                    target_horizon_months=target_horizon_months,
                    benchmark=etf,
                    purge_buffer=purge_buffer,
                )
                per_model[mtype] = result
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Skipping ensemble model %s for benchmark %s due to WFO failure. Error=%r",
                    mtype,
                    etf,
                    exc,
                )
                continue

        if not per_model:
            continue

        ics = [r.information_coefficient for r in per_model.values()]
        hit_rates = [r.hit_rate for r in per_model.values()]
        maes = [r.mean_absolute_error for r in per_model.values()]
        mean_ic = float(np.mean(ics)) if ics else float("nan")
        mean_hit_rate = float(np.mean(hit_rates)) if hit_rates else float("nan")
        mean_mae = float(np.mean(maes)) if maes else float("nan")

        ensemble_results[etf] = EnsembleWFOResult(
            benchmark=etf,
            target_horizon=target_horizon_months,
            mean_ic=mean_ic,
            mean_hit_rate=mean_hit_rate,
            mean_mae=mean_mae,
            model_results=per_model,
        )

    return ensemble_results


def get_ensemble_signals(
    X_full: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    ensemble_results: dict[str, EnsembleWFOResult],
    X_current: pd.DataFrame,
    train_window_months: int | None = None,
) -> pd.DataFrame:
    """
    Generate live predictions from each ensemble model per benchmark.

    v5.0: Predictions are combined using **inverse-variance weighting** (1/MAE²)
    rather than equal weights.  A model whose OOS MAE is half that of another
    receives 4× the weight, naturally down-weighting poorly-calibrated members.
    BayesianRidge posterior std is still used for ``prediction_std`` and the
    P(outperform) confidence tier.

    For each ETF, calls ``predict_current()`` on each of the four model types
    (ElasticNet, Ridge, BayesianRidge, GBT), computes per-model weights from
    ``1 / MAE²`` where MAE is the OOS mean absolute error from the WFO folds,
    and returns the weighted average point prediction.

    Args:
        X_full:                 Complete feature DataFrame.
        relative_return_matrix: DataFrame of relative returns.
        ensemble_results:       Output of ``run_ensemble_benchmarks()``.
        X_current:              Single-row DataFrame with current features.
        train_window_months:    Refit window size.

    Returns:
        DataFrame with one row per benchmark and columns:
          ``benchmark``, ``point_prediction``, ``prediction_std``,
          ``signal_to_noise``, ``mean_ic``, ``mean_hit_rate``, ``signal``.
    """
    from src.models.regularized_models import build_bayesian_ridge_pipeline  # noqa: F401

    rows = []
    for etf, ens_result in ensemble_results.items():
        if etf not in relative_return_matrix.columns:
            continue

        rel_series = relative_return_matrix[etf].rename(
            f"{etf}_{ens_result.target_horizon}m"
        )
        try:
            _, y_full = get_X_y_relative(X_full, rel_series, drop_na_target=True)
        except ValueError:
            continue

        # Collect (prediction, weight) pairs — weight = 1 / MAE²
        weighted_preds: list[tuple[float, float]] = []
        prediction_std = 0.0

        for mtype, wfo_result in ens_result.model_results.items():
            try:
                pred = predict_current(
                    X_full=X_full,
                    y_full=y_full,
                    X_current=X_current,
                    wfo_result=wfo_result,
                    model_type=cast(Literal["lasso", "ridge", "elasticnet", "bayesian_ridge", "gbt"], mtype),
                    train_window_months=train_window_months,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to generate ensemble prediction for benchmark %s model %s. Error=%r",
                    etf,
                    mtype,
                    exc,
                )
                continue

            point = pred["predicted_return"]

            # Inverse-variance weight: 1 / MAE².  Fall back to weight=1 when
            # MAE is zero or unavailable (prevents division-by-zero on perfect folds).
            mae = wfo_result.mean_absolute_error
            weight = 1.0 / (mae ** 2) if mae > 1e-9 else 1.0
            weighted_preds.append((point, weight))

            # Uncertainty from BayesianRidge — used for P(outperform) only
            if mtype == "bayesian_ridge" and "prediction_std" in pred:
                prediction_std = float(pred["prediction_std"])

        if not weighted_preds:
            continue

        # Normalised weighted average
        total_weight = sum(w for _, w in weighted_preds)
        raw_point_prediction = sum(p * w for p, w in weighted_preds) / total_weight
        point_prediction = float(apply_prediction_shrinkage(raw_point_prediction))
        signal_to_noise = (
            abs(point_prediction) / prediction_std
            if prediction_std > 0
            else 0.0
        )

        ic = ens_result.mean_ic
        if ic < _IC_THRESHOLD or abs(point_prediction) < _RETURN_THRESHOLD:
            signal = _SIGNAL_NEUTRAL
        elif point_prediction > 0:
            signal = _SIGNAL_OUTPERFORM
        else:
            signal = _SIGNAL_UNDERPERFORM

        confidence_tier, prob_outperform = get_confidence_tier(
            point_prediction, prediction_std
        )

        rows.append({
            "benchmark":        etf,
            "point_prediction": point_prediction,
            "prediction_std":   prediction_std,
            "signal_to_noise":  signal_to_noise,
            "mean_ic":          ens_result.mean_ic,
            "mean_hit_rate":    ens_result.mean_hit_rate,
            "signal":           signal,
            "prob_outperform":  prob_outperform,
            "confidence_tier":  confidence_tier,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "benchmark", "point_prediction", "prediction_std",
            "signal_to_noise", "mean_ic", "mean_hit_rate", "signal",
            "prob_outperform", "confidence_tier",
        ])

    return pd.DataFrame(rows).set_index("benchmark")


# ---------------------------------------------------------------------------
# v4.3 Confidence tier
# ---------------------------------------------------------------------------

def get_confidence_tier(y_hat: float, y_std: float) -> tuple[str, float]:
    """
    Compute a confidence tier and P(outperform) from a BayesianRidge prediction.

    Uses the standard-normal CDF to convert the signal-to-noise ratio into a
    probability that the predicted relative return is positive.  This is the
    uncalibrated Phase 1 estimate — posterior σ² from BayesianRidge without
    Platt scaling or isotonic regression calibration.

    Tier thresholds:
      - HIGH:     P(outperform) ≥ 0.70 or ≤ 0.30 (strong directional conviction)
      - MODERATE: P(outperform) ≥ 0.60 or ≤ 0.40
      - LOW:      otherwise (signal near 50/50)

    Args:
        y_hat: Point prediction (predicted relative return).
        y_std: Posterior standard deviation from BayesianRidge (σ_pred).

    Returns:
        ``(confidence_tier, prob_outperform)`` where ``prob_outperform`` is
        P(relative return > 0) ∈ [0, 1].
    """
    if y_std <= 0.0:
        prob = 0.5
    else:
        prob = float(_norm.cdf(y_hat / y_std))

    if prob >= 0.70 or prob <= 0.30:
        tier = "HIGH"
    elif prob >= 0.60 or prob <= 0.40:
        tier = "MODERATE"
    else:
        tier = "LOW"

    return tier, prob


# ---------------------------------------------------------------------------
# Current-period signal generation
# ---------------------------------------------------------------------------

_SIGNAL_OUTPERFORM = "OUTPERFORM"
_SIGNAL_UNDERPERFORM = "UNDERPERFORM"
_SIGNAL_NEUTRAL = "NEUTRAL"

_IC_THRESHOLD = 0.05          # minimum IC for a signal to be non-NEUTRAL
_RETURN_THRESHOLD = 0.01      # minimum |predicted return| for directional signal


def get_current_signals(
    X_full: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    wfo_results: dict[str, WFOResult],
    X_current: pd.DataFrame,
    model_type: Literal["lasso", "ridge", "elasticnet"] = "elasticnet",
    train_window_months: int | None = None,
) -> pd.DataFrame:
    """
    Generate live predictions for the current observation across all benchmarks.

    For each ETF in ``wfo_results``, calls ``predict_current()`` to refit the
    model on the most recent training window and predict on ``X_current``.

    Signal classification:
      - ``OUTPERFORM``:   predicted relative return > threshold AND IC ≥ 0.05
      - ``UNDERPERFORM``: predicted relative return < -threshold AND IC ≥ 0.05
      - ``NEUTRAL``:      |predicted return| < threshold OR IC < 0.05

    Args:
        X_full:                 Complete feature DataFrame (same as used in
                                ``run_all_benchmarks``).
        relative_return_matrix: DataFrame of relative returns (one column per
                                ETF).  Used to build the aligned y_full for
                                each benchmark.
        wfo_results:            Dict from ``run_all_benchmarks()`` mapping
                                ETF ticker → WFOResult.
        X_current:              Single-row DataFrame with current features.
        model_type:             Must match the model_type used in training.
        train_window_months:    Refit window size.  Defaults to
                                ``config.WFO_TRAIN_WINDOW_MONTHS``.

    Returns:
        DataFrame with one row per benchmark and columns:
          - ``benchmark``: ETF ticker
          - ``predicted_relative_return``: live model prediction
          - ``ic``: out-of-sample IC from WFO
          - ``hit_rate``: directional hit rate from WFO
          - ``signal``: "OUTPERFORM", "UNDERPERFORM", or "NEUTRAL"
          - ``top_feature``: name of the highest-weight feature (absolute coef)
    """
    rows = []

    for etf, wfo_result in wfo_results.items():
        if etf not in relative_return_matrix.columns:
            continue

        horizon = wfo_result.target_horizon
        rel_series = relative_return_matrix[etf].rename(f"{etf}_{horizon}m")

        try:
            _, y_full = get_X_y_relative(X_full, rel_series, drop_na_target=True)
        except ValueError:
            continue

        try:
            pred = predict_current(
                X_full=X_full,
                y_full=y_full,
                X_current=X_current,
                wfo_result=wfo_result,
                model_type=model_type,
                train_window_months=train_window_months,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to generate live signal for benchmark %s. Error=%r",
                etf,
                exc,
            )
            continue

        predicted = pred["predicted_return"]
        ic = pred["ic"]
        hit_rate = pred["hit_rate"]
        top_features = pred["top_features"]
        top_feature = top_features[0][0] if top_features else ""

        if ic < _IC_THRESHOLD or abs(predicted) < _RETURN_THRESHOLD:
            signal = _SIGNAL_NEUTRAL
        elif predicted > 0:
            signal = _SIGNAL_OUTPERFORM
        else:
            signal = _SIGNAL_UNDERPERFORM

        rows.append({
            "benchmark":                  etf,
            "predicted_relative_return":  predicted,
            "ic":                         ic,
            "hit_rate":                   hit_rate,
            "signal":                     signal,
            "top_feature":                top_feature,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "benchmark", "predicted_relative_return", "ic",
                "hit_rate", "signal", "top_feature",
            ]
        )

    df = pd.DataFrame(rows).set_index("benchmark")
    return df
