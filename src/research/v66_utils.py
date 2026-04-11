"""Shared helpers for the v66-v73 calibration and decision-layer cycle."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from config.features import MODEL_FEATURE_OVERRIDES
from results.research.v46_classification import compute_binary_metrics
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.models.forecast_diagnostics import summarize_prediction_diagnostics
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.models.policy_metrics import PolicySummary, evaluate_hold_fraction_series, evaluate_policy_series
from src.models.regularized_models import AdaptiveGapTimeSeriesSplit
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    compute_metrics,
)


def load_ensemble_oos_sequences(
    feature_df: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    shrinkage_alpha: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ensemble OOS sequences keyed by benchmark."""
    ensemble_results = run_ensemble_benchmarks(
        feature_df,
        relative_return_matrix,
        target_horizon_months=6,
        model_feature_overrides=MODEL_FEATURE_OVERRIDES,
    )
    sequences: dict[str, pd.DataFrame] = {}
    for benchmark, ens_result in ensemble_results.items():
        y_hat, y_true = reconstruct_ensemble_oos_predictions(
            ens_result,
            shrinkage_alpha=shrinkage_alpha,
        )
        if y_true.empty:
            continue
        sequences[benchmark] = pd.DataFrame(
            {
                "y_true": y_true.astype(float),
                "y_hat": y_hat.astype(float),
            },
            index=y_true.index,
        ).sort_index()
    return sequences


def fit_clipped_shrinkage_alpha(
    y_true_hist: np.ndarray,
    y_hat_hist: np.ndarray,
    default_alpha: float = 0.50,
) -> float:
    """Fit the no-intercept MSE-optimal shrinkage alpha and clip to [0, 1]."""
    denom = float(np.dot(y_hat_hist, y_hat_hist))
    if denom <= 1e-12:
        return float(default_alpha)
    alpha = float(np.dot(y_true_hist, y_hat_hist) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def apply_prequential_shrinkage(
    y_true: np.ndarray,
    y_hat_raw: np.ndarray,
    min_history: int,
    default_alpha: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply expanding-history shrinkage calibration to one forecast series."""
    calibrated = np.empty(len(y_hat_raw), dtype=float)
    alphas = np.empty(len(y_hat_raw), dtype=float)
    for idx in range(len(y_hat_raw)):
        if idx < min_history:
            alpha = default_alpha
        else:
            alpha = fit_clipped_shrinkage_alpha(
                y_true_hist=y_true[:idx],
                y_hat_hist=y_hat_raw[:idx],
                default_alpha=default_alpha,
            )
        calibrated[idx] = alpha * y_hat_raw[idx]
        alphas[idx] = alpha
    return calibrated, alphas


def apply_prequential_affine_calibration(
    y_true: np.ndarray,
    y_hat_raw: np.ndarray,
    min_history: int,
    ridge_alpha: float,
    default_alpha: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a conservative prequential affine recalibration a + b*y_hat."""
    calibrated = np.empty(len(y_hat_raw), dtype=float)
    intercepts = np.empty(len(y_hat_raw), dtype=float)
    slopes = np.empty(len(y_hat_raw), dtype=float)

    for idx in range(len(y_hat_raw)):
        if idx < min_history:
            intercept = 0.0
            slope = default_alpha
        else:
            model = Ridge(alpha=ridge_alpha, fit_intercept=True)
            model.fit(y_hat_raw[:idx].reshape(-1, 1), y_true[:idx])
            intercept = float(np.clip(model.intercept_, -0.05, 0.05))
            slope = float(np.clip(model.coef_[0], 0.0, 1.0))
        calibrated[idx] = intercept + slope * y_hat_raw[idx]
        intercepts[idx] = intercept
        slopes[idx] = slope
    return calibrated, intercepts, slopes


def benchmark_quality_weights(
    quality_df: pd.DataFrame,
    score_col: str = "nw_ic",
    lambda_mix: float = 0.25,
) -> dict[str, float]:
    """Convert benchmark diagnostics into conservative shrinkage-to-equal weights."""
    if quality_df.empty:
        equal = 1.0 / len(BENCHMARKS)
        return {benchmark: equal for benchmark in BENCHMARKS}

    scores = quality_df.set_index("benchmark")[score_col].astype(float).clip(lower=0.0)
    benchmarks = scores.index.tolist()
    equal_weight = 1.0 / len(benchmarks)
    score_sum = float(scores.sum())
    if score_sum <= 1e-12:
        return {benchmark: equal_weight for benchmark in benchmarks}

    normalized = scores / score_sum
    weights = ((1.0 - lambda_mix) * equal_weight + lambda_mix * normalized).to_dict()
    return {str(benchmark): float(weight) for benchmark, weight in weights.items()}


def build_consensus_frame(
    sequence_map: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate benchmark-level sequences into one cross-benchmark consensus frame."""
    if not sequence_map:
        return pd.DataFrame(columns=["predicted", "realized"])

    pred_frames: list[pd.Series] = []
    realized_frames: list[pd.Series] = []
    for benchmark, frame in sequence_map.items():
        pred_frames.append(frame["y_hat"].rename(benchmark))
        realized_frames.append(frame["y_true"].rename(benchmark))

    pred_df = pd.concat(pred_frames, axis=1).sort_index()
    realized_df = pd.concat(realized_frames, axis=1).sort_index()
    common_cols = [col for col in pred_df.columns if col in realized_df.columns]
    pred_df = pred_df[common_cols]
    realized_df = realized_df[common_cols]

    if weights is None:
        predicted = pred_df.mean(axis=1)
        realized = realized_df.mean(axis=1)
    else:
        aligned_weights = pd.Series(
            {col: float(weights.get(col, 0.0)) for col in common_cols},
            dtype=float,
        )
        weight_sum = float(aligned_weights.sum())
        if weight_sum <= 1e-12:
            aligned_weights = pd.Series(1.0 / len(common_cols), index=common_cols, dtype=float)
        else:
            aligned_weights = aligned_weights / weight_sum
        predicted = pred_df.mul(aligned_weights, axis=1).sum(axis=1)
        realized = realized_df.mul(aligned_weights, axis=1).sum(axis=1)

    return pd.DataFrame({"predicted": predicted, "realized": realized}).dropna()


def summarize_consensus_variant(
    variant: str,
    consensus_frame: pd.DataFrame,
    policy_name: str = "neutral_band_3pct",
) -> dict[str, Any]:
    """Return forecast and policy summaries for one consensus series."""
    metrics = compute_metrics(
        consensus_frame["realized"].to_numpy(dtype=float),
        consensus_frame["predicted"].to_numpy(dtype=float),
    )
    policy = evaluate_policy_series(
        predicted=consensus_frame["predicted"],
        realized_relative_return=consensus_frame["realized"],
        policy_name=policy_name,
    )
    return {
        "variant": variant,
        "n": int(metrics["n"]),
        "r2": float(metrics["r2"]),
        "ic": float(metrics["ic"]),
        "hit_rate": float(metrics["hit_rate"]),
        "mae": float(metrics["mae"]),
        "sigma_ratio": float(metrics["sigma_ratio"]),
        "policy": policy_name,
        "mean_policy_return": float(policy.mean_policy_return),
        "uplift_vs_sell_50": float(policy.uplift_vs_sell_50),
        "capture_ratio": float(policy.capture_ratio),
    }


def classification_probability_sequences(
    feature_df: pd.DataFrame,
    relative_return_matrix: dict[str, pd.Series],
) -> dict[str, pd.DataFrame]:
    """Rebuild v46-style probability sequences with dates for hybrid gating research."""
    from sklearn.linear_model import LogisticRegressionCV

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    sequences: dict[str, pd.DataFrame] = {}
    for benchmark in BENCHMARKS:
        rel_series = relative_return_matrix.get(benchmark)
        if rel_series is None or rel_series.empty:
            continue
        x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
        x_values = x_df[feature_cols].to_numpy()
        y_binary = (y.to_numpy() > 0).astype(int)

        n_obs = len(x_values)
        available = n_obs - MAX_TRAIN_MONTHS - GAP_MONTHS
        n_splits = max(1, available // TEST_SIZE_MONTHS)
        outer_splitter = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=MAX_TRAIN_MONTHS,
            test_size=TEST_SIZE_MONTHS,
            gap=GAP_MONTHS,
        )

        realized: list[int] = []
        probs: list[float] = []
        dates: list[pd.Timestamp] = []

        for train_idx, test_idx in outer_splitter.split(x_values):
            x_train = x_values[train_idx].copy()
            x_test = x_values[test_idx].copy()
            y_train = y_binary[train_idx]
            y_test = y_binary[test_idx]
            if len(np.unique(y_train)) < 2:
                continue

            medians = np.nanmedian(x_train, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            for col_idx in range(x_train.shape[1]):
                x_train[np.isnan(x_train[:, col_idx]), col_idx] = medians[col_idx]
                x_test[np.isnan(x_test[:, col_idx]), col_idx] = medians[col_idx]

            clf = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 20),
                cv=AdaptiveGapTimeSeriesSplit(n_splits=3, gap=GAP_MONTHS),
                solver="lbfgs",
                max_iter=5000,
                random_state=42,
            )
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_test)[:, 1]

            realized.extend(y_test.tolist())
            probs.extend(y_prob.tolist())
            dates.extend(list(y.iloc[test_idx].index))

        if dates:
            sequences[benchmark] = pd.DataFrame(
                {"y_true_binary": realized, "prob_outperform": probs},
                index=pd.DatetimeIndex(dates),
            ).sort_index()
    return sequences


def summarize_binary_variant(
    variant: str,
    sequence_map: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Pool binary probabilities and compute the standard v46 metric bundle."""
    pooled_true = np.concatenate(
        [frame["y_true_binary"].to_numpy(dtype=int) for frame in sequence_map.values()]
    )
    pooled_prob = np.concatenate(
        [frame["prob_outperform"].to_numpy(dtype=float) for frame in sequence_map.values()]
    )
    metrics = compute_binary_metrics(pooled_true, pooled_prob)
    return {"variant": variant, **metrics}


def evaluate_gated_policy_variant(
    variant: str,
    predicted: pd.Series,
    realized: pd.Series,
    probability: pd.Series,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Evaluate a regression-with-probability-gate decision rule."""
    aligned = pd.concat(
        [predicted.rename("predicted"), realized.rename("realized"), probability.rename("probability")],
        axis=1,
    ).dropna()
    if aligned.empty:
        empty_summary = evaluate_hold_fraction_series(
            pd.Series(dtype=float, name="hold_fraction"),
            pd.Series(dtype=float, name="realized"),
        )
        return {
            "variant": variant,
            "n": 0,
            "gate_lower": lower,
            "gate_upper": upper,
            "r2": float("nan"),
            "ic": float("nan"),
            "hit_rate": float("nan"),
            "policy_mean_return": float(empty_summary.mean_policy_return),
            "uplift_vs_sell_50": float(empty_summary.uplift_vs_sell_50),
            "capture_ratio": float(empty_summary.capture_ratio),
            "avg_hold_fraction": float(empty_summary.avg_hold_fraction),
        }

    hold_fraction = np.where(
        aligned["probability"] >= upper,
        np.where(aligned["predicted"] > 0.03, 1.0, 0.5),
        np.where(
            aligned["probability"] <= lower,
            np.where(aligned["predicted"] < -0.03, 0.0, 0.5),
            0.5,
        ),
    )
    policy_summary = evaluate_hold_fraction_series(
        pd.Series(hold_fraction, index=aligned.index, name="hold_fraction"),
        aligned["realized"],
    )
    metrics = compute_metrics(
        aligned["realized"].to_numpy(dtype=float),
        np.where(
            (aligned["probability"] >= upper) | (aligned["probability"] <= lower),
            aligned["predicted"].to_numpy(dtype=float),
            0.0,
        ),
    )
    return {
        "variant": variant,
        "n": int(metrics["n"]),
        "gate_lower": lower,
        "gate_upper": upper,
        "r2": float(metrics["r2"]),
        "ic": float(metrics["ic"]),
        "hit_rate": float(metrics["hit_rate"]),
        "policy_mean_return": float(policy_summary.mean_policy_return),
        "uplift_vs_sell_50": float(policy_summary.uplift_vs_sell_50),
        "capture_ratio": float(policy_summary.capture_ratio),
        "avg_hold_fraction": float(policy_summary.avg_hold_fraction),
    }
