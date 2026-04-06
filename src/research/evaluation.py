"""Reusable evaluation helpers for v9.x research scripts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

import config
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.models.wfo_engine import WFOResult, run_wfo
from src.reporting.backtest_report import compute_newey_west_ic, compute_oos_r_squared

logger = logging.getLogger(__name__)


BASELINE_STRATEGIES: tuple[str, ...] = ("historical_mean", "last_value", "zero")


@dataclass(frozen=True)
class PredictionSummary:
    """Canonical metric bundle for benchmark/model experiments."""

    n_obs: int
    ic: float
    hit_rate: float
    mae: float
    oos_r2: float
    nw_ic: float
    nw_p_value: float


@dataclass(frozen=True)
class BinaryPredictionSummary:
    """Classification-style metrics for binary target formulations."""

    n_obs: int
    brier_score: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    base_rate: float
    predicted_positive_rate: float


def _resolve_total_gap(
    target_horizon_months: int,
    purge_buffer: int | None = None,
) -> int:
    """Return the total purge gap used by the production WFO engine."""
    if purge_buffer is None:
        purge_buffer = (
            config.WFO_PURGE_BUFFER_6M
            if target_horizon_months <= 6
            else config.WFO_PURGE_BUFFER_12M
        )
    return target_horizon_months + purge_buffer


def build_wfo_splitter(
    n_obs: int,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
) -> TimeSeriesSplit:
    """Construct the exact production TimeSeriesSplit for a dataset size."""
    total_gap = _resolve_total_gap(
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    )
    available = n_obs - config.WFO_TRAIN_WINDOW_MONTHS - total_gap
    n_splits = max(1, available // config.WFO_TEST_WINDOW_MONTHS)
    return TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
        test_size=config.WFO_TEST_WINDOW_MONTHS,
        gap=total_gap,
    )


def iter_wfo_splits(
    X: pd.DataFrame,
    y: pd.Series,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Yield production-equivalent WFO train/test splits."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if y.isna().any():
        raise ValueError("y must not contain NaN values.")
    total_gap = _resolve_total_gap(
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    )
    min_required = (
        config.WFO_TRAIN_WINDOW_MONTHS
        + total_gap
        + config.WFO_TEST_WINDOW_MONTHS
    )
    if len(X) < min_required:
        raise ValueError(
            f"Dataset has only {len(X)} observations; need at least "
            f"{min_required}."
        )

    splitter = build_wfo_splitter(
        n_obs=len(X),
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    )
    x_arr = X.values
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_arr)):
        yield fold_idx, train_idx, test_idx


def summarize_predictions(
    predicted: pd.Series,
    realized: pd.Series,
    target_horizon_months: int = 6,
) -> PredictionSummary:
    """Compute the standard metric bundle from aligned predictions."""
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if aligned.empty:
        return PredictionSummary(
            n_obs=0,
            ic=float("nan"),
            hit_rate=float("nan"),
            mae=float("nan"),
            oos_r2=float("nan"),
            nw_ic=float("nan"),
            nw_p_value=float("nan"),
        )

    y_hat = aligned.iloc[:, 0]
    y_true = aligned.iloc[:, 1]

    if len(aligned) >= 2:
        if y_true.nunique(dropna=False) <= 1 or y_hat.nunique(dropna=False) <= 1:
            ic_value = float("nan")
        else:
            ic, _ = spearmanr(y_true, y_hat)
            ic_value = float(ic)
    else:
        ic_value = float("nan")

    try:
        nw_ic, nw_p_value = compute_newey_west_ic(
            y_hat,
            y_true,
            lags=max(1, target_horizon_months - 1),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not compute Newey-West IC summary; returning NaN diagnostics. Error=%r",
            exc,
        )
        nw_ic, nw_p_value = float("nan"), float("nan")

    return PredictionSummary(
        n_obs=int(len(aligned)),
        ic=ic_value,
        hit_rate=float(np.mean(np.sign(y_true) == np.sign(y_hat))),
        mae=float(mean_absolute_error(y_true, y_hat)),
        oos_r2=float(compute_oos_r_squared(y_hat, y_true)),
        nw_ic=float(nw_ic),
        nw_p_value=float(nw_p_value),
    )


def summarize_binary_predictions(
    predicted: pd.Series,
    realized: pd.Series,
    threshold: float = 0.5,
) -> BinaryPredictionSummary:
    """Compute binary-target metrics from continuous model outputs."""
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if aligned.empty:
        return BinaryPredictionSummary(
            n_obs=0,
            brier_score=float("nan"),
            accuracy=float("nan"),
            balanced_accuracy=float("nan"),
            precision=float("nan"),
            recall=float("nan"),
            base_rate=float("nan"),
            predicted_positive_rate=float("nan"),
        )

    y_score = aligned.iloc[:, 0].clip(0.0, 1.0)
    y_true = aligned.iloc[:, 1].astype(int)
    y_pred = (y_score >= threshold).astype(int)

    return BinaryPredictionSummary(
        n_obs=int(len(aligned)),
        brier_score=float(brier_score_loss(y_true, y_score)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        base_rate=float(y_true.mean()),
        predicted_positive_rate=float(y_pred.mean()),
    )


def evaluate_baseline_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate a simple benchmark strategy on the same production WFO folds."""
    if strategy not in BASELINE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {BASELINE_STRATEGIES}."
        )

    predictions: list[float] = []
    realized: list[float] = []
    test_dates: list[pd.Timestamp] = []

    for _, train_idx, test_idx in iter_wfo_splits(
        X,
        y,
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    ):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if strategy == "historical_mean":
            pred_value = float(y_train.mean())
        elif strategy == "last_value":
            pred_value = float(y_train.iloc[-1])
        else:
            pred_value = 0.0

        predictions.extend([pred_value] * len(test_idx))
        realized.extend(y_test.tolist())
        test_dates.extend(list(y_test.index))

    pred_series = pd.Series(predictions, index=test_dates, name="y_hat")
    realized_series = pd.Series(realized, index=test_dates, name="y_true")
    summary = summarize_predictions(
        pred_series,
        realized_series,
        target_horizon_months=target_horizon_months,
    )
    return {
        "n_obs": summary.n_obs,
        "ic": summary.ic,
        "hit_rate": summary.hit_rate,
        "mae": summary.mae,
        "oos_r2": summary.oos_r2,
        "nw_ic": summary.nw_ic,
        "nw_p_value": summary.nw_p_value,
        "strategy": strategy,
    }


def reconstruct_baseline_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Rebuild baseline OOS predictions on the exact production WFO folds."""
    if strategy not in BASELINE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {BASELINE_STRATEGIES}."
        )

    predictions: list[float] = []
    realized: list[float] = []
    test_dates: list[pd.Timestamp] = []

    for _, train_idx, test_idx in iter_wfo_splits(
        X,
        y,
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    ):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if strategy == "historical_mean":
            pred_value = float(y_train.mean())
        elif strategy == "last_value":
            pred_value = float(y_train.iloc[-1])
        else:
            pred_value = 0.0

        predictions.extend([pred_value] * len(test_idx))
        realized.extend(y_test.tolist())
        test_dates.extend(list(y_test.index))

    pred_series = pd.Series(predictions, index=pd.DatetimeIndex(test_dates), name="y_hat")
    realized_series = pd.Series(realized, index=pd.DatetimeIndex(test_dates), name="y_true")
    return pred_series, realized_series


def evaluate_binary_baseline_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate a simple baseline on a binary target using production WFO folds."""
    metrics = evaluate_baseline_strategy(
        X,
        y,
        strategy=strategy,
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    )
    predictions: list[float] = []
    realized: list[float] = []
    test_dates: list[pd.Timestamp] = []
    for _, train_idx, test_idx in iter_wfo_splits(
        X,
        y,
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
    ):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if strategy == "historical_mean":
            pred_value = float(y_train.mean())
        elif strategy == "last_value":
            pred_value = float(y_train.iloc[-1])
        else:
            pred_value = 0.0
        predictions.extend([pred_value] * len(test_idx))
        realized.extend(y_test.tolist())
        test_dates.extend(list(y_test.index))

    binary_summary = summarize_binary_predictions(
        pd.Series(predictions, index=test_dates, name="y_hat"),
        pd.Series(realized, index=test_dates, name="y_true"),
    )
    return {
        **metrics,
        "brier_score": binary_summary.brier_score,
        "accuracy": binary_summary.accuracy,
        "balanced_accuracy": binary_summary.balanced_accuracy,
        "precision": binary_summary.precision,
        "recall": binary_summary.recall,
        "base_rate": binary_summary.base_rate,
        "predicted_positive_rate": binary_summary.predicted_positive_rate,
    }


def evaluate_wfo_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    benchmark: str,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[WFOResult, dict[str, float | int | str]]:
    """Run the production WFO engine and package the standard metrics."""
    result = run_wfo(
        X,
        y,
        model_type=model_type,
        target_horizon_months=target_horizon_months,
        benchmark=benchmark,
        purge_buffer=purge_buffer,
        feature_columns=feature_columns,
    )
    y_true = pd.Series(result.y_true_all)
    y_hat = pd.Series(result.y_hat_all)
    summary = summarize_predictions(
        y_hat,
        y_true,
        target_horizon_months=target_horizon_months,
    )
    metrics = {
        "n_obs": summary.n_obs,
        "ic": summary.ic,
        "hit_rate": summary.hit_rate,
        "mae": summary.mae,
        "oos_r2": summary.oos_r2,
        "nw_ic": summary.nw_ic,
        "nw_p_value": summary.nw_p_value,
        "n_features": len(feature_columns) if feature_columns is not None else int(len(result.folds[0].feature_importances)) if result.folds else 0,
        "model_type": model_type,
        "benchmark": benchmark,
    }
    return result, metrics


def evaluate_binary_wfo_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    benchmark: str,
    target_horizon_months: int = 6,
    purge_buffer: int | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[WFOResult, dict[str, float | int | str]]:
    """Run the production WFO engine and add binary-target diagnostics."""
    result, metrics = evaluate_wfo_model(
        X,
        y,
        model_type=model_type,
        benchmark=benchmark,
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        feature_columns=feature_columns,
    )
    binary_summary = summarize_binary_predictions(
        pd.Series(result.y_hat_all),
        pd.Series(result.y_true_all),
    )
    return result, {
        **metrics,
        "brier_score": binary_summary.brier_score,
        "accuracy": binary_summary.accuracy,
        "balanced_accuracy": binary_summary.balanced_accuracy,
        "precision": binary_summary.precision,
        "recall": binary_summary.recall,
        "base_rate": binary_summary.base_rate,
        "predicted_positive_rate": binary_summary.predicted_positive_rate,
    }


def reconstruct_ensemble_oos_predictions(
    ens_result: EnsembleWFOResult,
) -> tuple[pd.Series, pd.Series]:
    """Rebuild inverse-variance ensemble OOS predictions from component folds."""
    model_results = ens_result.model_results
    if not model_results:
        empty = pd.Series(dtype=float)
        return empty, empty

    weights: dict[str, float] = {}
    for model_type, result in model_results.items():
        mae = result.mean_absolute_error
        weights[model_type] = 1.0 / (mae**2) if mae > 1e-9 else 1.0
    total_weight = sum(weights.values())

    ref = next(iter(model_results.values()))
    predictions: list[float] = []
    realized: list[float] = []
    dates: list[pd.Timestamp] = []

    for fold_idx in range(len(ref.folds)):
        fold_y_true: np.ndarray | None = None
        fold_y_hat: np.ndarray | None = None
        fold_dates: list[pd.Timestamp] = []

        for model_type, result in model_results.items():
            if fold_idx >= len(result.folds):
                continue
            fold = result.folds[fold_idx]
            weight = weights[model_type] / total_weight
            if fold_y_true is None:
                fold_y_true = fold.y_true.copy()
                fold_y_hat = np.zeros(len(fold.y_true), dtype=float)
                fold_dates = list(fold._test_dates)
            fold_y_hat = fold_y_hat + weight * fold.y_hat  # type: ignore[operator]

        if fold_y_true is not None and fold_y_hat is not None:
            predictions.extend(fold_y_hat.tolist())
            realized.extend(fold_y_true.tolist())
            dates.extend(fold_dates)

    pred_series = pd.Series(predictions, index=pd.DatetimeIndex(dates), name="y_hat")
    realized_series = pd.Series(realized, index=pd.DatetimeIndex(dates), name="y_true")
    return pred_series, realized_series


def evaluate_ensemble_result(
    ens_result: EnsembleWFOResult,
    target_horizon_months: int = 6,
) -> dict[str, float | int | str]:
    """Compute standard metrics for the inverse-variance ensemble itself."""
    y_hat, y_true = reconstruct_ensemble_oos_predictions(ens_result)
    summary = summarize_predictions(
        y_hat,
        y_true,
        target_horizon_months=target_horizon_months,
    )
    n_feature_rows = [
        len(result.folds[0].feature_importances)
        for result in ens_result.model_results.values()
        if result.folds
    ]
    return {
        "n_obs": summary.n_obs,
        "ic": summary.ic,
        "hit_rate": summary.hit_rate,
        "mae": summary.mae,
        "oos_r2": summary.oos_r2,
        "nw_ic": summary.nw_ic,
        "nw_p_value": summary.nw_p_value,
        "benchmark": ens_result.benchmark,
        "model_type": "ensemble",
        "n_features": int(np.mean(n_feature_rows)) if n_feature_rows else 0,
    }


def evaluate_binary_ensemble_result(
    ens_result: EnsembleWFOResult,
    target_horizon_months: int = 6,
) -> dict[str, float | int | str]:
    """Compute standard and binary-target metrics for the ensemble."""
    metrics = evaluate_ensemble_result(
        ens_result,
        target_horizon_months=target_horizon_months,
    )
    y_hat, y_true = reconstruct_ensemble_oos_predictions(ens_result)
    binary_summary = summarize_binary_predictions(y_hat, y_true)
    return {
        **metrics,
        "brier_score": binary_summary.brier_score,
        "accuracy": binary_summary.accuracy,
        "balanced_accuracy": binary_summary.balanced_accuracy,
        "precision": binary_summary.precision,
        "recall": binary_summary.recall,
        "base_rate": binary_summary.base_rate,
        "predicted_positive_rate": binary_summary.predicted_positive_rate,
    }


def classify_research_gate(
    oos_r2: float,
    ic: float,
    hit_rate: float,
) -> str:
    """Classify a candidate against the current production quality thresholds."""
    if (
        oos_r2 >= config.DIAG_MIN_OOS_R2
        and ic >= config.DIAG_MIN_IC
        and hit_rate >= config.DIAG_MIN_HIT_RATE
    ):
        return "PASS"
    if oos_r2 < 0.0 or ic < 0.03 or hit_rate < 0.52:
        return "FAIL"
    return "MARGINAL"


# ---------------------------------------------------------------------------
# v32.0 — Feature importance stability across WFO folds (Tier 4.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureImportanceStability:
    """Stability summary for feature importances across WFO folds.

    Attributes:
        stability_score: Mean pairwise Spearman rank-correlation between
            consecutive fold importance rankings.  Ranges [-1, 1]; values
            ≥ 0.7 indicate stable feature rankings.
        n_folds: Number of folds used to compute the metric.
        per_feature: DataFrame with columns ``mean_rank``, ``rank_std``, and
            ``mean_importance``, indexed by feature name.  Ranks are 1-based
            (rank 1 = highest importance).
        verdict: "STABLE" / "MARGINAL" / "UNSTABLE" based on stability_score.
    """

    stability_score: float
    n_folds: int
    per_feature: pd.DataFrame
    verdict: str


def compute_feature_importance_stability(
    result: WFOResult,
) -> FeatureImportanceStability | None:
    """Compute feature importance stability across WFO folds.

    For each consecutive pair of folds, the function converts each fold's
    importance dict to a rank series (rank 1 = highest importance) and
    computes the Spearman rank correlation.  The overall stability score is
    the mean of these pairwise correlations.

    Args:
        result: A completed ``WFOResult`` whose folds each contain a
            non-empty ``feature_importances`` dict.

    Returns:
        A ``FeatureImportanceStability`` instance, or ``None`` if there are
        fewer than two folds or all folds have empty importance dicts.
    """
    folds_with_data = [
        f for f in result.folds if f.feature_importances
    ]
    if len(folds_with_data) < 2:
        return None

    # Build per-fold rank series (rank 1 = most important).
    fold_ranks: list[pd.Series] = []
    for fold in folds_with_data:
        imp = pd.Series(fold.feature_importances).abs()
        # rankdata gives rank 1 to the smallest value; we want rank 1 for the
        # largest, so negate importance before ranking.
        ranks = imp.rank(ascending=False, method="average")
        fold_ranks.append(ranks)

    # Align all folds to the union of feature names.
    all_features = sorted(
        set().union(*[set(r.index) for r in fold_ranks])
    )
    aligned = pd.DataFrame(
        {i: r.reindex(all_features) for i, r in enumerate(fold_ranks)}
    )

    # Mean pairwise Spearman correlation between consecutive folds.
    pairwise_corrs: list[float] = []
    for i in range(len(fold_ranks) - 1):
        col_a = aligned.iloc[:, i]
        col_b = aligned.iloc[:, i + 1]
        mask = col_a.notna() & col_b.notna()
        if mask.sum() < 2:
            continue
        try:
            rho, _ = spearmanr(col_a[mask], col_b[mask])
            if not np.isnan(rho):
                pairwise_corrs.append(float(rho))
        except Exception:
            logger.warning(
                "compute_feature_importance_stability: spearmanr failed for "
                "fold pair (%d, %d); skipping",
                i,
                i + 1,
                exc_info=True,
            )

    if not pairwise_corrs:
        return None

    stability_score = float(np.mean(pairwise_corrs))

    # Per-feature summary statistics.
    # Re-read raw importances (not ranks) for the mean_importance column.
    raw_imp = pd.DataFrame(
        {i: pd.Series(f.feature_importances).abs().reindex(all_features)
         for i, f in enumerate(folds_with_data)}
    )
    per_feature = pd.DataFrame(
        {
            "mean_rank": aligned.mean(axis=1),
            "rank_std": aligned.std(axis=1),
            "mean_importance": raw_imp.mean(axis=1),
        },
        index=all_features,
    ).sort_values("mean_rank")

    if stability_score >= 0.7:
        verdict = "STABLE"
    elif stability_score >= 0.4:
        verdict = "MARGINAL"
    else:
        verdict = "UNSTABLE"

    return FeatureImportanceStability(
        stability_score=stability_score,
        n_folds=len(folds_with_data),
        per_feature=per_feature,
        verdict=verdict,
    )
