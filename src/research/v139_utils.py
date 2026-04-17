"""Shared helpers for the v139-v152 autoresearch follow-on cycle."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

import config
from config import features as config_features
from config.features import MODEL_FEATURE_OVERRIDES, PRIMARY_FORECAST_UNIVERSE
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.models.multi_benchmark_wfo import EnsembleWFOResult, run_ensemble_benchmarks
from src.research.v37_utils import compute_metrics, get_connection, load_feature_matrix, load_relative_series, pool_metrics

DEFAULT_BENCHMARKS = list(PRIMARY_FORECAST_UNIVERSE)


@contextmanager
def patched_config(**updates: Any) -> Iterator[None]:
    """Temporarily patch config values used by research harnesses."""
    originals_root: dict[str, Any] = {}
    originals_features: dict[str, Any] = {}
    for name, value in updates.items():
        originals_root[name] = getattr(config, name)
        setattr(config, name, value)
        if hasattr(config_features, name):
            originals_features[name] = getattr(config_features, name)
            setattr(config_features, name, value)
    try:
        yield
    finally:
        for name, value in originals_root.items():
            setattr(config, name, value)
        for name, value in originals_features.items():
            setattr(config_features, name, value)


def load_relative_matrix(benchmarks: list[str]) -> pd.DataFrame:
    """Load the pre-holdout relative-return matrix for the requested benchmarks."""
    conn = get_connection()
    try:
        rel_map: dict[str, pd.Series] = {}
        for benchmark in benchmarks:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if not rel_series.empty:
                rel_map[benchmark] = rel_series
        if not rel_map:
            raise RuntimeError("No relative-return series available for the requested benchmarks.")
        return pd.DataFrame(rel_map)
    finally:
        conn.close()


def prune_feature_overrides(
    feature_df: pd.DataFrame,
    model_feature_overrides: dict[str, list[str]],
    rho_threshold: float,
) -> dict[str, list[str]]:
    """Greedily prune highly correlated features within each model's feature set."""
    if not 0.50 <= rho_threshold <= 0.999:
        raise ValueError(f"rho_threshold must be in [0.50, 0.999], got {rho_threshold}")

    pruned: dict[str, list[str]] = {}
    for model_type, feature_list in model_feature_overrides.items():
        selected: list[str] = []
        corr_source = feature_df[[col for col in feature_list if col in feature_df.columns]].copy()
        if corr_source.empty:
            pruned[model_type] = []
            continue
        corr = corr_source.corr().abs()
        for column in feature_list:
            if column not in corr.columns:
                continue
            if not selected:
                selected.append(column)
                continue
            if all(float(corr.loc[column, kept]) < rho_threshold for kept in selected):
                selected.append(column)
        pruned[model_type] = selected
    return pruned


def reconstruct_two_model_blend_predictions(
    ens_result: EnsembleWFOResult,
    ridge_weight: float,
    shrinkage_alpha: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Rebuild OOS predictions using a fixed Ridge-vs-GBT blend weight."""
    if not 0.0 <= ridge_weight <= 1.0:
        raise ValueError(f"ridge_weight must be in [0, 1], got {ridge_weight}")
    if shrinkage_alpha is None:
        shrinkage_alpha = config.ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA

    ridge_result = ens_result.model_results.get("ridge")
    gbt_result = ens_result.model_results.get("gbt")
    if ridge_result is None or gbt_result is None:
        empty = pd.Series(dtype=float)
        return empty, empty

    predictions: list[float] = []
    realized: list[float] = []
    dates: list[pd.Timestamp] = []
    for fold_idx in range(min(len(ridge_result.folds), len(gbt_result.folds))):
        ridge_fold = ridge_result.folds[fold_idx]
        gbt_fold = gbt_result.folds[fold_idx]
        if len(ridge_fold.y_hat) != len(gbt_fold.y_hat):
            raise ValueError("Ridge and GBT fold lengths do not align.")
        if ridge_fold._test_dates != gbt_fold._test_dates:
            raise ValueError("Ridge and GBT fold dates do not align.")
        combined = ridge_weight * ridge_fold.y_hat + (1.0 - ridge_weight) * gbt_fold.y_hat
        predictions.extend((shrinkage_alpha * combined).tolist())
        realized.extend(ridge_fold.y_true.tolist())
        dates.extend(list(ridge_fold._test_dates))

    pred_series = pd.Series(predictions, index=pd.DatetimeIndex(dates), name="y_hat")
    realized_series = pd.Series(realized, index=pd.DatetimeIndex(dates), name="y_true")
    return pred_series, realized_series


def evaluate_ensemble_configuration(
    *,
    benchmarks: list[str] | None = None,
    config_overrides: dict[str, Any] | None = None,
    model_feature_overrides: dict[str, list[str]] | None = None,
    prediction_builder: Callable[[EnsembleWFOResult], tuple[pd.Series, pd.Series]] | None = None,
) -> dict[str, Any]:
    """Evaluate a research-only ensemble configuration on the current frame."""
    selected_benchmarks = list(benchmarks or DEFAULT_BENCHMARKS)
    overrides = config_overrides or {}
    builder = prediction_builder or (
        lambda ens_result: reconstruct_ensemble_oos_predictions(
            ens_result,
            shrinkage_alpha=config.ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA,
        )
    )

    with patched_config(**overrides):
        conn = get_connection()
        try:
            feature_df = load_feature_matrix(conn)
            relative_return_matrix = load_relative_matrix(selected_benchmarks)
            ensemble_results = run_ensemble_benchmarks(
                feature_df,
                relative_return_matrix,
                target_horizon_months=6,
                model_feature_overrides=model_feature_overrides or MODEL_FEATURE_OVERRIDES,
            )
        finally:
            conn.close()

    rows: list[dict[str, Any]] = []
    pooled_pred: list[np.ndarray] = []
    pooled_true: list[np.ndarray] = []
    for benchmark in selected_benchmarks:
        ens_result = ensemble_results.get(benchmark)
        if ens_result is None:
            continue
        y_hat, y_true = builder(ens_result)
        if y_hat.empty or y_true.empty:
            continue
        y_hat_arr = y_hat.to_numpy(dtype=float)
        y_true_arr = y_true.to_numpy(dtype=float)
        pooled_pred.append(y_hat_arr)
        pooled_true.append(y_true_arr)
        metrics = compute_metrics(y_true_arr, y_hat_arr)
        rows.append(
            {
                "benchmark": benchmark,
                **metrics,
                "_y_true": y_true_arr,
                "_y_hat": y_hat_arr,
            }
        )

    if not rows:
        raise RuntimeError("Configuration produced no benchmark rows.")

    pooled = pool_metrics(rows)
    return {
        "rows": rows,
        "feature_df": feature_df,
        "ensemble_results": ensemble_results,
        "pooled_oos_r2": float(pooled["r2"]),
        "pooled_ic": float(pooled["ic"]),
        "pooled_hit_rate": float(pooled["hit_rate"]),
        "pooled_sigma_ratio": float(pooled["sigma_ratio"]),
        "pooled_predictions": np.concatenate(pooled_pred),
        "pooled_realized": np.concatenate(pooled_true),
    }
