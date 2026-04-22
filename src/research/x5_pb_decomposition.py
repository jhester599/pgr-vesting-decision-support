"""Research-only x5 P/B decomposition utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.x2_absolute_classification import iter_absolute_wfo_splits


def build_pb_regressor(model_name: str) -> Pipeline:
    """Build an x5 P/B regressor by name."""
    if model_name in {"ridge_pb", "ridge_log_pb"}:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=5000.0)),
            ]
        )
    if model_name in {"hist_gbt_pb", "hist_gbt_log_pb"}:
        return Pipeline(
            steps=[
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_depth=2,
                        max_iter=120,
                        learning_rate=0.05,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                )
            ]
        )
    raise ValueError(f"Unsupported P/B regressor '{model_name}'.")


def _align_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    current_pb: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    selected = [feature for feature in feature_columns if feature in X.columns]
    if not selected:
        raise ValueError("No requested feature columns are present in X.")
    target = pd.to_numeric(y.copy(), errors="coerce")
    target.name = y.name or "target"
    current = pd.to_numeric(current_pb.copy(), errors="coerce")
    current.name = "current_pb"
    aligned = X[selected].join([target, current], how="inner")
    aligned = aligned.dropna(subset=[target.name, "current_pb"])
    if aligned.empty:
        raise ValueError("No aligned non-null P/B target observations.")
    return (
        aligned[selected].copy(),
        aligned[target.name].astype(float).copy(),
        aligned["current_pb"].astype(float).copy(),
    )


def _impute_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    test = x_test.copy()
    train[~np.isfinite(train)] = np.nan
    test[~np.isfinite(test)] = np.nan
    medians = np.zeros(train.shape[1], dtype=float)
    for col_idx in range(train.shape[1]):
        observed = train[:, col_idx][~np.isnan(train[:, col_idx])]
        if observed.size:
            medians[col_idx] = float(np.median(observed))
    for col_idx in range(train.shape[1]):
        train[np.isnan(train[:, col_idx]), col_idx] = medians[col_idx]
        test[np.isnan(test[:, col_idx]), col_idx] = medians[col_idx]
    return train, test


def _target_to_pb(values: pd.Series | np.ndarray, target_kind: str) -> Any:
    if target_kind == "pb":
        return values
    if target_kind == "log_pb":
        return np.exp(values)
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _pb_to_target(value: float, target_kind: str) -> float:
    if target_kind == "pb":
        return float(value)
    if target_kind == "log_pb":
        if value <= 0.0:
            return float("nan")
        return float(np.log(value))
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    target_kind: str,
    horizon_months: int,
    fold_count: int,
    n_features: int,
) -> dict[str, Any]:
    target_error = predictions["y_pred_target"] - predictions["y_true_target"]
    pb_error = predictions["y_pred_pb"] - predictions["y_true_pb"]
    return {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "target_kind": target_kind,
        "n_obs": int(len(predictions)),
        "fold_count": int(fold_count),
        "n_features": int(n_features),
        "target_mae": float(np.mean(np.abs(target_error))),
        "target_rmse": float(np.sqrt(np.mean(np.square(target_error)))),
        "pb_mae": float(np.mean(np.abs(pb_error))),
        "pb_rmse": float(np.sqrt(np.mean(np.square(pb_error)))),
        "mean_predicted_pb": float(predictions["y_pred_pb"].mean()),
        "mean_realized_pb": float(predictions["y_true_pb"].mean()),
    }


def _rows_for_fold(
    *,
    X_aligned: pd.DataFrame,
    y_aligned: pd.Series,
    current_aligned: pd.Series,
    target_kind: str,
    fold_idx: int,
    test_idx: np.ndarray,
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    y_pred_target: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true_target = y_aligned.iloc[test_idx].to_numpy(dtype=float)
    y_true_pb = _target_to_pb(y_true_target, target_kind)
    y_pred_pb = _target_to_pb(y_pred_target, target_kind)
    current_values = current_aligned.iloc[test_idx].to_numpy(dtype=float)
    for offset, row_idx in enumerate(test_idx):
        rows.append(
            {
                "date": X_aligned.index[row_idx],
                "fold_idx": int(fold_idx),
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
                "current_pb": float(current_values[offset]),
                "y_true_target": float(y_true_target[offset]),
                "y_pred_target": float(y_pred_target[offset]),
                "y_true_pb": float(y_true_pb[offset]),
                "y_pred_pb": float(y_pred_pb[offset]),
            }
        )
    return rows


def evaluate_pb_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_pb: pd.Series,
    model_name: str,
    feature_columns: list[str],
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one P/B regressor with strict WFO splits."""
    X_aligned, y_aligned, current_aligned = _align_inputs(
        X,
        y,
        current_pb,
        feature_columns,
    )
    _, splitter = iter_absolute_wfo_splits(
        len(X_aligned),
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
    )
    rows: list[dict[str, Any]] = []
    fold_count = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_aligned)):
        x_train, x_test = _impute_fold(
            X_aligned.iloc[train_idx].to_numpy(dtype=float),
            X_aligned.iloc[test_idx].to_numpy(dtype=float),
        )
        model = build_pb_regressor(model_name)
        model.fit(x_train, y_aligned.iloc[train_idx].to_numpy(dtype=float))
        rows.extend(
            _rows_for_fold(
                X_aligned=X_aligned,
                y_aligned=y_aligned,
                current_aligned=current_aligned,
                target_kind=target_kind,
                fold_idx=fold_idx,
                test_idx=test_idx,
                train_start_date=X_aligned.index[train_idx[0]],
                train_end_date=X_aligned.index[train_idx[-1]],
                y_pred_target=model.predict(x_test),
            )
        )
        fold_count += 1
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = _metrics_from_predictions(
        predictions,
        model_name=model_name,
        target_kind=target_kind,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=len(X_aligned.columns),
    )
    return predictions, metrics


def evaluate_pb_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_pb: pd.Series,
    baseline_name: str,
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a P/B baseline with fold-local history."""
    X_aligned, y_aligned, current_aligned = _align_inputs(
        X,
        y,
        current_pb,
        list(X.columns),
    )
    _, splitter = iter_absolute_wfo_splits(
        len(X_aligned),
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
    )
    rows: list[dict[str, Any]] = []
    fold_count = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_aligned)):
        if baseline_name == "no_change_pb":
            y_pred_target = [
                _pb_to_target(float(value), target_kind)
                for value in current_aligned.iloc[test_idx].to_numpy(dtype=float)
            ]
        elif baseline_name == "drift_pb":
            train_pb = _target_to_pb(
                y_aligned.iloc[train_idx].to_numpy(dtype=float),
                target_kind,
            )
            y_pred_target = np.repeat(
                _pb_to_target(float(np.mean(train_pb)), target_kind),
                len(test_idx),
            )
        else:
            raise ValueError(f"Unsupported P/B baseline '{baseline_name}'.")
        rows.extend(
            _rows_for_fold(
                X_aligned=X_aligned,
                y_aligned=y_aligned,
                current_aligned=current_aligned,
                target_kind=target_kind,
                fold_idx=fold_idx,
                test_idx=test_idx,
                train_start_date=X_aligned.index[train_idx[0]],
                train_end_date=X_aligned.index[train_idx[-1]],
                y_pred_target=np.asarray(y_pred_target, dtype=float),
            )
        )
        fold_count += 1
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = _metrics_from_predictions(
        predictions,
        model_name=baseline_name,
        target_kind=target_kind,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=0,
    )
    return predictions, metrics


def combine_decomposition_predictions(
    bvps_predictions: pd.DataFrame,
    pb_predictions: pd.DataFrame,
    *,
    horizon_months: int,
    bvps_model_name: str,
    pb_model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine fold-aligned BVPS and P/B predictions into future price."""
    merged = bvps_predictions.merge(
        pb_predictions,
        on=["date", "fold_idx"],
        how="inner",
        suffixes=("_bvps", "_pb"),
    )
    if merged.empty:
        raise ValueError("No aligned BVPS and P/B predictions to combine.")
    merged["implied_future_price"] = (
        merged["implied_future_bvps"] * merged["y_pred_pb"]
    )
    merged["true_future_price"] = (
        merged["true_future_bvps"] * merged["y_true_pb"]
    )
    price_error = merged["implied_future_price"] - merged["true_future_price"]
    current_price = merged["current_bvps"] * merged["current_pb"]
    true_up = merged["true_future_price"] > current_price
    predicted_up = merged["implied_future_price"] > current_price
    model_name = f"{bvps_model_name}__{pb_model_name}"
    metrics = {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "bvps_model_name": bvps_model_name,
        "pb_model_name": pb_model_name,
        "n_obs": int(len(merged)),
        "implied_price_mae": float(np.mean(np.abs(price_error))),
        "implied_price_rmse": float(np.sqrt(np.mean(np.square(price_error)))),
        "directional_hit_rate": float(np.mean(predicted_up == true_up)),
    }
    return merged.sort_values("date").reset_index(drop=True), metrics


def summarize_decomposition_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank recombined x5 decomposition rows by horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby("horizon_months", dropna=False):
        ranked = group.sort_values(
            [
                "implied_price_mae",
                "implied_price_rmse",
                "directional_hit_rate",
                "model_name",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)
