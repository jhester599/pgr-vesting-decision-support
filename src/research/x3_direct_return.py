"""Research-only x3 direct forward-return regression utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.x2_absolute_classification import iter_absolute_wfo_splits


def build_direct_return_regressor(model_name: str) -> Pipeline:
    """Build an x3 direct-return regressor by name."""
    if model_name in {"ridge_return", "ridge_log_return"}:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=5000.0)),
            ]
        )
    if model_name in {"hist_gbt_return", "hist_gbt_log_return"}:
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
    raise ValueError(f"Unsupported regressor '{model_name}'.")


def _align_regression_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    current_price: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    selected = [feature for feature in feature_columns if feature in X.columns]
    if not selected:
        raise ValueError("No requested feature columns are present in X.")

    target = pd.to_numeric(y.copy(), errors="coerce")
    target.name = y.name or "target"
    prices = pd.to_numeric(current_price.copy(), errors="coerce")
    prices.name = "current_price"
    aligned = X[selected].join([target, prices], how="inner")
    aligned = aligned.dropna(subset=[target.name, "current_price"])
    if aligned.empty:
        raise ValueError("No aligned non-null target observations.")
    return (
        aligned[selected].copy(),
        aligned[target.name].astype(float).copy(),
        aligned["current_price"].astype(float).copy(),
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


def _target_to_simple_return(values: pd.Series | np.ndarray, target_kind: str) -> Any:
    if target_kind == "return":
        return values
    if target_kind == "log_return":
        return np.expm1(values)
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _simple_return_to_target(value: float, target_kind: str) -> float:
    if target_kind == "return":
        return float(value)
    if target_kind == "log_return":
        if value <= -1.0:
            return float("nan")
        return float(np.log1p(value))
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _spearman_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2 or y_pred.nunique(dropna=True) < 2:
        return float("nan")
    corr = y_true.corr(y_pred, method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


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
    return_error = predictions["y_pred_return"] - predictions["y_true_return"]
    price_error = (
        predictions["implied_future_price"] - predictions["true_future_price"]
    )
    realized_up = predictions["y_true_return"] > 0.0
    predicted_up = predictions["y_pred_return"] > 0.0
    return {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "target_kind": target_kind,
        "n_obs": int(len(predictions)),
        "fold_count": int(fold_count),
        "n_features": int(n_features),
        "target_mae": float(np.mean(np.abs(target_error))),
        "target_rmse": float(np.sqrt(np.mean(np.square(target_error)))),
        "return_mae": float(np.mean(np.abs(return_error))),
        "return_rmse": float(np.sqrt(np.mean(np.square(return_error)))),
        "implied_price_mae": float(np.mean(np.abs(price_error))),
        "implied_price_rmse": float(np.sqrt(np.mean(np.square(price_error)))),
        "directional_hit_rate": float(np.mean(realized_up == predicted_up)),
        "spearman_ic": _spearman_ic(
            predictions["y_true_return"],
            predictions["y_pred_return"],
        ),
        "mean_predicted_return": float(predictions["y_pred_return"].mean()),
        "mean_realized_return": float(predictions["y_true_return"].mean()),
    }


def _prediction_rows(
    *,
    X_aligned: pd.DataFrame,
    y_aligned: pd.Series,
    current_aligned: pd.Series,
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None,
    train_window_months: int | None,
    test_window_months: int | None,
    model_name: str,
) -> tuple[list[dict[str, Any]], int]:
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
        model = build_direct_return_regressor(model_name)
        model.fit(x_train, y_aligned.iloc[train_idx].to_numpy(dtype=float))
        y_pred_target = model.predict(x_test)
        train_start_date = X_aligned.index[train_idx[0]]
        train_end_date = X_aligned.index[train_idx[-1]]
        rows.extend(
            _rows_for_fold(
                X_aligned=X_aligned,
                y_aligned=y_aligned,
                current_aligned=current_aligned,
                target_kind=target_kind,
                fold_idx=fold_idx,
                test_idx=test_idx,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                y_pred_target=y_pred_target,
            )
        )
        fold_count += 1
    return rows, fold_count


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
    y_true_return = _target_to_simple_return(y_true_target, target_kind)
    y_pred_return = _target_to_simple_return(y_pred_target, target_kind)
    current_prices = current_aligned.iloc[test_idx].to_numpy(dtype=float)
    for offset, row_idx in enumerate(test_idx):
        rows.append(
            {
                "date": X_aligned.index[row_idx],
                "fold_idx": int(fold_idx),
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
                "current_price": float(current_prices[offset]),
                "y_true_target": float(y_true_target[offset]),
                "y_pred_target": float(y_pred_target[offset]),
                "y_true_return": float(y_true_return[offset]),
                "y_pred_return": float(y_pred_return[offset]),
                "true_future_price": float(
                    current_prices[offset] * (1.0 + y_true_return[offset])
                ),
                "implied_future_price": float(
                    current_prices[offset] * (1.0 + y_pred_return[offset])
                ),
            }
        )
    return rows


def evaluate_direct_return_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_price: pd.Series,
    model_name: str,
    feature_columns: list[str],
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one direct-return regressor with strict WFO splits."""
    X_aligned, y_aligned, current_aligned = _align_regression_inputs(
        X,
        y,
        current_price,
        feature_columns,
    )
    rows, fold_count = _prediction_rows(
        X_aligned=X_aligned,
        y_aligned=y_aligned,
        current_aligned=current_aligned,
        target_kind=target_kind,
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
        model_name=model_name,
    )
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if predictions.empty:
        raise ValueError(f"No OOS predictions produced for {model_name}.")
    metrics = _metrics_from_predictions(
        predictions,
        model_name=model_name,
        target_kind=target_kind,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=len(X_aligned.columns),
    )
    return predictions, metrics


def evaluate_direct_return_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_price: pd.Series,
    baseline_name: str,
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a direct-return baseline with fold-local history."""
    X_aligned, y_aligned, current_aligned = _align_regression_inputs(
        X,
        y,
        current_price,
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
        if baseline_name == "no_change":
            pred_value = 0.0
        elif baseline_name == "drift":
            y_train = y_aligned.iloc[train_idx].to_numpy(dtype=float)
            simple_train = _target_to_simple_return(y_train, target_kind)
            pred_value = _simple_return_to_target(
                float(np.mean(simple_train)),
                target_kind,
            )
        else:
            raise ValueError(f"Unsupported baseline '{baseline_name}'.")
        y_pred_target = np.repeat(pred_value, len(test_idx)).astype(float)
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
                y_pred_target=y_pred_target,
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


def summarize_direct_return_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x3 model and baseline rows by horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    group_cols = ["horizon_months"]
    for _, group in detail_df.groupby(group_cols, dropna=False):
        ranked = group.sort_values(
            ["implied_price_mae", "return_rmse", "directional_hit_rate"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        no_change_rows = group[group["model_name"] == "no_change"]
        no_change_mae = (
            float(no_change_rows["implied_price_mae"].iloc[0])
            if not no_change_rows.empty
            else float("nan")
        )
        no_change_rmse = (
            float(no_change_rows["return_rmse"].iloc[0])
            if not no_change_rows.empty
            else float("nan")
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            out["beats_no_change"] = bool(
                row["implied_price_mae"] < no_change_mae
                and row["return_rmse"] < no_change_rmse
            )
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)
