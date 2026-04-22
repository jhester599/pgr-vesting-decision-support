"""Research-only x4 BVPS forecasting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.x2_absolute_classification import iter_absolute_wfo_splits


def normalize_bvps_monthly(
    pgr_monthly: pd.DataFrame,
    filing_lag_months: int = 2,
) -> pd.Series:
    """Return filing-lagged BVPS indexed to last business day of each month."""
    if "book_value_per_share" not in pgr_monthly.columns:
        raise ValueError("pgr_monthly must include book_value_per_share")
    result = pd.to_numeric(
        pgr_monthly["book_value_per_share"].copy(),
        errors="coerce",
    )
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if filing_lag_months > 0:
        result.index = result.index.shift(filing_lag_months, freq="MS")
    calendar_month_end = result.index + pd.offsets.MonthEnd(0)
    result.index = pd.DatetimeIndex(
        [pd.offsets.BMonthEnd().rollback(ts) for ts in calendar_month_end],
        name=result.index.name,
    )
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result.name = "current_bvps"
    return result


def build_bvps_regressor(model_name: str) -> Pipeline:
    """Build an x4 BVPS regressor by name."""
    if model_name in {"ridge_bvps_growth", "ridge_log_bvps_growth"}:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=5000.0)),
            ]
        )
    if model_name in {"hist_gbt_bvps_growth", "hist_gbt_log_bvps_growth"}:
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
    raise ValueError(f"Unsupported BVPS regressor '{model_name}'.")


def _align_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    current_bvps: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    selected = [feature for feature in feature_columns if feature in X.columns]
    if not selected:
        raise ValueError("No requested feature columns are present in X.")
    target = pd.to_numeric(y.copy(), errors="coerce")
    target.name = y.name or "target"
    current = pd.to_numeric(current_bvps.copy(), errors="coerce")
    current.name = "current_bvps"
    aligned = X[selected].join([target, current], how="inner")
    aligned = aligned.dropna(subset=[target.name, "current_bvps"])
    if aligned.empty:
        raise ValueError("No aligned non-null BVPS target observations.")
    return (
        aligned[selected].copy(),
        aligned[target.name].astype(float).copy(),
        aligned["current_bvps"].astype(float).copy(),
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


def _target_to_growth(values: pd.Series | np.ndarray, target_kind: str) -> Any:
    if target_kind == "growth":
        return values
    if target_kind == "log_growth":
        return np.expm1(values)
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _growth_to_target(value: float, target_kind: str) -> float:
    if target_kind == "growth":
        return float(value)
    if target_kind == "log_growth":
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
    growth_error = predictions["y_pred_growth"] - predictions["y_true_growth"]
    bvps_error = predictions["implied_future_bvps"] - predictions["true_future_bvps"]
    realized_up = predictions["y_true_growth"] > 0.0
    predicted_up = predictions["y_pred_growth"] > 0.0
    return {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "target_kind": target_kind,
        "n_obs": int(len(predictions)),
        "fold_count": int(fold_count),
        "n_features": int(n_features),
        "target_mae": float(np.mean(np.abs(target_error))),
        "target_rmse": float(np.sqrt(np.mean(np.square(target_error)))),
        "growth_mae": float(np.mean(np.abs(growth_error))),
        "growth_rmse": float(np.sqrt(np.mean(np.square(growth_error)))),
        "future_bvps_mae": float(np.mean(np.abs(bvps_error))),
        "future_bvps_rmse": float(np.sqrt(np.mean(np.square(bvps_error)))),
        "directional_hit_rate": float(np.mean(realized_up == predicted_up)),
        "spearman_ic": _spearman_ic(
            predictions["y_true_growth"],
            predictions["y_pred_growth"],
        ),
        "mean_predicted_growth": float(predictions["y_pred_growth"].mean()),
        "mean_realized_growth": float(predictions["y_true_growth"].mean()),
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
    y_true_growth = _target_to_growth(y_true_target, target_kind)
    y_pred_growth = _target_to_growth(y_pred_target, target_kind)
    current_values = current_aligned.iloc[test_idx].to_numpy(dtype=float)
    for offset, row_idx in enumerate(test_idx):
        rows.append(
            {
                "date": X_aligned.index[row_idx],
                "fold_idx": int(fold_idx),
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
                "current_bvps": float(current_values[offset]),
                "y_true_target": float(y_true_target[offset]),
                "y_pred_target": float(y_pred_target[offset]),
                "y_true_growth": float(y_true_growth[offset]),
                "y_pred_growth": float(y_pred_growth[offset]),
                "true_future_bvps": float(
                    current_values[offset] * (1.0 + y_true_growth[offset])
                ),
                "implied_future_bvps": float(
                    current_values[offset] * (1.0 + y_pred_growth[offset])
                ),
            }
        )
    return rows


def evaluate_bvps_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_bvps: pd.Series,
    model_name: str,
    feature_columns: list[str],
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one BVPS-growth regressor with strict WFO splits."""
    X_aligned, y_aligned, current_aligned = _align_inputs(
        X,
        y,
        current_bvps,
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
        model = build_bvps_regressor(model_name)
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
    if predictions.empty:
        raise ValueError(f"No OOS BVPS predictions produced for {model_name}.")
    metrics = _metrics_from_predictions(
        predictions,
        model_name=model_name,
        target_kind=target_kind,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=len(X_aligned.columns),
    )
    return predictions, metrics


def evaluate_bvps_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_bvps: pd.Series,
    baseline_name: str,
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a BVPS baseline with fold-local history."""
    X_aligned, y_aligned, current_aligned = _align_inputs(
        X,
        y,
        current_bvps,
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
        if baseline_name == "no_change_bvps":
            pred_value = 0.0
        elif baseline_name == "drift_bvps_growth":
            y_train = y_aligned.iloc[train_idx].to_numpy(dtype=float)
            growth_train = _target_to_growth(y_train, target_kind)
            pred_value = _growth_to_target(
                float(np.mean(growth_train)),
                target_kind,
            )
        else:
            raise ValueError(f"Unsupported BVPS baseline '{baseline_name}'.")
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
                y_pred_target=np.repeat(pred_value, len(test_idx)),
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


def summarize_bvps_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x4 model and baseline rows by horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby("horizon_months", dropna=False):
        ranked = group.sort_values(
            [
                "future_bvps_mae",
                "growth_rmse",
                "directional_hit_rate",
                "model_name",
                "target_kind",
            ],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        baseline_rows = group[group["model_name"] == "no_change_bvps"]
        baseline_mae = (
            float(baseline_rows["future_bvps_mae"].iloc[0])
            if not baseline_rows.empty
            else float("nan")
        )
        baseline_rmse = (
            float(baseline_rows["growth_rmse"].iloc[0])
            if not baseline_rows.empty
            else float("nan")
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            out["beats_no_change_bvps"] = bool(
                row["future_bvps_mae"] < baseline_mae
                and row["growth_rmse"] < baseline_rmse
            )
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)
