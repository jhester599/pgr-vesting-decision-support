"""Research-only x21 helpers for dividend size target-scale experiments."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric > 1e-12)
    return numeric


def build_scaled_size_targets(annual: pd.DataFrame) -> pd.DataFrame:
    """Add normalized dividend-size targets to an annual frame."""
    result = annual.copy()
    raw = pd.to_numeric(result["special_dividend_excess"], errors="coerce")
    result["target_raw_dollars"] = raw
    result["target_to_current_bvps"] = raw / _safe_scale(result["current_bvps"])
    result["target_to_persistent_bvps"] = raw / _safe_scale(result["persistent_bvps"])
    result["target_to_price"] = raw / _safe_scale(result["close_price"])
    return result


def back_transform_scaled_prediction(prediction: float, scale_value: float) -> float:
    """Convert a scaled size prediction back to dollars conservatively."""
    if not np.isfinite(scale_value) or scale_value <= 1e-12:
        return float("nan")
    return float(max(prediction, 0.0) * scale_value)


def iter_positive_expanding_splits(
    frame: pd.DataFrame,
    *,
    min_train_years: int = 4,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding splits for positive-only annual size modeling."""
    ordered = frame.sort_index()
    for test_pos in range(min_train_years, len(ordered)):
        yield np.arange(0, test_pos), np.array([test_pos])


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


def _aligned_scale_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    scale_column: str,
) -> pd.DataFrame:
    selected = [column for column in feature_columns if column in frame.columns]
    if not selected:
        raise ValueError("No requested feature columns are present.")
    columns = selected + [target_column, scale_column, "special_dividend_excess"]
    ordered_unique = list(dict.fromkeys(columns))
    aligned = frame[ordered_unique].copy()
    aligned = aligned.sort_index()
    aligned = aligned[aligned["special_dividend_excess"] > 0.0]
    aligned[target_column] = pd.to_numeric(aligned[target_column], errors="coerce")
    aligned[scale_column] = pd.to_numeric(aligned[scale_column], errors="coerce")
    aligned["special_dividend_excess"] = pd.to_numeric(
        aligned["special_dividend_excess"],
        errors="coerce",
    )
    aligned = aligned.dropna(subset=[target_column, scale_column, "special_dividend_excess"])
    aligned = aligned[aligned[scale_column] > 1e-12]
    if aligned.empty:
        raise ValueError("No positive annual observations with a valid target scale.")
    return aligned


def _predict_scaled_size(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
) -> float:
    if model_name == "historical_scaled_mean" or len(y_train) < 2:
        return float(np.mean(y_train))
    if model_name == "ridge_scaled":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1000.0)),
            ]
        )
        model.fit(x_train, y_train)
        prediction = float(model.predict(x_test)[0])
        return min(max(prediction, 0.0), float(np.max(y_train)))
    raise ValueError(f"Unsupported x21 model '{model_name}'.")


def evaluate_scaled_size_model(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    scale_column: str,
    target_scale_name: str,
    model_name: str,
    min_train_years: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one positive-only annual size model with back-transformed dollars."""
    aligned = _aligned_scale_frame(
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        scale_column=scale_column,
    )
    features = [feature for feature in feature_columns if feature in aligned.columns]
    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        iter_positive_expanding_splits(aligned, min_train_years=min_train_years)
    ):
        x_train, x_test = _impute_fold(
            aligned[features].iloc[train_idx].to_numpy(dtype=float),
            aligned[features].iloc[test_idx].to_numpy(dtype=float),
        )
        y_train = aligned[target_column].iloc[train_idx].to_numpy(dtype=float)
        scaled_pred = _predict_scaled_size(
            x_train,
            x_test,
            y_train,
            model_name=model_name,
        )
        scale_value = float(aligned[scale_column].iloc[test_idx[0]])
        dollar_pred = back_transform_scaled_prediction(scaled_pred, scale_value)
        rows.append(
            {
                "snapshot_date": aligned.index[test_idx[0]],
                "fold_idx": int(fold_idx),
                "train_start_date": aligned.index[train_idx[0]],
                "train_end_date": aligned.index[train_idx[-1]],
                "target_scale": target_scale_name,
                "scale_column": scale_column,
                "model_name": model_name,
                "actual_scaled_target": float(aligned[target_column].iloc[test_idx[0]]),
                "predicted_scaled_target": float(max(scaled_pred, 0.0)),
                "actual_dollars": float(aligned["special_dividend_excess"].iloc[test_idx[0]]),
                "predicted_dollars": float(dollar_pred),
            }
        )
    detail = pd.DataFrame(rows)
    metrics = {
        "target_scale": target_scale_name,
        "scale_column": scale_column,
        "model_name": model_name,
        "n_obs": int(len(detail)),
        "n_features": int(len(features)),
        "dollar_mae": float(np.mean(np.abs(detail["predicted_dollars"] - detail["actual_dollars"]))),
        "dollar_rmse": float(
            np.sqrt(np.mean(np.square(detail["predicted_dollars"] - detail["actual_dollars"])))
        ),
        "scaled_mae": float(
            np.mean(np.abs(detail["predicted_scaled_target"] - detail["actual_scaled_target"]))
        ),
        "mean_actual_dollars": float(detail["actual_dollars"].mean()),
        "mean_predicted_dollars": float(detail["predicted_dollars"].mean()),
    }
    return detail, metrics


def summarize_scaled_size_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x21 rows by dollar error."""
    if detail_df.empty:
        return pd.DataFrame()
    ranked = detail_df.copy()
    if "scaled_mae" not in ranked.columns:
        ranked["scaled_mae"] = np.nan
    ranked = ranked.sort_values(
        ["dollar_mae", "scaled_mae", "target_scale", "model_name"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked
