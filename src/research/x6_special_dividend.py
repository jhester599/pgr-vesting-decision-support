"""Research-only x6 special-dividend two-stage utilities."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def iter_annual_expanding_splits(
    frame: pd.DataFrame,
    *,
    min_train_years: int = 8,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding annual train/test splits with one held-out year."""
    if min_train_years <= 0:
        raise ValueError("min_train_years must be positive")
    ordered = frame.sort_index()
    n_obs = len(ordered)
    for test_pos in range(min_train_years, n_obs):
        train_idx = np.arange(0, test_pos)
        test_idx = np.array([test_pos])
        yield train_idx, test_idx


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


def _align_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    required = ["special_dividend_occurred", "special_dividend_excess"]
    selected = [feature for feature in feature_columns if feature in frame.columns]
    if not selected:
        raise ValueError("No requested feature columns are present.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required special-dividend columns: {missing}")
    aligned = frame[selected + required].copy()
    aligned = aligned.sort_index()
    aligned[required] = aligned[required].apply(pd.to_numeric, errors="coerce")
    aligned = aligned.dropna(subset=required)
    if aligned.empty:
        raise ValueError("No labeled annual special-dividend observations.")
    return aligned


def _stage1_probability(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
) -> float:
    if model_name == "historical_rate" or len(np.unique(y_train)) < 2:
        return float(np.mean(y_train))
    if model_name == "logistic_l2_balanced":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        return float(model.predict_proba(x_test)[0, 1])
    raise ValueError(f"Unsupported stage-1 model '{model_name}'.")


def _stage2_size(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train_occ: np.ndarray,
    size_train: np.ndarray,
    model_name: str,
) -> float:
    positive_mask = (y_train_occ == 1) & (size_train > 0.0)
    positive_sizes = size_train[positive_mask]
    if positive_sizes.size == 0:
        return 0.0
    if model_name == "historical_positive_mean" or positive_sizes.size < 2:
        return float(np.mean(positive_sizes))
    if model_name == "ridge_positive_excess":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1000.0)),
            ]
        )
        model.fit(x_train[positive_mask], positive_sizes)
        prediction = max(float(model.predict(x_test)[0]), 0.0)
        return min(prediction, float(np.max(positive_sizes)))
    raise ValueError(f"Unsupported stage-2 model '{model_name}'.")


def evaluate_special_dividend_two_stage(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    stage1_model_name: str,
    stage2_model_name: str,
    min_train_years: int = 8,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Evaluate one annual two-stage special-dividend model."""
    aligned = _align_frame(frame, feature_columns)
    features = [feature for feature in feature_columns if feature in aligned.columns]
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        iter_annual_expanding_splits(aligned, min_train_years=min_train_years)
    ):
        x_train, x_test = _impute_fold(
            aligned[features].iloc[train_idx].to_numpy(dtype=float),
            aligned[features].iloc[test_idx].to_numpy(dtype=float),
        )
        y_train_occ = aligned["special_dividend_occurred"].iloc[
            train_idx
        ].to_numpy(dtype=int)
        size_train = aligned["special_dividend_excess"].iloc[
            train_idx
        ].to_numpy(dtype=float)
        prob = _stage1_probability(
            x_train,
            x_test,
            y_train_occ,
            stage1_model_name,
        )
        size = _stage2_size(
            x_train,
            x_test,
            y_train_occ,
            size_train,
            stage2_model_name,
        )
        actual_occ = int(aligned["special_dividend_occurred"].iloc[test_idx[0]])
        actual_size = float(aligned["special_dividend_excess"].iloc[test_idx[0]])
        rows.append(
            {
                "snapshot_date": aligned.index[test_idx[0]],
                "fold_idx": int(fold_idx),
                "train_start_date": aligned.index[train_idx[0]],
                "train_end_date": aligned.index[train_idx[-1]],
                "actual_occurred": actual_occ,
                "actual_excess": actual_size,
                "stage1_probability": float(np.clip(prob, 0.0, 1.0)),
                "conditional_size_prediction": float(max(size, 0.0)),
                "expected_value_prediction": float(
                    np.clip(prob, 0.0, 1.0) * max(size, 0.0)
                ),
            }
        )
    predictions = pd.DataFrame(rows)
    metrics = _metrics_from_predictions(
        predictions,
        stage1_model_name=stage1_model_name,
        stage2_model_name=stage2_model_name,
        n_features=len(features),
    )
    return predictions, metrics


def _balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    recalls: list[float] = []
    for label in (0, 1):
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def _metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    stage1_model_name: str,
    stage2_model_name: str,
    n_features: int,
) -> dict[str, float | int | str]:
    y_true = predictions["actual_occurred"].astype(int)
    y_prob = predictions["stage1_probability"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    size_mask = predictions["actual_occurred"].astype(int) == 1
    size_error = (
        predictions.loc[size_mask, "conditional_size_prediction"]
        - predictions.loc[size_mask, "actual_excess"]
    )
    ev_error = (
        predictions["expected_value_prediction"] - predictions["actual_excess"]
    )
    return {
        "stage1_model_name": stage1_model_name,
        "stage2_model_name": stage2_model_name,
        "model_name": f"{stage1_model_name}__{stage2_model_name}",
        "n_obs": int(len(predictions)),
        "n_features": int(n_features),
        "stage1_brier": float(np.mean(np.square(y_prob - y_true))),
        "stage1_balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "stage2_positive_mae": (
            float(np.mean(np.abs(size_error))) if len(size_error) else float("nan")
        ),
        "expected_value_mae": float(np.mean(np.abs(ev_error))),
        "expected_value_rmse": float(np.sqrt(np.mean(np.square(ev_error)))),
        "actual_positive_rate": float(y_true.mean()),
        "mean_expected_value_prediction": float(
            predictions["expected_value_prediction"].mean()
        ),
        "mean_actual_excess": float(predictions["actual_excess"].mean()),
    }


def summarize_special_dividend_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x6 two-stage rows by expected-value error."""
    if detail_df.empty:
        return pd.DataFrame()
    detail = detail_df.copy()
    if "model_name" not in detail.columns:
        detail["model_name"] = (
            detail["stage1_model_name"].astype(str)
            + "__"
            + detail["stage2_model_name"].astype(str)
        )
    ranked = detail.sort_values(
        ["expected_value_mae", "stage1_brier", "model_name"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked
