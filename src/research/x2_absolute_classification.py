"""Research-only x2 absolute direction classification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.models.evaluation import summarize_binary_predictions


@dataclass(frozen=True)
class SplitConfig:
    """Fold sizing for research WFO evaluation."""

    train_window_months: int
    test_window_months: int
    gap_months: int


def resolve_split_config(
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> SplitConfig:
    """Resolve x2 WFO split settings from repo configuration."""
    if purge_buffer is None:
        purge_buffer = (
            config.WFO_PURGE_BUFFER_6M
            if target_horizon_months <= 6
            else config.WFO_PURGE_BUFFER_12M
        )
    return SplitConfig(
        train_window_months=(
            config.WFO_TRAIN_WINDOW_MONTHS
            if train_window_months is None
            else int(train_window_months)
        ),
        test_window_months=(
            config.WFO_TEST_WINDOW_MONTHS
            if test_window_months is None
            else int(test_window_months)
        ),
        gap_months=int(target_horizon_months + purge_buffer),
    )


def iter_absolute_wfo_splits(
    n_obs: int,
    *,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[SplitConfig, TimeSeriesSplit]:
    """Return the WFO split config and splitter used by x2."""
    split_config = resolve_split_config(
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
    )
    min_required = (
        split_config.train_window_months
        + split_config.gap_months
        + split_config.test_window_months
    )
    if n_obs < min_required:
        raise ValueError(
            f"Dataset has only {n_obs} observations; need at least "
            f"{min_required}."
        )
    available = n_obs - split_config.train_window_months - split_config.gap_months
    n_splits = max(1, available // split_config.test_window_months)
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=split_config.train_window_months,
        test_size=split_config.test_window_months,
        gap=split_config.gap_months,
    )
    return split_config, splitter


def build_absolute_classifier_pipeline(model_name: str) -> Pipeline:
    """Build an x2 classifier pipeline by name."""
    if model_name == "logistic_l2_balanced":
        return Pipeline(
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
    if model_name == "hist_gbt_depth2":
        return Pipeline(
            steps=[
                (
                    "classifier",
                    HistGradientBoostingClassifier(
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
    raise ValueError(f"Unsupported classifier '{model_name}'.")


def _align_xy(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    selected = [feature for feature in feature_columns if feature in X.columns]
    if not selected:
        raise ValueError("No requested feature columns are present in X.")
    aligned = X[selected].join(y, how="inner").dropna(subset=[y.name])
    if aligned.empty:
        raise ValueError("No aligned non-null target observations.")
    return aligned[selected].copy(), aligned[y.name].astype(int).copy()


def _impute_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    test = x_test.copy()
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(train.shape[1]):
        train[np.isnan(train[:, col_idx]), col_idx] = medians[col_idx]
        test[np.isnan(test[:, col_idx]), col_idx] = medians[col_idx]
    return train, test


def _metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    horizon_months: int,
    fold_count: int,
    n_features: int,
) -> dict[str, Any]:
    y_prob = predictions["y_prob"].astype(float)
    y_true = predictions["y_true"].astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found",
            category=UserWarning,
        )
        binary = summarize_binary_predictions(y_prob, y_true)
    clipped = np.clip(y_prob.to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "n_obs": int(binary.n_obs),
        "fold_count": int(fold_count),
        "n_features": int(n_features),
        "brier_score": float(binary.brier_score),
        "accuracy": float(binary.accuracy),
        "balanced_accuracy": float(binary.balanced_accuracy),
        "precision": float(binary.precision),
        "recall": float(binary.recall),
        "base_rate": float(binary.base_rate),
        "predicted_positive_rate": float(binary.predicted_positive_rate),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def evaluate_absolute_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    feature_columns: list[str],
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one absolute direction classifier with strict WFO splits."""
    X_aligned, y_aligned = _align_xy(X, y, feature_columns)
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
        y_train = y_aligned.iloc[train_idx].to_numpy(dtype=int)
        if len(np.unique(y_train)) < 2:
            continue
        x_train, x_test = _impute_fold(
            X_aligned.iloc[train_idx].to_numpy(dtype=float),
            X_aligned.iloc[test_idx].to_numpy(dtype=float),
        )
        model = build_absolute_classifier_pipeline(model_name)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        for offset, row_idx in enumerate(test_idx):
            rows.append(
                {
                    "date": X_aligned.index[row_idx],
                    "fold_idx": int(fold_idx),
                    "y_true": int(y_aligned.iloc[row_idx]),
                    "y_prob": float(y_prob[offset]),
                }
            )
        fold_count += 1

    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if predictions.empty:
        raise ValueError(f"No OOS predictions produced for {model_name}.")
    metrics = _metrics_from_predictions(
        predictions,
        model_name=model_name,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=len(X_aligned.columns),
    )
    return predictions, metrics


def evaluate_absolute_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    baseline_name: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a simple absolute-direction baseline with fold-local history."""
    X_aligned, y_aligned = _align_xy(X, y, list(X.columns))
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
        y_train = y_aligned.iloc[train_idx].to_numpy(dtype=int)
        if baseline_name == "base_rate":
            prob = float(np.mean(y_train))
        elif baseline_name == "always_up":
            prob = 1.0
        else:
            raise ValueError(f"Unsupported baseline '{baseline_name}'.")
        for row_idx in test_idx:
            rows.append(
                {
                    "date": X_aligned.index[row_idx],
                    "fold_idx": int(fold_idx),
                    "y_true": int(y_aligned.iloc[row_idx]),
                    "y_prob": prob,
                }
            )
        fold_count += 1

    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = _metrics_from_predictions(
        predictions,
        model_name=baseline_name,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=0,
    )
    return predictions, metrics


def summarize_absolute_classification_results(
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank x2 model and baseline rows by horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby("horizon_months", dropna=False):
        ranked = group.sort_values(
            ["balanced_accuracy", "brier_score", "log_loss"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        base_rate_rows = group[group["model_name"] == "base_rate"]
        base_rate_brier = (
            float(base_rate_rows["brier_score"].iloc[0])
            if not base_rate_rows.empty
            else float("nan")
        )
        base_rate_ba = (
            float(base_rate_rows["balanced_accuracy"].iloc[0])
            if not base_rate_rows.empty
            else float("nan")
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            out["beats_base_rate"] = bool(
                row["balanced_accuracy"] > base_rate_ba
                and row["brier_score"] < base_rate_brier
            )
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)
