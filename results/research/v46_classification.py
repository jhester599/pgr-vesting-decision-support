"""v46 - Binary classification: will PGR outperform the benchmark over 6M?"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.models.evaluation import summarize_binary_predictions
from src.models.regularized_models import AdaptiveGapTimeSeriesSplit
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    print_footer,
    print_header,
    save_results,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute the binary metric bundle from probabilities."""
    y_true_series = pd.Series(y_true, name="y_true")
    y_prob_series = pd.Series(y_prob, name="y_prob")
    summary = summarize_binary_predictions(y_prob_series, y_true_series)
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "n": float(summary.n_obs),
        "accuracy": float(summary.accuracy),
        "balanced_accuracy": float(summary.balanced_accuracy),
        "brier_score": float(summary.brier_score),
        "log_loss": float(log_loss(y_true, y_prob_clipped, labels=[0, 1])),
        "precision": float(summary.precision),
        "recall": float(summary.recall),
        "base_rate": float(summary.base_rate),
        "predicted_positive_rate": float(summary.predicted_positive_rate),
    }


def classification_wfo(
    x_values: np.ndarray,
    y_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOS binary labels and predicted probabilities."""
    n_obs = len(x_values)
    available = n_obs - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    outer_splitter = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )

    all_y_true: list[int] = []
    all_y_prob: list[float] = []

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

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    return np.asarray(all_y_true, dtype=int), np.asarray(all_y_prob, dtype=float)


def main() -> None:
    """Run the phase-2 binary classification experiment."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        rows: list[dict[str, float | str]] = []
        pooled_true: list[np.ndarray] = []
        pooled_prob: list[np.ndarray] = []

        print_header("v46", "Binary Classification - Logistic Regression CV")
        print(f"\n  {'Benchmark':<10}  {'N':>5}  {'Accuracy':>9}  {'BalAcc':>8}  {'Brier':>7}  {'LogLoss':>8}")

        for etf in BENCHMARKS:
            rel_series = load_relative_series(conn, etf, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            x_values = x_df[feature_cols].to_numpy()
            y_binary = (y.to_numpy() > 0).astype(int)

            y_true, y_prob = classification_wfo(x_values, y_binary)
            if len(y_true) == 0:
                continue

            metrics = compute_binary_metrics(y_true, y_prob)
            print(
                f"  {etf:<10}  {int(metrics['n']):>5}  {metrics['accuracy']:>9.4f}  "
                f"{metrics['balanced_accuracy']:>8.4f}  {metrics['brier_score']:>7.4f}  "
                f"{metrics['log_loss']:>8.4f}"
            )
            rows.append({"benchmark": etf, "version": "v46", **metrics})
            pooled_true.append(y_true)
            pooled_prob.append(y_prob)

        if pooled_true:
            pooled_metrics = compute_binary_metrics(
                np.concatenate(pooled_true),
                np.concatenate(pooled_prob),
            )
            print(
                f"\n  {'POOLED':<10}  {int(pooled_metrics['n']):>5}  {pooled_metrics['accuracy']:>9.4f}  "
                f"{pooled_metrics['balanced_accuracy']:>8.4f}  {pooled_metrics['brier_score']:>7.4f}  "
                f"{pooled_metrics['log_loss']:>8.4f}"
            )
            rows.append({"benchmark": "POOLED", "version": "v46", **pooled_metrics})

        print_footer()
        save_results(pd.DataFrame(rows), "v46_classification_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
