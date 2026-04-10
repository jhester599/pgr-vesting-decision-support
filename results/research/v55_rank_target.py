"""v55 - Rank-based target transforms within each WFO fold."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    load_research_baseline_results,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def transform_targets(y_train: np.ndarray, percentile: bool) -> np.ndarray:
    """Convert raw training targets to centered rank or percentile space."""
    if percentile:
        raw_ranks = rankdata(y_train, method="average")
        ranked = (raw_ranks - 0.5) / len(y_train)
    else:
        ranked = rankdata(y_train, method="average") / len(y_train)
    return ranked - 0.5


def rank_target_wfo(
    x: np.ndarray,
    y: np.ndarray,
    percentile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run WFO with a rank-transformed training target."""
    available = len(x) - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )

    all_y_true: list[float] = []
    all_y_hat: list[float] = []
    for train_idx, test_idx in tscv.split(x):
        x_train = x[train_idx].copy()
        x_test = x[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for col_idx in range(x_train.shape[1]):
            x_train[np.isnan(x_train[:, col_idx]), col_idx] = medians[col_idx]
            x_test[np.isnan(x_test[:, col_idx]), col_idx] = medians[col_idx]

        y_train_transformed = transform_targets(y_train, percentile=percentile)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
            ]
        )
        pipe.fit(x_train, y_train_transformed)
        y_hat = pipe.predict(x_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run rank-target experiments against the v38 research baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, percentile in [
            ("A_rank_transform", False),
            ("B_percentile_transform", True),
        ]:
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
                y_true, y_hat = rank_target_wfo(
                    x_df[feature_cols].to_numpy(),
                    y.to_numpy(),
                    percentile=percentile,
                )
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": benchmark,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )

            pooled = pool_metrics(rows)
            print_header("v55", f"Rank-Based Targets - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print("  Note: predictions are evaluated on raw y_true vs centered rank-space y_hat.")
            print("  IC and hit rate are more informative here than raw-scale R2.")
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={"variant": variant_name, "version": "v55"},
                )
            )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v55_rank_target_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
