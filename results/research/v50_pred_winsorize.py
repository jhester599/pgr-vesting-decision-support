"""v50 - Prediction winsorization: clip OOS predictions at training percentiles."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
CLIP_LEVELS: dict[str, tuple[float, float]] = {
    "A_p5_p95": (5.0, 95.0),
    "B_p10_p90": (10.0, 90.0),
}


def load_v38_alpha() -> float:
    """Load the best phase-1 shrinkage alpha, falling back to 0.5."""
    best_path = Path("results/research/v38_shrinkage_best_results.csv")
    if best_path.exists():
        df = pd.read_csv(best_path)
        pooled = df[df["benchmark"] == "POOLED"]
        if not pooled.empty and "optimal_alpha" in pooled.columns:
            return float(pooled.iloc[0]["optimal_alpha"])

    grid_path = Path("results/research/v38_shrinkage_results.csv")
    if grid_path.exists():
        df = pd.read_csv(grid_path)
        if {"alpha", "r2"}.issubset(df.columns):
            return float(df.loc[df["r2"].idxmax(), "alpha"])

    return 0.5


def pred_winsorize_wfo(
    x: np.ndarray,
    y: np.ndarray,
    clip_pct: tuple[float, float],
    post_shrink_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Ridge WFO and clip OOS predictions to training-prediction bounds."""
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

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
            ]
        )
        pipe.fit(x_train, y_train)

        y_hat_train = pipe.predict(x_train)
        lower = float(np.percentile(y_hat_train, clip_pct[0]))
        upper = float(np.percentile(y_hat_train, clip_pct[1]))
        y_hat_test = np.clip(pipe.predict(x_test), lower, upper)

        if post_shrink_alpha is not None:
            y_hat_test = post_shrink_alpha * y_hat_test

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat_test.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run prediction-winsorization experiments against the v38 baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        optimal_alpha = load_v38_alpha()
        output_frames: list[pd.DataFrame] = []

        for variant_name, clip_pct in CLIP_LEVELS.items():
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
                y_true, y_hat = pred_winsorize_wfo(
                    x_df[feature_cols].to_numpy(),
                    y.to_numpy(),
                    clip_pct=clip_pct,
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
            print_header("v50", f"Prediction Winsorization - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={"variant": variant_name, "version": "v50"},
                )
            )

        rows_combo: list[dict[str, object]] = []
        combo_variant = f"C_clip_shrink_alpha_{optimal_alpha:.2f}"
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            y_true, y_hat = pred_winsorize_wfo(
                x_df[feature_cols].to_numpy(),
                y.to_numpy(),
                clip_pct=CLIP_LEVELS["A_p5_p95"],
                post_shrink_alpha=optimal_alpha,
            )
            metrics = compute_metrics(y_true, y_hat)
            rows_combo.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled_combo = pool_metrics(rows_combo)
        print_header("v50", f"Prediction Winsorization - {combo_variant}")
        print_per_benchmark(rows_combo)
        print_pooled(pooled_combo)
        print_delta(pooled_combo, baseline)
        print_footer()
        output_frames.append(
            build_results_df(
                rows_combo,
                pooled_combo,
                extra_cols={
                    "variant": combo_variant,
                    "version": "v50",
                    "optimal_alpha": optimal_alpha,
                },
            )
        )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v50_pred_winsorize_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
