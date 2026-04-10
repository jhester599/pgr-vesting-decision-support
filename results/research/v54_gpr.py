"""v54 - Gaussian Process Regression variants on the shared 7-feature set."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process")

from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    SHARED_7_FEATURES,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    custom_wfo,
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


def make_gpr_matern() -> Pipeline:
    """Return a Matern-5/2 GPR pipeline."""
    kernel = (
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1.0)
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    normalize_y=True,
                    random_state=42,
                ),
            ),
        ]
    )


def make_gpr_rbf() -> Pipeline:
    """Return an RBF-kernel GPR pipeline."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    normalize_y=True,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    """Run Gaussian-process experiments against the v38 research baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, factory in [
            ("A_gpr_matern52", make_gpr_matern),
            ("B_gpr_rbf", make_gpr_rbf),
        ]:
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                feature_cols = [col for col in SHARED_7_FEATURES if col in x_df.columns]
                y_true, y_hat = custom_wfo(
                    x_df[feature_cols].to_numpy(),
                    y.to_numpy(),
                    factory,
                    MAX_TRAIN_MONTHS,
                    TEST_SIZE_MONTHS,
                    GAP_MONTHS,
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
            print_header("v54", f"GPR - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={"variant": variant_name, "version": "v54"},
                )
            )

        rows_blend: list[dict[str, object]] = []
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in SHARED_7_FEATURES if col in x_df.columns]
            x = x_df[feature_cols].to_numpy()

            def ridge_factory() -> Pipeline:
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                    ]
                )

            y_true_ridge, y_hat_ridge = custom_wfo(
                x,
                y.to_numpy(),
                ridge_factory,
                MAX_TRAIN_MONTHS,
                TEST_SIZE_MONTHS,
                GAP_MONTHS,
            )
            y_true_gpr, y_hat_gpr = custom_wfo(
                x,
                y.to_numpy(),
                make_gpr_matern,
                MAX_TRAIN_MONTHS,
                TEST_SIZE_MONTHS,
                GAP_MONTHS,
            )

            n_obs = min(len(y_true_ridge), len(y_true_gpr))
            y_true = y_true_ridge[-n_obs:]
            y_hat = 0.80 * y_hat_ridge[-n_obs:] + 0.20 * y_hat_gpr[-n_obs:]
            metrics = compute_metrics(y_true, y_hat)
            rows_blend.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled_blend = pool_metrics(rows_blend)
        print_header("v54", "GPR - C_ridge_gpr_80_20")
        print_per_benchmark(rows_blend)
        print_pooled(pooled_blend)
        print_delta(pooled_blend, baseline)
        print_footer()
        output_frames.append(
            build_results_df(
                rows_blend,
                pooled_blend,
                extra_cols={"variant": "C_ridge_gpr_80_20", "version": "v54"},
            )
        )

        save_results(pd.concat(output_frames, ignore_index=True), "v54_gpr_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
