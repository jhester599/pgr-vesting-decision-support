"""v53 - ARDRegression: automatic relevance determination variants."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.models.wfo_engine import run_wfo
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GBT_FEATURES_13,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
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

EXTENDED_FEATURES: list[str] = RIDGE_FEATURES_12 + [
    "pif_growth_yoy",
    "investment_book_yield",
    "yield_curvature",
    "mom_3m",
]


def run_ard(
    x: np.ndarray,
    y: np.ndarray,
    threshold_lambda: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an ARDRegression WFO loop over a provided feature matrix."""

    def factory() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ARDRegression(
                        alpha_1=1e-6,
                        alpha_2=1e-6,
                        lambda_1=1e-6,
                        lambda_2=1e-6,
                        threshold_lambda=threshold_lambda,
                        fit_intercept=True,
                        max_iter=300,
                    ),
                ),
            ]
        )

    return custom_wfo(
        x,
        y,
        factory,
        MAX_TRAIN_MONTHS,
        TEST_SIZE_MONTHS,
        GAP_MONTHS,
    )


def main() -> None:
    """Run ARD experiments against the v38 research baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, feature_list in [
            ("A_default_ard", RIDGE_FEATURES_12),
            ("C_ard_extended", EXTENDED_FEATURES),
        ]:
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                feature_cols = [col for col in feature_list if col in x_df.columns]
                y_true, y_hat = run_ard(
                    x_df[feature_cols].to_numpy(),
                    y.to_numpy(),
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
            print_header("v53", f"ARDRegression - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={"variant": variant_name, "version": "v53"},
                )
            )

        rows_blend: list[dict[str, object]] = []
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            y_true_ard, y_hat_ard = run_ard(x_df[feature_cols].to_numpy(), y.to_numpy())
            gbt_result = run_wfo(
                x_df,
                y,
                model_type="gbt",
                target_horizon_months=6,
                benchmark=benchmark,
                feature_columns=GBT_FEATURES_13,
            )

            n_obs = min(len(y_true_ard), len(gbt_result.y_true_all))
            y_true = y_true_ard[-n_obs:]
            y_hat = 0.80 * y_hat_ard[-n_obs:] + 0.20 * gbt_result.y_hat_all[-n_obs:]
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
        print_header("v53", "ARDRegression - B_ard_gbt")
        print_per_benchmark(rows_blend)
        print_pooled(pooled_blend)
        print_delta(pooled_blend, baseline)
        print_footer()
        output_frames.append(
            build_results_df(
                rows_blend,
                pooled_blend,
                extra_cols={"variant": "B_ard_gbt", "version": "v53"},
            )
        )

        save_results(pd.concat(output_frames, ignore_index=True), "v53_ard_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
