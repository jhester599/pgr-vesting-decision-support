"""v45 - BayesianRidge as primary model: default, tight prior, and 80/20 +GBT."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.models.regularized_models import build_bayesian_ridge_pipeline
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

VARIANTS: dict[str, dict[str, float]] = {
    "A_default_bayesian_ridge": {
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6,
    },
    "B_tight_prior": {
        "alpha_1": 1e-5,
        "alpha_2": 1e-5,
        "lambda_1": 1e-4,
        "lambda_2": 1e-4,
    },
}


def run_bayesian_ridge_variant(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
    alpha_1: float,
    alpha_2: float,
    lambda_1: float,
    lambda_2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a custom BayesianRidge WFO over the lean Ridge feature set."""
    x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
    feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
    x_values = x_df[feature_cols].to_numpy()

    def factory():
        return build_bayesian_ridge_pipeline(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

    return custom_wfo(
        x_values,
        y.to_numpy(),
        factory,
        MAX_TRAIN_MONTHS,
        TEST_SIZE_MONTHS,
        GAP_MONTHS,
    )


def main() -> None:
    """Run BayesianRidge replacement experiments against the v38 baseline."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, hyperparams in VARIANTS.items():
            rows: list[dict[str, object]] = []
            for etf in BENCHMARKS:
                rel_series = load_relative_series(conn, etf, horizon=6)
                if rel_series.empty:
                    continue
                y_true, y_hat = run_bayesian_ridge_variant(df, rel_series, **hyperparams)
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": etf,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )

            pooled = pool_metrics(rows)
            print_header("v45", f"BayesianRidge - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(rows, pooled, extra_cols={"variant": variant_name})
            )

        rows_blend: list[dict[str, object]] = []
        for etf in BENCHMARKS:
            rel_series = load_relative_series(conn, etf, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)

            y_true_br, y_hat_br = run_bayesian_ridge_variant(
                df,
                rel_series,
                **VARIANTS["A_default_bayesian_ridge"],
            )
            gbt_result = run_wfo(
                x_df,
                y,
                model_type="gbt",
                target_horizon_months=6,
                benchmark=etf,
                feature_columns=GBT_FEATURES_13,
            )
            y_true_gbt = gbt_result.y_true_all
            y_hat_gbt = gbt_result.y_hat_all

            n_obs = min(len(y_true_br), len(y_true_gbt))
            y_true = y_true_br[-n_obs:]
            y_hat = 0.80 * y_hat_br[-n_obs:] + 0.20 * y_hat_gbt[-n_obs:]
            metrics = compute_metrics(y_true, y_hat)
            rows_blend.append(
                {
                    "benchmark": etf,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled_blend = pool_metrics(rows_blend)
        print_header("v45", "BayesianRidge - C_bayesian_ridge_gbt")
        print_per_benchmark(rows_blend)
        print_pooled(pooled_blend)
        print_delta(pooled_blend, baseline)
        print_footer()
        output_frames.append(
            build_results_df(rows_blend, pooled_blend, extra_cols={"variant": "C_bayesian_ridge_gbt"})
        )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v45_bayesian_ridge_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
