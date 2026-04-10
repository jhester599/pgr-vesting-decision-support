"""v40 - Compare Ridge-only, constrained GBT, and 80/20 post-hoc reweighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
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
    GBT_FEATURES_13,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    custom_wfo,
    get_connection,
    load_baseline_results,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def run_variant_a_ridge_only(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Ridge-only WFO with the extended alpha grid."""
    x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
    feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
    x_values = x_df[feature_cols].to_numpy()

    def factory() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
            ]
        )

    return custom_wfo(
        x_values,
        y.to_numpy(),
        factory,
        MAX_TRAIN_MONTHS,
        TEST_SIZE_MONTHS,
        GAP_MONTHS,
    )


def run_variant_b_constrained_gbt(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a deliberately constrained GBT to reduce amplitude noise."""
    x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
    feature_cols = [col for col in GBT_FEATURES_13 if col in x_df.columns]
    x_values = x_df[feature_cols].to_numpy()

    def factory() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        max_depth=2,
                        n_estimators=50,
                        learning_rate=0.01,
                        min_samples_leaf=10,
                        subsample=0.7,
                        random_state=42,
                    ),
                ),
            ]
        )

    return custom_wfo(
        x_values,
        y.to_numpy(),
        factory,
        MAX_TRAIN_MONTHS,
        TEST_SIZE_MONTHS,
        GAP_MONTHS,
    )


def run_variant_c_reweight(
    y_true_ridge: np.ndarray,
    y_hat_ridge: np.ndarray,
    y_true_gbt: np.ndarray,
    y_hat_gbt: np.ndarray,
    ridge_weight: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend the overlapping OOS segments from Ridge and constrained GBT."""
    n_obs = min(len(y_true_ridge), len(y_true_gbt))
    y_true = y_true_ridge[-n_obs:]
    y_hat = ridge_weight * y_hat_ridge[-n_obs:] + (1.0 - ridge_weight) * y_hat_gbt[-n_obs:]
    return y_true, y_hat


def main() -> None:
    """Run three GBT simplification variants and save pooled summaries."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_baseline_results()

        variants: dict[str, list[dict[str, object]]] = {
            "ridge_only": [],
            "constrained_gbt": [],
            "reweight_80_20": [],
        }

        for etf in BENCHMARKS:
            rel_series = load_relative_series(conn, etf, horizon=6)
            if rel_series.empty:
                continue

            y_true_ridge, y_hat_ridge = run_variant_a_ridge_only(df, rel_series)
            metrics_ridge = compute_metrics(y_true_ridge, y_hat_ridge)
            variants["ridge_only"].append(
                {
                    "benchmark": etf,
                    **metrics_ridge,
                    "_y_true": y_true_ridge,
                    "_y_hat": y_hat_ridge,
                }
            )

            y_true_gbt, y_hat_gbt = run_variant_b_constrained_gbt(df, rel_series)
            metrics_gbt = compute_metrics(y_true_gbt, y_hat_gbt)
            variants["constrained_gbt"].append(
                {
                    "benchmark": etf,
                    **metrics_gbt,
                    "_y_true": y_true_gbt,
                    "_y_hat": y_hat_gbt,
                }
            )

            y_true_blend, y_hat_blend = run_variant_c_reweight(
                y_true_ridge,
                y_hat_ridge,
                y_true_gbt,
                y_hat_gbt,
            )
            metrics_blend = compute_metrics(y_true_blend, y_hat_blend)
            variants["reweight_80_20"].append(
                {
                    "benchmark": etf,
                    **metrics_blend,
                    "_y_true": y_true_blend,
                    "_y_hat": y_hat_blend,
                }
            )

        output_frames: list[pd.DataFrame] = []
        for variant_name, rows in variants.items():
            pooled = pool_metrics(rows)
            print_header("v40", f"GBT Constraint - Variant: {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(rows, pooled, extra_cols={"variant": variant_name})
            )

        save_results(pd.concat(output_frames, ignore_index=True), "v40_gbt_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
