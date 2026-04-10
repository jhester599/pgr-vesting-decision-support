"""v39 - Ridge alpha grid extension across production WFO splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
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

ALPHA_GRIDS: dict[str, np.ndarray] = {
    "current_logspace(-4,4)": np.logspace(-4, 4, 50),
    "extended_logspace(0,6)": np.logspace(0, 6, 100),
    "aggressive_logspace(2,6)": np.logspace(2, 6, 100),
}


def run_ridge_grid(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Ridge-only WFO with a custom alpha grid."""
    x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
    feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
    x_values = x_df[feature_cols].to_numpy()

    def factory() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=alphas, cv=None, fit_intercept=True)),
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


def main() -> None:
    """Compare current, extended, and aggressive Ridge alpha ranges."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)

        all_rows: list[dict[str, object]] = []
        print(f"\n{'Grid':<30}  {'R2_pooled':>10}  {'IC_pooled':>10}  {'HitRate':>8}")
        for grid_name, alphas in ALPHA_GRIDS.items():
            rows: list[dict[str, object]] = []
            for etf in BENCHMARKS:
                rel_series = load_relative_series(conn, etf, horizon=6)
                if rel_series.empty:
                    continue
                y_true, y_hat = run_ridge_grid(df, rel_series, alphas)
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": etf,
                        "grid": grid_name,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )
            pooled = pool_metrics(rows)
            print(
                f"  {grid_name:<30}  {pooled['r2']:>10.4f}  "
                f"{pooled['ic']:>10.4f}  {pooled['hit_rate']:>8.4f}"
            )
            all_rows.extend(rows)
            all_rows.append({"benchmark": "POOLED", "grid": grid_name, **pooled})

        summary_rows = [row for row in all_rows if row["benchmark"] == "POOLED"]
        best_grid = max(summary_rows, key=lambda row: float(row["r2"]))
        best_grid_name = str(best_grid["grid"])

        best_rows = [
            row for row in all_rows
            if row.get("grid") == best_grid_name and row["benchmark"] != "POOLED"
        ]
        pooled_best = pool_metrics(best_rows)

        print(f"\nBest grid: {best_grid_name} (R2={float(best_grid['r2']):.4f})")
        print_header("v39", "Ridge Alpha Extension")
        print_per_benchmark(best_rows)
        print_pooled(pooled_best)
        print_delta(pooled_best, load_baseline_results())
        print_footer()

        output_rows: list[dict[str, object]] = []
        for row in all_rows:
            clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
            output_rows.append(clean_row)
        save_results(pd.DataFrame(output_rows), "v39_ridge_alpha_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
