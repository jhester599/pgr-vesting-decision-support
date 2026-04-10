"""v43 - Feature reduction experiments using 7-feature subsets."""

from __future__ import annotations

import pandas as pd
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.models.wfo_engine import run_wfo
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    SHARED_7_FEATURES,
    build_results_df,
    compute_metrics,
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

RIDGE_7: list[str] = [
    "yield_slope",
    "npw_growth_yoy",
    "investment_income_growth_yoy",
    "real_yield_change_6m",
    "combined_ratio_ttm",
    "mom_12m",
    "credit_spread_hy",
]
GBT_7: list[str] = [
    "mom_12m",
    "vol_63d",
    "yield_slope",
    "credit_spread_hy",
    "vix",
    "pif_growth_yoy",
    "investment_book_yield",
]

VARIANTS: dict[str, dict[str, str | list[str]]] = {
    "A_ridge7": {"model_type": "ridge", "features": RIDGE_7},
    "B_gbt7": {"model_type": "gbt", "features": GBT_7},
    "C_shared7_ridge": {"model_type": "ridge", "features": SHARED_7_FEATURES},
}


def main() -> None:
    """Run the phase-1 feature reduction variants."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, cfg in VARIANTS.items():
            rows: list[dict[str, object]] = []
            for etf in BENCHMARKS:
                rel_series = load_relative_series(conn, etf, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
                result = run_wfo(
                    x_df,
                    y,
                    model_type=str(cfg["model_type"]),
                    target_horizon_months=6,
                    benchmark=etf,
                    feature_columns=list(cfg["features"]),  # type: ignore[arg-type]
                )
                y_true = result.y_true_all
                y_hat = result.y_hat_all
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
            print_header("v43", f"Feature Reduction - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(rows, pooled, extra_cols={"variant": variant_name})
            )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v43_feature_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
