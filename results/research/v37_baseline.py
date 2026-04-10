"""v37 - Baseline measurement of current v11.0 lean Ridge+GBT ensemble."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from config.features import MODEL_FEATURE_OVERRIDES
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.research.v37_utils import (
    BENCHMARKS,
    build_results_df,
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)


def main() -> None:
    """Run the production ensemble and save baseline pooled metrics."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)

        rel_matrix: dict[str, pd.Series] = {}
        for etf in BENCHMARKS:
            rel_series = load_relative_series(conn, etf, horizon=6)
            if not rel_series.empty:
                rel_matrix[etf] = rel_series

        ensemble_results = run_ensemble_benchmarks(
            df,
            pd.DataFrame(rel_matrix),
            target_horizon_months=6,
            model_feature_overrides=MODEL_FEATURE_OVERRIDES,
        )

        rows: list[dict[str, object]] = []
        for etf in BENCHMARKS:
            if etf not in ensemble_results:
                continue
            y_hat, y_true = reconstruct_ensemble_oos_predictions(ensemble_results[etf])
            if y_true.empty or y_hat.empty:
                continue
            metrics = compute_metrics(y_true.to_numpy(), y_hat.to_numpy())
            rows.append(
                {
                    "benchmark": etf,
                    **metrics,
                    "_y_true": y_true.to_numpy(),
                    "_y_hat": y_hat.to_numpy(),
                }
            )

        if not rows:
            raise RuntimeError("No benchmark results were produced for v37 baseline.")

        pooled = pool_metrics(rows)

        print_header("v37", "Baseline - v11.0 Ridge+GBT Ensemble")
        print_per_benchmark(rows)
        print_pooled(pooled)
        print_footer()

        results_df = build_results_df(
            rows,
            pooled,
            extra_cols={"version": "v37", "experiment": "baseline"},
        )
        save_results(results_df, "v37_baseline_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
