"""v38 - Post-hoc prediction shrinkage: y_hat_shrunk = alpha * y_hat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path

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

ALPHAS: list[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def main() -> None:
    """Sweep shrinkage factors over production OOS predictions."""
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

        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for etf in BENCHMARKS:
            if etf not in ensemble_results:
                continue
            y_hat, y_true = reconstruct_ensemble_oos_predictions(ensemble_results[etf])
            if y_true.empty or y_hat.empty:
                continue
            raw[etf] = (y_hat.to_numpy(), y_true.to_numpy())

        if not raw:
            raise RuntimeError("No raw ensemble predictions available for v38 shrinkage.")

        alpha_results: list[dict[str, float | str]] = []
        print("\nAlpha sweep:")
        print(f"  {'alpha':>6}  {'R2_pooled':>10}  {'IC_pooled':>10}  {'HitRate':>8}")
        for alpha in ALPHAS:
            rows: list[dict[str, object]] = []
            for etf, (y_hat_raw, y_true) in raw.items():
                y_hat_shrunk = alpha * y_hat_raw
                metrics = compute_metrics(y_true, y_hat_shrunk)
                rows.append(
                    {
                        "benchmark": etf,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat_shrunk,
                    }
                )
            pooled = pool_metrics(rows)
            print(
                f"  {alpha:>6.2f}  {pooled['r2']:>10.4f}  "
                f"{pooled['ic']:>10.4f}  {pooled['hit_rate']:>8.4f}"
            )
            alpha_results.append(
                {
                    "alpha": alpha,
                    "r2": pooled["r2"],
                    "ic": pooled["ic"],
                    "hit_rate": pooled["hit_rate"],
                    "mae": pooled["mae"],
                    "sigma_ratio": pooled["sigma_ratio"],
                    "version": "v38",
                }
            )

        best = max(alpha_results, key=lambda row: float(row["r2"]))
        optimal_alpha = float(best["alpha"])
        print(f"\nOptimal alpha: {optimal_alpha:.2f} (R2={float(best['r2']):.4f})")

        raw_true = np.concatenate([value[1] for value in raw.values()])
        raw_hat = np.concatenate([value[0] for value in raw.values()])
        raw_metrics = compute_metrics(raw_true, raw_hat)
        if abs(float(best["ic"]) - raw_metrics["ic"]) >= 1e-10:
            raise AssertionError("IC changed with shrinkage - this should be invariant.")
        if abs(float(best["hit_rate"]) - raw_metrics["hit_rate"]) >= 1e-10:
            raise AssertionError("Hit rate changed with shrinkage - this should be invariant.")
        print("Sanity check passed: IC and hit rate are invariant to linear scaling.")

        final_rows: list[dict[str, object]] = []
        for etf, (y_hat_raw, y_true) in raw.items():
            y_hat_shrunk = optimal_alpha * y_hat_raw
            metrics = compute_metrics(y_true, y_hat_shrunk)
            final_rows.append(
                {
                    "benchmark": etf,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat_shrunk,
                }
            )
        pooled_best = pool_metrics(final_rows)

        print_header("v38", f"Prediction Shrinkage (optimal alpha={optimal_alpha:.2f})")
        print_per_benchmark(final_rows)
        print_pooled(pooled_best)
        print_delta(pooled_best, load_baseline_results())
        print_footer()

        save_results(pd.DataFrame(alpha_results), "v38_shrinkage_results.csv")
        best_df = build_results_df(
            final_rows,
            pooled_best,
            extra_cols={"version": "v38", "optimal_alpha": optimal_alpha},
        )
        save_results(best_df, "v38_shrinkage_best_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
