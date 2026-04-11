"""v70 - Per-benchmark prequential shrinkage calibration."""

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

from src.research.v37_utils import (
    BENCHMARKS,
    build_results_df,
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_research_baseline_results,
    load_relative_series,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)
from src.research.v66_utils import (
    apply_prequential_shrinkage,
    load_ensemble_oos_sequences,
)


VARIANTS = {
    "A_prior12": 12,
    "B_prior24": 24,
}


def main() -> None:
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_matrix = {
            benchmark: load_relative_series(conn, benchmark, horizon=6)
            for benchmark in BENCHMARKS
        }
        raw_sequences = load_ensemble_oos_sequences(
            feature_df,
            pd.DataFrame(rel_matrix),
            shrinkage_alpha=1.0,
        )

        baseline_df = load_research_baseline_results()
        result_frames: list[pd.DataFrame] = []

        for variant, min_history in VARIANTS.items():
            rows: list[dict[str, object]] = []
            pooled_true: list[np.ndarray] = []
            pooled_hat: list[np.ndarray] = []

            print_header("v70", f"Per-Benchmark Shrinkage ({variant})")
            for benchmark in BENCHMARKS:
                frame = raw_sequences.get(benchmark)
                if frame is None or frame.empty:
                    continue
                y_true = frame["y_true"].to_numpy(dtype=float)
                y_hat_raw = frame["y_hat"].to_numpy(dtype=float)
                y_hat, alphas = apply_prequential_shrinkage(
                    y_true=y_true,
                    y_hat_raw=y_hat_raw,
                    min_history=min_history,
                )
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "variant": variant,
                        "min_history": min_history,
                        "mean_alpha": float(np.mean(alphas)),
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )
                pooled_true.append(y_true)
                pooled_hat.append(y_hat)

            pooled_metrics = compute_metrics(
                np.concatenate(pooled_true),
                np.concatenate(pooled_hat),
            )
            print_per_benchmark(rows)
            print_pooled(pooled_metrics)
            print_delta(pooled_metrics, baseline_df)
            print_footer()

            result_frames.append(
                build_results_df(
                    rows,
                    pooled_metrics,
                    extra_cols={
                        "version": "v70",
                        "variant": variant,
                        "min_history": min_history,
                    },
                )
            )

        save_results(
            pd.concat(result_frames, ignore_index=True),
            "v70_benchmark_shrinkage_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
