"""v72 - Benchmark-quality-weighted consensus on top of the v38 baseline."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.models.forecast_diagnostics import summarize_prediction_diagnostics
from src.research.v37_utils import (
    BENCHMARKS,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    print_footer,
    print_header,
    save_results,
)
from src.research.v66_utils import (
    benchmark_quality_weights,
    build_consensus_frame,
    load_ensemble_oos_sequences,
    summarize_consensus_variant,
)


def main() -> None:
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_matrix = {
            benchmark: load_relative_series(conn, benchmark, horizon=6)
            for benchmark in BENCHMARKS
        }
        sequences = load_ensemble_oos_sequences(
            feature_df,
            pd.DataFrame(rel_matrix),
            shrinkage_alpha=0.50,
        )
        quality_rows: list[dict[str, float | int | str]] = []
        for benchmark, frame in sequences.items():
            summary = summarize_prediction_diagnostics(frame["y_hat"], frame["y_true"])
            quality_rows.append({"benchmark": benchmark, **summary})
        quality_df = pd.DataFrame(quality_rows)
        weights = benchmark_quality_weights(quality_df, score_col="nw_ic", lambda_mix=0.25)

        equal_frame = build_consensus_frame(sequences)
        weighted_frame = build_consensus_frame(sequences, weights=weights)
        result_rows = [
            {
                **summarize_consensus_variant("equal_weight", equal_frame),
                "version": "v72",
                "weight_mode": "equal",
                "lambda_mix": 0.0,
            },
            {
                **summarize_consensus_variant("quality_weighted", weighted_frame),
                "version": "v72",
                "weight_mode": "nw_ic_shrink_to_equal",
                "lambda_mix": 0.25,
            },
        ]

        print_header("v72", "Benchmark-Quality-Weighted Consensus")
        print(pd.DataFrame(result_rows).to_string(index=False, float_format="{:.4f}".format))
        print_footer()
        save_results(pd.DataFrame(result_rows), "v72_quality_weighted_consensus_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
