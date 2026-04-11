"""v73 - Hybrid decision-layer gating using v38 regression and v46 probabilities."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

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
    build_consensus_frame,
    classification_probability_sequences,
    evaluate_gated_policy_variant,
    load_ensemble_oos_sequences,
)


VARIANTS = (
    ("A_gate_40_60", 0.40, 0.60),
    ("B_gate_35_65", 0.35, 0.65),
)


def main() -> None:
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_matrix = {
            benchmark: load_relative_series(conn, benchmark, horizon=6)
            for benchmark in BENCHMARKS
        }
        regression_sequences = load_ensemble_oos_sequences(
            feature_df,
            pd.DataFrame(rel_matrix),
            shrinkage_alpha=0.50,
        )
        probability_sequences = classification_probability_sequences(feature_df, rel_matrix)

        regression_consensus = build_consensus_frame(regression_sequences)
        probability_consensus = build_consensus_frame(
            {
                benchmark: frame.rename(columns={"prob_outperform": "y_hat", "y_true_binary": "y_true"})
                for benchmark, frame in probability_sequences.items()
            }
        )
        result_rows: list[dict[str, float | int | str]] = []
        for variant, lower, upper in VARIANTS:
            result_rows.append(
                {
                    **evaluate_gated_policy_variant(
                        variant=variant,
                        predicted=regression_consensus["predicted"],
                        realized=regression_consensus["realized"],
                        probability=probability_consensus["predicted"],
                        lower=lower,
                        upper=upper,
                    ),
                    "version": "v73",
                }
            )

        print_header("v73", "Hybrid Regression + Classification Gate")
        print(pd.DataFrame(result_rows).to_string(index=False, float_format="{:.4f}".format))
        print_footer()
        save_results(pd.DataFrame(result_rows), "v73_hybrid_decision_gate_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
