"""v92 - Probability calibration and abstention-threshold design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import (
    aggregate_probability_sequences,
    binary_metric_bundle,
    build_quality_weighted_regression_consensus,
    classifier_hold_fraction,
    load_research_inputs,
    prequential_logistic_calibration,
    probability_candidate_sequences,
    resolve_best_feature_set,
    resolve_primary_target,
    summarize_hold_series,
    write_markdown_summary,
)


THRESHOLDS = (
    (0.40, 0.60),
    (0.35, 0.65),
    (0.30, 0.70),
)


def _bundle_from_sequence_map(
    sequence_map: dict[str, pd.DataFrame],
    probability_column: str,
) -> dict[str, float]:
    y_true = np.concatenate(
        [frame["y_true"].to_numpy(dtype=int) for frame in sequence_map.values()]
    )
    y_prob = np.concatenate(
        [frame[probability_column].to_numpy(dtype=float) for frame in sequence_map.values()]
    )
    return binary_metric_bundle(y_true, y_prob).__dict__


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_set_name = resolve_best_feature_set()

    regression_consensus, weights, _ = build_quality_weighted_regression_consensus(feature_df, rel_map)
    candidates = probability_candidate_sequences(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_set_name=feature_set_name,
    )

    rows: list[dict[str, object]] = []
    print_header("v92", "Calibration and Abstention Thresholds")

    for candidate_name, sequence_map in candidates.items():
        if not sequence_map:
            continue

        calibrated_map: dict[str, pd.DataFrame] = {}
        for benchmark, frame in sequence_map.items():
            calibrated = frame.copy()
            calibrated["y_prob_cal"] = prequential_logistic_calibration(
                calibrated["y_true"].to_numpy(dtype=int),
                calibrated["y_prob"].to_numpy(dtype=float),
            )
            calibrated_map[benchmark] = calibrated

        raw_bundle = _bundle_from_sequence_map(sequence_map, "y_prob")
        cal_bundle = _bundle_from_sequence_map(calibrated_map, "y_prob_cal")

        for calibration_name, sequence_source, bundle, prob_col in (
            ("raw", sequence_map, raw_bundle, "y_prob"),
            ("prequential_logistic", calibrated_map, cal_bundle, "y_prob_cal"),
        ):
            agg_prob = aggregate_probability_sequences(
                sequence_source,
                probability_column=prob_col,
                weights=weights,
            )
            for lower, upper in THRESHOLDS:
                hold_fraction = classifier_hold_fraction(agg_prob, lower=lower, upper=upper)
                summary = summarize_hold_series(
                    variant=f"{candidate_name}__{calibration_name}__{lower:.2f}_{upper:.2f}",
                    hold_fraction=hold_fraction,
                    realized_relative_return=regression_consensus["realized"],
                )
                rows.append(
                    {
                        "candidate_name": candidate_name,
                        "calibration": calibration_name,
                        "target": target_name,
                        "feature_set": feature_set_name,
                        "lower_threshold": lower,
                        "upper_threshold": upper,
                        "abstention_rate": float((hold_fraction == 0.5).mean()),
                        **bundle,
                        **summary,
                    }
                )

    results_df = pd.DataFrame(rows)
    ranked = results_df.sort_values(
        ["mean_policy_return", "balanced_accuracy", "ece_10"],
        ascending=[False, False, True],
    )
    best_variant = str(ranked.iloc[0]["variant"]) if not ranked.empty else ""
    results_df["selected_variant"] = best_variant
    results_df["selected_next"] = results_df["variant"] == best_variant

    pooled = results_df[
        [
            "candidate_name",
            "calibration",
            "lower_threshold",
            "upper_threshold",
            "balanced_accuracy",
            "ece_10",
            "abstention_rate",
            "mean_policy_return",
            "selected_next",
        ]
    ].sort_values("mean_policy_return", ascending=False)
    print(pooled.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v92_calibration_and_abstention_results.csv")
    write_markdown_summary(
        Path("results/research/v92_calibration_and_abstention_summary.md"),
        "v92 Calibration and Abstention Summary",
        [
            f"Selected probability path: `{best_variant}`.",
            "",
            pooled.head(12).to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
