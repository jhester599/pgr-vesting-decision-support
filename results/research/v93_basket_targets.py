"""v93 - Basket-target versus benchmark-panel target formulations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import (
    aggregate_probability_sequences,
    basket_probability_candidates,
    binary_metric_bundle,
    build_quality_weighted_regression_consensus,
    build_basket_target_series,
    classifier_hold_fraction,
    load_research_inputs,
    prequential_logistic_calibration,
    probability_candidate_sequences,
    resolve_best_feature_set,
    resolve_primary_target,
    summarize_hold_series,
    write_markdown_summary,
)


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_set_name = resolve_best_feature_set()

    regression_consensus, weights, _ = build_quality_weighted_regression_consensus(feature_df, rel_map)
    benchmark_candidates = probability_candidate_sequences(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_set_name=feature_set_name,
    )
    basket_candidates = basket_probability_candidates(
        feature_df,
        rel_map,
        feature_set_name=feature_set_name,
    )

    rows: list[dict[str, object]] = []
    print_header("v93", "Basket Target Formulations")

    benchmark_candidate_name = "pooled_fixed_effects_logistic_balanced"
    benchmark_sequence_map = benchmark_candidates[benchmark_candidate_name]
    calibrated_benchmark_map: dict[str, pd.DataFrame] = {}
    for benchmark, frame in benchmark_sequence_map.items():
        calibrated = frame.copy()
        calibrated["y_prob_cal"] = prequential_logistic_calibration(
            calibrated["y_true"].to_numpy(dtype=int),
            calibrated["y_prob"].to_numpy(dtype=float),
        )
        calibrated_benchmark_map[benchmark] = calibrated
    benchmark_prob = aggregate_probability_sequences(
        calibrated_benchmark_map,
        probability_column="y_prob_cal",
        weights=weights,
    )
    basket_target = build_basket_target_series(rel_map, "basket_underperform_0pct")
    aligned_benchmark = pd.concat([benchmark_prob.rename("y_prob"), basket_target.rename("y_true")], axis=1).dropna()
    benchmark_bundle = binary_metric_bundle(
        aligned_benchmark["y_true"].to_numpy(dtype=int),
        aligned_benchmark["y_prob"].to_numpy(dtype=float),
    )
    benchmark_hold = classifier_hold_fraction(benchmark_prob, lower=0.35, upper=0.65)
    rows.append(
        {
            "candidate_name": "benchmark_panel_primary",
            "target": target_name,
            "feature_set": feature_set_name,
            "calibration": "prequential_logistic_per_benchmark",
            **benchmark_bundle.__dict__,
            "abstention_rate": float((benchmark_hold == 0.5).mean()),
            **summarize_hold_series(
                "benchmark_panel_primary",
                benchmark_hold,
                regression_consensus["realized"],
            ),
        }
    )

    for candidate_name, pred_df in basket_candidates.items():
        calibrated = pred_df.copy()
        calibrated["y_prob_cal"] = prequential_logistic_calibration(
            calibrated["y_true"].to_numpy(dtype=int),
            calibrated["y_prob"].to_numpy(dtype=float),
        )
        bundle = binary_metric_bundle(
            calibrated["y_true"].to_numpy(dtype=int),
            calibrated["y_prob_cal"].to_numpy(dtype=float),
        )
        prob_sell = pd.Series(
            calibrated["y_prob_cal"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(calibrated["date"]),
            name="prob_sell",
        )
        hold_fraction = classifier_hold_fraction(prob_sell, lower=0.35, upper=0.65)
        rows.append(
            {
                "candidate_name": candidate_name,
                "target": candidate_name,
                "feature_set": feature_set_name,
                "calibration": "prequential_logistic",
                **bundle.__dict__,
                "abstention_rate": float((hold_fraction == 0.5).mean()),
                **summarize_hold_series(
                    candidate_name,
                    hold_fraction,
                    regression_consensus["realized"],
                ),
            }
        )

    results_df = pd.DataFrame(rows)
    ranked = results_df.sort_values(
        ["mean_policy_return", "balanced_accuracy", "ece_10"],
        ascending=[False, False, True],
    )
    best_candidate = str(ranked.iloc[0]["candidate_name"]) if not ranked.empty else ""
    results_df["selected_target_candidate"] = best_candidate
    results_df["selected_next"] = results_df["candidate_name"] == best_candidate

    display_df = results_df[
        [
            "candidate_name",
            "balanced_accuracy",
            "brier_score",
            "ece_10",
            "abstention_rate",
            "mean_policy_return",
            "selected_next",
        ]
    ].sort_values("mean_policy_return", ascending=False)
    print(display_df.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v93_basket_targets_results.csv")
    write_markdown_summary(
        Path("results/research/v93_basket_targets_summary.md"),
        "v93 Basket Target Summary",
        [
            f"Selected target formulation: `{best_candidate}`.",
            "",
            display_df.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
