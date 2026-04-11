"""v94 - Hybrid classifier + regression gate architecture."""

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
    build_quality_weighted_regression_consensus,
    classifier_hold_fraction,
    hybrid_hold_fraction,
    load_research_inputs,
    prequential_logistic_calibration,
    probability_candidate_sequences,
    resolve_best_feature_set,
    resolve_primary_target,
    summarize_hold_series,
    summarize_regression_baseline,
    write_markdown_summary,
)


def _best_v92_candidate_name() -> str:
    path = Path("results/research/v92_calibration_and_abstention_results.csv")
    if not path.exists():
        return "pooled_fixed_effects_logistic_balanced"
    df = pd.read_csv(path)
    selected = df.loc[df["selected_next"]]
    if not selected.empty:
        return str(selected.iloc[0]["candidate_name"])
    ranked = df.sort_values("mean_policy_return", ascending=False)
    return str(ranked.iloc[0]["candidate_name"])


def _best_v93_basket_candidate_name() -> str:
    path = Path("results/research/v93_basket_targets_results.csv")
    if not path.exists():
        return "basket_underperform_0pct"
    df = pd.read_csv(path)
    basket_only = df[df["candidate_name"] != "benchmark_panel_primary"]
    selected = basket_only.loc[basket_only["selected_next"]]
    if not selected.empty:
        return str(selected.iloc[0]["candidate_name"])
    ranked = basket_only.sort_values("mean_policy_return", ascending=False)
    return str(ranked.iloc[0]["candidate_name"])


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_set_name = resolve_best_feature_set()
    regression_consensus, weights, _ = build_quality_weighted_regression_consensus(feature_df, rel_map)

    benchmark_source_name = _best_v92_candidate_name()
    benchmark_candidates = probability_candidate_sequences(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_set_name=feature_set_name,
    )
    benchmark_sequence_map = benchmark_candidates[benchmark_source_name]
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

    basket_source_name = _best_v93_basket_candidate_name()
    basket_candidates = basket_probability_candidates(
        feature_df,
        rel_map,
        feature_set_name=feature_set_name,
    )
    basket_pred = basket_candidates[basket_source_name].copy()
    basket_pred["y_prob_cal"] = prequential_logistic_calibration(
        basket_pred["y_true"].to_numpy(dtype=int),
        basket_pred["y_prob"].to_numpy(dtype=float),
    )
    basket_prob = pd.Series(
        basket_pred["y_prob_cal"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(basket_pred["date"]),
        name="prob_sell",
    )

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    print_header("v94", "Hybrid Gate Architectures")

    regression_summary = summarize_regression_baseline(regression_consensus)
    summary_rows.append(regression_summary)
    regression_hold = pd.Series(
        pd.Series(regression_consensus["predicted"]).map(
            lambda value: 1.0 if value > 0.03 else (0.0 if value < -0.03 else 0.5)
        ).to_numpy(dtype=float),
        index=regression_consensus.index,
        name="hold_fraction",
    )
    detail_rows.extend(
        pd.DataFrame(
            {
                "date": regression_consensus.index,
                "variant": "regression_only_quality_weighted",
                "hold_fraction": regression_hold.to_numpy(dtype=float),
                "realized": regression_consensus["realized"].to_numpy(dtype=float),
            }
        ).to_dict(orient="records")
    )

    variants = {
        "classifier_only_benchmark_panel": classifier_hold_fraction(benchmark_prob, 0.35, 0.65),
        "hybrid_benchmark_panel_35_65": hybrid_hold_fraction(
            regression_consensus["predicted"],
            benchmark_prob,
            0.35,
            0.65,
        ),
        "hybrid_benchmark_panel_30_70": hybrid_hold_fraction(
            regression_consensus["predicted"],
            benchmark_prob,
            0.30,
            0.70,
        ),
        "classifier_only_best_basket": classifier_hold_fraction(basket_prob, 0.35, 0.65),
        "hybrid_best_basket_35_65": hybrid_hold_fraction(
            regression_consensus["predicted"],
            basket_prob,
            0.35,
            0.65,
        ),
    }

    for variant_name, hold_fraction in variants.items():
        summary_rows.append(
            summarize_hold_series(
                variant_name,
                hold_fraction,
                regression_consensus["realized"],
            )
        )
        aligned = pd.concat(
            [
                hold_fraction.rename("hold_fraction"),
                regression_consensus["realized"].rename("realized"),
            ],
            axis=1,
        ).dropna()
        detail_rows.extend(
            pd.DataFrame(
                {
                    "date": aligned.index,
                    "variant": variant_name,
                    "hold_fraction": aligned["hold_fraction"].to_numpy(dtype=float),
                    "realized": aligned["realized"].to_numpy(dtype=float),
                }
            ).to_dict(orient="records")
        )

    results_df = pd.DataFrame(summary_rows)
    ranked = results_df.loc[
        results_df["variant"] != "regression_only_quality_weighted"
    ].sort_values(
        ["mean_policy_return", "capture_ratio"],
        ascending=[False, False],
    )
    best_variant = str(ranked.iloc[0]["variant"]) if not ranked.empty else "regression_only_quality_weighted"
    results_df["selected_variant"] = best_variant
    results_df["selected_next"] = results_df["variant"] == best_variant

    display_df = results_df[
        ["variant", "mean_policy_return", "uplift_vs_sell_50", "capture_ratio", "hold_fraction_changes", "selected_next"]
    ].sort_values("mean_policy_return", ascending=False)
    print(display_df.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v94_hybrid_gate_results.csv")
    detail_df = pd.DataFrame(detail_rows).sort_values(["variant", "date"])
    detail_df.to_csv("results/research/v94_hybrid_gate_detail.csv", index=False)
    write_markdown_summary(
        Path("results/research/v94_hybrid_gate_summary.md"),
        "v94 Hybrid Gate Summary",
        [
            f"Best non-regression policy variant: `{best_variant}`.",
            f"Benchmark probability source: `{benchmark_source_name}`.",
            f"Basket probability source: `{basket_source_name}`.",
            "",
            display_df.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
