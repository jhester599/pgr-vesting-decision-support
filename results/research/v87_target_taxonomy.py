"""v87 - Problem framing and target taxonomy for classification research."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import (
    available_feature_families,
    binary_metric_bundle,
    build_basket_target_series,
    build_target_series,
    choose_primary_target,
    describe_target_series,
    evaluate_binary_time_series,
    evaluate_separate_benchmark_binary,
    load_research_inputs,
    logistic_factory,
    write_markdown_summary,
)


TARGETS = (
    "benchmark_underperform_0pct",
    "actionable_sell_3pct",
    "basket_underperform_0pct",
    "breadth_underperform_majority",
)


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    families = available_feature_families(feature_df)
    lean_features = families["lean_baseline"]
    model_factory = logistic_factory(class_weight="balanced", c_value=0.5)

    rows: list[dict[str, object]] = []
    target_descriptions: list[dict[str, object]] = []

    print_header("v87", "Classification Target Taxonomy")
    for target_name in TARGETS:
        if target_name.startswith("benchmark_") or target_name.startswith("actionable_"):
            target_pool: list[pd.Series] = []
            for benchmark, rel_series in rel_map.items():
                if rel_series.empty:
                    continue
                target_pool.append(build_target_series(rel_series, target_name))
            if target_pool:
                target_descriptions.append(
                    describe_target_series(pd.concat(target_pool).sort_index(), target_name)
                )
            metrics_df, _ = evaluate_separate_benchmark_binary(
                feature_df,
                rel_map,
                target_name=target_name,
                feature_columns=lean_features,
                model_factory=model_factory,
            )
            if not metrics_df.empty:
                metrics_df["feature_set"] = "lean_baseline"
                metrics_df["scope"] = "benchmark_panel"
                rows.extend(metrics_df.to_dict(orient="records"))
        else:
            target = build_basket_target_series(rel_map, target_name)
            target_descriptions.append(describe_target_series(target, target_name))
            x_df = feature_df[lean_features].copy()
            pred_df = evaluate_binary_time_series(x_df, target, model_factory)
            if pred_df.empty:
                continue
            metrics = binary_metric_bundle(
                pred_df["y_true"].to_numpy(dtype=int),
                pred_df["y_prob"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "benchmark": "BASKET",
                    "target": target_name,
                    "feature_set": "lean_baseline",
                    "scope": "basket_time_series",
                    "n_features": len(lean_features),
                    **metrics.__dict__,
                }
            )

    results_df = pd.DataFrame(rows)
    target_desc_df = pd.DataFrame(target_descriptions)
    if not target_desc_df.empty:
        results_df = results_df.merge(target_desc_df, on="target", how="left", suffixes=("", "_target"))

    recommended_target = choose_primary_target(results_df)
    results_df["recommended_target"] = recommended_target
    results_df["recommended_next"] = results_df["target"] == recommended_target

    pooled_view = results_df.loc[
        results_df["benchmark"].isin(["POOLED", "BASKET"]),
        [
            "benchmark",
            "target",
            "balanced_accuracy",
            "brier_score",
            "ece_10",
            "base_rate",
            "recommended_next",
        ],
    ].copy()
    print(pooled_view.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v87_target_taxonomy_results.csv")
    summary_path = Path("results/research/v87_target_taxonomy_summary.md")
    write_markdown_summary(
        summary_path,
        "v87 Target Taxonomy Summary",
        [
            f"Recommended forward binary target: `{recommended_target}`.",
            "",
            "Top pooled / basket rows:",
            pooled_view.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
