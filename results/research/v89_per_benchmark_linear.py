"""v89 - Per-benchmark linear classifier comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import (
    feature_set_from_name,
    load_research_inputs,
    logistic_factory,
    resolve_best_feature_set,
    resolve_primary_target,
    write_markdown_summary,
    evaluate_separate_benchmark_binary,
)


MODEL_VARIANTS = {
    "logistic_l2": logistic_factory(class_weight=None, c_value=0.5),
    "logistic_balanced": logistic_factory(class_weight="balanced", c_value=0.5),
    "logistic_l1_balanced": logistic_factory(class_weight="balanced", penalty="l1", c_value=0.25),
}


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_set_name = resolve_best_feature_set()
    feature_columns = feature_set_from_name(feature_df, feature_set_name)

    rows: list[dict[str, object]] = []
    print_header("v89", "Per-Benchmark Linear Classifiers")
    for model_name, factory in MODEL_VARIANTS.items():
        metrics_df, _ = evaluate_separate_benchmark_binary(
            feature_df,
            rel_map,
            target_name=target_name,
            feature_columns=feature_columns,
            model_factory=factory,
        )
        if metrics_df.empty:
            continue
        metrics_df["feature_set"] = feature_set_name
        metrics_df["model_name"] = model_name
        rows.extend(metrics_df.to_dict(orient="records"))

    results_df = pd.DataFrame(rows)
    pooled = results_df.loc[
        results_df["benchmark"] == "POOLED",
        ["model_name", "balanced_accuracy", "brier_score", "ece_10", "feature_set"],
    ].sort_values("balanced_accuracy", ascending=False)
    best_model = str(pooled.iloc[0]["model_name"]) if not pooled.empty else "logistic_balanced"
    results_df["selected_model"] = best_model
    results_df["selected_next"] = results_df["model_name"] == best_model

    print(pooled.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v89_per_benchmark_linear_results.csv")
    write_markdown_summary(
        Path("results/research/v89_per_benchmark_linear_summary.md"),
        "v89 Per-Benchmark Linear Summary",
        [
            f"Forward separate-model reference: `{best_model}` with feature set `{feature_set_name}`.",
            "",
            pooled.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
