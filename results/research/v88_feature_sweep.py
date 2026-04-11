"""v88 - Stepwise feature-family sweep for the chosen binary target."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import (
    build_feature_sets,
    choose_best_feature_set,
    evaluate_separate_benchmark_binary,
    load_research_inputs,
    logistic_factory,
    resolve_primary_target,
    write_markdown_summary,
)


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_sets = build_feature_sets(feature_df)
    model_factory = logistic_factory(class_weight="balanced", c_value=0.5)

    rows: list[dict[str, object]] = []
    print_header("v88", "Feature Family Sweep")
    for feature_set_name, features in feature_sets.items():
        metrics_df, _ = evaluate_separate_benchmark_binary(
            feature_df,
            rel_map,
            target_name=target_name,
            feature_columns=features,
            model_factory=model_factory,
        )
        if metrics_df.empty:
            continue
        metrics_df["feature_set"] = feature_set_name
        rows.extend(metrics_df.to_dict(orient="records"))

    results_df = pd.DataFrame(rows)
    best_feature_set = choose_best_feature_set(results_df)
    results_df["selected_feature_set"] = best_feature_set
    results_df["selected_next"] = results_df["feature_set"] == best_feature_set

    pooled = results_df.loc[
        results_df["benchmark"] == "POOLED",
        ["feature_set", "balanced_accuracy", "brier_score", "ece_10", "n_features", "selected_next"],
    ].sort_values("balanced_accuracy", ascending=False)
    print(pooled.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v88_feature_sweep_results.csv")
    summary_path = Path("results/research/v88_feature_sweep_summary.md")
    write_markdown_summary(
        summary_path,
        "v88 Feature Sweep Summary",
        [
            f"Forward feature set for later versions: `{best_feature_set}`.",
            "",
            pooled.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
