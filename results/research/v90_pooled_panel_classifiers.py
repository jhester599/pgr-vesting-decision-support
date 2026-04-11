"""v90 - Pooled panel classifiers versus separate benchmark models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import (
    build_panel_dataset,
    evaluate_panel_binary,
    evaluate_separate_benchmark_binary,
    feature_set_from_name,
    load_research_inputs,
    logistic_factory,
    resolve_best_feature_set,
    resolve_primary_target,
    summarize_prediction_rows,
    write_markdown_summary,
)


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_set_name = resolve_best_feature_set()
    feature_columns = feature_set_from_name(feature_df, feature_set_name)

    rows: list[dict[str, object]] = []

    print_header("v90", "Pooled Panel Classifiers")

    separate_df, _ = evaluate_separate_benchmark_binary(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_columns=feature_columns,
        model_factory=logistic_factory(class_weight="balanced", c_value=0.5),
    )
    if not separate_df.empty:
        separate_df["feature_set"] = feature_set_name
        separate_df["model_name"] = "separate_logistic_balanced"
        separate_df["panel_structure"] = "separate"
        rows.extend(separate_df.to_dict(orient="records"))

    panel_shared = build_panel_dataset(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_columns=feature_columns,
        include_benchmark_dummies=False,
    )
    shared_preds = evaluate_panel_binary(
        panel_shared,
        target_name=target_name,
        feature_columns=feature_columns,
        model_factory=logistic_factory(class_weight="balanced", c_value=0.5),
    )
    shared_summary = summarize_prediction_rows(
        shared_preds,
        group_label="pooled_shared",
        target_name=target_name,
        feature_set_name=feature_set_name,
        model_name="pooled_shared_logistic_balanced",
    )
    if not shared_summary.empty:
        shared_summary["panel_structure"] = "shared"
        rows.extend(shared_summary.to_dict(orient="records"))

    panel_fe = build_panel_dataset(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_columns=feature_columns,
        include_benchmark_dummies=True,
    )
    fe_features = feature_columns + [column for column in panel_fe.columns if column.startswith("bm_")]
    fe_preds = evaluate_panel_binary(
        panel_fe,
        target_name=target_name,
        feature_columns=fe_features,
        model_factory=logistic_factory(class_weight="balanced", c_value=0.5),
    )
    fe_summary = summarize_prediction_rows(
        fe_preds,
        group_label="pooled_fixed_effects",
        target_name=target_name,
        feature_set_name=feature_set_name,
        model_name="pooled_fixed_effects_logistic_balanced",
    )
    if not fe_summary.empty:
        fe_summary["panel_structure"] = "fixed_effects"
        rows.extend(fe_summary.to_dict(orient="records"))

    results_df = pd.DataFrame(rows)
    pooled = results_df.loc[
        results_df["benchmark"] == "POOLED",
        ["model_name", "panel_structure", "balanced_accuracy", "brier_score", "ece_10"],
    ].sort_values("balanced_accuracy", ascending=False)
    best_model = str(pooled.iloc[0]["model_name"]) if not pooled.empty else "pooled_fixed_effects_logistic_balanced"
    results_df["selected_model"] = best_model
    results_df["selected_next"] = results_df["model_name"] == best_model

    print(pooled.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v90_pooled_panel_classifiers_results.csv")
    write_markdown_summary(
        Path("results/research/v90_pooled_panel_classifiers_summary.md"),
        "v90 Pooled Panel Summary",
        [
            f"Forward pooled reference: `{best_model}`.",
            "",
            pooled.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
