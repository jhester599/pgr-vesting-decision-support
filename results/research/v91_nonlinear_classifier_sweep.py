"""v91 - Conservative nonlinear classifier family sweep."""

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
    feature_set_from_name,
    hist_gbt_factory,
    load_research_inputs,
    logistic_factory,
    resolve_best_feature_set,
    resolve_primary_target,
    summarize_prediction_rows,
    write_markdown_summary,
)


MODEL_FACTORIES = {
    "logistic_fixed_effects_balanced": logistic_factory(class_weight="balanced", c_value=0.5),
    "histgb_depth2": hist_gbt_factory(max_depth=2, max_iter=120, learning_rate=0.05, min_samples_leaf=10),
    "histgb_depth3": hist_gbt_factory(max_depth=3, max_iter=160, learning_rate=0.04, min_samples_leaf=12),
}


def main() -> None:
    feature_df, rel_map = load_research_inputs()
    target_name = resolve_primary_target()
    feature_set_name = resolve_best_feature_set()
    feature_columns = feature_set_from_name(feature_df, feature_set_name)

    panel_fe = build_panel_dataset(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_columns=feature_columns,
        include_benchmark_dummies=True,
    )
    fe_features = feature_columns + [column for column in panel_fe.columns if column.startswith("bm_")]

    rows: list[dict[str, object]] = []
    print_header("v91", "Nonlinear Classifier Sweep")
    for model_name, factory in MODEL_FACTORIES.items():
        pred_df = evaluate_panel_binary(
            panel_fe,
            target_name=target_name,
            feature_columns=fe_features,
            model_factory=factory,
        )
        summary_df = summarize_prediction_rows(
            pred_df,
            group_label="pooled_fixed_effects",
            target_name=target_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
        )
        if summary_df.empty:
            continue
        summary_df["model_family"] = "nonlinear" if model_name.startswith("histgb") else "linear_reference"
        rows.extend(summary_df.to_dict(orient="records"))

    results_df = pd.DataFrame(rows)
    pooled = results_df.loc[
        results_df["benchmark"] == "POOLED",
        ["model_name", "model_family", "balanced_accuracy", "brier_score", "ece_10"],
    ].sort_values("balanced_accuracy", ascending=False)
    best_model = str(pooled.iloc[0]["model_name"]) if not pooled.empty else "logistic_fixed_effects_balanced"
    results_df["selected_model"] = best_model
    results_df["selected_next"] = results_df["model_name"] == best_model

    print(pooled.to_string(index=False, float_format="{:.4f}".format))
    print_footer()

    save_results(results_df, "v91_nonlinear_classifier_sweep_results.csv")
    write_markdown_summary(
        Path("results/research/v91_nonlinear_classifier_sweep_summary.md"),
        "v91 Nonlinear Sweep Summary",
        [
            f"Forward model-family winner: `{best_model}`.",
            "",
            pooled.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
