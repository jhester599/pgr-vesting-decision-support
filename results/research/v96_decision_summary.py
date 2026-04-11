"""v96 - Final summary and candidacy decision for the classification program."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import write_markdown_summary


def _selected_row(path: str, selected_col: str) -> pd.Series | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    selected = df.loc[df[selected_col]]
    if not selected.empty:
        return selected.iloc[0]
    return df.iloc[0] if not df.empty else None


def main() -> None:
    selected_rows: list[dict[str, object]] = []

    v87 = _selected_row("results/research/v87_target_taxonomy_results.csv", "recommended_next")
    v88 = _selected_row("results/research/v88_feature_sweep_results.csv", "selected_next")
    v89 = _selected_row("results/research/v89_per_benchmark_linear_results.csv", "selected_next")
    v90 = _selected_row("results/research/v90_pooled_panel_classifiers_results.csv", "selected_next")
    v91 = _selected_row("results/research/v91_nonlinear_classifier_sweep_results.csv", "selected_next")
    v92 = _selected_row("results/research/v92_calibration_and_abstention_results.csv", "selected_next")
    v93 = _selected_row("results/research/v93_basket_targets_results.csv", "selected_next")
    v94 = _selected_row("results/research/v94_hybrid_gate_results.csv", "selected_next")
    v95 = _selected_row("results/research/v95_policy_replay_results.csv", "selected_next")

    if v87 is not None:
        selected_rows.append(
            {"stage": "v87", "selection": str(v87["target"]), "notes": "forward binary target"}
        )
    if v88 is not None:
        selected_rows.append(
            {"stage": "v88", "selection": str(v88["feature_set"]), "notes": "forward feature set"}
        )
    if v89 is not None:
        selected_rows.append(
            {"stage": "v89", "selection": str(v89["model_name"]), "notes": "best separate linear model"}
        )
    if v90 is not None:
        selected_rows.append(
            {"stage": "v90", "selection": str(v90["model_name"]), "notes": "best pooled / panel linear model"}
        )
    if v91 is not None:
        selected_rows.append(
            {"stage": "v91", "selection": str(v91["model_name"]), "notes": "best nonlinear family result"}
        )
    if v92 is not None:
        selected_rows.append(
            {"stage": "v92", "selection": str(v92["variant"]), "notes": "best calibration + abstention path"}
        )
    if v93 is not None:
        selected_rows.append(
            {"stage": "v93", "selection": str(v93["candidate_name"]), "notes": "best target formulation"}
        )
    if v94 is not None:
        selected_rows.append(
            {"stage": "v94", "selection": str(v94["variant"]), "notes": "best classifier / hybrid policy candidate"}
        )
    if v95 is not None:
        selected_rows.append(
            {"stage": "v95", "selection": str(v95["variant"]), "notes": "best replay candidate"}
        )

    summary_df = pd.DataFrame(selected_rows)

    recommendation_status = "continue_research_no_promotion"
    rationale: list[str] = []
    if v94 is not None and v95 is not None:
        regression_df = pd.read_csv("results/research/v94_hybrid_gate_results.csv")
        regression_row = regression_df.loc[
            regression_df["variant"] == "regression_only_quality_weighted"
        ].iloc[0]
        best_replay = pd.read_csv("results/research/v95_policy_replay_results.csv")
        chosen = best_replay.loc[best_replay["selected_next"]].iloc[0]
        if (
            float(chosen["mean_policy_return"]) > float(regression_row["mean_policy_return"])
            and float(chosen["agreement_with_regression_rate"]) >= 0.70
            and int(chosen["hold_fraction_changes"]) <= int(regression_row["hold_fraction_changes"]) + 4
        ):
            recommendation_status = "shadow_candidate_only"
            rationale.append(
                "The best classification-led path improved replay utility without causing an excessive stability break."
            )
        else:
            rationale.append(
                "The best classification-led path did not clear a conservative promotion gate versus the regression baseline."
            )

    print_header("v96", "Classification Program Decision Summary")
    print(summary_df.to_string(index=False))
    print(f"\nRecommendation status: {recommendation_status}")
    print_footer()

    save_results(summary_df, "v96_decision_summary_results.csv")
    write_markdown_summary(
        Path("results/research/v96_decision_summary.md"),
        "v96 Classification Program Decision Summary",
        [
            f"Recommendation status: `{recommendation_status}`.",
            *rationale,
            "",
            summary_df.to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
