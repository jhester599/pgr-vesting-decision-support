"""v120 - Promotion-style assessment of the prospective shadow replay."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v102_utils import write_results, write_summary
from src.research.v118_utils import summarize_prospective_gate_assessment


def main() -> None:
    scorecard = pd.read_csv(Path("results/research/v119_disagreement_scorecard_results.csv"))
    monitoring = pd.read_csv(
        Path("results/research/v115_classifier_monitoring_summary_results.csv")
    )
    assessment = summarize_prospective_gate_assessment(scorecard, monitoring)
    write_results("v120_prospective_shadow_gate_assessment_results.csv", assessment)
    row = assessment.iloc[0]
    write_summary(
        "v120_prospective_shadow_gate_assessment_summary.md",
        "v120 Prospective Shadow Gate Assessment",
        [
            f"- selected candidate: {row['selected_variant']}",
            f"- decision: {row['decision']}",
            f"- agreement pass: {bool(row['agreement_pass'])}",
            f"- uplift pass: {bool(row['uplift_pass'])}",
            f"- disagreement-month pass: {bool(row['disagreement_pass'])}",
            f"- churn pass: {bool(row['churn_pass'])}",
            f"- matured live monitoring observations: {int(row['matured_live_monitoring_n'])}",
        ],
    )


if __name__ == "__main__":
    main()
