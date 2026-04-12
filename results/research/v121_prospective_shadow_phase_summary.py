"""v121 - Final summary for the prospective shadow-monitoring phase."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v102_utils import write_results, write_summary


def main() -> None:
    scorecard = pd.read_csv(Path("results/research/v119_disagreement_scorecard_results.csv"))
    assessment = pd.read_csv(
        Path("results/research/v120_prospective_shadow_gate_assessment_results.csv")
    )
    score = scorecard.iloc[0]
    gate = assessment.iloc[0]
    summary_df = pd.DataFrame(
        [
            {
                "selected_variant": score["selected_variant"],
                "agreement_rate": score["agreement_rate"],
                "disagreement_months": score["disagreement_months"],
                "cumulative_shadow_minus_live_all": score["cumulative_shadow_minus_live_all"],
                "cumulative_shadow_minus_live_disagreement": score[
                    "cumulative_shadow_minus_live_disagreement"
                ],
                "next_step_decision": gate["decision"],
            }
        ]
    )
    write_results("v121_prospective_shadow_phase_summary_results.csv", summary_df)
    write_summary(
        "v121_prospective_shadow_phase_summary.md",
        "v121 Prospective Shadow Monitoring Phase Summary",
        [
            f"- selected candidate: {score['selected_variant']}",
            f"- agreement rate: {float(score['agreement_rate']):.4f}",
            f"- disagreement months: {int(score['disagreement_months'])}",
            f"- cumulative shadow minus live (all months): {float(score['cumulative_shadow_minus_live_all']):.4f}",
            f"- cumulative shadow minus live (disagreement months): {float(score['cumulative_shadow_minus_live_disagreement']):.4f}",
            f"- next step: {gate['decision']}",
        ],
    )


if __name__ == "__main__":
    main()
