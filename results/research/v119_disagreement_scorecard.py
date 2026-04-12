"""v119 - Disagreement-focused scorecard for prospective shadow monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v102_utils import write_results, write_summary
from src.research.v118_utils import summarize_disagreement_scorecard


def main() -> None:
    detail = pd.read_csv(
        Path("results/research/v118_prospective_shadow_replay_results.csv"),
        parse_dates=["date"],
    )
    scorecard = summarize_disagreement_scorecard(detail)
    write_results("v119_disagreement_scorecard_results.csv", scorecard)
    row = scorecard.iloc[0]
    write_summary(
        "v119_disagreement_scorecard_summary.md",
        "v119 Disagreement Scorecard",
        [
            f"- selected candidate: {row['selected_variant']}",
            f"- agreement rate: {float(row['agreement_rate']):.4f}",
            f"- disagreement months: {int(row['disagreement_months'])}",
            f"- cumulative shadow minus live (all months): {float(row['cumulative_shadow_minus_live_all']):.4f}",
            f"- cumulative shadow minus live (disagreement months): {float(row['cumulative_shadow_minus_live_disagreement']):.4f}",
            f"- max consecutive disagreements: {int(row['max_consecutive_disagreements'])}",
        ],
    )


if __name__ == "__main__":
    main()
