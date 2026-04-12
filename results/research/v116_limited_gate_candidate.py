"""v116 - Limited production gate candidacy evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import RESULTS_DIR
from src.research.v102_utils import write_results, write_summary


def main() -> None:
    ranked = pd.read_csv(RESULTS_DIR / "v113_constrained_candidate_selection_results.csv")
    monitoring = pd.read_csv(RESULTS_DIR / "v115_classifier_monitoring_summary_results.csv")
    selected = ranked.iloc[0]
    monitor = monitoring.iloc[0]
    matured_ok = int(monitor["matured_n"]) == 0 or pd.notna(monitor["brier_score"])
    ready = bool(selected["promotion_eligible"]) and matured_ok
    result = pd.DataFrame(
        [
            {
                "selected_variant": selected["variant"],
                "gate_style": selected["gate_style"],
                "promotion_eligible": bool(selected["promotion_eligible"]),
                "matured_monitoring_ready": matured_ok,
                "limited_gate_candidate_ready": ready,
            }
        ]
    )
    write_results("v116_limited_gate_candidate_results.csv", result)
    write_summary(
        "v116_limited_gate_candidate_summary.md",
        "v116 Limited Gate Candidate",
        [
            f"- selected variant: {selected['variant']}",
            f"- promotion eligible from v113: {bool(selected['promotion_eligible'])}",
            f"- matured monitoring ready: {matured_ok}",
            f"- limited gate candidate ready: {ready}",
            "- Recommendation: keep the gate shadow-only unless this candidate remains promotion-eligible and matured monitoring stays acceptable.",
        ],
    )


if __name__ == "__main__":
    main()
