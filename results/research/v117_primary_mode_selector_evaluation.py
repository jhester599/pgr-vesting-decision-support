"""v117 - Evaluate classifier influence on recommendation mode selection."""

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
    v116 = pd.read_csv(RESULTS_DIR / "v116_limited_gate_candidate_results.csv")
    selected = v116.iloc[0]
    defer = not bool(selected["limited_gate_candidate_ready"])
    result = pd.DataFrame(
        [
            {
                "selected_variant": selected["selected_variant"],
                "limited_gate_candidate_ready": bool(selected["limited_gate_candidate_ready"]),
                "primary_mode_selector_ready": False if defer else False,
                "decision": "defer_classifier_mode_promotion",
            }
        ]
    )
    write_results("v117_primary_mode_selector_evaluation_results.csv", result)
    write_summary(
        "v117_primary_mode_selector_evaluation.md",
        "v117 Primary Recommendation-Mode Selector Evaluation",
        [
            f"- selected variant: {selected['selected_variant']}",
            f"- limited gate candidate ready: {bool(selected['limited_gate_candidate_ready'])}",
            "- decision: defer classifier-led recommendation-mode promotion pending longer shadow evidence and matured monitoring.",
        ],
    )


if __name__ == "__main__":
    main()
