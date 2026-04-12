"""v114 - Shadow gate overlay selection summary."""

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
    selected = ranked.iloc[0]
    summary_df = pd.DataFrame(
        [
            {
                "selected_variant": selected["variant"],
                "source_version": selected["source_version"],
                "gate_style": selected["gate_style"],
                "threshold": selected["threshold"],
                "promotion_eligible": bool(selected["promotion_eligible"]),
            }
        ]
    )
    write_results("v114_shadow_gate_overlay_summary_results.csv", summary_df)
    write_summary(
        "v114_shadow_gate_overlay_summary.md",
        "v114 Shadow Gate Overlay Summary",
        [
            "Selected shadow overlay candidate for monthly production artifacts:",
            f"- variant: {selected['variant']}",
            f"- source version: {selected['source_version']}",
            f"- gate style: {selected['gate_style']}",
            f"- threshold: {float(selected['threshold']):.2f}",
            f"- promotion eligible: {bool(selected['promotion_eligible'])}",
        ],
    )


if __name__ == "__main__":
    main()
