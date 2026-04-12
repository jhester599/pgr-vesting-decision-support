"""v118 - Prospective shadow replay for the selected gate candidate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v102_utils import write_results, write_summary
from src.research.v118_utils import build_prospective_shadow_replay


def main() -> None:
    detail = build_prospective_shadow_replay()
    write_results("v118_prospective_shadow_replay_results.csv", detail)
    selected_variant = str(detail["selected_variant"].iloc[0])
    agreement_rate = float((~detail["would_change"]).mean())
    cumulative_diff = float(detail["shadow_minus_live"].sum())
    disagreements = int(detail["would_change"].sum())
    write_summary(
        "v118_prospective_shadow_replay_summary.md",
        "v118 Prospective Shadow Replay",
        [
            f"- selected candidate: {selected_variant}",
            f"- review months: {len(detail)}",
            f"- agreement rate: {agreement_rate:.4f}",
            f"- disagreement months: {disagreements}",
            f"- cumulative shadow minus live: {cumulative_diff:.4f}",
        ],
    )


if __name__ == "__main__":
    main()
