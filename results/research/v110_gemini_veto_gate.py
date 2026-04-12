"""v110 - Gemini-style veto gate backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v102_utils import (
    build_overlay_alignment_frame,
    evaluate_overlay_variant,
    veto_overlay_hold_fraction,
    write_results,
    write_summary,
)


def main() -> None:
    aligned, baseline_hold = build_overlay_alignment_frame()
    rows: list[dict[str, object]] = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        hold = veto_overlay_hold_fraction(
            aligned,
            baseline_hold,
            threshold=threshold,
        )
        rows.append(
            evaluate_overlay_variant(
                aligned,
                variant=f"gemini_veto_{threshold:.2f}",
                hold_fraction=hold,
                baseline_hold=baseline_hold,
                gate_style="veto_regression_sell",
                threshold=threshold,
            )
        )
    results = pd.DataFrame(rows).sort_values(
        ["mean_policy_return", "agreement_with_regression_rate"],
        ascending=[False, False],
    )
    write_results("v110_gemini_veto_gate_results.csv", results)
    best = results.iloc[0]
    write_summary(
        "v110_gemini_veto_gate_summary.md",
        "v110 Gemini-Style Veto Gate",
        [
            "Best candidate:",
            f"- variant: {best['variant']}",
            f"- mean policy return: {best['mean_policy_return']:.4f}",
            f"- agreement with regression: {best['agreement_with_regression_rate']:.4f}",
            f"- hold-fraction changes: {int(best['hold_fraction_changes'])}",
            f"- unnecessary action rate: {best['unnecessary_action_rate']:.4f}",
        ],
    )


if __name__ == "__main__":
    main()
