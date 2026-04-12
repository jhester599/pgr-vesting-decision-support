"""v112 - Narrow target reformulation research."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v102_utils import (
    ACTION_THRESHOLD,
    build_target_probability_alignment,
    evaluate_overlay_variant,
    write_results,
    write_summary,
)


def _target_hold_fraction(aligned: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "actionable_sell_3pct":
        hold = pd.Series(0.5, index=aligned.index, name="hold_fraction")
        hold.loc[
            (aligned["prob_target"] >= 0.70) & (aligned["predicted"] < -ACTION_THRESHOLD)
        ] = 0.0
        return hold
    if target_name == "actionable_hold_3pct":
        hold = pd.Series(0.5, index=aligned.index, name="hold_fraction")
        hold.loc[
            (aligned["prob_target"] >= 0.70) & (aligned["predicted"] > ACTION_THRESHOLD)
        ] = 1.0
        return hold
    hold = pd.Series(0.5, index=aligned.index, name="hold_fraction")
    high_conf = aligned["prob_target"] >= 0.70
    hold.loc[high_conf & (aligned["predicted"] < -ACTION_THRESHOLD)] = 0.0
    hold.loc[high_conf & (aligned["predicted"] > ACTION_THRESHOLD)] = 1.0
    return hold


def main() -> None:
    rows: list[dict[str, object]] = []
    for target_name in [
        "actionable_sell_3pct",
        "actionable_hold_3pct",
        "deviate_from_default_50pct_sell",
    ]:
        aligned, baseline_hold = build_target_probability_alignment(target_name)
        hold = _target_hold_fraction(aligned, target_name)
        rows.append(
            evaluate_overlay_variant(
                aligned.rename(columns={"prob_target": "prob_sell"}),
                variant=target_name,
                hold_fraction=hold,
                baseline_hold=baseline_hold,
                gate_style="target_reformulation",
                threshold=0.70,
            )
        )
    results = pd.DataFrame(rows).sort_values(
        ["mean_policy_return", "agreement_with_regression_rate"],
        ascending=[False, False],
    )
    write_results("v112_target_reformulation_results.csv", results)
    best = results.iloc[0]
    write_summary(
        "v112_target_reformulation_summary.md",
        "v112 Target Reformulation",
        [
            "Best target formulation:",
            f"- target: {best['variant']}",
            f"- mean policy return: {best['mean_policy_return']:.4f}",
            f"- agreement with regression: {best['agreement_with_regression_rate']:.4f}",
            f"- hold-fraction changes: {int(best['hold_fraction_changes'])}",
        ],
    )


if __name__ == "__main__":
    main()
