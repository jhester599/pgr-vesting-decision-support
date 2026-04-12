"""v113 - Constrained selection of promotable policy candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import RESULTS_DIR
from src.research.v102_utils import write_results, write_summary


def _load_results(path: Path, source_version: str, gate_style: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_version"] = source_version
    df["gate_style"] = gate_style
    return df


def main() -> None:
    baseline = pd.read_csv(RESULTS_DIR / "v95_policy_replay_results.csv")
    regression_row = baseline[baseline["variant"] == "regression_only_quality_weighted"].iloc[0]
    baseline_mean = float(regression_row["mean_policy_return"])
    baseline_changes = int(regression_row["hold_fraction_changes"])

    candidate_frames = [
        _load_results(
            RESULTS_DIR / "v110_gemini_veto_gate_results.csv",
            "v110",
            "veto_regression_sell",
        ),
        _load_results(
            RESULTS_DIR / "v111_permission_overlay_results.csv",
            "v111",
            "permission_to_deviate",
        ),
        _load_results(
            RESULTS_DIR / "v112_target_reformulation_results.csv",
            "v112",
            "target_reformulation",
        ),
    ]
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["uplift_vs_regression"] = (
        candidates["mean_policy_return"] - baseline_mean
    )
    candidates["changes_vs_regression"] = (
        candidates["hold_fraction_changes"] - baseline_changes
    )
    candidates["promotion_eligible"] = (
        (candidates["uplift_vs_regression"] > 0.0)
        & (candidates["agreement_with_regression_rate"] >= 0.70)
        & (candidates["hold_fraction_changes"] <= baseline_changes + 4)
    )
    ranked = candidates.sort_values(
        [
            "promotion_eligible",
            "uplift_vs_regression",
            "agreement_with_regression_rate",
            "hold_fraction_changes",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    write_results("v113_constrained_candidate_selection_results.csv", ranked)
    best = ranked.iloc[0]
    write_summary(
        "v113_constrained_candidate_selection_summary.md",
        "v113 Constrained Candidate Selection",
        [
            "Top candidate:",
            f"- variant: {best['variant']}",
            f"- source version: {best['source_version']}",
            f"- gate style: {best['gate_style']}",
            f"- uplift vs regression: {best['uplift_vs_regression']:.4f}",
            f"- agreement with regression: {best['agreement_with_regression_rate']:.4f}",
            f"- promotion eligible: {bool(best['promotion_eligible'])}",
        ],
    )


if __name__ == "__main__":
    main()
