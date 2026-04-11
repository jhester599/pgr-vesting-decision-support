"""v95 - Policy replay and promotion-style evaluation for classifier hybrids."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.research.v37_utils import print_footer, print_header, save_results
from src.research.v87_utils import summarize_monthly_policy_path, write_markdown_summary


def main() -> None:
    detail_path = Path("results/research/v94_hybrid_gate_detail.csv")
    if not detail_path.exists():
        raise FileNotFoundError("Run v94_hybrid_gate.py before v95_policy_replay.py.")

    detail_df = pd.read_csv(detail_path, parse_dates=["date"])
    summary_rows: list[dict[str, object]] = []
    regression_ref = detail_df.loc[
        detail_df["variant"] == "regression_only_quality_weighted",
        ["date", "hold_fraction"],
    ].rename(columns={"hold_fraction": "regression_hold"})

    print_header("v95", "Policy Replay and Promotion-Style Evaluation")
    for variant, frame in detail_df.groupby("variant"):
        hold_series = pd.Series(frame["hold_fraction"].to_numpy(dtype=float), index=frame["date"], name="hold_fraction")
        realized_series = pd.Series(frame["realized"].to_numpy(dtype=float), index=frame["date"], name="realized")
        row = summarize_monthly_policy_path(variant, hold_series, realized_series)
        aligned = frame.merge(regression_ref, on="date", how="left")
        row["agreement_with_regression_rate"] = float(
            (aligned["hold_fraction"] == aligned["regression_hold"]).mean()
        )
        row["mean_abs_hold_diff_vs_regression"] = float(
            (aligned["hold_fraction"] - aligned["regression_hold"]).abs().mean()
        )
        summary_rows.append(row)

    results_df = pd.DataFrame(summary_rows)
    non_reg = results_df.loc[
        results_df["variant"] != "regression_only_quality_weighted"
    ].sort_values(
        ["mean_policy_return", "capture_ratio", "agreement_with_regression_rate"],
        ascending=[False, False, False],
    )
    best_variant = str(non_reg.iloc[0]["variant"]) if not non_reg.empty else "regression_only_quality_weighted"
    results_df["selected_variant"] = best_variant
    results_df["selected_next"] = results_df["variant"] == best_variant

    print(
        results_df[
            [
                "variant",
                "mean_policy_return",
                "capture_ratio",
                "hold_fraction_changes",
                "agreement_with_regression_rate",
                "selected_next",
            ]
        ].sort_values("mean_policy_return", ascending=False).to_string(
            index=False,
            float_format="{:.4f}".format,
        )
    )
    print_footer()

    save_results(results_df, "v95_policy_replay_results.csv")
    write_markdown_summary(
        Path("results/research/v95_policy_replay_summary.md"),
        "v95 Policy Replay Summary",
        [
            f"Best replay candidate: `{best_variant}`.",
            "",
            results_df.sort_values("mean_policy_return", ascending=False).to_markdown(index=False),
        ],
    )


if __name__ == "__main__":
    main()
