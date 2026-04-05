"""v22 implementation step for the promoted visible cross-check."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts import monthly_decision
from src.database import db_client
from src.research.v22 import build_promoted_cross_check_summary, v22_promoted_cross_check_spec


DEFAULT_OUTPUT_DIR = os.path.join("results", "v22")


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v22_cross_check_promotion(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    as_of: date | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    run_as_of = as_of or date.today()

    cross_check = build_promoted_cross_check_summary(conn, run_as_of, target_horizon_months=6)
    shadow_summary, _ = monthly_decision._build_shadow_baseline_summary(conn, run_as_of, 6)  # noqa: SLF001
    if cross_check is None or shadow_summary is None:
        raise RuntimeError("Unable to build the v22 promotion snapshots.")

    agreement = {
        "signal_agrees_with_shadow": cross_check.consensus == shadow_summary.consensus,
        "mode_agrees_with_shadow": cross_check.recommendation_mode == shadow_summary.recommendation_mode,
        "sell_agrees_with_shadow": abs(cross_check.sell_pct - shadow_summary.sell_pct) < 1e-9,
    }

    spec = v22_promoted_cross_check_spec()
    snapshot_df = pd.DataFrame(
        [
            {
                "path_name": "visible_cross_check",
                "candidate_name": cross_check.candidate_name,
                "policy_name": cross_check.policy_name,
                "consensus": cross_check.consensus,
                "recommendation_mode": cross_check.recommendation_mode,
                "sell_pct": cross_check.sell_pct,
                "mean_predicted": cross_check.mean_predicted,
                "mean_ic": cross_check.mean_ic,
                "mean_hit_rate": cross_check.mean_hit_rate,
                "aggregate_oos_r2": cross_check.aggregate_oos_r2,
                "aggregate_nw_ic": cross_check.aggregate_nw_ic,
                **agreement,
                "notes": spec.notes,
            },
            {
                "path_name": "simpler_baseline",
                "candidate_name": shadow_summary.candidate_name,
                "policy_name": shadow_summary.policy_name,
                "consensus": shadow_summary.consensus,
                "recommendation_mode": shadow_summary.recommendation_mode,
                "sell_pct": shadow_summary.sell_pct,
                "mean_predicted": shadow_summary.mean_predicted,
                "mean_ic": shadow_summary.mean_ic,
                "mean_hit_rate": shadow_summary.mean_hit_rate,
                "aggregate_oos_r2": shadow_summary.aggregate_oos_r2,
                "aggregate_nw_ic": shadow_summary.aggregate_nw_ic,
                "signal_agrees_with_shadow": True,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
                "notes": "Active recommendation layer remains the simpler diversification-first baseline.",
            },
        ]
    )

    decision_df = pd.DataFrame(
        [
            {
                "status": "implemented_visible_cross_check_promotion",
                "recommended_path": spec.candidate_name,
                "as_of": run_as_of.isoformat(),
                "rationale": (
                    "v22 promotes the v21 historical winner as the visible production cross-check while "
                    "keeping the v13.1 simpler diversification-first recommendation layer unchanged."
                ),
            }
        ]
    )

    stamp = datetime.today().strftime("%Y%m%d")
    snapshot_path = Path(output_dir) / f"v22_cross_check_snapshot_{stamp}.csv"
    decision_path = Path(output_dir) / f"v22_decision_{stamp}.csv"
    snapshot_df.to_csv(snapshot_path, index=False)
    decision_df.to_csv(decision_path, index=False)

    result_lines = [
        "# V22 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v22 is a narrow implementation step: keep the v13.1 simpler diversification-first recommendation layer active, but replace the visible production cross-check with the v21 historical winner.",
        "",
        "## Implemented Path",
        "",
        f"- Promoted cross-check candidate: `{spec.candidate_name}`",
        f"- Members: `{', '.join(spec.members)}`",
        f"- As-of validation date: `{run_as_of.isoformat()}`",
        "",
        "## Current Snapshot",
        "",
        f"- Visible cross-check signal: `{cross_check.consensus}`",
        f"- Visible cross-check recommendation mode: `{cross_check.recommendation_mode}`",
        f"- Visible cross-check sell %: `{cross_check.sell_pct:.0%}`",
        f"- Visible cross-check predicted 6M relative return: `{cross_check.mean_predicted:+.2%}`",
        f"- Signal agrees with simpler baseline: `{agreement['signal_agrees_with_shadow']}`",
        f"- Recommendation mode agrees with simpler baseline: `{agreement['mode_agrees_with_shadow']}`",
        f"- Sell % agrees with simpler baseline: `{agreement['sell_agrees_with_shadow']}`",
        "",
        "## Decision",
        "",
        "- Status: `implemented_visible_cross_check_promotion`",
        "- The active recommendation layer is unchanged; only the displayed cross-check candidate changed.",
    ]
    _write_text(Path("docs") / "results" / "V22_RESULTS_SUMMARY.md", result_lines)

    closeout_lines = [
        "# V22 Closeout And V23 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v22 implemented the v21 recommendation by promoting `ensemble_ridge_gbt_v18` as the visible production cross-check.",
        "- The active recommendation path remains the simpler diversification-first baseline from v13.1.",
        "",
        "## Recommended V23 Scope",
        "",
        "- Test whether the same promotion conclusion still holds when the benchmark universe is extended backward with research-only pre-inception proxy histories.",
        "- Focus on VOO, VXUS, and VMBS, which currently bind the common OOS start date.",
    ]
    _write_text(Path("docs") / "closeouts" / "V22_CLOSEOUT_AND_V23_NEXT.md", closeout_lines)

    plan_lines = [
        "# v22 Cross-Check Promotion Plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Objective",
        "",
        "- Implement the v21 recommendation without changing the active recommendation layer.",
        "",
        "## Scope",
        "",
        "- Promote `ensemble_ridge_gbt_v18` as the visible production cross-check.",
        "- Keep the promoted simpler diversification-first baseline as the active recommendation layer.",
        "- Leave the underlying live 4-model production signal path in place for now.",
    ]
    _write_text(Path("docs") / "plans" / "codex-v22-plan.md", plan_lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v22 visible cross-check promotion step.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_v22_cross_check_promotion(
        output_dir=args.output_dir,
        as_of=args.as_of,
    )
