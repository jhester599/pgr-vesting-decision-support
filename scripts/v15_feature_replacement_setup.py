"""Prepare v15 fixed-budget feature-replacement inputs and swap queues."""

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
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_feature_columns
from src.research.v15 import (
    V15_FORECAST_UNIVERSE,
    V15_RESEARCH_STATUS,
    base_model_specs,
    build_existing_feature_inventory,
    build_inventory_template,
    build_swap_queue,
    normalize_inventory,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v15")


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v15_setup(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    inventory_csv: str | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn)
    available_features = set(get_feature_columns(df))
    specs = base_model_specs()
    stamp = datetime.today().strftime("%Y%m%d")

    template_path = Path(output_dir) / "feature_candidate_inventory_template.csv"
    if not template_path.exists():
        build_inventory_template().to_csv(template_path, index=False)

    existing_inventory = build_existing_feature_inventory(specs)
    existing_inventory.to_csv(Path(output_dir) / f"v15_existing_feature_inventory_{stamp}.csv", index=False)

    availability_df = pd.DataFrame(
        [
            {
                "feature_name": feature,
                "available_now": feature in available_features,
                "first_non_null_date": str(df[feature].dropna().index.min().date()) if feature in df.columns and not df[feature].dropna().empty else "",
                "non_null_rows": int(df[feature].notna().sum()) if feature in df.columns else 0,
            }
            for feature in sorted(available_features)
        ]
    )
    availability_df.to_csv(Path(output_dir) / f"v15_available_feature_coverage_{stamp}.csv", index=False)

    swap_queue_path = None
    if inventory_csv:
        inventory_df = pd.read_csv(inventory_csv)
        normalized = normalize_inventory(inventory_df)
        normalized.to_csv(Path(output_dir) / f"v15_feature_inventory_normalized_{stamp}.csv", index=False)
        queue_df = build_swap_queue(normalized, specs, available_features)
        swap_queue_path = Path(output_dir) / f"v15_swap_queue_{stamp}.csv"
        queue_df.to_csv(swap_queue_path, index=False)

    plan_lines = [
        "# codex-v15-plan.md",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Goal",
        "",
        "Run a fixed-budget feature-replacement cycle on the v14 leading replacement candidate without expanding model complexity or feature count materially.",
        "",
        "## Starting Point",
        "",
        f"- Research status: `{V15_RESEARCH_STATUS}`",
        f"- Forecast universe: `{', '.join(V15_FORECAST_UNIVERSE)}`",
        "- Leading prediction-layer candidate from v14: `ensemble_ridge_gbt`",
        "- Working model baselines:",
        f"  - ridge: `{', '.join(specs['ridge_lean_v1'].features)}`",
        f"  - gbt: `{', '.join(specs['gbt_lean_plus_two'].features)}`",
        "",
        "## Rules",
        "",
        "- keep the v13.1 recommendation layer fixed",
        "- test one replacement at a time",
        "- prefer one-for-one swaps",
        "- only consider broader changes after the one-for-one queue is exhausted",
        "- include both PGR-specific features and benchmark-predictive/shared-regime features",
        "",
        "## Setup Artifacts",
        "",
        f"- candidate inventory template: `{template_path}`",
        f"- existing baseline feature inventory: `results/v15/v15_existing_feature_inventory_{stamp}.csv`",
        f"- currently available feature coverage: `results/v15/v15_available_feature_coverage_{stamp}.csv`",
    ]
    if swap_queue_path is not None:
        plan_lines.append(f"- generated swap queue: `{swap_queue_path}`")
    _write_text(Path("docs") / "plans" / "codex-v15-plan.md", plan_lines)

    summary_lines = [
        "# V15 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Status",
        "",
        "v15 setup is complete. The repo is prepared for fixed-budget feature replacement once the external deep-research candidate lists are ready.",
        "",
        "## Prepared Inputs",
        "",
        f"- Forecast universe: `{', '.join(V15_FORECAST_UNIVERSE)}`",
        "- Baseline candidate carried forward from v14: `ensemble_ridge_gbt`",
        "- Ridge baseline: `ridge_lean_v1`",
        "- GBT baseline: `gbt_lean_plus_two`",
        "- Candidate inventory template written to `results/v15/feature_candidate_inventory_template.csv`",
        "- Current feature coverage inventory written under `results/v15/`",
    ]
    if swap_queue_path is not None:
        summary_lines += [
            f"- Swap queue generated at `{swap_queue_path}`",
            "- The next step is to run the one-for-one replacement queue in priority order.",
        ]
    else:
        summary_lines += [
            "- Awaiting external deep-research outputs before generating the swap queue.",
        ]
    _write_text(Path("docs") / "results" / "V15_RESULTS_SUMMARY.md", summary_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v15 fixed-budget feature replacement inputs.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--inventory-csv",
        default="",
        help="Optional candidate inventory CSV. If provided, a swap queue is generated immediately.",
    )
    args = parser.parse_args()
    run_v15_setup(
        output_dir=args.output_dir,
        inventory_csv=args.inventory_csv or None,
    )


if __name__ == "__main__":
    main()
