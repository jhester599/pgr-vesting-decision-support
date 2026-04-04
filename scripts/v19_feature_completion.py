"""v19 completion pass for the remaining v15 feature inventory."""

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
from scripts.v15_execute import (
    _benchmark_dataset_map,
    _build_final_bakeoff_specs,
    _evaluate_baseline_historical_mean,
    _evaluate_model_spec,
    _run_swap_phase,
    _summarize_model_bakeoff,
)
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db
from src.research.v15 import (
    V15_FORECAST_UNIVERSE,
    base_model_specs,
    build_confirmation_queue,
    build_swap_queue,
    choose_best_confirmed_swaps,
    choose_phase0_winners,
    deployed_model_specs,
    normalize_inventory,
)
from src.research.v19 import (
    BLOCKED_FEATURE_REASONS,
    build_v19_traceability_matrix,
    fetch_v19_public_macro,
    upsert_v19_public_macro,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v19")
DEFAULT_INVENTORY_CSV = os.path.join(
    "results", "v15", "v15_feature_candidate_inventory_from_reports_20260404.csv"
)

V19_FALLBACK_SWAPS: dict[str, list[tuple[str, str]]] = {
    # gasoline_retail_sales_delta was proposed as a competitor to wti_return_3m,
    # but the lean v15/v16 survivor specs never carried WTI directly. Evaluate
    # it against the nearest surviving generic benchmark-macro slot instead of
    # leaving it untested.
    "gasoline_retail_sales_delta": [("gbt_lean_plus_two", "yield_curvature")],
}


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v19_feature_completion(
    *,
    inventory_csv: str = DEFAULT_INVENTORY_CSV,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    horizon: int = 6,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")

    conn = db_client.get_connection(config.DB_PATH)

    public_macro_df, public_macro_summary = fetch_v19_public_macro()
    public_macro_summary["upserted_rows"] = 0
    if not public_macro_df.empty:
        upsert_v19_public_macro(conn, public_macro_df)
        public_macro_summary["upserted_rows"] = public_macro_summary["rows_loaded"]
    public_macro_summary.to_csv(
        Path(output_dir) / f"v19_public_macro_backfill_{stamp}.csv",
        index=False,
    )

    original_min_obs = config.WFO_MIN_GAINSHARE_OBS
    # Research-only relaxation: several late-added EDGAR breadth fields have
    # ~25 monthly observations, which is enough for a one-feature-at-a-time
    # evaluation pass but below the stricter production promotion guard.
    config.WFO_MIN_GAINSHARE_OBS = min(original_min_obs, 24)
    try:
        feature_matrix = build_feature_matrix_from_db(conn, force_refresh=True)
    finally:
        config.WFO_MIN_GAINSHARE_OBS = original_min_obs
    feature_columns = set(feature_matrix.columns)
    coverage_rows = []
    for feature in sorted(feature_columns):
        coverage_rows.append(
            {
                "feature_name": feature,
                "non_null_rows": int(feature_matrix[feature].notna().sum()),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(
        Path(output_dir) / f"v19_feature_coverage_{stamp}.csv",
        index=False,
    )

    benchmark_data = _benchmark_dataset_map(conn, feature_matrix, list(V15_FORECAST_UNIVERSE), horizon)
    inventory_df = normalize_inventory(pd.read_csv(inventory_csv))

    phase0_specs = base_model_specs()
    phase0_queue = build_swap_queue(inventory_df, phase0_specs, feature_columns)
    fallback_rows: list[dict[str, object]] = []
    queued_pairs = {
        (str(row["candidate_name"]), str(row["candidate_feature"]), str(row["replace_feature"]))
        for _, row in phase0_queue.iterrows()
    }
    inventory_lookup = inventory_df.set_index("feature_name", drop=False)
    for feature_name, replacements in V19_FALLBACK_SWAPS.items():
        if feature_name not in inventory_lookup.index or feature_name not in feature_columns:
            continue
        source_row = inventory_lookup.loc[feature_name]
        for candidate_name, replace_feature in replacements:
            spec = phase0_specs[candidate_name]
            queue_key = (candidate_name, feature_name, replace_feature)
            if queue_key in queued_pairs or replace_feature not in spec.features:
                continue
            fallback_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_type": spec.model_type,
                    "candidate_feature": feature_name,
                    "replace_feature": replace_feature,
                    "candidate_available_now": True,
                    "replacement_present_in_model": True,
                    "priority_rank": source_row["priority_rank"],
                    "category": source_row["category"],
                    "likely_source": source_row["likely_source"],
                    "implementation_difficulty": source_row["implementation_difficulty"],
                    "likely_signal_quality": source_row["likely_signal_quality"],
                    "research_source": source_row.get("research_source", ""),
                    "status": "fallback_family_test",
                    "notes": (
                        f"Fallback family test for {feature_name}: original competitor "
                        f"'{source_row['replace_or_compete_with']}' is not present in the "
                        "surviving lean model specs."
                    ),
                }
            )
    if fallback_rows:
        phase0_queue = pd.concat([phase0_queue, pd.DataFrame(fallback_rows)], ignore_index=True)
    phase0_queue = phase0_queue[phase0_queue["candidate_available_now"] == True].copy()  # noqa: E712
    phase0_detail, phase0_summary = _run_swap_phase(phase0_queue, phase0_specs, benchmark_data, horizon)
    phase0_detail.to_csv(Path(output_dir) / f"v19_0_core_detail_{stamp}.csv", index=False)
    phase0_summary.to_csv(Path(output_dir) / f"v19_0_core_summary_{stamp}.csv", index=False)

    phase1_winners = choose_phase0_winners(phase0_summary)
    phase1_winners.to_csv(Path(output_dir) / f"v19_1_phase0_winners_{stamp}.csv", index=False)
    phase1_specs = deployed_model_specs()
    phase1_queue = build_confirmation_queue(phase1_winners, phase1_specs)
    phase1_detail, phase1_summary = _run_swap_phase(phase1_queue, phase1_specs, benchmark_data, horizon)
    phase1_detail.to_csv(Path(output_dir) / f"v19_1_confirmation_detail_{stamp}.csv", index=False)
    phase1_summary.to_csv(Path(output_dir) / f"v19_1_confirmation_summary_{stamp}.csv", index=False)

    phase2_best = choose_best_confirmed_swaps(phase1_summary)
    phase2_best.to_csv(Path(output_dir) / f"v19_2_best_swaps_{stamp}.csv", index=False)
    final_specs = _build_final_bakeoff_specs(phase2_best)
    bakeoff_rows: list[pd.DataFrame] = []
    for spec_name, spec in final_specs.items():
        bakeoff_rows.append(
            _evaluate_model_spec(
                spec_name=spec_name,
                model_type=spec.model_type,
                feature_columns=spec.features,
                benchmark_data=benchmark_data,
                horizon=horizon,
            )
        )
    bakeoff_rows.append(_evaluate_baseline_historical_mean(benchmark_data, horizon))
    phase2_detail = pd.concat(bakeoff_rows, ignore_index=True)
    phase2_summary = _summarize_model_bakeoff(phase2_detail)
    phase2_detail.to_csv(Path(output_dir) / f"v19_2_bakeoff_detail_{stamp}.csv", index=False)
    phase2_summary.to_csv(Path(output_dir) / f"v19_2_bakeoff_summary_{stamp}.csv", index=False)

    traceability_df = build_v19_traceability_matrix(
        inventory_df,
        feature_columns=feature_columns,
        phase0_summary=phase0_summary,
        blocked_reasons=BLOCKED_FEATURE_REASONS,
    )
    traceability_df.to_csv(
        Path(output_dir) / f"v19_feature_traceability_{stamp}.csv",
        index=False,
    )

    tested_count = int((traceability_df["evaluation_status"] == "tested").sum())
    blocked_count = int((traceability_df["evaluation_status"] == "blocked").sum())
    available_count = int(traceability_df["available_in_matrix"].sum())
    summary_lines = [
        "# V19 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- Backfilled the remaining public macro / valuation series needed to finish the v15 inventory.",
        "- Relaxed the research-only EDGAR breadth gate to 24 non-null rows so newer live-parser fields could be evaluated.",
        "- Re-ran the fixed-budget feature replacement cycle on the full now-available inventory.",
        "- Produced a final tested/blocked traceability matrix for all original 46 features.",
        "",
        "## Coverage",
        "",
        f"- available in feature matrix: `{available_count}`",
        f"- tested through v19 swap phase: `{tested_count}`",
        f"- blocked after source audit: `{blocked_count}`",
        "",
        "## Blocked Features",
        "",
    ]
    for feature_name, reason in BLOCKED_FEATURE_REASONS.items():
        summary_lines.append(f"- `{feature_name}`: {reason}")

    if not phase2_summary.empty:
        leader = phase2_summary.iloc[0]
        summary_lines += [
            "",
            "## Current Leader",
            "",
            f"- candidate: `{leader['candidate_name']}`",
            f"- model type: `{leader['model_type']}`",
            f"- mean IC: `{leader['mean_ic']:.4f}`",
            f"- mean hit rate: `{leader['mean_hit_rate']:.4f}`",
            f"- mean OOS R^2: `{leader['mean_oos_r2']:.4f}`",
            f"- mean sign-policy return: `{leader['mean_policy_return_sign']:.4f}`",
        ]

    summary_lines += [
        "",
        "## Artifacts",
        "",
        f"- `{output_dir}/v19_public_macro_backfill_{stamp}.csv`",
        f"- `{output_dir}/v19_0_core_summary_{stamp}.csv`",
        f"- `{output_dir}/v19_1_confirmation_summary_{stamp}.csv`",
        f"- `{output_dir}/v19_2_bakeoff_summary_{stamp}.csv`",
        f"- `{output_dir}/v19_feature_traceability_{stamp}.csv`",
    ]

    _write_text(Path("docs") / "results" / "V19_RESULTS_SUMMARY.md", summary_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the remaining v15 feature inventory as v19.")
    parser.add_argument("--inventory-csv", default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()
    run_v19_feature_completion(
        inventory_csv=args.inventory_csv,
        output_dir=args.output_dir,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
