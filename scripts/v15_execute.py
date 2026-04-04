"""Execute the v15 fixed-budget feature-replacement cycle."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import (
    evaluate_baseline_strategy,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
)
from src.research.policy_metrics import evaluate_policy_series
from src.research.v15 import (
    V15_FORECAST_UNIVERSE,
    apply_one_for_one_swap,
    base_model_specs,
    build_confirmation_queue,
    build_swap_queue,
    choose_best_confirmed_swaps,
    choose_phase0_winners,
    deployed_model_specs,
    normalize_inventory,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v15")
DEFAULT_INVENTORY_CSV = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "v15_feature_candidate_inventory_from_reports_20260404.csv",
)


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_dataset_map(
    conn: Any,
    df: pd.DataFrame,
    benchmarks: list[str],
    horizon: int,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, horizon)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue
        datasets[benchmark] = (X_aligned, y_aligned)
    return datasets


def _evaluate_model_spec(
    spec_name: str,
    model_type: str,
    feature_columns: list[str],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        selected = [feature for feature in feature_columns if feature in X_aligned.columns]
        result, metrics = evaluate_wfo_model(
            X_aligned,
            y_aligned,
            model_type=model_type,
            benchmark=benchmark,
            target_horizon_months=horizon,
            feature_columns=selected,
        )
        pred_series = pd.Series(
            result.y_hat_all,
            index=pd.DatetimeIndex(result.test_dates_all),
            name="y_hat",
        )
        realized = pd.Series(
            result.y_true_all,
            index=pd.DatetimeIndex(result.test_dates_all),
            name="y_true",
        )
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        neutral_policy = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
        rows.append(
            {
                "candidate_name": spec_name,
                "model_type": model_type,
                "benchmark": benchmark,
                "n_features": len(selected),
                "feature_columns": ",".join(selected),
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_uplift_vs_sell_50_sign": sign_policy.uplift_vs_sell_50,
                "policy_return_neutral_3pct": neutral_policy.mean_policy_return,
                "policy_uplift_vs_sell_50_neutral_3pct": neutral_policy.uplift_vs_sell_50,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_baseline_historical_mean(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        metrics = evaluate_baseline_strategy(
            X_aligned,
            y_aligned,
            strategy="historical_mean",
            target_horizon_months=horizon,
        )
        pred_series, realized = reconstruct_baseline_predictions(
            X_aligned,
            y_aligned,
            strategy="historical_mean",
            target_horizon_months=horizon,
        )
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        neutral_policy = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
        rows.append(
            {
                "candidate_name": "baseline_historical_mean",
                "model_type": "baseline",
                "benchmark": benchmark,
                "n_features": 0,
                "feature_columns": "",
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_uplift_vs_sell_50_sign": sign_policy.uplift_vs_sell_50,
                "policy_return_neutral_3pct": neutral_policy.mean_policy_return,
                "policy_uplift_vs_sell_50_neutral_3pct": neutral_policy.uplift_vs_sell_50,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _summarize_phase_rows(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    group_cols = [
        "candidate_name",
        "model_type",
        "candidate_feature",
        "replace_feature",
        "priority_rank",
        "research_source",
        "candidate_available_now",
        "status",
        "notes",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in detail_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key, strict=False))
        rows.append(
            {
                **key_dict,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_ic": float(group["ic"].mean()),
                "mean_ic_delta": float(group["ic_delta"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_hit_rate_delta": float(group["hit_rate_delta"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_oos_r2_delta": float(group["oos_r2_delta"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_sign_delta": float(group["policy_return_sign_delta"].mean()),
                "mean_policy_return_neutral_3pct": float(group["policy_return_neutral_3pct"].mean()),
                "mean_policy_return_neutral_3pct_delta": float(group["policy_return_neutral_3pct_delta"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_mae_delta": float(group["mae_delta"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=[
            "mean_policy_return_sign_delta",
            "mean_oos_r2_delta",
            "mean_ic_delta",
            "priority_rank",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _run_swap_phase(
    queue_df: pd.DataFrame,
    specs: dict[str, Any],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if queue_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    baseline_cache: dict[str, pd.DataFrame] = {}
    detail_rows: list[dict[str, Any]] = []

    for spec_name, spec in specs.items():
        baseline_cache[spec_name] = _evaluate_model_spec(
            spec_name=spec_name,
            model_type=spec.model_type,
            feature_columns=spec.features,
            benchmark_data=benchmark_data,
            horizon=horizon,
        )

    for row in queue_df.itertuples(index=False):
        spec = specs[str(row.candidate_name)]
        swapped_features = apply_one_for_one_swap(
            spec,
            str(row.replace_feature),
            str(row.candidate_feature),
        )
        swapped_detail = _evaluate_model_spec(
            spec_name=str(row.candidate_name),
            model_type=str(row.model_type),
            feature_columns=swapped_features,
            benchmark_data=benchmark_data,
            horizon=horizon,
        )
        baseline_detail = baseline_cache[str(row.candidate_name)]
        merged = swapped_detail.merge(
            baseline_detail[
                [
                    "benchmark",
                    "ic",
                    "hit_rate",
                    "oos_r2",
                    "mae",
                    "policy_return_sign",
                    "policy_return_neutral_3pct",
                ]
            ],
            on="benchmark",
            suffixes=("", "_baseline"),
        )
        for merged_row in merged.itertuples(index=False):
            detail_rows.append(
                {
                    "candidate_name": row.candidate_name,
                    "model_type": row.model_type,
                    "candidate_feature": row.candidate_feature,
                    "replace_feature": row.replace_feature,
                    "priority_rank": getattr(row, "priority_rank", None),
                    "research_source": getattr(row, "research_source", ""),
                    "candidate_available_now": getattr(row, "candidate_available_now", True),
                    "status": getattr(row, "status", "queued"),
                    "notes": getattr(row, "notes", ""),
                    "benchmark": merged_row.benchmark,
                    "ic": merged_row.ic,
                    "ic_delta": merged_row.ic - merged_row.ic_baseline,
                    "hit_rate": merged_row.hit_rate,
                    "hit_rate_delta": merged_row.hit_rate - merged_row.hit_rate_baseline,
                    "oos_r2": merged_row.oos_r2,
                    "oos_r2_delta": merged_row.oos_r2 - merged_row.oos_r2_baseline,
                    "mae": merged_row.mae,
                    "mae_delta": merged_row.mae - merged_row.mae_baseline,
                    "policy_return_sign": merged_row.policy_return_sign,
                    "policy_return_sign_delta": (
                        merged_row.policy_return_sign - merged_row.policy_return_sign_baseline
                    ),
                    "policy_return_neutral_3pct": merged_row.policy_return_neutral_3pct,
                    "policy_return_neutral_3pct_delta": (
                        merged_row.policy_return_neutral_3pct
                        - merged_row.policy_return_neutral_3pct_baseline
                    ),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    return detail_df, _summarize_phase_rows(detail_df)


def _build_final_bakeoff_specs(best_swaps_df: pd.DataFrame) -> dict[str, Any]:
    baselines = deployed_model_specs()
    final_specs = {**baselines}
    for row in best_swaps_df.itertuples(index=False):
        spec_name = str(row.candidate_name)
        base_spec = baselines[spec_name]
        final_specs[f"{spec_name}__v15_best"] = type(base_spec)(
            candidate_name=f"{spec_name}__v15_best",
            model_type=base_spec.model_type,
            features=apply_one_for_one_swap(
                base_spec,
                str(row.replace_feature),
                str(row.candidate_feature),
            ),
        )
    return final_specs


def _summarize_model_bakeoff(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for candidate_name, group in detail_df.groupby("candidate_name", dropna=False):
        rows.append(
            {
                "candidate_name": candidate_name,
                "model_type": group["model_type"].iloc[0],
                "n_features": int(group["n_features"].iloc[0]),
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_neutral_3pct": float(group["policy_return_neutral_3pct"].mean()),
                "mean_mae": float(group["mae"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_v15_execution(
    *,
    inventory_csv: str = DEFAULT_INVENTORY_CSV,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    horizon: int = 6,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)
    available_features = set(df.columns)
    benchmark_data = _benchmark_dataset_map(conn, df, list(V15_FORECAST_UNIVERSE), horizon)
    stamp = datetime.today().strftime("%Y%m%d")

    inventory_df = normalize_inventory(pd.read_csv(inventory_csv))

    # v15.0: exhaustive feature screening on the v14 survivor models
    phase0_specs = base_model_specs()
    phase0_queue = build_swap_queue(inventory_df, phase0_specs, available_features)
    phase0_queue = phase0_queue[phase0_queue["candidate_available_now"] == True].copy()  # noqa: E712
    phase0_detail, phase0_summary = _run_swap_phase(phase0_queue, phase0_specs, benchmark_data, horizon)
    phase0_detail.to_csv(Path(output_dir) / f"v15_0_core_detail_{stamp}.csv", index=False)
    phase0_summary.to_csv(Path(output_dir) / f"v15_0_core_summary_{stamp}.csv", index=False)

    # v15.1: cross-model confirmation on winners only
    phase1_winners = choose_phase0_winners(phase0_summary)
    phase1_winners.to_csv(Path(output_dir) / f"v15_1_phase0_winners_{stamp}.csv", index=False)
    phase1_specs = deployed_model_specs()
    phase1_queue = build_confirmation_queue(phase1_winners, phase1_specs)
    phase1_detail, phase1_summary = _run_swap_phase(phase1_queue, phase1_specs, benchmark_data, horizon)
    phase1_detail.to_csv(Path(output_dir) / f"v15_1_confirmation_detail_{stamp}.csv", index=False)
    phase1_summary.to_csv(Path(output_dir) / f"v15_1_confirmation_summary_{stamp}.csv", index=False)

    # v15.2: final cross-model bakeoff
    phase2_best = choose_best_confirmed_swaps(phase1_summary)
    phase2_best.to_csv(Path(output_dir) / f"v15_2_best_swaps_{stamp}.csv", index=False)
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
    phase2_detail.to_csv(Path(output_dir) / f"v15_2_bakeoff_detail_{stamp}.csv", index=False)
    phase2_summary.to_csv(Path(output_dir) / f"v15_2_bakeoff_summary_{stamp}.csv", index=False)

    top_line = phase2_summary.iloc[0] if not phase2_summary.empty else None
    lines = [
        "# V15 Execution Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- `v15.0`: exhaustive feature screening on Ridge and GBT",
        "- `v15.1`: winning-feature confirmation across all deployed model types",
        "- `v15.2`: final cross-model bakeoff",
        "",
        "## Output Artifacts",
        "",
        f"- `results/v15/v15_0_core_summary_{stamp}.csv`",
        f"- `results/v15/v15_1_confirmation_summary_{stamp}.csv`",
        f"- `results/v15/v15_2_bakeoff_summary_{stamp}.csv`",
    ]
    if top_line is not None:
        lines += [
            "",
            "## Current Leader",
            "",
            f"- candidate: `{top_line['candidate_name']}`",
            f"- model type: `{top_line['model_type']}`",
            f"- mean IC: `{top_line['mean_ic']:.4f}`",
            f"- mean hit rate: `{top_line['mean_hit_rate']:.4f}`",
            f"- mean OOS R²: `{top_line['mean_oos_r2']:.4f}`",
            f"- mean sign-policy return: `{top_line['mean_policy_return_sign']:.4f}`",
        ]
    _write_text(Path("docs") / "results" / "V15_EXECUTION_SUMMARY.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v15 feature-replacement execution cycle.")
    parser.add_argument(
        "--inventory-csv",
        default=DEFAULT_INVENTORY_CSV,
        help=f"Candidate inventory CSV. Default: {DEFAULT_INVENTORY_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--horizon",
        default="6",
        help="Target horizon in months.",
    )
    args = parser.parse_args()
    run_v15_execution(
        inventory_csv=args.inventory_csv,
        output_dir=args.output_dir,
        horizon=int(args.horizon),
    )


if __name__ == "__main__":
    main()
