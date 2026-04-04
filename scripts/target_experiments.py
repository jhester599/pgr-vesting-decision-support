"""v9.3 target-formulation experiments."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import (
    BASELINE_STRATEGIES,
    classify_research_gate,
    evaluate_binary_baseline_strategy,
    evaluate_binary_ensemble_result,
    evaluate_binary_wfo_model,
    evaluate_wfo_model,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")
TARGET_VARIANTS: dict[str, dict[str, float | bool]] = {
    "continuous_relative_return": {"binary": False, "threshold": 0.0},
    "binary_outperform": {"binary": True, "threshold": 0.0},
    "binary_outperform_3pct": {"binary": True, "threshold": 0.03},
}


def _transform_target(y: pd.Series, variant_name: str) -> pd.Series:
    """Transform the relative-return series into the chosen target variant."""
    variant = TARGET_VARIANTS[variant_name]
    if not bool(variant["binary"]):
        return y
    threshold = float(variant["threshold"])
    return (y > threshold).astype(float).rename(f"{y.name}_{variant_name}")


def run_target_experiments(
    conn: Any,
    benchmarks: list[str],
    horizons: list[int],
    model_types: list[str],
    baseline_strategies: list[str],
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare continuous vs binary target formulations on the same WFO path."""
    if not benchmarks:
        raise ValueError("benchmarks list must not be empty.")

    df = build_feature_matrix_from_db(conn)
    detail_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        rel_matrix = pd.DataFrame(
            {
                benchmark: load_relative_return_matrix(conn, benchmark, horizon)
                for benchmark in benchmarks
            }
        ).dropna(axis=1, how="all")

        for variant_name, variant in TARGET_VARIANTS.items():
            is_binary = bool(variant["binary"])
            transformed_rel_matrix = rel_matrix.apply(
                lambda col: _transform_target(col, variant_name)
            )
            ensemble_results = run_ensemble_benchmarks(
                df,
                transformed_rel_matrix,
                target_horizon_months=horizon,
            )

            for benchmark in benchmarks:
                rel_series = load_relative_return_matrix(conn, benchmark, horizon)
                if rel_series.empty:
                    continue
                try:
                    X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
                except ValueError:
                    continue
                y_target = _transform_target(y_aligned, variant_name)

                for strategy in baseline_strategies:
                    if is_binary:
                        metrics = evaluate_binary_baseline_strategy(
                            X_aligned,
                            y_target,
                            strategy=strategy,
                            target_horizon_months=horizon,
                        )
                    else:
                        from src.research.evaluation import evaluate_baseline_strategy

                        metrics = evaluate_baseline_strategy(
                            X_aligned,
                            y_target,
                            strategy=strategy,
                            target_horizon_months=horizon,
                        )
                    detail_rows.append(
                        {
                            "target_variant": variant_name,
                            "item_type": "baseline",
                            "item_name": strategy,
                            "benchmark": benchmark,
                            "horizon_months": horizon,
                            "gate_status": classify_research_gate(
                                float(metrics["oos_r2"]),
                                float(metrics["ic"]),
                                float(metrics["hit_rate"]),
                            ),
                            **metrics,
                        }
                    )

                for model_type in model_types:
                    if is_binary:
                        _, metrics = evaluate_binary_wfo_model(
                            X_aligned,
                            y_target,
                            model_type=model_type,
                            benchmark=benchmark,
                            target_horizon_months=horizon,
                        )
                    else:
                        _, metrics = evaluate_wfo_model(
                            X_aligned,
                            y_target,
                            model_type=model_type,
                            benchmark=benchmark,
                            target_horizon_months=horizon,
                        )
                    detail_rows.append(
                        {
                            "target_variant": variant_name,
                            "item_type": "model",
                            "item_name": model_type,
                            "benchmark": benchmark,
                            "horizon_months": horizon,
                            "gate_status": classify_research_gate(
                                float(metrics["oos_r2"]),
                                float(metrics["ic"]),
                                float(metrics["hit_rate"]),
                            ),
                            **metrics,
                        }
                    )

                ens_result = ensemble_results.get(benchmark)
                if ens_result is not None:
                    if is_binary:
                        metrics = evaluate_binary_ensemble_result(
                            ens_result,
                            target_horizon_months=horizon,
                        )
                    else:
                        from src.research.evaluation import evaluate_ensemble_result

                        metrics = evaluate_ensemble_result(
                            ens_result,
                            target_horizon_months=horizon,
                        )
                    detail_rows.append(
                        {
                            "target_variant": variant_name,
                            "item_type": "ensemble",
                            "item_name": "ensemble",
                            "benchmark": benchmark,
                            "horizon_months": horizon,
                            "gate_status": classify_research_gate(
                                float(metrics["oos_r2"]),
                                float(metrics["ic"]),
                                float(metrics["hit_rate"]),
                            ),
                            **metrics,
                        }
                    )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_target_experiments(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"target_experiments_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"target_experiments_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote target experiment detail to {detail_path}")
    print(f"Wrote target experiment summary to {summary_path}")
    return detail_df, summary_df


def summarize_target_experiments(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate target-experiment rows by target variant and candidate."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["target_variant", "item_type", "item_name", "horizon_months"]
    for keys, group in detail_df.groupby(group_cols, dropna=False):
        target_variant, item_type, item_name, horizon = keys
        row = {
            "target_variant": target_variant,
            "item_type": item_type,
            "item_name": item_name,
            "horizon_months": horizon,
            "n_benchmarks": int(group["benchmark"].nunique()),
            "mean_ic": float(group["ic"].mean()),
            "median_ic": float(group["ic"].median()),
            "mean_hit_rate": float(group["hit_rate"].mean()),
            "mean_oos_r2": float(group["oos_r2"].mean()),
            "mean_mae": float(group["mae"].mean()),
            "pass_rate": float((group["gate_status"] == "PASS").mean()),
            "gate_status": classify_research_gate(
                float(group["oos_r2"].mean()),
                float(group["ic"].mean()),
                float(group["hit_rate"].mean()),
            ),
        }
        if "brier_score" in group.columns:
            row.update(
                {
                    "mean_brier_score": float(group["brier_score"].mean()),
                    "mean_accuracy": float(group["accuracy"].mean()),
                    "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                    "mean_precision": float(group["precision"].mean()),
                    "mean_recall": float(group["recall"].mean()),
                    "mean_base_rate": float(group["base_rate"].mean()),
                    "mean_predicted_positive_rate": float(group["predicted_positive_rate"].mean()),
                }
            )
        rows.append(row)

    sort_cols = ["target_variant", "mean_ic", "mean_oos_r2"]
    return pd.DataFrame(rows).sort_values(by=sort_cols, ascending=[True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 target-formulation experiments.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(config.ETF_BENCHMARK_UNIVERSE),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--horizons",
        default="6",
        help="Comma-separated target horizons in months.",
    )
    parser.add_argument(
        "--model-types",
        default=",".join(config.ENSEMBLE_MODELS),
        help="Comma-separated model types to evaluate.",
    )
    parser.add_argument(
        "--baseline-strategies",
        default=",".join(BASELINE_STRATEGIES),
        help="Comma-separated simple baseline strategies.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    horizons = [int(value.strip()) for value in args.horizons.split(",") if value.strip()]
    model_types = [value.strip() for value in args.model_types.split(",") if value.strip()]
    baseline_strategies = [value.strip() for value in args.baseline_strategies.split(",") if value.strip()]

    conn = db_client.get_connection(config.DB_PATH)
    run_target_experiments(
        conn=conn,
        benchmarks=benchmarks,
        horizons=horizons,
        model_types=model_types,
        baseline_strategies=baseline_strategies,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
