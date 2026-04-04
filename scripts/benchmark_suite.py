"""v9.0 benchmark suite for production models and simple baselines."""

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
    evaluate_baseline_strategy,
    evaluate_ensemble_result,
    evaluate_wfo_model,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")


def run_benchmark_suite(
    conn: Any,
    benchmarks: list[str],
    horizons: list[int],
    model_types: list[str],
    baseline_strategies: list[str],
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run production-model and baseline evaluations across the benchmark set."""
    if not benchmarks:
        raise ValueError("benchmarks list must not be empty.")

    df = build_feature_matrix_from_db(conn)
    detail_records: list[dict[str, Any]] = []

    for horizon in horizons:
        rel_matrix = pd.DataFrame(
            {
                benchmark: load_relative_return_matrix(conn, benchmark, horizon)
                for benchmark in benchmarks
            }
        )
        rel_matrix = rel_matrix.dropna(axis=1, how="all")
        ensemble_results = run_ensemble_benchmarks(
            df,
            rel_matrix,
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

            for strategy in baseline_strategies:
                metrics = evaluate_baseline_strategy(
                    X_aligned,
                    y_aligned,
                    strategy=strategy,
                    target_horizon_months=horizon,
                )
                detail_records.append(
                    {
                        "item_type": "baseline",
                        "item_name": strategy,
                        "benchmark": benchmark,
                        "horizon_months": horizon,
                        "n_features": 0,
                        "gate_status": classify_research_gate(
                            float(metrics["oos_r2"]),
                            float(metrics["ic"]),
                            float(metrics["hit_rate"]),
                        ),
                        **metrics,
                    }
                )

            for model_type in model_types:
                _, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=model_type,
                    benchmark=benchmark,
                    target_horizon_months=horizon,
                )
                detail_records.append(
                    {
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
                metrics = evaluate_ensemble_result(
                    ens_result,
                    target_horizon_months=horizon,
                )
                detail_records.append(
                    {
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

    detail_df = pd.DataFrame(detail_records)
    summary_df = summarize_benchmark_suite(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"benchmark_suite_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"benchmark_suite_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote detail results to {detail_path}")
    print(f"Wrote summary results to {summary_path}")
    return detail_df, summary_df


def summarize_benchmark_suite(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark-suite detail rows into strategy/model summaries."""
    if detail_df.empty:
        return pd.DataFrame(
            columns=[
                "item_type",
                "item_name",
                "horizon_months",
                "n_benchmarks",
                "mean_ic",
                "median_ic",
                "mean_hit_rate",
                "mean_oos_r2",
                "mean_mae",
                "pass_rate",
                "gate_status",
            ]
        )

    rows: list[dict[str, Any]] = []
    grouped = detail_df.groupby(["item_type", "item_name", "horizon_months"], dropna=False)
    for (item_type, item_name, horizon), group in grouped:
        mean_ic = float(group["ic"].mean())
        mean_hit_rate = float(group["hit_rate"].mean())
        mean_oos_r2 = float(group["oos_r2"].mean())
        rows.append(
            {
                "item_type": item_type,
                "item_name": item_name,
                "horizon_months": horizon,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_ic": mean_ic,
                "median_ic": float(group["ic"].median()),
                "mean_hit_rate": mean_hit_rate,
                "mean_oos_r2": mean_oos_r2,
                "mean_mae": float(group["mae"].mean()),
                "pass_rate": float((group["gate_status"] == "PASS").mean()),
                "gate_status": classify_research_gate(mean_oos_r2, mean_ic, mean_hit_rate),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["horizon_months", "mean_ic", "mean_oos_r2"],
        ascending=[True, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v9 benchmark suite.")
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
    run_benchmark_suite(
        conn=conn,
        benchmarks=benchmarks,
        horizons=horizons,
        model_types=model_types,
        baseline_strategies=baseline_strategies,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
