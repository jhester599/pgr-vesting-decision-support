"""v9.7 pooled benchmark-family experiments."""

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
from src.research.benchmark_sets import BENCHMARK_POOLS
from src.research.evaluation import (
    BASELINE_STRATEGIES,
    classify_research_gate,
    evaluate_baseline_strategy,
    evaluate_ensemble_result,
    evaluate_wfo_model,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")


def build_pooled_relative_targets(
    conn: Any,
    pool_definitions: dict[str, list[str]],
    target_horizon_months: int,
) -> pd.DataFrame:
    """Build averaged relative-return targets for benchmark families/pools."""
    pool_series: dict[str, pd.Series] = {}
    for pool_name, tickers in pool_definitions.items():
        matrix = pd.DataFrame(
            {
                ticker: load_relative_return_matrix(conn, ticker, target_horizon_months)
                for ticker in tickers
            }
        ).dropna(axis=1, how="all")
        if matrix.empty:
            continue
        pool_series[pool_name] = matrix.mean(axis=1, skipna=True).rename(pool_name)
    if not pool_series:
        return pd.DataFrame()
    return pd.DataFrame(pool_series)


def summarize_pooled_experiments(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pooled-target experiments into candidate summaries."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["pooled_target", "item_type", "item_name", "horizon_months"]
    for keys, group in detail_df.groupby(group_cols, dropna=False):
        pooled_target, item_type, item_name, horizon = keys
        mean_ic = float(group["ic"].mean())
        mean_hit_rate = float(group["hit_rate"].mean())
        mean_oos_r2 = float(group["oos_r2"].mean())
        rows.append(
            {
                "pooled_target": pooled_target,
                "item_type": item_type,
                "item_name": item_name,
                "horizon_months": horizon,
                "n_rows": int(len(group)),
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
        by=["mean_ic", "mean_oos_r2"],
        ascending=[False, False],
    )


def run_pooled_benchmark_experiments(
    conn: Any,
    horizons: list[int],
    model_types: list[str],
    baseline_strategies: list[str],
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models against pooled benchmark-family targets."""
    df = build_feature_matrix_from_db(conn)
    detail_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        pooled_targets = build_pooled_relative_targets(
            conn=conn,
            pool_definitions=BENCHMARK_POOLS,
            target_horizon_months=horizon,
        )
        if pooled_targets.empty:
            continue
        ensemble_results = run_ensemble_benchmarks(
            df,
            pooled_targets,
            target_horizon_months=horizon,
        )

        for pooled_target in pooled_targets.columns:
            rel_series = pooled_targets[pooled_target].dropna()
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
                detail_rows.append(
                    {
                        "pooled_target": pooled_target,
                        "item_type": "baseline",
                        "item_name": strategy,
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
                _, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=model_type,
                    benchmark=pooled_target,
                    target_horizon_months=horizon,
                )
                detail_rows.append(
                    {
                        "pooled_target": pooled_target,
                        "item_type": "model",
                        "item_name": model_type,
                        "horizon_months": horizon,
                        "gate_status": classify_research_gate(
                            float(metrics["oos_r2"]),
                            float(metrics["ic"]),
                            float(metrics["hit_rate"]),
                        ),
                        **metrics,
                    }
                )

            ens_result = ensemble_results.get(pooled_target)
            if ens_result is not None:
                metrics = evaluate_ensemble_result(
                    ens_result,
                    target_horizon_months=horizon,
                )
                detail_rows.append(
                    {
                        "pooled_target": pooled_target,
                        "item_type": "ensemble",
                        "item_name": "ensemble",
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
    summary_df = summarize_pooled_experiments(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"pooled_benchmark_experiments_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"pooled_benchmark_experiments_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote pooled benchmark detail to {detail_path}")
    print(f"Wrote pooled benchmark summary to {summary_path}")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 pooled benchmark-family experiments.")
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

    horizons = [int(value.strip()) for value in args.horizons.split(",") if value.strip()]
    model_types = [value.strip() for value in args.model_types.split(",") if value.strip()]
    baseline_strategies = [value.strip() for value in args.baseline_strategies.split(",") if value.strip()]

    conn = db_client.get_connection(config.DB_PATH)
    run_pooled_benchmark_experiments(
        conn=conn,
        horizons=horizons,
        model_types=model_types,
        baseline_strategies=baseline_strategies,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
