"""v9.6 policy-level evaluation for normalized vesting decisions."""

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
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    reconstruct_ensemble_oos_predictions,
)
from src.research.policy_metrics import (
    FIXED_POLICIES,
    SIGNAL_POLICIES,
    evaluate_policy_series,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")


def _evaluate_fixed_policy(
    realized: pd.Series,
    benchmark: str,
    horizon: int,
    policy_name: str,
    universe_name: str,
) -> list[dict[str, Any]]:
    summary = evaluate_policy_series(
        predicted=pd.Series(0.0, index=realized.index, name="y_hat"),
        realized_relative_return=realized,
        policy_name=policy_name,
    )
    return [
        {
            "universe_name": universe_name,
            "benchmark": benchmark,
            "horizon_months": horizon,
            "candidate_type": "heuristic",
            "candidate_name": policy_name,
            "policy_name": policy_name,
            "n_obs": summary.n_obs,
            "avg_hold_fraction": summary.avg_hold_fraction,
            "mean_policy_return": summary.mean_policy_return,
            "median_policy_return": summary.median_policy_return,
            "cumulative_policy_return": summary.cumulative_policy_return,
            "positive_utility_rate": summary.positive_utility_rate,
            "regret_vs_oracle": summary.regret_vs_oracle,
            "uplift_vs_sell_all": summary.uplift_vs_sell_all,
            "uplift_vs_sell_50": summary.uplift_vs_sell_50,
            "uplift_vs_hold_all": summary.uplift_vs_hold_all,
            "capture_ratio": summary.capture_ratio,
        }
    ]


def _evaluate_signal_policies(
    predicted: pd.Series,
    realized: pd.Series,
    benchmark: str,
    horizon: int,
    candidate_type: str,
    candidate_name: str,
    universe_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name in SIGNAL_POLICIES:
        summary = evaluate_policy_series(
            predicted=predicted,
            realized_relative_return=realized,
            policy_name=policy_name,
        )
        rows.append(
            {
                "universe_name": universe_name,
                "benchmark": benchmark,
                "horizon_months": horizon,
                "candidate_type": candidate_type,
                "candidate_name": candidate_name,
                "policy_name": policy_name,
                "n_obs": summary.n_obs,
                "avg_hold_fraction": summary.avg_hold_fraction,
                "mean_policy_return": summary.mean_policy_return,
                "median_policy_return": summary.median_policy_return,
                "cumulative_policy_return": summary.cumulative_policy_return,
                "positive_utility_rate": summary.positive_utility_rate,
                "regret_vs_oracle": summary.regret_vs_oracle,
                "uplift_vs_sell_all": summary.uplift_vs_sell_all,
                "uplift_vs_sell_50": summary.uplift_vs_sell_50,
                "uplift_vs_hold_all": summary.uplift_vs_hold_all,
                "capture_ratio": summary.capture_ratio,
            }
        )
    return rows


def summarize_policy_evaluation(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-benchmark policy utility into candidate summaries."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = [
        "universe_name",
        "candidate_type",
        "candidate_name",
        "policy_name",
        "horizon_months",
    ]
    for keys, group in detail_df.groupby(group_cols, dropna=False):
        universe_name, candidate_type, candidate_name, policy_name, horizon = keys
        rows.append(
            {
                "universe_name": universe_name,
                "candidate_type": candidate_type,
                "candidate_name": candidate_name,
                "policy_name": policy_name,
                "horizon_months": horizon,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_policy_return": float(group["mean_policy_return"].mean()),
                "median_policy_return": float(group["mean_policy_return"].median()),
                "mean_positive_utility_rate": float(group["positive_utility_rate"].mean()),
                "mean_regret_vs_oracle": float(group["regret_vs_oracle"].mean()),
                "mean_uplift_vs_sell_all": float(group["uplift_vs_sell_all"].mean()),
                "mean_uplift_vs_sell_50": float(group["uplift_vs_sell_50"].mean()),
                "mean_uplift_vs_hold_all": float(group["uplift_vs_hold_all"].mean()),
                "mean_capture_ratio": float(group["capture_ratio"].mean()),
                "mean_hold_fraction": float(group["avg_hold_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["universe_name", "mean_policy_return", "mean_uplift_vs_sell_50"],
        ascending=[True, False, False],
    )


def run_policy_evaluation(
    conn: Any,
    benchmarks: list[str],
    horizons: list[int],
    model_types: list[str],
    baseline_strategies: list[str],
    output_dir: str,
    universe_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare candidate predictors against simple decision policies."""
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

            for policy_name in FIXED_POLICIES:
                detail_rows.extend(
                    _evaluate_fixed_policy(
                        realized=y_aligned,
                        benchmark=benchmark,
                        horizon=horizon,
                        policy_name=policy_name,
                        universe_name=universe_name,
                    )
                )

            for strategy in baseline_strategies:
                pred_series, realized = reconstruct_baseline_predictions(
                    X_aligned,
                    y_aligned,
                    strategy=strategy,
                    target_horizon_months=horizon,
                )
                detail_rows.extend(
                    _evaluate_signal_policies(
                        predicted=pred_series,
                        realized=realized,
                        benchmark=benchmark,
                        horizon=horizon,
                        candidate_type="baseline",
                        candidate_name=strategy,
                        universe_name=universe_name,
                    )
                )

            for model_type in model_types:
                result, _ = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=model_type,
                    benchmark=benchmark,
                    target_horizon_months=horizon,
                )
                detail_rows.extend(
                    _evaluate_signal_policies(
                        predicted=pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat"),
                        realized=pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true"),
                        benchmark=benchmark,
                        horizon=horizon,
                        candidate_type="model",
                        candidate_name=model_type,
                        universe_name=universe_name,
                    )
                )

            ens_result = ensemble_results.get(benchmark)
            if ens_result is not None:
                pred_series, realized = reconstruct_ensemble_oos_predictions(ens_result)
                detail_rows.extend(
                    _evaluate_signal_policies(
                        predicted=pred_series,
                        realized=realized,
                        benchmark=benchmark,
                        horizon=horizon,
                        candidate_type="ensemble",
                        candidate_name="ensemble",
                        universe_name=universe_name,
                    )
                )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_policy_evaluation(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"policy_evaluation_detail_{universe_name}_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"policy_evaluation_summary_{universe_name}_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote policy evaluation detail to {detail_path}")
    print(f"Wrote policy evaluation summary to {summary_path}")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 policy-level evaluation.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(config.ETF_BENCHMARK_UNIVERSE),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--universe-name",
        default="full21",
        help="Label for the evaluated benchmark universe.",
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
    run_policy_evaluation(
        conn=conn,
        benchmarks=benchmarks,
        horizons=horizons,
        model_types=model_types,
        baseline_strategies=baseline_strategies,
        output_dir=args.output_dir,
        universe_name=args.universe_name,
    )


if __name__ == "__main__":
    main()
