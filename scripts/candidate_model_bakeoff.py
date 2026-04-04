"""v9.8 lean candidate bakeoff on the best reduced benchmark universe."""

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
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import BASELINE_STRATEGIES, evaluate_baseline_strategy, evaluate_wfo_model, reconstruct_baseline_predictions
from src.research.policy_metrics import evaluate_policy_series


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")
DEFAULT_BENCHMARKS = ["VXUS", "VEA", "VHT", "VPU", "BNDX", "BND", "VNQ"]


def _dedupe(features: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for feature in features:
        if feature in seen:
            continue
        ordered.append(feature)
        seen.add(feature)
    return ordered


def candidate_feature_sets() -> dict[str, dict[str, Any]]:
    """Return the manually curated lean-candidate bakeoff definitions."""
    base = list(config.MODEL_FEATURE_BASE_GROUP_B)
    return {
        "elasticnet_current": {
            "model_type": "elasticnet",
            "features": list(config.MODEL_FEATURE_OVERRIDES["elasticnet"]),
            "notes": "Current production ElasticNet feature set.",
        },
        "elasticnet_lean_v1": {
            "model_type": "elasticnet",
            "features": _dedupe(
                [feature for feature in base if feature not in {"vol_63d", "real_rate_10y", "vmt_yoy"}]
                + ["investment_income_growth_yoy", "roe_net_income_ttm", "underwriting_income"]
                + ["medical_cpi_yoy", "roe"]
            ),
            "notes": "Drops weaker macro features and adds the strongest v9 add-one winners.",
        },
        "ridge_current": {
            "model_type": "ridge",
            "features": list(config.MODEL_FEATURE_OVERRIDES["ridge"]),
            "notes": "Current production Ridge feature set.",
        },
        "ridge_lean_v1": {
            "model_type": "ridge",
            "features": _dedupe(
                [feature for feature in base if feature not in {"mom_3m", "mom_6m", "vmt_yoy"}]
                + ["combined_ratio_ttm", "investment_income_growth_yoy", "roe_net_income_ttm", "npw_growth_yoy"]
            ),
            "notes": "Simpler Ridge set that keeps the best company features and removes weaker momentum/miles features.",
        },
        "gbt_current": {
            "model_type": "gbt",
            "features": list(config.MODEL_FEATURE_OVERRIDES["gbt"]),
            "notes": "Current production GBT feature set.",
        },
        "gbt_lean_plus_two": {
            "model_type": "gbt",
            "features": _dedupe(base + ["pif_growth_yoy", "investment_book_yield"]),
            "notes": "Adds the two strongest v9 GBT add-one winners.",
        },
        "gbt_lean_plus_three": {
            "model_type": "gbt",
            "features": _dedupe(base + ["pif_growth_yoy", "investment_book_yield", "underwriting_income_growth_yoy"]),
            "notes": "Adds the two strongest wins plus a third underwriting-growth feature.",
        },
        "bayesian_ridge_current": {
            "model_type": "bayesian_ridge",
            "features": list(config.MODEL_FEATURE_OVERRIDES["bayesian_ridge"]),
            "notes": "Current production BayesianRidge feature set.",
        },
        "bayesian_ridge_lean_v1": {
            "model_type": "bayesian_ridge",
            "features": _dedupe(base + ["combined_ratio_ttm", "npw_growth_yoy", "roe"]),
            "notes": "Smaller BayesianRidge set meant to test whether instability is partly feature-cost driven.",
        },
    }


def summarize_candidate_bakeoff(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate rows into a ranked summary."""
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
                "median_ic": float(group["ic"].median()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_tiered": float(group["policy_return_tiered"].mean()),
                "mean_policy_uplift_vs_sell_50_sign": float(group["policy_uplift_vs_sell_50_sign"].mean()),
                "mean_policy_uplift_vs_sell_50_tiered": float(group["policy_uplift_vs_sell_50_tiered"].mean()),
                "notes": group["notes"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    )


def run_candidate_model_bakeoff(
    conn: Any,
    benchmarks: list[str],
    horizon: int,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run curated lean-candidate comparisons on a reduced benchmark universe."""
    df = build_feature_matrix_from_db(conn)
    feature_columns = set(df.columns)
    detail_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, horizon)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue

        for candidate_name, spec in candidate_feature_sets().items():
            selected = [feature for feature in spec["features"] if feature in feature_columns]
            result, metrics = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=str(spec["model_type"]),
                benchmark=benchmark,
                target_horizon_months=horizon,
                feature_columns=selected,
            )
            pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
            realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
            sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
            tiered_policy = evaluate_policy_series(pred_series, realized, "tiered_25_50_100")
            detail_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_type": spec["model_type"],
                    "benchmark": benchmark,
                    "horizon_months": horizon,
                    "n_features": len(selected),
                    "feature_columns": ",".join(selected),
                    "notes": spec["notes"],
                    "policy_return_sign": sign_policy.mean_policy_return,
                    "policy_return_tiered": tiered_policy.mean_policy_return,
                    "policy_uplift_vs_sell_50_sign": sign_policy.uplift_vs_sell_50,
                    "policy_uplift_vs_sell_50_tiered": tiered_policy.uplift_vs_sell_50,
                    **metrics,
                }
            )

        for strategy in BASELINE_STRATEGIES:
            metrics = evaluate_baseline_strategy(
                X_aligned,
                y_aligned,
                strategy=strategy,
                target_horizon_months=horizon,
            )
            pred_series, realized = reconstruct_baseline_predictions(
                X_aligned,
                y_aligned,
                strategy=strategy,
                target_horizon_months=horizon,
            )
            sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
            tiered_policy = evaluate_policy_series(pred_series, realized, "tiered_25_50_100")
            detail_rows.append(
                {
                    "candidate_name": f"baseline_{strategy}",
                    "model_type": "baseline",
                    "benchmark": benchmark,
                    "horizon_months": horizon,
                    "n_features": 0,
                    "feature_columns": "",
                    "notes": "Simple non-ML baseline.",
                    "policy_return_sign": sign_policy.mean_policy_return,
                    "policy_return_tiered": tiered_policy.mean_policy_return,
                    "policy_uplift_vs_sell_50_sign": sign_policy.uplift_vs_sell_50,
                    "policy_uplift_vs_sell_50_tiered": tiered_policy.uplift_vs_sell_50,
                    **metrics,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_candidate_bakeoff(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"candidate_model_bakeoff_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"candidate_model_bakeoff_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote candidate bakeoff detail to {detail_path}")
    print(f"Wrote candidate bakeoff summary to {summary_path}")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 lean-candidate bakeoff.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--horizon",
        default="6",
        help="Target horizon in months.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    conn = db_client.get_connection(config.DB_PATH)
    run_candidate_model_bakeoff(
        conn=conn,
        benchmarks=benchmarks,
        horizon=int(args.horizon),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
