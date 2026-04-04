"""v9.2 exhaustive per-feature experiments across model types."""

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
from src.processing.feature_engineering import build_feature_matrix_from_db, get_feature_columns, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import classify_research_gate, evaluate_wfo_model


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")
DEFAULT_BENCHMARKS = [
    "VTI",
    "VOO",
    "VFH",
    "BND",
    "VHT",
    "GLD",
    "VNQ",
    "VXUS",
]


def _current_baseline_features(
    df: pd.DataFrame,
    model_type: str,
) -> list[str]:
    """Return the current production feature set for one model, filtered to df."""
    configured = config.MODEL_FEATURE_OVERRIDES.get(model_type)
    all_features = get_feature_columns(df)
    if not configured:
        return all_features
    selected = [feature for feature in configured if feature in all_features]
    return selected or all_features


def _experiment_specs(
    feature: str,
    model_type: str,
    baseline_features: list[str],
) -> list[tuple[str, list[str], bool]]:
    """Return the feature-set experiments to run for one feature/model pair."""
    specs: list[tuple[str, list[str], bool]] = [("single_feature", [feature], feature in baseline_features)]
    if feature in baseline_features:
        remaining = [col for col in baseline_features if col != feature]
        if remaining:
            specs.append(("drop_from_baseline", remaining, True))
    else:
        specs.append(("add_to_baseline", baseline_features + [feature], False))
    return specs


def run_feature_experiments(
    conn: Any,
    benchmarks: list[str],
    horizons: list[int],
    model_types: list[str],
    output_dir: str,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exhaustive single-feature, drop-one, and add-one experiments."""
    if not benchmarks:
        raise ValueError("benchmarks list must not be empty.")

    df = build_feature_matrix_from_db(conn)
    all_features = [feature for feature in get_feature_columns(df) if features is None or feature in features]
    records: list[dict[str, Any]] = []

    for horizon in horizons:
        for benchmark in benchmarks:
            rel_series = load_relative_return_matrix(conn, benchmark, horizon)
            if rel_series.empty:
                continue
            try:
                X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
            except ValueError:
                continue

            for model_type in model_types:
                baseline_features = [feature for feature in _current_baseline_features(X_aligned, model_type) if feature in X_aligned.columns]
                _, baseline_metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=model_type,
                    benchmark=benchmark,
                    target_horizon_months=horizon,
                    feature_columns=baseline_features,
                )
                records.append(
                    {
                        "experiment_mode": "baseline",
                        "feature": "__baseline__",
                        "feature_in_baseline": True,
                        "benchmark": benchmark,
                        "horizon_months": horizon,
                        "model_type": model_type,
                        "n_features": len(baseline_features),
                        "feature_columns": ",".join(baseline_features),
                        "gate_status": classify_research_gate(
                            float(baseline_metrics["oos_r2"]),
                            float(baseline_metrics["ic"]),
                            float(baseline_metrics["hit_rate"]),
                        ),
                        **baseline_metrics,
                    }
                )

                for feature in all_features:
                    for mode, selected_columns, in_baseline in _experiment_specs(
                        feature=feature,
                        model_type=model_type,
                        baseline_features=baseline_features,
                    ):
                        _, metrics = evaluate_wfo_model(
                            X_aligned,
                            y_aligned,
                            model_type=model_type,
                            benchmark=benchmark,
                            target_horizon_months=horizon,
                            feature_columns=selected_columns,
                        )
                        records.append(
                            {
                                "experiment_mode": mode,
                                "feature": feature,
                                "feature_in_baseline": in_baseline,
                                "benchmark": benchmark,
                                "horizon_months": horizon,
                                "model_type": model_type,
                                "n_features": len(selected_columns),
                                "feature_columns": ",".join(selected_columns),
                                "gate_status": classify_research_gate(
                                    float(metrics["oos_r2"]),
                                    float(metrics["ic"]),
                                    float(metrics["hit_rate"]),
                                ),
                                **metrics,
                            }
                        )

    detail_df = pd.DataFrame(records)
    summary_df = summarize_feature_experiments(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"feature_experiments_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"feature_experiments_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote feature experiment detail to {detail_path}")
    print(f"Wrote feature experiment summary to {summary_path}")
    return detail_df, summary_df


def summarize_feature_experiments(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-benchmark experiment rows and compute deltas vs baseline."""
    if detail_df.empty:
        return pd.DataFrame()

    baseline_df = detail_df[detail_df["experiment_mode"] == "baseline"][
        ["benchmark", "horizon_months", "model_type", "ic", "hit_rate", "oos_r2", "mae"]
    ].rename(
        columns={
            "ic": "baseline_ic",
            "hit_rate": "baseline_hit_rate",
            "oos_r2": "baseline_oos_r2",
            "mae": "baseline_mae",
        }
    )
    merged = detail_df.merge(
        baseline_df,
        on=["benchmark", "horizon_months", "model_type"],
        how="left",
    )
    merged["delta_ic"] = merged["ic"] - merged["baseline_ic"]
    merged["delta_hit_rate"] = merged["hit_rate"] - merged["baseline_hit_rate"]
    merged["delta_oos_r2"] = merged["oos_r2"] - merged["baseline_oos_r2"]
    merged["delta_mae"] = merged["mae"] - merged["baseline_mae"]

    rows: list[dict[str, Any]] = []
    grouped = merged.groupby(
        ["experiment_mode", "model_type", "feature", "feature_in_baseline", "horizon_months"],
        dropna=False,
    )
    for (mode, model_type, feature, in_baseline, horizon), group in grouped:
        rows.append(
            {
                "experiment_mode": mode,
                "model_type": model_type,
                "feature": feature,
                "feature_in_baseline": bool(in_baseline),
                "horizon_months": horizon,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_n_features": float(group["n_features"].mean()),
                "mean_ic": float(group["ic"].mean()),
                "median_ic": float(group["ic"].median()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_delta_ic": float(group["delta_ic"].mean()),
                "median_delta_ic": float(group["delta_ic"].median()),
                "mean_delta_hit_rate": float(group["delta_hit_rate"].mean()),
                "mean_delta_oos_r2": float(group["delta_oos_r2"].mean()),
                "mean_delta_mae": float(group["delta_mae"].mean()),
                "positive_delta_ic_rate": float((group["delta_ic"] > 0).mean()),
                "positive_delta_oos_r2_rate": float((group["delta_oos_r2"] > 0).mean()),
                "gate_pass_rate": float((group["gate_status"] == "PASS").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["experiment_mode", "model_type", "mean_delta_ic", "mean_delta_oos_r2"],
        ascending=[True, True, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 exhaustive per-feature experiments.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
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
        "--features",
        default="",
        help="Optional comma-separated subset of features to test.",
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
    features = [value.strip() for value in args.features.split(",") if value.strip()] or None

    conn = db_client.get_connection(config.DB_PATH)
    run_feature_experiments(
        conn=conn,
        benchmarks=benchmarks,
        horizons=horizons,
        model_types=model_types,
        output_dir=args.output_dir,
        features=features,
    )


if __name__ == "__main__":
    main()
