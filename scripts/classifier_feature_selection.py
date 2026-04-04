"""v9 follow-up: one-feature-at-a-time Ridge classifier selection."""

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
from scripts.candidate_model_bakeoff import DEFAULT_BENCHMARKS, candidate_feature_sets
from scripts.confirmatory_classifier_experiments import evaluate_confirmatory_classifier
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_feature_columns, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import evaluate_wfo_model
from src.research.policy_metrics import evaluate_hold_fraction_series, evaluate_policy_series


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")
DEFAULT_REGRESSION_CANDIDATE = "ridge_lean_v1"
DEFAULT_CLASSIFIER_MODEL_TYPE = "ridge"
DEFAULT_THRESHOLDS = (0.50, 0.55, 0.60)


def _feature_set_key(features: list[str]) -> str:
    return "|".join(features)


def _evaluate_classifier_feature_set(
    regression_candidate_name: str,
    classifier_model_type: str,
    classifier_features: list[str],
    thresholds: list[float],
    benchmark_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate one classifier feature set across the reduced benchmark universe."""
    detail_rows: list[dict[str, Any]] = []
    if not benchmark_contexts:
        return detail_rows
    feature_columns = set(benchmark_contexts[0]["X"].columns)
    selected = [feature for feature in classifier_features if feature in feature_columns]
    if not selected:
        return detail_rows

    for context in benchmark_contexts:
        benchmark = context["benchmark"]
        X_aligned = context["X"]
        y_reg = context["y_reg"]
        y_binary = context["y_binary"]
        reg_pred = context["reg_pred"]
        reg_realized = context["reg_realized"]
        reg_metrics = context["reg_metrics"]
        prob_series, _, cls_metrics = evaluate_confirmatory_classifier(
            X_aligned,
            y_binary,
            model_type=classifier_model_type,
            feature_columns=selected,
            target_horizon_months=context["horizon"],
        )

        aligned = pd.concat([reg_pred, reg_realized, prob_series], axis=1).dropna()
        if aligned.empty:
            continue

        regression_sign_policy = evaluate_policy_series(
            aligned["y_hat_reg"],
            aligned["y_true_reg"],
            "sign_hold_vs_sell",
        )
        detail_rows.append(
            {
                "benchmark": benchmark,
                "policy_name": "regression_sign",
                "threshold": float("nan"),
                "classifier_model_type": classifier_model_type,
                "regression_candidate": regression_candidate_name,
                "n_features": len(selected),
                "feature_set_key": _feature_set_key(selected),
                "feature_columns": ",".join(selected),
                "regression_ic": reg_metrics["ic"],
                "regression_oos_r2": reg_metrics["oos_r2"],
                "brier_score": cls_metrics["brier_score"],
                "accuracy": cls_metrics["accuracy"],
                "balanced_accuracy": cls_metrics["balanced_accuracy"],
                "precision": cls_metrics["precision"],
                "recall": cls_metrics["recall"],
                "base_rate": cls_metrics["base_rate"],
                "predicted_positive_rate": cls_metrics["predicted_positive_rate"],
                "avg_hold_fraction": regression_sign_policy.avg_hold_fraction,
                "mean_policy_return": regression_sign_policy.mean_policy_return,
                "uplift_vs_sell_50": regression_sign_policy.uplift_vs_sell_50,
                "uplift_vs_regression_sign": 0.0,
            }
        )

        for threshold in thresholds:
            classifier_signal = aligned["p_outperform"] - threshold
            classifier_policy = evaluate_policy_series(
                classifier_signal,
                aligned["y_true_reg"],
                "sign_hold_vs_sell",
            )
            hybrid_hold = (
                (aligned["y_hat_reg"] > 0.0) & (aligned["p_outperform"] >= threshold)
            ).astype(float)
            hybrid_policy = evaluate_hold_fraction_series(
                hybrid_hold.rename("hold_fraction"),
                aligned["y_true_reg"],
            )

            for policy_name, policy_summary in (
                ("classifier_only", classifier_policy),
                ("hybrid_confirm", hybrid_policy),
            ):
                detail_rows.append(
                    {
                        "benchmark": benchmark,
                        "policy_name": policy_name,
                        "threshold": threshold,
                        "classifier_model_type": classifier_model_type,
                        "regression_candidate": regression_candidate_name,
                        "n_features": len(selected),
                        "feature_set_key": _feature_set_key(selected),
                        "feature_columns": ",".join(selected),
                        "regression_ic": reg_metrics["ic"],
                        "regression_oos_r2": reg_metrics["oos_r2"],
                        "brier_score": cls_metrics["brier_score"],
                        "accuracy": cls_metrics["accuracy"],
                        "balanced_accuracy": cls_metrics["balanced_accuracy"],
                        "precision": cls_metrics["precision"],
                        "recall": cls_metrics["recall"],
                        "base_rate": cls_metrics["base_rate"],
                        "predicted_positive_rate": cls_metrics["predicted_positive_rate"],
                        "avg_hold_fraction": policy_summary.avg_hold_fraction,
                        "mean_policy_return": policy_summary.mean_policy_return,
                        "uplift_vs_sell_50": policy_summary.uplift_vs_sell_50,
                        "uplift_vs_regression_sign": (
                            policy_summary.mean_policy_return
                            - regression_sign_policy.mean_policy_return
                        ),
                    }
                )

    return detail_rows


def summarize_classifier_feature_selection(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows by feature set and policy type."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["experiment_mode", "step", "candidate_feature", "policy_name", "threshold", "feature_set_key"]
    for keys, group in detail_df.groupby(group_cols, dropna=False):
        experiment_mode, step, candidate_feature, policy_name, threshold, feature_set_key = keys
        rows.append(
            {
                "experiment_mode": experiment_mode,
                "step": step,
                "candidate_feature": candidate_feature,
                "policy_name": policy_name,
                "threshold": threshold,
                "feature_set_key": feature_set_key,
                "n_features": int(group["n_features"].iloc[0]),
                "feature_columns": group["feature_columns"].iloc[0],
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_regression_ic": float(group["regression_ic"].mean()),
                "mean_regression_oos_r2": float(group["regression_oos_r2"].mean()),
                "mean_brier_score": float(group["brier_score"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_precision": float(group["precision"].mean()),
                "mean_recall": float(group["recall"].mean()),
                "mean_policy_return": float(group["mean_policy_return"].mean()),
                "mean_uplift_vs_sell_50": float(group["uplift_vs_sell_50"].mean()),
                "mean_uplift_vs_regression_sign": float(group["uplift_vs_regression_sign"].mean()),
                "mean_avg_hold_fraction": float(group["avg_hold_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _best_hybrid_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    hybrid = summary_df.loc[summary_df["policy_name"] == "hybrid_confirm"].copy()
    if hybrid.empty:
        return hybrid
    hybrid = hybrid.sort_values(
        by=[
            "experiment_mode",
            "step",
            "candidate_feature",
            "mean_uplift_vs_regression_sign",
            "mean_balanced_accuracy",
            "mean_brier_score",
            "n_features",
        ],
        ascending=[True, True, True, False, False, True, True],
    )
    return hybrid.groupby(
        ["experiment_mode", "step", "candidate_feature", "feature_set_key"], as_index=False
    ).head(1)


def _choose_forward_winner(candidate_rows: pd.DataFrame) -> pd.Series:
    """Choose the next forward-selection winner by a tradeoff-aware rule."""
    max_uplift = float(candidate_rows["mean_uplift_vs_regression_sign"].max())
    uplift_pool = candidate_rows.loc[
        candidate_rows["mean_uplift_vs_regression_sign"] >= max_uplift - 0.003
    ].copy()
    max_bal_acc = float(uplift_pool["mean_balanced_accuracy"].max())
    bal_pool = uplift_pool.loc[
        uplift_pool["mean_balanced_accuracy"] >= max_bal_acc - 0.01
    ].copy()
    return bal_pool.sort_values(
        by=["n_features", "mean_brier_score", "candidate_feature"],
        ascending=[True, True, True],
    ).iloc[0]


def _choose_final_recommendation(step_rows: pd.DataFrame) -> pd.Series:
    """Choose the smallest near-best step on uplift and balanced accuracy."""
    max_uplift = float(step_rows["mean_uplift_vs_regression_sign"].max())
    max_bal_acc = float(step_rows["mean_balanced_accuracy"].max())
    pool = step_rows.loc[
        (step_rows["mean_uplift_vs_regression_sign"] >= max_uplift - 0.003)
        & (step_rows["mean_balanced_accuracy"] >= max_bal_acc - 0.01)
    ].copy()
    return pool.sort_values(
        by=["n_features", "mean_brier_score", "step"],
        ascending=[True, True, True],
    ).iloc[0]


def run_classifier_feature_selection(
    conn: Any,
    benchmarks: list[str],
    regression_candidate_name: str,
    classifier_model_type: str,
    thresholds: list[float],
    horizon: int,
    output_dir: str,
    max_features: int = 8,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run single-feature and greedy forward selection for the Ridge classifier."""
    df = build_feature_matrix_from_db(conn)
    feature_pool = [
        feature
        for feature in get_feature_columns(df)
        if features is None or feature in features
    ]
    feature_columns = set(df.columns)
    regression_candidate = candidate_feature_sets()[regression_candidate_name]
    regression_features = [
        feature for feature in regression_candidate["features"] if feature in feature_columns
    ]

    benchmark_contexts: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, horizon)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_reg = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue
        y_binary = (y_reg > 0.0).astype(int).rename(f"{benchmark}_outperform")
        reg_result, reg_metrics = evaluate_wfo_model(
            X_aligned,
            y_reg,
            model_type=str(regression_candidate["model_type"]),
            benchmark=benchmark,
            target_horizon_months=horizon,
            feature_columns=regression_features,
        )
        benchmark_contexts.append(
            {
                "benchmark": benchmark,
                "X": X_aligned,
                "y_reg": y_reg,
                "y_binary": y_binary,
                "reg_pred": pd.Series(
                    reg_result.y_hat_all,
                    index=pd.DatetimeIndex(reg_result.test_dates_all),
                    name="y_hat_reg",
                ),
                "reg_realized": pd.Series(
                    reg_result.y_true_all,
                    index=pd.DatetimeIndex(reg_result.test_dates_all),
                    name="y_true_reg",
                ),
                "reg_metrics": reg_metrics,
                "horizon": horizon,
            }
        )

    detail_rows: list[dict[str, Any]] = []
    forward_trace_rows: list[dict[str, Any]] = []

    # Single-feature sweep.
    for feature in feature_pool:
        rows = _evaluate_classifier_feature_set(
            regression_candidate_name=regression_candidate_name,
            classifier_model_type=classifier_model_type,
            classifier_features=[feature],
            thresholds=thresholds,
            benchmark_contexts=benchmark_contexts,
        )
        for row in rows:
            row.update(
                {
                    "experiment_mode": "single_feature",
                    "step": 1,
                    "candidate_feature": feature,
                }
            )
        detail_rows.extend(rows)

    summary_df = summarize_classifier_feature_selection(pd.DataFrame(detail_rows))
    best_single = _best_hybrid_rows(summary_df)
    first_winner = _choose_forward_winner(best_single)
    selected_features = first_winner["feature_columns"].split(",")
    forward_trace_rows.append(first_winner.to_dict())

    # Greedy forward selection.
    for step in range(2, max_features + 1):
        remaining = [feature for feature in feature_pool if feature not in selected_features]
        if not remaining:
            break

        step_rows: list[dict[str, Any]] = []
        for feature in remaining:
            rows = _evaluate_classifier_feature_set(
                regression_candidate_name=regression_candidate_name,
                classifier_model_type=classifier_model_type,
                classifier_features=selected_features + [feature],
                thresholds=thresholds,
                benchmark_contexts=benchmark_contexts,
            )
            for row in rows:
                row.update(
                    {
                        "experiment_mode": "forward_add",
                        "step": step,
                        "candidate_feature": feature,
                    }
                )
            step_rows.extend(rows)
            detail_rows.extend(rows)

        summary_df = summarize_classifier_feature_selection(pd.DataFrame(detail_rows))
        best_step_rows = _best_hybrid_rows(summary_df)
        candidate_step_rows = best_step_rows.loc[
            (best_step_rows["experiment_mode"] == "forward_add")
            & (best_step_rows["step"] == step)
        ].copy()
        if candidate_step_rows.empty:
            break
        winner = _choose_forward_winner(candidate_step_rows)
        selected_features = winner["feature_columns"].split(",")
        forward_trace_rows.append(winner.to_dict())

    final_trace_df = pd.DataFrame(forward_trace_rows)
    final_recommendation = _choose_final_recommendation(final_trace_df)

    summary_df = summarize_classifier_feature_selection(pd.DataFrame(detail_rows))
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"classifier_feature_selection_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"classifier_feature_selection_summary_{stamp}.csv")
    trace_path = os.path.join(output_dir, f"classifier_forward_selection_trace_{stamp}.csv")
    recommendation_path = os.path.join(
        output_dir, f"classifier_feature_selection_recommendation_{stamp}.md"
    )
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    final_trace_df.to_csv(trace_path, index=False)
    with open(recommendation_path, "w", encoding="utf-8") as fh:
        fh.write("# Ridge Confirmatory Classifier Feature Recommendation\n\n")
        fh.write(f"- Regression candidate: `{regression_candidate_name}`\n")
        fh.write(f"- Classifier model type: `{classifier_model_type}`\n")
        fh.write(f"- Recommended feature count: `{int(final_recommendation['n_features'])}`\n")
        fh.write(f"- Recommended features: `{final_recommendation['feature_columns']}`\n")
        fh.write(
            f"- Best threshold: `{final_recommendation['threshold']:.2f}`\n"
            if pd.notna(final_recommendation["threshold"])
            else "- Best threshold: `N/A`\n"
        )
        fh.write(
            f"- Mean balanced accuracy: `{final_recommendation['mean_balanced_accuracy']:.4f}`\n"
        )
        fh.write(f"- Mean Brier score: `{final_recommendation['mean_brier_score']:.4f}`\n")
        fh.write(
            f"- Mean hybrid uplift vs regression sign: `{final_recommendation['mean_uplift_vs_regression_sign']:.4f}`\n"
        )
        fh.write(
            f"- Mean policy return: `{final_recommendation['mean_policy_return']:.4f}`\n"
        )
    print(f"Wrote classifier feature detail to {detail_path}")
    print(f"Wrote classifier feature summary to {summary_path}")
    print(f"Wrote classifier forward trace to {trace_path}")
    print(f"Wrote classifier recommendation to {recommendation_path}")
    return pd.DataFrame(detail_rows), summary_df, final_trace_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-feature-at-a-time selection for the Ridge confirmatory classifier."
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--regression-candidate",
        default=DEFAULT_REGRESSION_CANDIDATE,
        help="Regression candidate name from candidate_feature_sets().",
    )
    parser.add_argument(
        "--classifier-model-type",
        default=DEFAULT_CLASSIFIER_MODEL_TYPE,
        help="Classifier model type. Default: ridge",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS),
        help="Comma-separated probability thresholds.",
    )
    parser.add_argument(
        "--horizon",
        default="6",
        help="Target horizon in months.",
    )
    parser.add_argument(
        "--max-features",
        default="8",
        help="Maximum number of forward-selected features.",
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
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    features = [value.strip() for value in args.features.split(",") if value.strip()] or None
    conn = db_client.get_connection(config.DB_PATH)
    run_classifier_feature_selection(
        conn=conn,
        benchmarks=benchmarks,
        regression_candidate_name=args.regression_candidate,
        classifier_model_type=args.classifier_model_type,
        thresholds=thresholds,
        horizon=int(args.horizon),
        output_dir=args.output_dir,
        max_features=int(args.max_features),
        features=features,
    )


if __name__ == "__main__":
    main()
