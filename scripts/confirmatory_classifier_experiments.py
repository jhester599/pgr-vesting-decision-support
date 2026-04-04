"""v9 confirmatory classifier experiments on the reduced benchmark universe."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts.candidate_model_bakeoff import DEFAULT_BENCHMARKS, candidate_feature_sets
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import (
    iter_wfo_splits,
    summarize_binary_predictions,
    evaluate_wfo_model,
)
from src.research.policy_metrics import evaluate_hold_fraction_series, evaluate_policy_series


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")
DEFAULT_CONFIRMATORY_CANDIDATES = (
    "ridge_lean_v1",
    "gbt_lean_plus_two",
    "elasticnet_lean_v1",
)
DEFAULT_THRESHOLDS = (0.50, 0.55, 0.60)


def _build_classifier_pipeline(model_type: str) -> Pipeline:
    """Construct a modest classifier counterpart for the lean regression models."""
    if model_type == "elasticnet":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        C=1.0,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_type == "ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        C=1.0,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_type == "gbt":
        return Pipeline(
            steps=[
                (
                    "classifier",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=2,
                        n_estimators=100,
                        random_state=42,
                    ),
                )
            ]
        )
    raise ValueError(f"Unsupported classifier model_type '{model_type}'.")


def evaluate_confirmatory_classifier(
    X: pd.DataFrame,
    y_binary: pd.Series,
    model_type: str,
    feature_columns: list[str],
    target_horizon_months: int = 6,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Run a point-in-time safe classifier on the production WFO splits."""
    selected = [feature for feature in feature_columns if feature in X.columns]
    if not selected:
        raise ValueError("No selected classifier features were found in X.")

    X_selected = X[selected].copy()
    aligned = pd.concat([X_selected, y_binary], axis=1).dropna(subset=[y_binary.name])
    X_aligned = aligned[selected]
    y_aligned = aligned[y_binary.name].astype(int)

    probabilities: list[float] = []
    realized: list[int] = []
    dates: list[pd.Timestamp] = []

    for _, train_idx, test_idx in iter_wfo_splits(
        X_aligned,
        y_aligned,
        target_horizon_months=target_horizon_months,
    ):
        X_train = X_aligned.iloc[train_idx].to_numpy(copy=True)
        X_test = X_aligned.iloc[test_idx].to_numpy(copy=True)
        y_train = y_aligned.iloc[train_idx].to_numpy(copy=True)
        y_test = y_aligned.iloc[test_idx].to_numpy(copy=True)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            train_medians = np.nanmedian(X_train, axis=0)
        train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
        for col_idx in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, col_idx]), col_idx] = train_medians[col_idx]
            X_test[np.isnan(X_test[:, col_idx]), col_idx] = train_medians[col_idx]

        if len(np.unique(y_train)) < 2:
            prob_pos = np.full(len(test_idx), float(np.mean(y_train)))
        else:
            pipeline = _build_classifier_pipeline(model_type)
            pipeline.fit(X_train, y_train)
            classifier = pipeline[-1]
            if hasattr(classifier, "predict_proba"):
                prob_pos = pipeline.predict_proba(X_test)[:, 1]
            else:
                decision = pipeline.decision_function(X_test)
                prob_pos = 1.0 / (1.0 + np.exp(-decision))

        probabilities.extend(prob_pos.tolist())
        realized.extend(y_test.tolist())
        dates.extend(list(y_aligned.index[test_idx]))

    prob_series = pd.Series(probabilities, index=pd.DatetimeIndex(dates), name="p_outperform")
    realized_series = pd.Series(realized, index=pd.DatetimeIndex(dates), name="y_true_binary")
    summary = summarize_binary_predictions(prob_series, realized_series)
    metrics = {
        "n_obs": float(summary.n_obs),
        "brier_score": float(summary.brier_score),
        "accuracy": float(summary.accuracy),
        "balanced_accuracy": float(summary.balanced_accuracy),
        "precision": float(summary.precision),
        "recall": float(summary.recall),
        "base_rate": float(summary.base_rate),
        "predicted_positive_rate": float(summary.predicted_positive_rate),
    }
    return prob_series, realized_series, metrics


def summarize_confirmatory_classifier_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate confirmatory-classifier experiment rows into a ranked summary."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["candidate_name", "model_type", "policy_name", "threshold"]
    for keys, group in detail_df.groupby(group_cols, dropna=False):
        candidate_name, model_type, policy_name, threshold = keys
        rows.append(
            {
                "candidate_name": candidate_name,
                "model_type": model_type,
                "policy_name": policy_name,
                "threshold": threshold,
                "n_features": int(group["n_features"].iloc[0]),
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
                "notes": group["notes"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_uplift_vs_regression_sign", "mean_policy_return", "mean_balanced_accuracy"],
        ascending=[False, False, False],
    )


def run_confirmatory_classifier_experiments(
    conn: Any,
    benchmarks: list[str],
    candidates: list[str],
    thresholds: list[float],
    horizon: int,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether a binary outperform classifier improves lean regression decisions."""
    df = build_feature_matrix_from_db(conn)
    feature_columns = set(df.columns)
    detail_rows: list[dict[str, Any]] = []
    all_candidates = candidate_feature_sets()

    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, horizon)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_reg = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue
        y_binary = (y_reg > 0.0).astype(int).rename(f"{benchmark}_outperform")

        for candidate_name in candidates:
            spec = all_candidates[candidate_name]
            selected = [feature for feature in spec["features"] if feature in feature_columns]
            reg_result, reg_metrics = evaluate_wfo_model(
                X_aligned,
                y_reg,
                model_type=str(spec["model_type"]),
                benchmark=benchmark,
                target_horizon_months=horizon,
                feature_columns=selected,
            )
            reg_pred = pd.Series(
                reg_result.y_hat_all,
                index=pd.DatetimeIndex(reg_result.test_dates_all),
                name="y_hat_reg",
            )
            reg_realized = pd.Series(
                reg_result.y_true_all,
                index=pd.DatetimeIndex(reg_result.test_dates_all),
                name="y_true_reg",
            )

            prob_series, _, cls_metrics = evaluate_confirmatory_classifier(
                X_aligned,
                y_binary,
                model_type=str(spec["model_type"]),
                feature_columns=selected,
                target_horizon_months=horizon,
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
                    "candidate_name": candidate_name,
                    "model_type": spec["model_type"],
                    "n_features": len(selected),
                    "policy_name": "regression_sign",
                    "threshold": np.nan,
                    "notes": spec["notes"],
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
                    (f"classifier_only_{threshold:.2f}", classifier_policy),
                    (f"hybrid_confirm_{threshold:.2f}", hybrid_policy),
                ):
                    detail_rows.append(
                        {
                            "benchmark": benchmark,
                            "candidate_name": candidate_name,
                            "model_type": spec["model_type"],
                            "n_features": len(selected),
                            "policy_name": policy_name,
                            "threshold": threshold,
                            "notes": spec["notes"],
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

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_confirmatory_classifier_results(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"confirmatory_classifier_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"confirmatory_classifier_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote confirmatory classifier detail to {detail_path}")
    print(f"Wrote confirmatory classifier summary to {summary_path}")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 confirmatory classifier experiments.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CONFIRMATORY_CANDIDATES),
        help="Comma-separated candidate names from candidate_feature_sets().",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS),
        help="Comma-separated classifier probability thresholds.",
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
    candidates = [value.strip() for value in args.candidates.split(",") if value.strip()]
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    conn = db_client.get_connection(config.DB_PATH)
    run_confirmatory_classifier_experiments(
        conn=conn,
        benchmarks=benchmarks,
        candidates=candidates,
        thresholds=thresholds,
        horizon=int(args.horizon),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
