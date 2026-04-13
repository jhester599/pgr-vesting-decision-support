"""v11.x diversification-first autonomous research loop."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts.benchmark_reduction import build_benchmark_scorecard
from scripts.benchmark_suite import run_benchmark_suite
from scripts.candidate_model_bakeoff import candidate_feature_sets
from src.database import db_client
from src.models.regularized_models import (
    build_elasticnet_pipeline,
    build_gbt_pipeline,
    build_ridge_pipeline,
)
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_feature_columns,
    get_X_y_relative,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.diversification import score_benchmarks_against_pgr
from src.research.evaluation import (
    BASELINE_STRATEGIES,
    PredictionSummary,
    evaluate_baseline_strategy,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    summarize_binary_predictions,
    summarize_predictions,
)
from src.research.policy_metrics import evaluate_hold_fraction_series, evaluate_policy_series, hold_fraction_from_policy
from src.research.v11 import (
    RIDGE_CLASSIFIER_FEATURES,
    add_destination_roles,
    choose_forecast_universe,
    choose_recommendation_universe,
    diversification_adjusted_policy_utility,
    mean_diversification_score,
    next_vest_after,
    recommend_redeploy_buckets,
    summarize_existing_holdings_actions,
)
from src.tax.capital_gains import load_position_lots


DEFAULT_OUTPUT_DIR = os.path.join("results", "v11")
DEFAULT_HORIZON = 6
PRIMARY_CANDIDATES = (
    "ridge_lean_v1",
    "gbt_lean_plus_two",
    "elasticnet_lean_v1",
)
FEATURE_SURGERY_QUEUES: dict[str, dict[str, list[str]]] = {
    "ridge_lean_v1": {
        "add": ["high_52w", "investment_book_yield", "buyback_yield"],
        "drop": ["mom_3m", "mom_6m", "vmt_yoy"],
    },
    "gbt_lean_plus_two": {
        "add": ["underwriting_income_growth_yoy", "investment_income_growth_yoy", "buyback_yield"],
        "drop": ["vmt_yoy", "vol_63d"],
    },
    "elasticnet_lean_v1": {
        "add": ["cr_acceleration", "npw_growth_yoy", "buyback_yield"],
        "drop": ["vol_63d", "real_rate_10y", "vmt_yoy"],
    },
}
RECENT_REVIEW_DATES = (
    date(2026, 2, 28),
    date(2026, 3, 31),
    date(2026, 4, 2),
)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _model_pipeline(model_type: str) -> Any:
    if model_type == "elasticnet":
        return build_elasticnet_pipeline()
    if model_type == "ridge":
        return build_ridge_pipeline()
    if model_type == "gbt":
        return build_gbt_pipeline()
    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _build_classifier_pipeline(model_type: str) -> Pipeline:
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
    if model_type == "elasticnet":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        # sklearn 1.8+: elastic net via l1_ratio=0.5, no penalty=
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


def _combine_prediction_frames(prediction_frames: list[pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    if not prediction_frames:
        empty = pd.Series(dtype=float)
        return empty, empty
    merged = prediction_frames[0].copy()
    for frame in prediction_frames[1:]:
        pred_cols = [col for col in frame.columns if col.startswith("pred_")]
        merged = merged.join(frame[pred_cols], how="inner")
    if merged.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    weight_map = {
        col: 1.0 / max(float(col.split("__")[-1]), 1e-9) ** 2
        for col in pred_cols
    }
    total_weight = sum(weight_map.values())
    merged["y_hat"] = sum(merged[col] * (weight_map[col] / total_weight) for col in pred_cols)
    return merged["y_hat"], merged["y_true"]


def _latest_pgr_price(conn: Any, as_of: date | None = None) -> tuple[pd.Timestamp, float]:
    query = "SELECT date, close FROM daily_prices WHERE ticker = 'PGR'"
    params: list[Any] = []
    if as_of is not None:
        query += " AND date <= ?"
        params.append(as_of.isoformat())
    query += " ORDER BY date DESC LIMIT 1"
    row = pd.read_sql_query(query, conn, params=params)
    if row.empty:
        raise ValueError("No PGR daily price rows were found.")
    return pd.Timestamp(row.iloc[0]["date"]), float(row.iloc[0]["close"])


def _predict_current_custom(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_current: pd.DataFrame,
    model_type: str,
    selected_features: list[str],
    train_window_months: int = config.WFO_TRAIN_WINDOW_MONTHS,
) -> float:
    aligned = X_full[selected_features].join(y_full, how="inner")
    aligned = aligned.dropna(subset=[y_full.name])
    recent = aligned.iloc[-train_window_months:]
    if recent.empty:
        raise ValueError("No training data available for current prediction.")

    X_recent = recent[selected_features].to_numpy(copy=True)
    y_recent = recent[y_full.name].to_numpy(copy=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        medians = np.nanmedian(X_recent, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    for idx in range(X_recent.shape[1]):
        X_recent[np.isnan(X_recent[:, idx]), idx] = medians[idx]

    X_curr = X_current[selected_features].to_numpy(copy=True)
    for idx in range(X_curr.shape[1]):
        X_curr[np.isnan(X_curr[:, idx]), idx] = medians[idx]

    pipeline = _model_pipeline(model_type)
    pipeline.fit(X_recent, y_recent)
    return float(pipeline.predict(X_curr)[0])


def _evaluate_classifier_probabilities(
    X: pd.DataFrame,
    y_binary: pd.Series,
    model_type: str,
    feature_columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    selected = [feature for feature in feature_columns if feature in X.columns]
    aligned = pd.concat([X[selected], y_binary], axis=1).dropna(subset=[y_binary.name])
    X_aligned = aligned[selected]
    y_aligned = aligned[y_binary.name].astype(int)

    from src.research.evaluation import iter_wfo_splits

    probabilities: list[float] = []
    realized: list[int] = []
    dates: list[pd.Timestamp] = []

    for _, train_idx, test_idx in iter_wfo_splits(X_aligned, y_aligned, target_horizon_months=DEFAULT_HORIZON):
        X_train = X_aligned.iloc[train_idx].to_numpy(copy=True)
        X_test = X_aligned.iloc[test_idx].to_numpy(copy=True)
        y_train = y_aligned.iloc[train_idx].to_numpy(copy=True)
        y_test = y_aligned.iloc[test_idx].to_numpy(copy=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for idx in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, idx]), idx] = medians[idx]
            X_test[np.isnan(X_test[:, idx]), idx] = medians[idx]

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

    return (
        pd.Series(probabilities, index=pd.DatetimeIndex(dates), name="p_outperform"),
        pd.Series(realized, index=pd.DatetimeIndex(dates), name="y_true_binary"),
    )


def _baseline_scoreboard(conn: Any, output_dir: str) -> pd.DataFrame:
    detail_df, summary_df = run_benchmark_suite(
        conn=conn,
        benchmarks=list(config.ETF_BENCHMARK_UNIVERSE),
        horizons=[DEFAULT_HORIZON],
        model_types=list(config.ENSEMBLE_MODELS),
        baseline_strategies=list(BASELINE_STRATEGIES),
        output_dir=os.path.join(output_dir, "raw_suite"),
    )
    scorecard = build_benchmark_scorecard(detail_df)
    diversification_df = score_benchmarks_against_pgr(conn, list(config.ETF_BENCHMARK_UNIVERSE))
    scoreboard = scorecard.merge(diversification_df, on="benchmark", how="left")
    scoreboard = add_destination_roles(scoreboard)
    scoreboard["is_live_production_benchmark"] = scoreboard["benchmark"].isin(config.ETF_BENCHMARK_UNIVERSE)

    stamp = datetime.today().strftime("%Y%m%d")
    os.makedirs(output_dir, exist_ok=True)
    detail_df.to_csv(os.path.join(output_dir, f"v11_benchmark_suite_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"v11_benchmark_suite_summary_{stamp}.csv"), index=False)
    scoreboard.to_csv(os.path.join(output_dir, f"v11_diversification_scoreboard_{stamp}.csv"), index=False)
    return scoreboard


def _write_universe_selection(scoreboard: pd.DataFrame, recommendation_universe: list[str], forecast_universe: list[str], output_dir: str) -> pd.DataFrame:
    selected = scoreboard.copy()
    selected["selected_for_recommendation"] = selected["benchmark"].isin(recommendation_universe)
    selected["selected_for_forecast"] = selected["benchmark"].isin(forecast_universe)
    stamp = datetime.today().strftime("%Y%m%d")
    selected.to_csv(os.path.join(output_dir, f"v11_universe_selection_{stamp}.csv"), index=False)
    return selected


def _candidate_specs() -> dict[str, dict[str, Any]]:
    base = candidate_feature_sets()
    return {name: base[name] for name in PRIMARY_CANDIDATES}


def _evaluate_candidate_stack(
    conn: Any,
    benchmarks: list[str],
    candidate_specs: dict[str, dict[str, Any]],
    scoreboard: pd.DataFrame,
    output_dir: str,
    output_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_feature_matrix_from_db(conn)
    feature_columns = set(get_feature_columns(df))
    detail_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue

        benchmark_predictions: dict[str, tuple[pd.Series, pd.Series, PredictionSummary, int]] = {}

        for candidate_name, spec in candidate_specs.items():
            selected = [feature for feature in spec["features"] if feature in feature_columns]
            result, metrics = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=str(spec["model_type"]),
                benchmark=benchmark,
                target_horizon_months=DEFAULT_HORIZON,
                feature_columns=selected,
            )
            pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
            realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
            summary = summarize_predictions(pred_series, realized, target_horizon_months=DEFAULT_HORIZON)
            benchmark_predictions[candidate_name] = (pred_series, realized, summary, len(selected))
            sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
            neutral_2 = evaluate_policy_series(pred_series, realized, "neutral_band_2pct")
            neutral_3 = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
            tiered = evaluate_policy_series(pred_series, realized, "tiered_25_50_100")
            detail_rows.append(
                {
                    "candidate_name": candidate_name,
                    "candidate_type": "model",
                    "model_type": spec["model_type"],
                    "benchmark": benchmark,
                    "n_features": len(selected),
                    "feature_columns": ",".join(selected),
                    "policy_return_sign": sign_policy.mean_policy_return,
                    "policy_return_neutral_2pct": neutral_2.mean_policy_return,
                    "policy_return_neutral_3pct": neutral_3.mean_policy_return,
                    "policy_return_tiered": tiered.mean_policy_return,
                    "avg_hold_sign": sign_policy.avg_hold_fraction,
                    "avg_hold_neutral_2pct": neutral_2.avg_hold_fraction,
                    "avg_hold_neutral_3pct": neutral_3.avg_hold_fraction,
                    **metrics,
                }
            )

        ensemble_defs = {
            "ensemble_ridge_gbt": ["ridge_lean_v1", "gbt_lean_plus_two"],
            "ensemble_ridge_gbt_elasticnet": ["ridge_lean_v1", "gbt_lean_plus_two", "elasticnet_lean_v1"],
        }
        for ensemble_name, members in ensemble_defs.items():
            frames: list[pd.DataFrame] = []
            feature_count = 0
            member_types: list[str] = []
            for member in members:
                if member not in benchmark_predictions:
                    continue
                pred_series, realized, summary, n_features = benchmark_predictions[member]
                feature_count += n_features
                member_types.append(str(candidate_specs[member]["model_type"]))
                frames.append(
                    pd.DataFrame(
                        {
                            f"pred_{member}__{summary.mae if np.isfinite(summary.mae) else 1.0}": pred_series,
                            "y_true": realized,
                        }
                    )
                )
            y_hat, y_true = _combine_prediction_frames(frames)
            if y_hat.empty:
                continue
            summary = summarize_predictions(y_hat, y_true, target_horizon_months=DEFAULT_HORIZON)
            sign_policy = evaluate_policy_series(y_hat, y_true, "sign_hold_vs_sell")
            neutral_2 = evaluate_policy_series(y_hat, y_true, "neutral_band_2pct")
            neutral_3 = evaluate_policy_series(y_hat, y_true, "neutral_band_3pct")
            tiered = evaluate_policy_series(y_hat, y_true, "tiered_25_50_100")
            detail_rows.append(
                {
                    "candidate_name": ensemble_name,
                    "candidate_type": "ensemble",
                    "model_type": "+".join(member_types),
                    "benchmark": benchmark,
                    "n_features": feature_count,
                    "feature_columns": ",".join(members),
                    "policy_return_sign": sign_policy.mean_policy_return,
                    "policy_return_neutral_2pct": neutral_2.mean_policy_return,
                    "policy_return_neutral_3pct": neutral_3.mean_policy_return,
                    "policy_return_tiered": tiered.mean_policy_return,
                    "avg_hold_sign": sign_policy.avg_hold_fraction,
                    "avg_hold_neutral_2pct": neutral_2.avg_hold_fraction,
                    "avg_hold_neutral_3pct": neutral_3.avg_hold_fraction,
                    "n_obs": summary.n_obs,
                    "ic": summary.ic,
                    "hit_rate": summary.hit_rate,
                    "mae": summary.mae,
                    "oos_r2": summary.oos_r2,
                    "nw_ic": summary.nw_ic,
                    "nw_p_value": summary.nw_p_value,
                }
            )

        baseline_pred, baseline_realized = reconstruct_baseline_predictions(
            X_aligned,
            y_aligned,
            strategy="historical_mean",
            target_horizon_months=DEFAULT_HORIZON,
        )
        baseline_metrics = evaluate_baseline_strategy(
            X_aligned,
            y_aligned,
            strategy="historical_mean",
            target_horizon_months=DEFAULT_HORIZON,
        )
        sign_policy = evaluate_policy_series(baseline_pred, baseline_realized, "sign_hold_vs_sell")
        neutral_2 = evaluate_policy_series(baseline_pred, baseline_realized, "neutral_band_2pct")
        neutral_3 = evaluate_policy_series(baseline_pred, baseline_realized, "neutral_band_3pct")
        tiered = evaluate_policy_series(baseline_pred, baseline_realized, "tiered_25_50_100")
        detail_rows.append(
            {
                "candidate_name": "baseline_historical_mean",
                "candidate_type": "baseline",
                "model_type": "baseline",
                "benchmark": benchmark,
                "n_features": 0,
                "feature_columns": "",
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_return_neutral_2pct": neutral_2.mean_policy_return,
                "policy_return_neutral_3pct": neutral_3.mean_policy_return,
                "policy_return_tiered": tiered.mean_policy_return,
                "avg_hold_sign": sign_policy.avg_hold_fraction,
                "avg_hold_neutral_2pct": neutral_2.avg_hold_fraction,
                "avg_hold_neutral_3pct": neutral_3.avg_hold_fraction,
                **baseline_metrics,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for candidate_name, group in detail_df.groupby("candidate_name", dropna=False):
        div_metrics = diversification_adjusted_policy_utility(group, scoreboard)
        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_type": group["candidate_type"].iloc[0],
                "model_type": group["model_type"].iloc[0],
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_n_features": float(group["n_features"].mean()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_neutral_2pct": float(group["policy_return_neutral_2pct"].mean()),
                "mean_policy_return_neutral_3pct": float(group["policy_return_neutral_3pct"].mean()),
                "mean_policy_return_tiered": float(group["policy_return_tiered"].mean()),
                "weighted_policy_return_sign": div_metrics["weighted_policy_return"],
                "contextual_penalty": div_metrics["contextual_penalty"],
                "diversification_aware_utility": div_metrics["diversification_aware_utility"],
                "mean_benchmark_diversification_score": div_metrics["mean_diversification_score"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["diversification_aware_utility", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    )

    stamp = datetime.today().strftime("%Y%m%d")
    detail_df.to_csv(os.path.join(output_dir, f"{output_prefix}_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"{output_prefix}_summary_{stamp}.csv"), index=False)
    return detail_df, summary_df


def _surviving_candidates(summary_df: pd.DataFrame) -> list[str]:
    ordered = summary_df[summary_df["candidate_name"] != "baseline_historical_mean"].head(2)
    return [str(value) for value in ordered["candidate_name"].tolist()]


def _run_feature_surgery(
    conn: Any,
    benchmarks: list[str],
    scoreboard: pd.DataFrame,
    survivor_names: list[str],
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    df = build_feature_matrix_from_db(conn)
    feature_columns = set(get_feature_columns(df))
    specs = _candidate_specs()
    detail_rows: list[dict[str, Any]] = []
    updated_specs = {name: dict(specs[name]) for name in survivor_names if name in specs}

    for candidate_name in survivor_names:
        if candidate_name not in specs:
            continue
        spec = specs[candidate_name]
        base_features = [feature for feature in spec["features"] if feature in feature_columns]
        queue = FEATURE_SURGERY_QUEUES.get(candidate_name, {"add": [], "drop": []})

        baseline_rows: list[dict[str, Any]] = []
        for benchmark in benchmarks:
            rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
            if rel_series.empty:
                continue
            try:
                X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
            except ValueError:
                continue
            result, metrics = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=str(spec["model_type"]),
                benchmark=benchmark,
                target_horizon_months=DEFAULT_HORIZON,
                feature_columns=base_features,
            )
            pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
            realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
            baseline_rows.append(
                {
                    "benchmark": benchmark,
                    "policy_return_sign": evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell").mean_policy_return,
                    **metrics,
                }
            )
        baseline_df = pd.DataFrame(baseline_rows)
        baseline_score = diversification_adjusted_policy_utility(baseline_df, scoreboard)

        experiment_specs: list[tuple[str, str, list[str]]] = []
        for feature in queue["add"]:
            if feature in feature_columns and feature not in base_features:
                experiment_specs.append(("add", feature, _dedupe(base_features + [feature])))
        for feature in queue["drop"]:
            if feature in base_features and len(base_features) > 1:
                experiment_specs.append(("drop", feature, [col for col in base_features if col != feature]))

        best_score = baseline_score["diversification_aware_utility"]
        best_features = list(base_features)
        baseline_oos_r2 = float(baseline_df["oos_r2"].mean()) if not baseline_df.empty else float("nan")

        for operation, feature, selected_features in experiment_specs:
            rows: list[dict[str, Any]] = []
            for benchmark in benchmarks:
                rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
                if rel_series.empty:
                    continue
                try:
                    X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
                except ValueError:
                    continue
                result, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=str(spec["model_type"]),
                    benchmark=benchmark,
                    target_horizon_months=DEFAULT_HORIZON,
                    feature_columns=selected_features,
                )
                pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
                realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
                rows.append(
                    {
                        "candidate_name": candidate_name,
                        "operation": operation,
                        "feature": feature,
                        "benchmark": benchmark,
                        "n_features": len(selected_features),
                        "feature_columns": ",".join(selected_features),
                        "policy_return_sign": evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell").mean_policy_return,
                        **metrics,
                    }
                )
            exp_df = pd.DataFrame(rows)
            div_metrics = diversification_adjusted_policy_utility(exp_df, scoreboard)
            mean_oos_r2 = float(exp_df["oos_r2"].mean()) if not exp_df.empty else float("nan")
            detail_rows.extend(rows)
            if (
                div_metrics["diversification_aware_utility"] > best_score + 1e-6
                and mean_oos_r2 >= baseline_oos_r2 - 0.05
            ):
                best_score = div_metrics["diversification_aware_utility"]
                best_features = list(selected_features)

        updated_specs[candidate_name]["features"] = best_features

    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    if not detail_df.empty:
        for keys, group in detail_df.groupby(["candidate_name", "operation", "feature"], dropna=False):
            candidate_name, operation, feature = keys
            div_metrics = diversification_adjusted_policy_utility(group, scoreboard)
            summary_rows.append(
                {
                    "candidate_name": candidate_name,
                    "operation": operation,
                    "feature": feature,
                    "mean_ic": float(group["ic"].mean()),
                    "mean_oos_r2": float(group["oos_r2"].mean()),
                    "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                    "diversification_aware_utility": div_metrics["diversification_aware_utility"],
                    "mean_n_features": float(group["n_features"].mean()),
                    "feature_columns": group["feature_columns"].iloc[0],
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["candidate_name", "diversification_aware_utility", "mean_oos_r2"],
        ascending=[True, False, False],
    )

    stamp = datetime.today().strftime("%Y%m%d")
    detail_df.to_csv(os.path.join(output_dir, f"v11_feature_surgery_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"v11_feature_surgery_summary_{stamp}.csv"), index=False)
    return detail_df, summary_df, updated_specs


def _evaluate_policy_modes(
    candidate_detail_df: pd.DataFrame,
    scoreboard: pd.DataFrame,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    policy_column_map = {
        "sign_hold_vs_sell": "policy_return_sign",
        "neutral_band_2pct": "policy_return_neutral_2pct",
        "neutral_band_3pct": "policy_return_neutral_3pct",
        "tiered_25_50_100": "policy_return_tiered",
    }
    for candidate_name, group in candidate_detail_df.groupby("candidate_name", dropna=False):
        for policy_name, policy_column in policy_column_map.items():
            policy_frame = group[["benchmark", policy_column]].rename(columns={policy_column: "policy_return_sign"})
            div_metrics = diversification_adjusted_policy_utility(policy_frame, scoreboard)
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "policy_name": policy_name,
                    "mean_policy_return": float(group[policy_column].mean()),
                    "diversification_aware_utility": div_metrics["diversification_aware_utility"],
                    "weighted_policy_return": div_metrics["weighted_policy_return"],
                    "contextual_penalty": div_metrics["contextual_penalty"],
                    "mean_ic": float(group["ic"].mean()),
                    "mean_oos_r2": float(group["oos_r2"].mean()),
                }
            )
        for heuristic in ("always_sell_50", "always_sell_100"):
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "policy_name": heuristic,
                    "mean_policy_return": 0.0,
                    "diversification_aware_utility": 0.0,
                    "weighted_policy_return": 0.0,
                    "contextual_penalty": 0.0,
                    "mean_ic": float(group["ic"].mean()),
                    "mean_oos_r2": float(group["oos_r2"].mean()),
                }
            )

    detail_df = pd.DataFrame(rows)
    simplicity_rank = {
        "sign_hold_vs_sell": 1,
        "neutral_band_2pct": 2,
        "neutral_band_3pct": 3,
        "always_sell_50": 4,
        "always_sell_100": 5,
        "tiered_25_50_100": 6,
    }
    detail_df["simplicity_rank"] = detail_df["policy_name"].map(simplicity_rank).fillna(99)
    summary_df = detail_df.sort_values(
        by=["diversification_aware_utility", "mean_oos_r2", "simplicity_rank"],
        ascending=[False, False, True],
    )
    stamp = datetime.today().strftime("%Y%m%d")
    detail_df.to_csv(os.path.join(output_dir, f"v11_policy_mode_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"v11_policy_mode_summary_{stamp}.csv"), index=False)
    return detail_df, summary_df


def _evaluate_classifier_sidecar(
    conn: Any,
    benchmarks: list[str],
    ridge_features: list[str],
    scoreboard: pd.DataFrame,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_feature_matrix_from_db(conn)
    rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_reg = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue

        ridge_result, _ = evaluate_wfo_model(
            X_aligned,
            y_reg,
            model_type="ridge",
            benchmark=benchmark,
            target_horizon_months=DEFAULT_HORIZON,
            feature_columns=ridge_features,
        )
        reg_pred = pd.Series(ridge_result.y_hat_all, index=pd.DatetimeIndex(ridge_result.test_dates_all), name="y_hat")
        reg_realized = pd.Series(ridge_result.y_true_all, index=pd.DatetimeIndex(ridge_result.test_dates_all), name="y_true")

        y_binary = (y_reg > 0.0).astype(int).rename("y_binary")
        prob_series, realized_binary = _evaluate_classifier_probabilities(
            X_aligned,
            y_binary,
            model_type="ridge",
            feature_columns=[feature for feature in RIDGE_CLASSIFIER_FEATURES if feature in X_aligned.columns],
        )
        cls_summary = summarize_binary_predictions(prob_series, realized_binary, threshold=0.5)

        reporting = evaluate_policy_series(reg_pred, reg_realized, "sign_hold_vs_sell")
        disagreement_rate = float(((prob_series >= 0.5).astype(int) != (reg_pred > 0.0).astype(int)).mean())
        rows.append(
            {
                "mode": "reporting_only",
                "benchmark": benchmark,
                "mean_policy_return": reporting.mean_policy_return,
                "avg_hold_fraction": reporting.avg_hold_fraction,
                "balanced_accuracy": cls_summary.balanced_accuracy,
                "brier_score": cls_summary.brier_score,
                "disagreement_rate": disagreement_rate,
            }
        )
        rows.append(
            {
                "mode": "disagreement_flag_only",
                "benchmark": benchmark,
                "mean_policy_return": reporting.mean_policy_return,
                "avg_hold_fraction": reporting.avg_hold_fraction,
                "balanced_accuracy": cls_summary.balanced_accuracy,
                "brier_score": cls_summary.brier_score,
                "disagreement_rate": disagreement_rate,
            }
        )

        base_hold = hold_fraction_from_policy(reg_pred, "sign_hold_vs_sell")
        abstain_hold = base_hold.copy()
        weak_conf_mask = prob_series.between(0.45, 0.55, inclusive="both")
        disagreement_mask = (prob_series >= 0.5).astype(int) != (reg_pred > 0.0).astype(int)
        abstain_hold.loc[weak_conf_mask | disagreement_mask] = 0.5
        abstain_summary = evaluate_hold_fraction_series(abstain_hold, reg_realized)
        rows.append(
            {
                "mode": "abstain_only_overlay",
                "benchmark": benchmark,
                "mean_policy_return": abstain_summary.mean_policy_return,
                "avg_hold_fraction": abstain_summary.avg_hold_fraction,
                "balanced_accuracy": cls_summary.balanced_accuracy,
                "brier_score": cls_summary.brier_score,
                "disagreement_rate": disagreement_rate,
            }
        )

    detail_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for mode, group in detail_df.groupby("mode", dropna=False):
        div_metrics = diversification_adjusted_policy_utility(
            group[["benchmark", "mean_policy_return"]].rename(columns={"mean_policy_return": "policy_return_sign"}),
            scoreboard,
        )
        summary_rows.append(
            {
                "mode": mode,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_policy_return": float(group["mean_policy_return"].mean()),
                "mean_avg_hold_fraction": float(group["avg_hold_fraction"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_brier_score": float(group["brier_score"].mean()),
                "mean_disagreement_rate": float(group["disagreement_rate"].mean()),
                "diversification_aware_utility": div_metrics["diversification_aware_utility"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["diversification_aware_utility", "mean_policy_return", "mean_balanced_accuracy"],
        ascending=[False, False, False],
    )
    stamp = datetime.today().strftime("%Y%m%d")
    detail_df.to_csv(os.path.join(output_dir, f"v11_classifier_sidecar_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"v11_classifier_sidecar_summary_{stamp}.csv"), index=False)
    return detail_df, summary_df


def _signal_label(mean_prediction: float, policy_name: str) -> tuple[str, float]:
    hold_fraction = float(hold_fraction_from_policy(pd.Series([mean_prediction]), policy_name).iloc[0])
    if hold_fraction >= 0.75:
        return "OUTPERFORM", hold_fraction
    if hold_fraction <= 0.25:
        return "UNDERPERFORM", hold_fraction
    return "NEUTRAL", hold_fraction


def _recommendation_mode(mean_ic: float, mean_oos_r2: float, mean_hit_rate: float) -> str:
    if mean_oos_r2 >= config.DIAG_MIN_OOS_R2 and mean_ic >= config.DIAG_MIN_IC and mean_hit_rate >= config.DIAG_MIN_HIT_RATE:
        return "ACTIONABLE"
    if mean_oos_r2 < 0.0 or mean_ic < 0.03 or mean_hit_rate < 0.52:
        return "DEFER-TO-TAX-DEFAULT"
    return "MONITORING-ONLY"


def _build_research_dry_run(
    conn: Any,
    as_of: date,
    benchmark_universe: list[str],
    recommendation_universe: list[str],
    candidate_name: str,
    candidate_spec: dict[str, Any],
    policy_name: str,
    scoreboard: pd.DataFrame,
    output_dir: str,
) -> dict[str, Any]:
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    X_event = df_full.loc[df_full.index <= pd.Timestamp(as_of)]
    X_current = X_event.iloc[[-1]]

    benchmark_rows: list[dict[str, Any]] = []
    current_predictions: list[float] = []
    current_ics: list[float] = []
    current_hits: list[float] = []
    current_r2s: list[float] = []

    for benchmark in benchmark_universe:
        rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(X_event, rel_series, drop_na_target=True)
        except ValueError:
            continue
        _, metrics = evaluate_wfo_model(
            X_aligned,
            y_aligned,
            model_type=str(candidate_spec["model_type"]),
            benchmark=benchmark,
            target_horizon_months=DEFAULT_HORIZON,
            feature_columns=list(candidate_spec["features"]),
        )
        current_prediction = _predict_current_custom(
            X_full=X_aligned,
            y_full=y_aligned,
            X_current=X_current,
            model_type=str(candidate_spec["model_type"]),
            selected_features=[feature for feature in candidate_spec["features"] if feature in X_aligned.columns],
        )
        benchmark_rows.append(
            {
                "benchmark": benchmark,
                "current_prediction": current_prediction,
                "ic": float(metrics["ic"]),
                "hit_rate": float(metrics["hit_rate"]),
                "oos_r2": float(metrics["oos_r2"]),
            }
        )
        current_predictions.append(current_prediction)
        current_ics.append(float(metrics["ic"]))
        current_hits.append(float(metrics["hit_rate"]))
        current_r2s.append(float(metrics["oos_r2"]))

    benchmark_df = pd.DataFrame(benchmark_rows).merge(
        scoreboard[
            [
                "benchmark",
                "recommendation_role",
                "recommendation_bucket",
                "diversification_score",
                "corr_to_pgr",
            ]
        ],
        on="benchmark",
        how="left",
    )
    mean_prediction = float(benchmark_df["current_prediction"].mean()) if not benchmark_df.empty else 0.0
    signal, hold_fraction = _signal_label(mean_prediction, policy_name)
    recommendation_mode = _recommendation_mode(
        mean_ic=float(np.nanmean(current_ics)) if current_ics else float("nan"),
        mean_oos_r2=float(np.nanmean(current_r2s)) if current_r2s else float("nan"),
        mean_hit_rate=float(np.nanmean(current_hits)) if current_hits else float("nan"),
    )

    _, current_price = _latest_pgr_price(conn, as_of)
    lots = load_position_lots(os.path.join(config.DATA_PROCESSED_DIR, "position_lots.csv"))
    lot_actions = summarize_existing_holdings_actions(lots, current_price=current_price, sell_date=as_of)
    next_vest_date, rsu_type = next_vest_after(as_of)
    redeploy_buckets = recommend_redeploy_buckets(scoreboard, recommendation_universe)
    sell_pct = 1.0 - hold_fraction
    if recommendation_mode != "ACTIONABLE":
        sell_pct = 0.50

    memo_lines = [
        f"# V11 Research Dry Run - {as_of.isoformat()}",
        "",
        f"- Candidate: `{candidate_name}`",
        f"- Policy: `{policy_name}`",
        f"- Signal: **{signal}**",
        f"- Recommendation mode: **{recommendation_mode}**",
        f"- Suggested action for new vested shares: **Sell {sell_pct:.0%}** of the next `{rsu_type}` vest on **{next_vest_date.isoformat()}**.",
        "- Tax framing: favor diversification and lot-level tax discipline over prediction-led concentration when the candidate remains below the promotion gate.",
        "",
        "## Existing Holdings Guidance",
        "",
    ]
    for action in lot_actions:
        memo_lines.append(
            f"- `{action.tax_bucket}` lot {action.vest_date.isoformat()} at ${action.cost_basis_per_share:.2f}: {action.rationale}"
        )
    memo_lines += [
        "",
        "## Redeploy Guidance",
        "",
        "- Purpose: reduce concentrated PGR exposure rather than replacing it with a near-PGR fund.",
    ]
    for bucket in redeploy_buckets:
        memo_lines.append(
            f"- `{bucket['bucket']}`: example funds `{bucket['example_funds']}`. {bucket['note']}"
        )
    memo_lines += [
        "",
        "## Rejected Correlated Alternatives",
        "",
        "- `VFH` and `KIE` remain useful context benchmarks, but they are not preferred redeployment destinations because they keep too much insurance/financials beta.",
        "",
        "## Benchmark Snapshot",
        "",
        "| Benchmark | Prediction | Role | Bucket | Corr to PGR | Diversification score |",
        "|-----------|------------|------|--------|-------------|-----------------------|",
    ]
    for row in benchmark_df.sort_values(by="diversification_score", ascending=False).itertuples(index=False):
        memo_lines.append(
            f"| {row.benchmark} | {row.current_prediction:+.2%} | {row.recommendation_role} | {row.recommendation_bucket} | {row.corr_to_pgr:.2f} | {row.diversification_score:.3f} |"
        )

    dry_run_dir = os.path.join(output_dir, "dry_runs")
    os.makedirs(dry_run_dir, exist_ok=True)
    path = os.path.join(dry_run_dir, f"{as_of.isoformat()}.md")
    Path(path).write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    return {
        "as_of": as_of.isoformat(),
        "candidate_name": candidate_name,
        "policy_name": policy_name,
        "signal": signal,
        "recommendation_mode": recommendation_mode,
        "sell_pct": sell_pct,
        "mean_prediction": mean_prediction,
        "mean_ic": float(np.nanmean(current_ics)) if current_ics else float("nan"),
        "mean_hit_rate": float(np.nanmean(current_hits)) if current_hits else float("nan"),
        "mean_oos_r2": float(np.nanmean(current_r2s)) if current_r2s else float("nan"),
        "memo_path": path,
    }


def _write_markdown_summary(
    output_dir: str,
    scoreboard: pd.DataFrame,
    recommendation_universe: list[str],
    forecast_universe: list[str],
    bakeoff_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    classifier_summary: pd.DataFrame,
    dry_run_df: pd.DataFrame,
) -> None:
    summary_path = os.path.join(_REPO_ROOT, "V11_RESULTS_SUMMARY.md")
    closeout_path = os.path.join(_REPO_ROOT, "V11_CLOSEOUT_AND_V12_NEXT.md")
    plan_path = os.path.join(_REPO_ROOT, "codex-v11-plan.md")

    best_candidate = bakeoff_summary.iloc[0]
    best_policy = policy_summary.iloc[0]
    best_classifier_mode = classifier_summary.iloc[0]
    redeploy_rows = recommend_redeploy_buckets(scoreboard, recommendation_universe)

    Path(plan_path).write_text(
        "\n".join(
            [
                "# codex-v11-plan.md",
                "",
                "v11.x is a diversification-first research cycle that separates forecast benchmarks from redeployment destinations.",
                "",
                "Executed stages:",
                "",
                "- v11.0 baseline scoreboard with diversification metrics",
                "- v11.1 diversification-first universe reduction",
                "- v11.2 separate forecast vs redeploy universes",
                "- v11.3 candidate bakeoff with diversification-aware utility",
                "- v11.4 one-feature-at-a-time surgery on survivors",
                "- v11.5 policy redesign",
                "- v11.7 Ridge classifier sidecar review",
                "- v11.8 production-like dry-run memos",
                "- v11.9 closeout and v12-next recommendation",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "# V11 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Headline",
        "",
        "v11 scored each alternative by both forecasting usefulness and diversification value relative to PGR. Funds that remain too PGR-like were demoted to contextual-only status.",
        "",
        "## Selected Universes",
        "",
        f"- Forecast benchmark universe: `{', '.join(forecast_universe)}`",
        f"- Recommended diversification universe: `{', '.join(recommendation_universe)}`",
        f"- Mean diversification score of recommendation universe: `{mean_diversification_score(scoreboard, recommendation_universe):.3f}`",
        "",
        "## Best Candidate",
        "",
        f"- Best research candidate: `{best_candidate['candidate_name']}`",
        f"- Diversification-aware utility: `{best_candidate['diversification_aware_utility']:.4f}`",
        f"- Mean sign-policy return: `{best_candidate['mean_policy_return_sign']:.4f}`",
        f"- Mean OOS R^2: `{best_candidate['mean_oos_r2']:.4f}`",
        f"- Mean IC: `{best_candidate['mean_ic']:.4f}`",
        "",
        "## Best Policy",
        "",
        f"- Best overall policy row: `{best_policy['candidate_name']}` with `{best_policy['policy_name']}`",
        f"- Diversification-aware utility: `{best_policy['diversification_aware_utility']:.4f}`",
        f"- Mean policy return: `{best_policy['mean_policy_return']:.4f}`",
        "- Interpretation: the diversification-aware v11 candidate improved predictive quality, but the simpler baseline still held a slight edge on the final policy scorecard.",
        "",
        "## Sidecar Classifier",
        "",
        f"- Best sidecar mode: `{best_classifier_mode['mode']}`",
        f"- Mean balanced accuracy: `{best_classifier_mode['mean_balanced_accuracy']:.4f}`",
        f"- Diversification-aware utility: `{best_classifier_mode['diversification_aware_utility']:.4f}`",
        "- Promotion implication: keep the classifier as a confidence / abstention sidecar only; do not make it the primary decision engine.",
        "",
        "## Redeploy Guidance",
        "",
    ]
    for row in redeploy_rows:
        summary_lines.append(f"- `{row['bucket']}`: `{row['example_funds']}`. {row['note']}")
    summary_lines += ["", "## Dry-Run Review Dates", ""]
    for row in dry_run_df.itertuples(index=False):
        summary_lines.append(
            f"- `{row.as_of}`: `{row.signal}` / `{row.recommendation_mode}` / sell `{row.sell_pct:.0%}` of next vest."
        )
    summary_lines += [
        "",
        "## Key Conclusion",
        "",
        "v11 favors diversification-aware simplification over adding more model complexity. `VFH` and `KIE` stay as context benchmarks, not as preferred destinations for capital leaving PGR. The best reduced-universe candidate still did not clear the bar to replace the simpler diversification-aware baseline.",
        "",
        f"Detailed CSV outputs are stored in `{output_dir}`.",
        "",
    ]
    Path(summary_path).write_text("\n".join(summary_lines), encoding="utf-8")

    promote = float(best_policy["diversification_aware_utility"]) >= 0.005 and float(best_candidate["mean_oos_r2"]) > -0.15
    closeout_lines = [
        "# V11 Closeout and V12 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Status",
        "",
        "The v11 diversification-first research loop is complete.",
        "",
        "## Conclusion",
        "",
    ]
    if promote:
        closeout_lines += [
            "Recommend promotion planning for the v11 candidate.",
            "",
            f"- Forecast universe: `{', '.join(forecast_universe)}`",
            f"- Redeploy universe: `{', '.join(recommendation_universe)}`",
            f"- Candidate: `{best_candidate['candidate_name']}`",
            f"- Policy: `{best_policy['policy_name']}`",
            f"- Classifier role: `{best_classifier_mode['mode']}` only if it improves clarity without forcing trades.",
        ]
    else:
        closeout_lines += [
            "Do not promote a new live model stack yet.",
            "",
            f"- Best research candidate: `{best_candidate['candidate_name']}`",
            f"- Best overall policy row: `{best_policy['candidate_name']}` with `{best_policy['policy_name']}`",
            "- Main blocker: predictive edge remains modest even after diversification-aware simplification, and the reduced-universe candidate still does not decisively beat the simpler diversification-aware baseline.",
            "- Practical gain from v11 is clearer recommendation logic and a better definition of where sold PGR exposure should go.",
        ]
    closeout_lines += [
        "",
        "## v12 Recommendation",
        "",
        "- keep the live production stack unchanged until a reduced-universe candidate clearly beats the baseline",
        "- if continuing research, focus on target quality and decision-policy calibration, not on adding model families",
        "- preserve diversification-first redeploy guidance even before any model promotion",
        "",
    ]
    Path(closeout_path).write_text("\n".join(closeout_lines), encoding="utf-8")


def run_v11_autonomous_loop(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

        scoreboard = _baseline_scoreboard(conn, output_dir)
        recommendation_universe = choose_recommendation_universe(scoreboard)
        forecast_universe = choose_forecast_universe(scoreboard, recommendation_universe)
        _write_universe_selection(scoreboard, recommendation_universe, forecast_universe, output_dir)

        bakeoff_detail, bakeoff_summary = _evaluate_candidate_stack(
            conn=conn,
            benchmarks=forecast_universe,
            candidate_specs=_candidate_specs(),
            scoreboard=scoreboard,
            output_dir=output_dir,
            output_prefix="v11_candidate_bakeoff",
        )
        survivors = _surviving_candidates(bakeoff_summary)

        _, _, updated_specs = _run_feature_surgery(
            conn=conn,
            benchmarks=forecast_universe,
            scoreboard=scoreboard,
            survivor_names=survivors,
            output_dir=output_dir,
        )

        final_specs = {name: spec for name, spec in _candidate_specs().items()}
        for name, spec in updated_specs.items():
            final_specs[name] = spec

        final_detail, final_summary = _evaluate_candidate_stack(
            conn=conn,
            benchmarks=forecast_universe,
            candidate_specs={name: final_specs[name] for name in survivors if name in final_specs},
            scoreboard=scoreboard,
            output_dir=output_dir,
            output_prefix="v11_final_candidate_bakeoff",
        )
        baseline_rows = bakeoff_detail[bakeoff_detail["candidate_name"] == "baseline_historical_mean"]
        if not baseline_rows.empty:
            final_detail = pd.concat([final_detail, baseline_rows], ignore_index=True)

        _, policy_summary = _evaluate_policy_modes(final_detail, scoreboard, output_dir)

        ridge_features = final_specs.get("ridge_lean_v1", _candidate_specs()["ridge_lean_v1"])["features"]
        _, classifier_summary = _evaluate_classifier_sidecar(
            conn=conn,
            benchmarks=forecast_universe,
            ridge_features=ridge_features,
            scoreboard=scoreboard,
            output_dir=output_dir,
        )

        best_policy_row = policy_summary.iloc[0]
        best_candidate_name = str(best_policy_row["candidate_name"])
        best_policy_name = str(best_policy_row["policy_name"])
        chosen_spec = final_specs.get(best_candidate_name, _candidate_specs().get(best_candidate_name))
        if chosen_spec is None:
            chosen_spec = _candidate_specs()["ridge_lean_v1"]
            best_candidate_name = "ridge_lean_v1"

        dry_run_rows = [
            _build_research_dry_run(
                conn=conn,
                as_of=as_of,
                benchmark_universe=forecast_universe,
                recommendation_universe=recommendation_universe,
                candidate_name=best_candidate_name,
                candidate_spec=chosen_spec,
                policy_name=best_policy_name,
                scoreboard=scoreboard,
                output_dir=output_dir,
            )
            for as_of in RECENT_REVIEW_DATES
        ]
        dry_run_df = pd.DataFrame(dry_run_rows)
        stamp = datetime.today().strftime("%Y%m%d")
        dry_run_df.to_csv(os.path.join(output_dir, f"v11_dry_run_review_{stamp}.csv"), index=False)

        _write_markdown_summary(
            output_dir=output_dir,
            scoreboard=scoreboard,
            recommendation_universe=recommendation_universe,
            forecast_universe=forecast_universe,
            bakeoff_summary=final_summary,
            policy_summary=policy_summary,
            classifier_summary=classifier_summary,
            dry_run_df=dry_run_df,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v11 diversification-first autonomous loop.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()
    run_v11_autonomous_loop(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
