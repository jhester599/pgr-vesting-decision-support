"""v14 narrow prediction-layer bakeoff on a reduced benchmark universe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts.benchmark_reduction import build_benchmark_scorecard
from scripts.benchmark_suite import run_benchmark_suite
from scripts.candidate_model_bakeoff import candidate_feature_sets
from src.database import db_client
from src.models.multi_benchmark_wfo import get_ensemble_signals, run_ensemble_benchmarks
from src.processing.feature_engineering import build_feature_matrix_from_db, get_feature_columns, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.diversification import score_benchmarks_against_pgr
from src.research.evaluation import (
    BASELINE_STRATEGIES,
    classify_research_gate,
    evaluate_baseline_strategy,
    evaluate_ensemble_result,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    reconstruct_ensemble_oos_predictions,
)
from src.research.policy_metrics import evaluate_policy_series, hold_fraction_from_policy
from src.research.v11 import (
    add_destination_roles,
    choose_recommendation_universe,
    diversification_adjusted_policy_utility,
    mean_diversification_score,
    next_vest_after,
    recommend_redeploy_buckets,
    summarize_existing_holdings_actions,
)
from src.research.v12 import recent_monthly_review_dates, signal_from_prediction
from src.research.v14 import (
    V14_BALANCED_CORE7,
    UniverseSelection,
    choose_feature_surgery_candidates,
    count_signal_changes,
    select_best_universe,
)
from src.tax.capital_gains import load_position_lots


DEFAULT_OUTPUT_DIR = os.path.join("results", "v14")
DEFAULT_HORIZON = 6
DEFAULT_REVIEW_MONTHS = 6
DEFAULT_UNIVERSE_CANDIDATES: dict[str, list[str]] = {
    "v13_forecast_core9": list(config.V13_SHADOW_FORECAST_UNIVERSE),
    "v13_redeploy_core8": list(config.V13_REDEPLOY_UNIVERSE),
    "balanced_core7": list(V14_BALANCED_CORE7),
}
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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _latest_csv(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern))
    return files[-1] if files else None


def _candidate_specs() -> dict[str, dict[str, Any]]:
    base = candidate_feature_sets()
    return {
        "ridge_lean_v1": base["ridge_lean_v1"],
        "gbt_lean_plus_two": base["gbt_lean_plus_two"],
        "elasticnet_lean_v1": base["elasticnet_lean_v1"],
    }


def _build_scoreboard(conn: Any, output_dir: str) -> pd.DataFrame:
    v11_dir = Path("results") / "v11"
    existing = _latest_csv(v11_dir, "v11_diversification_scoreboard_*.csv")
    if existing is not None:
        return pd.read_csv(existing)

    detail_df, _ = run_benchmark_suite(
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
    return scoreboard


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

    from src.models.regularized_models import (
        build_elasticnet_pipeline,
        build_gbt_pipeline,
        build_ridge_pipeline,
    )

    if model_type == "elasticnet":
        pipeline = build_elasticnet_pipeline()
    elif model_type == "ridge":
        pipeline = build_ridge_pipeline()
    elif model_type == "gbt":
        pipeline = build_gbt_pipeline()
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    pipeline.fit(X_recent, y_recent)
    return float(pipeline.predict(X_curr)[0])


def _predict_current_baseline(
    y_full: pd.Series,
    strategy: str = "historical_mean",
    train_window_months: int = config.WFO_TRAIN_WINDOW_MONTHS,
) -> float:
    recent = y_full.dropna().iloc[-train_window_months:]
    if recent.empty:
        raise ValueError("No training target history available for baseline prediction.")
    if strategy == "historical_mean":
        return float(recent.mean())
    if strategy == "last_value":
        return float(recent.iloc[-1])
    if strategy == "zero":
        return 0.0
    raise ValueError(f"Unsupported baseline strategy '{strategy}'.")


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


def _evaluate_candidate_stack(
    conn: Any,
    df_full: pd.DataFrame,
    benchmarks: list[str],
    scoreboard: pd.DataFrame,
    candidate_specs: dict[str, dict[str, Any]],
    output_dir: str,
    output_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = set(get_feature_columns(df_full))
    detail_rows: list[dict[str, Any]] = []

    rel_matrix = pd.DataFrame(
        {
            benchmark: load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
            for benchmark in benchmarks
        }
    ).dropna(axis=1, how="all")
    ensemble_results = run_ensemble_benchmarks(
        df_full,
        rel_matrix,
        target_horizon_months=DEFAULT_HORIZON,
    )

    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df_full, rel_series, drop_na_target=True)
        except ValueError:
            continue

        model_predictions: dict[str, tuple[pd.Series, pd.Series, float]] = {}

        ens_result = ensemble_results.get(benchmark)
        if ens_result is not None:
            ens_metrics = evaluate_ensemble_result(ens_result, target_horizon_months=DEFAULT_HORIZON)
            ens_pred, ens_realized = reconstruct_ensemble_oos_predictions(ens_result)
            ens_sign = evaluate_policy_series(ens_pred, ens_realized, "sign_hold_vs_sell")
            ens_neutral3 = evaluate_policy_series(ens_pred, ens_realized, "neutral_band_3pct")
            ens_tiered = evaluate_policy_series(ens_pred, ens_realized, "tiered_25_50_100")
            detail_rows.append(
                {
                    "candidate_name": "live_production_ensemble",
                    "candidate_type": "live_ensemble",
                    "model_type": "ensemble",
                    "benchmark": benchmark,
                    "n_features": int(ens_metrics["n_features"]),
                    "feature_columns": "",
                    "policy_return_sign": ens_sign.mean_policy_return,
                    "policy_return_neutral_3pct": ens_neutral3.mean_policy_return,
                    "policy_return_tiered": ens_tiered.mean_policy_return,
                    "policy_uplift_vs_sell_50_neutral_3pct": ens_neutral3.uplift_vs_sell_50,
                    "avg_hold_neutral_3pct": ens_neutral3.avg_hold_fraction,
                    **ens_metrics,
                }
            )

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
            model_predictions[candidate_name] = (pred_series, realized, float(metrics["mae"]))
            sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
            neutral3 = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
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
                    "policy_return_neutral_3pct": neutral3.mean_policy_return,
                    "policy_return_tiered": tiered.mean_policy_return,
                    "policy_uplift_vs_sell_50_neutral_3pct": neutral3.uplift_vs_sell_50,
                    "avg_hold_neutral_3pct": neutral3.avg_hold_fraction,
                    **metrics,
                }
            )

        ensemble_members = ["ridge_lean_v1", "gbt_lean_plus_two"]
        if all(member in model_predictions for member in ensemble_members):
            frames: list[pd.DataFrame] = []
            feature_count = 0
            for member in ensemble_members:
                pred_series, realized, mae = model_predictions[member]
                feature_count += len(candidate_specs[member]["features"])
                frames.append(
                    pd.DataFrame(
                        {
                            f"pred_{member}__{mae if np.isfinite(mae) else 1.0}": pred_series,
                            "y_true": realized,
                        }
                    )
                )
            y_hat, y_true = _combine_prediction_frames(frames)
            if not y_hat.empty:
                from src.research.evaluation import summarize_predictions

                summary = summarize_predictions(y_hat, y_true, target_horizon_months=DEFAULT_HORIZON)
                sign_policy = evaluate_policy_series(y_hat, y_true, "sign_hold_vs_sell")
                neutral3 = evaluate_policy_series(y_hat, y_true, "neutral_band_3pct")
                tiered = evaluate_policy_series(y_hat, y_true, "tiered_25_50_100")
                detail_rows.append(
                    {
                        "candidate_name": "ensemble_ridge_gbt",
                        "candidate_type": "ensemble",
                        "model_type": "ridge+gbt",
                        "benchmark": benchmark,
                        "n_features": feature_count,
                        "feature_columns": ",".join(ensemble_members),
                        "policy_return_sign": sign_policy.mean_policy_return,
                        "policy_return_neutral_3pct": neutral3.mean_policy_return,
                        "policy_return_tiered": tiered.mean_policy_return,
                        "policy_uplift_vs_sell_50_neutral_3pct": neutral3.uplift_vs_sell_50,
                        "avg_hold_neutral_3pct": neutral3.avg_hold_fraction,
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
        baseline_sign = evaluate_policy_series(baseline_pred, baseline_realized, "sign_hold_vs_sell")
        baseline_neutral3 = evaluate_policy_series(baseline_pred, baseline_realized, "neutral_band_3pct")
        baseline_tiered = evaluate_policy_series(baseline_pred, baseline_realized, "tiered_25_50_100")
        detail_rows.append(
            {
                "candidate_name": "baseline_historical_mean",
                "candidate_type": "baseline",
                "model_type": "baseline",
                "benchmark": benchmark,
                "n_features": 0,
                "feature_columns": "",
                "policy_return_sign": baseline_sign.mean_policy_return,
                "policy_return_neutral_3pct": baseline_neutral3.mean_policy_return,
                "policy_return_tiered": baseline_tiered.mean_policy_return,
                "policy_uplift_vs_sell_50_neutral_3pct": baseline_neutral3.uplift_vs_sell_50,
                "avg_hold_neutral_3pct": baseline_neutral3.avg_hold_fraction,
                **baseline_metrics,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for candidate_name, group in detail_df.groupby("candidate_name", dropna=False):
        div_metrics = diversification_adjusted_policy_utility(
            group,
            scoreboard[scoreboard["benchmark"].isin(benchmarks)],
            policy_column="policy_return_neutral_3pct",
        )
        mean_ic = float(group["ic"].mean())
        mean_hit_rate = float(group["hit_rate"].mean())
        mean_oos_r2 = float(group["oos_r2"].mean())
        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_type": str(group["candidate_type"].iloc[0]),
                "model_type": str(group["model_type"].iloc[0]),
                "n_benchmarks": int(group["benchmark"].nunique()),
                "n_features": int(group["n_features"].max()),
                "mean_ic": mean_ic,
                "mean_hit_rate": mean_hit_rate,
                "mean_oos_r2": mean_oos_r2,
                "mean_mae": float(group["mae"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_neutral_3pct": float(group["policy_return_neutral_3pct"].mean()),
                "mean_policy_return_tiered": float(group["policy_return_tiered"].mean()),
                "mean_policy_uplift_vs_sell_50_neutral_3pct": float(group["policy_uplift_vs_sell_50_neutral_3pct"].mean()),
                "mean_avg_hold_neutral_3pct": float(group["avg_hold_neutral_3pct"].mean()),
                "weighted_policy_return": div_metrics["weighted_policy_return"],
                "contextual_penalty": div_metrics["contextual_penalty"],
                "diversification_aware_utility": div_metrics["diversification_aware_utility"],
                "mean_diversification_score": div_metrics["mean_diversification_score"],
                "gate_status": classify_research_gate(mean_oos_r2, mean_ic, mean_hit_rate),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["diversification_aware_utility", "mean_policy_return_neutral_3pct", "mean_oos_r2"],
        ascending=[False, False, False],
    )

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_df.to_csv(os.path.join(output_dir, f"{output_prefix}_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"{output_prefix}_summary_{stamp}.csv"), index=False)
    return detail_df, summary_df


def _evaluate_universe_candidates(
    conn: Any,
    df_full: pd.DataFrame,
    scoreboard: pd.DataFrame,
    output_dir: str,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    candidate_specs = _candidate_specs()
    universe_rows: list[dict[str, Any]] = []
    universe_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for universe_name, benchmarks in DEFAULT_UNIVERSE_CANDIDATES.items():
        detail_df, summary_df = _evaluate_candidate_stack(
            conn=conn,
            df_full=df_full,
            benchmarks=benchmarks,
            scoreboard=scoreboard,
            candidate_specs=candidate_specs,
            output_dir=output_dir,
            output_prefix=f"v14_{universe_name}",
        )
        universe_results[universe_name] = (detail_df, summary_df)
        nonbaseline = summary_df[~summary_df["candidate_name"].astype(str).str.startswith("baseline_")]
        best_nonbaseline = nonbaseline.iloc[0] if not nonbaseline.empty else summary_df.iloc[0]
        subset = scoreboard[scoreboard["benchmark"].isin(benchmarks)]
        universe_rows.append(
            {
                "universe_name": universe_name,
                "benchmarks": ",".join(benchmarks),
                "n_benchmarks": len(benchmarks),
                "contextual_only_count": int((subset["recommendation_role"] == "contextual_only").sum()),
                "mean_diversification_score": mean_diversification_score(scoreboard, benchmarks),
                "best_nonbaseline_candidate": best_nonbaseline["candidate_name"],
                "best_nonbaseline_policy_return": best_nonbaseline["mean_policy_return_neutral_3pct"],
                "best_nonbaseline_oos_r2": best_nonbaseline["mean_oos_r2"],
                "best_nonbaseline_div_utility": best_nonbaseline["diversification_aware_utility"],
            }
        )

    universe_summary = pd.DataFrame(universe_rows).sort_values(
        by=["best_nonbaseline_policy_return", "n_benchmarks", "best_nonbaseline_oos_r2"],
        ascending=[False, True, False],
    )
    stamp = datetime.today().strftime("%Y%m%d")
    universe_summary.to_csv(os.path.join(output_dir, f"v14_universe_selection_{stamp}.csv"), index=False)
    return universe_summary, universe_results


def _run_feature_surgery(
    conn: Any,
    df_full: pd.DataFrame,
    benchmarks: list[str],
    scoreboard: pd.DataFrame,
    output_dir: str,
    selected_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    candidate_specs = _candidate_specs()
    surgery_targets = choose_feature_surgery_candidates(selected_summary)
    feature_columns = set(get_feature_columns(df_full))
    detail_rows: list[dict[str, Any]] = []
    updated_specs: dict[str, dict[str, Any]] = {}

    for candidate_name in surgery_targets:
        if candidate_name not in FEATURE_SURGERY_QUEUES:
            continue
        base_spec = candidate_specs[candidate_name]
        base_summary = selected_summary[selected_summary["candidate_name"] == candidate_name]
        if base_summary.empty:
            continue
        base_policy = float(base_summary["mean_policy_return_neutral_3pct"].iloc[0])
        base_oos_r2 = float(base_summary["mean_oos_r2"].iloc[0])
        best_spec = base_spec
        best_policy = base_policy
        best_oos_r2 = base_oos_r2

        for operation, features in FEATURE_SURGERY_QUEUES[candidate_name].items():
            for feature in features:
                if operation == "add":
                    trial_features = _dedupe(list(base_spec["features"]) + [feature])
                else:
                    trial_features = [value for value in base_spec["features"] if value != feature]
                trial_features = [value for value in trial_features if value in feature_columns]
                trial_spec = {
                    "model_type": base_spec["model_type"],
                    "features": trial_features,
                    "notes": f"{candidate_name} {operation} {feature}",
                }
                _, summary_df = _evaluate_candidate_stack(
                    conn=conn,
                    df_full=df_full,
                    benchmarks=benchmarks,
                    scoreboard=scoreboard,
                    candidate_specs={candidate_name: trial_spec},
                    output_dir=output_dir,
                    output_prefix=f"v14_feature_surgery_{candidate_name}_{operation}_{feature}",
                )
                trial_row = summary_df.iloc[0]
                detail_rows.append(
                    {
                        "candidate_name": candidate_name,
                        "operation": operation,
                        "feature": feature,
                        "n_features": len(trial_features),
                        "mean_policy_return_neutral_3pct": trial_row["mean_policy_return_neutral_3pct"],
                        "mean_oos_r2": trial_row["mean_oos_r2"],
                        "diversification_aware_utility": trial_row["diversification_aware_utility"],
                    }
                )
                if (
                    float(trial_row["mean_policy_return_neutral_3pct"]) > best_policy
                    and float(trial_row["mean_oos_r2"]) >= best_oos_r2 - 0.02
                ):
                    best_spec = trial_spec
                    best_policy = float(trial_row["mean_policy_return_neutral_3pct"])
                    best_oos_r2 = float(trial_row["mean_oos_r2"])
        updated_specs[candidate_name] = best_spec

    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for candidate_name, spec in updated_specs.items():
        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "model_type": spec["model_type"],
                "n_features": len(spec["features"]),
                "feature_columns": ",".join(spec["features"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_df.to_csv(os.path.join(output_dir, f"v14_feature_surgery_detail_{stamp}.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"v14_feature_surgery_summary_{stamp}.csv"), index=False)
    return detail_df, summary_df, updated_specs


def _load_april_manifest() -> dict[str, Any]:
    manifest_path = Path("results") / "monthly_decisions" / "2026-04" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing production manifest at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_current_snapshot(
    conn: Any,
    df_full: pd.DataFrame,
    as_of: date,
    benchmarks: list[str],
    candidate_name: str,
    candidate_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    as_of_ts = pd.Timestamp(as_of)
    X_event = df_full.loc[df_full.index <= as_of_ts]
    X_current = X_event.iloc[[-1]]
    feature_columns = set(get_feature_columns(df_full))
    benchmark_rows: list[dict[str, Any]] = []

    if candidate_name == "live_production_ensemble":
        rel_matrix = pd.DataFrame(
            {
                benchmark: load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
                for benchmark in benchmarks
            }
        ).dropna(axis=1, how="all")
        ensemble_results = run_ensemble_benchmarks(
            X_event,
            rel_matrix,
            target_horizon_months=DEFAULT_HORIZON,
        )
        signal_df = get_ensemble_signals(X_event, rel_matrix, ensemble_results, X_current)
        for row in signal_df.reset_index().itertuples(index=False):
            ens_result = ensemble_results[str(row.benchmark)]
            benchmark_rows.append(
                {
                    "benchmark": row.benchmark,
                    "current_prediction": float(row.point_prediction),
                    "ic": float(ens_result.mean_ic),
                    "hit_rate": float(ens_result.mean_hit_rate),
                    "oos_r2": float(evaluate_ensemble_result(ens_result, target_horizon_months=DEFAULT_HORIZON)["oos_r2"]),
                }
            )
    else:
        for benchmark in benchmarks:
            rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
            if rel_series.empty:
                continue
            try:
                X_aligned, y_aligned = get_X_y_relative(X_event, rel_series, drop_na_target=True)
            except ValueError:
                continue

            if candidate_name == "baseline_historical_mean":
                current_prediction = _predict_current_baseline(y_aligned, strategy="historical_mean")
                metrics = evaluate_baseline_strategy(
                    X_aligned,
                    y_aligned,
                    strategy="historical_mean",
                    target_horizon_months=DEFAULT_HORIZON,
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
                continue

            if candidate_name == "ensemble_ridge_gbt":
                member_rows: list[tuple[float, float]] = []
                member_metrics: list[dict[str, float]] = []
                for member in ("ridge_lean_v1", "gbt_lean_plus_two"):
                    spec = candidate_specs[member]
                    selected = [feature for feature in spec["features"] if feature in feature_columns]
                    _, metrics = evaluate_wfo_model(
                        X_aligned,
                        y_aligned,
                        model_type=str(spec["model_type"]),
                        benchmark=benchmark,
                        target_horizon_months=DEFAULT_HORIZON,
                        feature_columns=selected,
                    )
                    current_prediction = _predict_current_custom(
                        X_full=X_aligned,
                        y_full=y_aligned,
                        X_current=X_current,
                        model_type=str(spec["model_type"]),
                        selected_features=selected,
                    )
                    member_rows.append((current_prediction, float(metrics["mae"])))
                    member_metrics.append(
                        {
                            "ic": float(metrics["ic"]),
                            "hit_rate": float(metrics["hit_rate"]),
                            "oos_r2": float(metrics["oos_r2"]),
                        }
                    )
                weights = [1.0 / max(mae, 1e-9) ** 2 for _, mae in member_rows]
                total_weight = sum(weights)
                combined_prediction = sum(pred * weight for (pred, _), weight in zip(member_rows, weights)) / total_weight
                benchmark_rows.append(
                    {
                        "benchmark": benchmark,
                        "current_prediction": combined_prediction,
                        "ic": float(np.mean([row["ic"] for row in member_metrics])),
                        "hit_rate": float(np.mean([row["hit_rate"] for row in member_metrics])),
                        "oos_r2": float(np.mean([row["oos_r2"] for row in member_metrics])),
                    }
                )
                continue

            spec = candidate_specs[candidate_name]
            selected = [feature for feature in spec["features"] if feature in feature_columns]
            _, metrics = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=str(spec["model_type"]),
                benchmark=benchmark,
                target_horizon_months=DEFAULT_HORIZON,
                feature_columns=selected,
            )
            current_prediction = _predict_current_custom(
                X_full=X_aligned,
                y_full=y_aligned,
                X_current=X_current,
                model_type=str(spec["model_type"]),
                selected_features=selected,
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

    snapshot_df = pd.DataFrame(benchmark_rows)
    mean_prediction = float(snapshot_df["current_prediction"].mean()) if not snapshot_df.empty else 0.0
    policy_name = config.V13_SHADOW_BASELINE_POLICY if candidate_name == "baseline_historical_mean" else "neutral_band_3pct"
    hold_fraction = float(hold_fraction_from_policy(pd.Series([mean_prediction]), policy_name).iloc[0])
    return {
        "candidate_name": candidate_name,
        "policy_name": policy_name,
        "signal": signal_from_prediction(mean_prediction, threshold=0.03),
        "sell_pct": 1.0 - hold_fraction,
        "mean_prediction": mean_prediction,
        "mean_ic": float(snapshot_df["ic"].mean()) if not snapshot_df.empty else float("nan"),
        "mean_hit_rate": float(snapshot_df["hit_rate"].mean()) if not snapshot_df.empty else float("nan"),
        "mean_oos_r2": float(snapshot_df["oos_r2"].mean()) if not snapshot_df.empty else float("nan"),
        "snapshot_df": snapshot_df,
    }


def _build_baseline_snapshot(
    conn: Any,
    df_full: pd.DataFrame,
    candidate_specs: dict[str, dict[str, Any]],
    output_dir: str,
) -> pd.DataFrame:
    manifest = _load_april_manifest()
    as_of = date.fromisoformat(str(manifest["as_of_date"]))
    live = _build_current_snapshot(
        conn=conn,
        df_full=df_full,
        as_of=as_of,
        benchmarks=list(config.V13_SHADOW_FORECAST_UNIVERSE),
        candidate_name="live_production_ensemble",
        candidate_specs=candidate_specs,
    )
    shadow = _build_current_snapshot(
        conn=conn,
        df_full=df_full,
        as_of=as_of,
        benchmarks=list(config.V13_SHADOW_FORECAST_UNIVERSE),
        candidate_name="baseline_historical_mean",
        candidate_specs=candidate_specs,
    )

    baseline_df = pd.DataFrame(
        [
            {
                "as_of": as_of.isoformat(),
                "recommendation_layer_mode": config.RECOMMENDATION_LAYER_MODE,
                "manifest_git_sha": str(manifest.get("git_sha", "")),
                "manifest_warnings": " | ".join(manifest.get("warnings", [])),
                "live_signal": live["signal"],
                "live_mean_prediction": live["mean_prediction"],
                "shadow_signal": shadow["signal"],
                "shadow_mean_prediction": shadow["mean_prediction"],
                "sell_pct_agreement": abs(float(live["sell_pct"]) - float(shadow["sell_pct"])) < 1e-9,
                "signal_agreement": live["signal"] == shadow["signal"],
            }
        ]
    )
    stamp = datetime.today().strftime("%Y%m%d")
    baseline_df.to_csv(os.path.join(output_dir, f"v14_baseline_snapshot_{stamp}.csv"), index=False)
    return baseline_df


def _shadow_review_dates(as_of: date, months: int) -> list[date]:
    dates = recent_monthly_review_dates(as_of, months=months - 1)
    if as_of not in dates:
        dates.append(as_of)
    return dates


def _choose_replacement_candidate(summary_df: pd.DataFrame) -> str:
    rows = summary_df[
        ~summary_df["candidate_name"].astype(str).isin({"baseline_historical_mean", "live_production_ensemble"})
    ].copy()
    if rows.empty:
        return "live_production_ensemble"
    rows = rows.sort_values(
        by=["diversification_aware_utility", "mean_policy_return_neutral_3pct", "mean_oos_r2"],
        ascending=[False, False, False],
    )
    return str(rows.iloc[0]["candidate_name"])


def _write_shadow_review_memo(
    memo_path: Path,
    as_of: date,
    live_snapshot: dict[str, Any],
    candidate_snapshot: dict[str, Any],
    existing_holdings: list[dict[str, Any]],
    redeploy_buckets: list[dict[str, Any]],
    next_vest_date: date,
    next_vest_type: str,
) -> None:
    lines = [
        f"# V14 Shadow Review - {as_of.isoformat()}",
        "",
        "## Recommendation Layer",
        "",
        f"- Active recommendation layer remains `baseline_historical_mean + {config.V13_SHADOW_BASELINE_POLICY}`.",
        f"- Next vest: **{next_vest_date.isoformat()}** (`{next_vest_type}`), default sell **50%** unless model quality improves materially.",
        "",
        "## Prediction-Layer Cross-Check",
        "",
        "| Path | Signal | Predicted 6M Relative Return | Mean IC | Mean Hit Rate | Mean OOS R^2 |",
        "|------|--------|-----------------------------|---------|---------------|--------------|",
        f"| Live 4-model stack | {live_snapshot['signal']} | {live_snapshot['mean_prediction']:+.2%} | {live_snapshot['mean_ic']:.4f} | {live_snapshot['mean_hit_rate']:.1%} | {live_snapshot['mean_oos_r2']:.2%} |",
        f"| V14 candidate `{candidate_snapshot['candidate_name']}` | {candidate_snapshot['signal']} | {candidate_snapshot['mean_prediction']:+.2%} | {candidate_snapshot['mean_ic']:.4f} | {candidate_snapshot['mean_hit_rate']:.1%} | {candidate_snapshot['mean_oos_r2']:.2%} |",
        "",
        f"- Agreement: **{'yes' if live_snapshot['signal'] == candidate_snapshot['signal'] else 'no'}**",
        "",
        "## Existing Holdings Guidance",
        "",
    ]
    for row in existing_holdings[:5]:
        lines.append(
            "- "
            f"{row['tax_bucket']}: {row['vest_date']} @ ${row['cost_basis_per_share']:.2f} "
            f"({row['shares']:.2f} share(s)). {row['rationale']}"
        )
    lines += ["", "## Redeploy Guidance", ""]
    for bucket in redeploy_buckets:
        lines.append(f"- `{bucket['bucket']}`: {bucket['example_funds']}. {bucket['note']}")
    lines.append("")
    memo_path.write_text("\n".join(lines), encoding="utf-8")


def _run_shadow_reviews(
    conn: Any,
    df_full: pd.DataFrame,
    scoreboard: pd.DataFrame,
    benchmarks: list[str],
    candidate_name: str,
    candidate_specs: dict[str, dict[str, Any]],
    output_dir: str,
    review_months: int,
) -> pd.DataFrame:
    as_of = date.fromisoformat(str(_load_april_manifest()["as_of_date"]))
    review_dates = _shadow_review_dates(as_of, review_months)
    redeploy_universe = choose_recommendation_universe(scoreboard)
    redeploy_buckets = recommend_redeploy_buckets(scoreboard, redeploy_universe)
    lots = load_position_lots(os.path.join(config.DATA_PROCESSED_DIR, "position_lots.csv"))
    price_row = pd.read_sql_query(
        "SELECT date, close FROM daily_prices WHERE ticker='PGR' ORDER BY date DESC LIMIT 1",
        conn,
    ).iloc[0]
    lot_actions = [
        {
            "vest_date": action.vest_date,
            "shares": action.shares,
            "cost_basis_per_share": action.cost_basis_per_share,
            "tax_bucket": action.tax_bucket,
            "rationale": action.rationale,
        }
        for action in summarize_existing_holdings_actions(
            lots,
            current_price=float(price_row["close"]),
            sell_date=as_of,
        )
    ]

    review_rows: list[dict[str, Any]] = []
    memo_dir = Path(output_dir) / "shadow_reviews"
    memo_dir.mkdir(parents=True, exist_ok=True)
    for review_date in review_dates:
        live_snapshot = _build_current_snapshot(
            conn=conn,
            df_full=df_full,
            as_of=review_date,
            benchmarks=benchmarks,
            candidate_name="live_production_ensemble",
            candidate_specs=candidate_specs,
        )
        candidate_snapshot = _build_current_snapshot(
            conn=conn,
            df_full=df_full,
            as_of=review_date,
            benchmarks=benchmarks,
            candidate_name=candidate_name,
            candidate_specs=candidate_specs,
        )
        next_vest_date, next_vest_type = next_vest_after(review_date)
        memo_path = memo_dir / f"{review_date.isoformat()}.md"
        _write_shadow_review_memo(
            memo_path=memo_path,
            as_of=review_date,
            live_snapshot=live_snapshot,
            candidate_snapshot=candidate_snapshot,
            existing_holdings=lot_actions,
            redeploy_buckets=redeploy_buckets,
            next_vest_date=next_vest_date,
            next_vest_type=next_vest_type,
        )
        review_rows.append(
            {
                "as_of": review_date.isoformat(),
                "live_signal": live_snapshot["signal"],
                "candidate_signal": candidate_snapshot["signal"],
                "live_mean_prediction": live_snapshot["mean_prediction"],
                "candidate_mean_prediction": candidate_snapshot["mean_prediction"],
                "signal_agreement": live_snapshot["signal"] == candidate_snapshot["signal"],
                "memo_path": str(memo_path),
            }
        )

    review_df = pd.DataFrame(review_rows)
    stamp = datetime.today().strftime("%Y%m%d")
    review_df.to_csv(os.path.join(output_dir, f"v14_shadow_review_{stamp}.csv"), index=False)
    return review_df


def _write_markdown_outputs(
    output_dir: str,
    baseline_df: pd.DataFrame,
    selected_universe: UniverseSelection,
    final_summary: pd.DataFrame,
    shadow_review_df: pd.DataFrame,
) -> None:
    best_row = final_summary.iloc[0]
    baseline_row = final_summary[final_summary["candidate_name"] == "baseline_historical_mean"]
    live_row = final_summary[final_summary["candidate_name"] == "live_production_ensemble"]
    replacement_rows = final_summary[
        ~final_summary["candidate_name"].astype(str).isin({"baseline_historical_mean", "live_production_ensemble"})
    ]
    replacement_row = replacement_rows.iloc[0] if not replacement_rows.empty else live_row.iloc[0]

    summary_path = Path("docs") / "results" / "V14_RESULTS_SUMMARY.md"
    closeout_path = Path("docs") / "closeouts" / "V14_CLOSEOUT_AND_V15_NEXT.md"

    shadow_signal_changes = count_signal_changes(shadow_review_df["candidate_signal"].astype(str).tolist())
    live_signal_changes = count_signal_changes(shadow_review_df["live_signal"].astype(str).tolist())
    agreement_rate = float(shadow_review_df["signal_agreement"].mean()) if not shadow_review_df.empty else float("nan")
    replacement_policy = float(replacement_row["mean_policy_return_neutral_3pct"])
    baseline_policy = float(baseline_row.iloc[0]["mean_policy_return_neutral_3pct"])
    live_policy = float(live_row.iloc[0]["mean_policy_return_neutral_3pct"])
    replacement_oos_r2 = float(replacement_row["mean_oos_r2"])
    live_oos_r2 = float(live_row.iloc[0]["mean_oos_r2"])

    summary_lines = [
        "# V14 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Headline",
        "",
        "v14 tested whether the underlying prediction layer can be simplified or replaced on a reduced, diversification-aware benchmark universe without changing the promoted v13.1 recommendation layer.",
        "",
        "## Post-v13 Baseline Snapshot",
        "",
        f"- April production as-of date: `{baseline_df.iloc[0]['as_of']}`",
        f"- Recommendation-layer mode: `{baseline_df.iloc[0]['recommendation_layer_mode']}`",
        f"- Live vs simpler-baseline signal agreement at freeze point: `{bool(baseline_df.iloc[0]['signal_agreement'])}`",
        f"- Live vs simpler-baseline sell agreement at freeze point: `{bool(baseline_df.iloc[0]['sell_pct_agreement'])}`",
        "",
        "## Universe Selection",
        "",
        f"- Selected v14 forecast universe: `{selected_universe.universe_name}`",
        f"- Benchmarks: `{', '.join(selected_universe.benchmarks)}`",
        "",
        "## Final Candidate Table",
        "",
        f"- Best overall row: `{best_row['candidate_name']}`",
        f"- Best replacement candidate: `{replacement_row['candidate_name']}`",
        f"- Reduced-universe live stack policy / OOS R^2: `{live_policy:.4f}` / `{live_oos_r2:.4f}`",
        f"- Historical-mean baseline policy / OOS R^2: `{baseline_policy:.4f}` / `{float(baseline_row.iloc[0]['mean_oos_r2']):.4f}`",
        f"- Replacement candidate policy / OOS R^2: `{replacement_policy:.4f}` / `{replacement_oos_r2:.4f}`",
        "",
        "## Shadow Review Window",
        "",
        f"- Review snapshots: `{len(shadow_review_df)}`",
        f"- Live signal changes: `{live_signal_changes}`",
        f"- Candidate signal changes: `{shadow_signal_changes}`",
        f"- Live / candidate agreement rate: `{agreement_rate:.1%}`",
        "",
        "## Key Conclusion",
        "",
    ]
    if replacement_policy > live_policy and replacement_policy >= baseline_policy - 0.002:
        summary_lines.append(
            f"`{replacement_row['candidate_name']}` improves on the reduced-universe live stack and stays within range of the historical-mean baseline. It is the narrow candidate worth carrying into v15 feature-replacement work."
        )
    else:
        summary_lines.append(
            f"`{replacement_row['candidate_name']}` is the best replacement candidate, but it still does not clearly beat the simpler baseline. v14 therefore supports continued research rather than immediate model-stack promotion."
        )
    summary_lines += ["", f"Detailed CSV outputs are stored in `{output_dir}`.", ""]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    closeout_lines = [
        "# V14 Closeout And V15 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Status",
        "",
        "The v14 reduced-universe prediction-layer study is complete.",
        "",
        "## Recommendation",
        "",
    ]
    if replacement_policy > live_policy and replacement_policy >= baseline_policy - 0.002 and replacement_oos_r2 >= live_oos_r2:
        closeout_lines += [
            f"- Continue shadowing `{replacement_row['candidate_name']}` as the leading replacement candidate.",
            "- Do not replace the promoted v13.1 recommendation layer.",
            "- Use v15 as a fixed-budget feature-replacement cycle for the surviving lean candidate(s).",
        ]
    else:
        closeout_lines += [
            "- Do not promote a new prediction layer yet.",
            f"- Best replacement candidate: `{replacement_row['candidate_name']}`",
            "- Main blocker: the reduced-universe candidate still does not decisively beat the historical-mean baseline while keeping the story simpler than the current live stack.",
            "- Use v15 for fixed-budget feature replacement rather than broader methodology expansion.",
        ]
    closeout_lines += [
        "",
        "## v15 Direction",
        "",
        "- keep the v13.1 recommendation layer fixed",
        f"- start from `{replacement_row['candidate_name']}` and only swap features one-for-one or in strict fixed-budget fashion",
        "- include both PGR-specific features and benchmark-predictive regime features in the candidate ideation list",
        "- only promote a replacement candidate if it beats the current live stack and is meaningfully closer to or ahead of the historical-mean baseline",
        "",
    ]
    closeout_path.write_text("\n".join(closeout_lines), encoding="utf-8")


def run_v14_prediction_layer_study(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    review_months: int = DEFAULT_REVIEW_MONTHS,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df_full = build_feature_matrix_from_db(conn)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

        scoreboard = _build_scoreboard(conn, output_dir)
        stamp = datetime.today().strftime("%Y%m%d")
        scoreboard.to_csv(os.path.join(output_dir, f"v14_diversification_scoreboard_{stamp}.csv"), index=False)

        candidate_specs = _candidate_specs()
        baseline_df = _build_baseline_snapshot(conn, df_full, candidate_specs, output_dir)
        universe_summary, _ = _evaluate_universe_candidates(
            conn=conn,
            df_full=df_full,
            scoreboard=scoreboard,
            output_dir=output_dir,
        )
        selected_universe = select_best_universe(universe_summary)

        _, selected_summary = _evaluate_candidate_stack(
            conn=conn,
            df_full=df_full,
            benchmarks=selected_universe.benchmarks,
            scoreboard=scoreboard,
            candidate_specs=candidate_specs,
            output_dir=output_dir,
            output_prefix="v14_selected_universe",
        )
        _, surgery_summary, updated_specs = _run_feature_surgery(
            conn=conn,
            df_full=df_full,
            benchmarks=selected_universe.benchmarks,
            scoreboard=scoreboard,
            output_dir=output_dir,
            selected_summary=selected_summary,
        )

        final_specs = candidate_specs.copy()
        for row in surgery_summary.itertuples(index=False):
            final_specs[str(row.candidate_name)] = {
                "model_type": str(row.model_type),
                "features": [value for value in str(row.feature_columns).split(",") if value],
                "notes": "Updated via v14 minimal feature surgery.",
            }
        final_specs.update(updated_specs)

        _, final_summary = _evaluate_candidate_stack(
            conn=conn,
            df_full=df_full,
            benchmarks=selected_universe.benchmarks,
            scoreboard=scoreboard,
            candidate_specs=final_specs,
            output_dir=output_dir,
            output_prefix="v14_final_candidate_bakeoff",
        )
        replacement_candidate = _choose_replacement_candidate(final_summary)
        shadow_review_df = _run_shadow_reviews(
            conn=conn,
            df_full=df_full,
            scoreboard=scoreboard,
            benchmarks=selected_universe.benchmarks,
            candidate_name=replacement_candidate,
            candidate_specs=final_specs,
            output_dir=output_dir,
            review_months=review_months,
        )
        _write_markdown_outputs(
            output_dir=output_dir,
            baseline_df=baseline_df,
            selected_universe=selected_universe,
            final_summary=final_summary,
            shadow_review_df=shadow_review_df,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v14 reduced-universe prediction-layer study.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--review-months",
        type=int,
        default=DEFAULT_REVIEW_MONTHS,
        help=f"Number of recent monthly snapshots to review. Default: {DEFAULT_REVIEW_MONTHS}",
    )
    args = parser.parse_args()
    run_v14_prediction_layer_study(
        output_dir=args.output_dir,
        review_months=args.review_months,
    )


if __name__ == "__main__":
    main()
