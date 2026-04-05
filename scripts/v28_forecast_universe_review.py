"""v28 forecast-universe review for buyable-first benchmark pruning."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.reporting.decision_rendering import determine_recommendation_mode
from src.research.evaluation import (
    evaluate_baseline_strategy,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    summarize_predictions,
)
from src.research.policy_metrics import evaluate_policy_series
from src.research.v12 import aggregate_health_from_prediction_frames, signal_from_prediction
from src.research.v20 import summarize_v20_review, v20_model_specs
from src.research.v21 import common_historical_dates
from src.research.v28 import choose_v28_decision, v28_universe_manifest, v28_universe_specs


DEFAULT_OUTPUT_DIR = os.path.join("results", "v28")
DEFAULT_HORIZON = 6


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_dataset_map(
    conn: Any,
    df: pd.DataFrame,
    benchmarks: list[str],
    horizon: int,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, horizon)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue
        datasets[benchmark] = (X_aligned, y_aligned)
    return datasets


def _evaluate_model_prediction_series(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    model_type: str,
    feature_columns: list[str],
    benchmark: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    X_aligned, y_aligned = benchmark_data[benchmark]
    selected = [feature for feature in feature_columns if feature in X_aligned.columns]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        result, metrics = evaluate_wfo_model(
            X_aligned,
            y_aligned,
            model_type=model_type,
            benchmark=benchmark,
            target_horizon_months=DEFAULT_HORIZON,
            feature_columns=selected,
        )
    pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
    realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
    frame = pd.DataFrame({"y_hat": pred_series, "y_true": realized}).sort_index()
    summary = summarize_predictions(pred_series, realized, target_horizon_months=DEFAULT_HORIZON)
    sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
    return frame, {
        "benchmark": benchmark,
        "n_features": len(selected),
        "feature_columns": ",".join(selected),
        "mae": float(metrics["mae"]),
        "model_type": model_type,
        "policy_return_sign": float(sign_policy.mean_policy_return),
        "ic": float(summary.ic),
        "hit_rate": float(summary.hit_rate),
        "oos_r2": float(summary.oos_r2),
    }


def _combine_member_frames(member_frames: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    merged = member_frames[0][0].rename(columns={"y_hat": "pred_0"})
    for idx, (frame, _) in enumerate(member_frames[1:], start=1):
        merged = merged.join(frame[["y_hat"]].rename(columns={"y_hat": f"pred_{idx}"}), how="inner")
    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    weights = [1.0 / max(mae, 1e-9) ** 2 for _, mae in member_frames]
    total_weight = sum(weights)
    merged["y_hat"] = 0.0
    for idx, weight in enumerate(weights):
        merged["y_hat"] += merged[f"pred_{idx}"] * (weight / total_weight)
    return merged[["y_hat", "y_true"]]


def _build_prediction_maps(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    model_specs = v20_model_specs()
    required_specs = {
        "elasticnet_current": model_specs["elasticnet_current"],
        "ridge_current": model_specs["ridge_current"],
        "bayesian_ridge_current": model_specs["bayesian_ridge_current"],
        "gbt_current": model_specs["gbt_current"],
        "ridge_lean_v1__v18": model_specs["ridge_lean_v1__v18"],
        "gbt_lean_plus_two__v18": model_specs["gbt_lean_plus_two__v18"],
    }

    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    manifest_rows: list[dict[str, object]] = []
    for spec_name, spec in required_specs.items():
        model_frames[spec_name] = {}
        for benchmark in benchmark_data:
            frame, meta = _evaluate_model_prediction_series(
                benchmark_data=benchmark_data,
                model_type=spec.model_type,
                feature_columns=spec.features,
                benchmark=benchmark,
            )
            model_frames[spec_name][benchmark] = frame
            manifest_rows.append(
                {
                    "entry_name": spec_name,
                    "entry_type": spec.candidate_type,
                    "model_type": spec.model_type,
                    "benchmark": benchmark,
                    "n_features": meta["n_features"],
                    "feature_columns": meta["feature_columns"],
                    "mae": meta["mae"],
                    "policy_return_sign": meta["policy_return_sign"],
                    "ic": meta["ic"],
                    "hit_rate": meta["hit_rate"],
                    "oos_r2": meta["oos_r2"],
                    "notes": spec.notes,
                }
            )

    prediction_map: dict[str, dict[str, pd.DataFrame]] = {"shadow_baseline": {}}
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        evaluate_baseline_strategy(
            X_aligned,
            y_aligned,
            strategy=config.V13_SHADOW_BASELINE_STRATEGY,
            target_horizon_months=DEFAULT_HORIZON,
        )
        pred_series, realized = reconstruct_baseline_predictions(
            X_aligned,
            y_aligned,
            strategy=config.V13_SHADOW_BASELINE_STRATEGY,
            target_horizon_months=DEFAULT_HORIZON,
        )
        prediction_map["shadow_baseline"][benchmark] = pd.DataFrame(
            {"y_hat": pred_series, "y_true": realized}
        ).sort_index()

    ensemble_specs = {
        "live_production_ensemble_reduced": [
            "elasticnet_current",
            "ridge_current",
            "bayesian_ridge_current",
            "gbt_current",
        ],
        "ensemble_ridge_gbt_v18": ["ridge_lean_v1__v18", "gbt_lean_plus_two__v18"],
    }
    for ensemble_name, members in ensemble_specs.items():
        benchmark_frames: dict[str, pd.DataFrame] = {}
        for benchmark in benchmark_data:
            member_frames: list[tuple[pd.DataFrame, float]] = []
            for member_name in members:
                frame = model_frames[member_name][benchmark]
                manifest_match = next(
                    row for row in manifest_rows
                    if row["entry_name"] == member_name and row["benchmark"] == benchmark
                )
                member_frames.append((frame, float(manifest_match["mae"])))
            benchmark_frames[benchmark] = _combine_member_frames(member_frames)
        prediction_map[ensemble_name] = benchmark_frames

    return prediction_map, pd.DataFrame(manifest_rows)


def _build_path_snapshot(
    prediction_map: dict[str, dict[str, pd.DataFrame]],
    path_name: str,
    as_of: pd.Timestamp,
) -> dict[str, object]:
    signal_rows: list[dict[str, object]] = []
    pooled_frames: list[pd.DataFrame] = []

    for benchmark, frame in prediction_map[path_name].items():
        upto = frame.loc[:as_of]
        if upto.empty:
            continue
        current_pred = float(upto["y_hat"].iloc[-1])
        summary = summarize_predictions(upto["y_hat"], upto["y_true"], target_horizon_months=DEFAULT_HORIZON)
        signal_rows.append(
            {
                "benchmark": benchmark,
                "predicted_relative_return": current_pred,
                "ic": float(summary.ic),
                "hit_rate": float(summary.hit_rate),
                "signal": signal_from_prediction(current_pred),
            }
        )
        pooled_frames.append(upto[["y_hat", "y_true"]].reset_index(drop=True))

    if not signal_rows:
        raise ValueError(f"No signal rows available for path {path_name} at {as_of.date().isoformat()}.")

    signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(pooled_frames, DEFAULT_HORIZON)
    mean_pred = float(signals["predicted_relative_return"].mean())
    mean_ic = float(signals["ic"].mean())
    mean_hr = float(signals["hit_rate"].mean())
    outperform_count = int((signals["signal"] == "OUTPERFORM").sum())
    underperform_count = int((signals["signal"] == "UNDERPERFORM").sum())
    total = len(signals)
    if outperform_count > total / 2:
        consensus = "OUTPERFORM"
    elif underperform_count > total / 2:
        consensus = "UNDERPERFORM"
    else:
        consensus = "NEUTRAL"
    recommendation_mode = determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        representative_cpcv=None,
    )
    return {
        "as_of": as_of.date().isoformat(),
        "path_name": path_name,
        "consensus": consensus,
        "recommendation_mode": str(recommendation_mode["label"]),
        "sell_pct": float(recommendation_mode["sell_pct"]),
        "mean_predicted": mean_pred,
        "mean_ic": mean_ic,
        "mean_hit_rate": mean_hr,
        "aggregate_oos_r2": float(aggregate_health["oos_r2"]) if aggregate_health else float("nan"),
    }


def _build_review_detail(
    prediction_map: dict[str, dict[str, pd.DataFrame]],
    review_dates: list[pd.Timestamp],
    universe_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for as_of in review_dates:
        shadow = _build_path_snapshot(prediction_map, "shadow_baseline", as_of)
        live = _build_path_snapshot(prediction_map, "live_production_ensemble_reduced", as_of)
        rows.append(
            {
                **shadow,
                "universe_name": universe_name,
                "signal_agrees_with_shadow": True,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
            }
        )
        for path_name in ("live_production_ensemble_reduced", "ensemble_ridge_gbt_v18"):
            snapshot = live if path_name == "live_production_ensemble_reduced" else _build_path_snapshot(
                prediction_map,
                path_name,
                as_of,
            )
            rows.append(
                {
                    **snapshot,
                    "universe_name": universe_name,
                    "signal_agrees_with_shadow": snapshot["consensus"] == shadow["consensus"],
                    "mode_agrees_with_shadow": snapshot["recommendation_mode"] == shadow["recommendation_mode"],
                    "sell_agrees_with_shadow": snapshot["sell_pct"] == shadow["sell_pct"],
                    "signal_agrees_with_live": snapshot["consensus"] == live["consensus"],
                    "mode_agrees_with_live": snapshot["recommendation_mode"] == live["recommendation_mode"],
                    "sell_agrees_with_live": snapshot["sell_pct"] == live["sell_pct"],
                }
            )
    return pd.DataFrame(rows)


def _metric_summary(
    prediction_map: dict[str, dict[str, pd.DataFrame]],
    universe_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path_name, benchmark_map in prediction_map.items():
        for benchmark, frame in benchmark_map.items():
            summary = summarize_predictions(frame["y_hat"], frame["y_true"], target_horizon_months=DEFAULT_HORIZON)
            sign_policy = evaluate_policy_series(frame["y_hat"], frame["y_true"], "sign_hold_vs_sell")
            rows.append(
                {
                    "universe_name": universe_name,
                    "candidate_name": path_name,
                    "benchmark": benchmark,
                    "ic": float(summary.ic),
                    "hit_rate": float(summary.hit_rate),
                    "oos_r2": float(summary.oos_r2),
                    "mae": float(summary.mae),
                    "policy_return_sign": float(sign_policy.mean_policy_return),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    for candidate_name, group in detail.groupby("candidate_name", dropna=False):
        summary_rows.append(
            {
                "universe_name": universe_name,
                "candidate_name": candidate_name,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def run_v28_review() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)
    universe_specs = v28_universe_specs()

    metric_frames: list[pd.DataFrame] = []
    review_summaries: list[pd.DataFrame] = []
    review_details: list[pd.DataFrame] = []
    manifest = v28_universe_manifest()
    review_window_rows: list[dict[str, object]] = []

    for spec in universe_specs.values():
        benchmark_data = _benchmark_dataset_map(conn, df, spec.benchmarks, DEFAULT_HORIZON)
        prediction_map, _ = _build_prediction_maps(benchmark_data)
        metric_frames.append(_metric_summary(prediction_map, spec.universe_name))

        review_dates = common_historical_dates(prediction_map, list(benchmark_data.keys()))
        detail = _build_review_detail(prediction_map, review_dates, spec.universe_name)
        review_details.append(detail)
        summary = summarize_v20_review(detail)
        if not summary.empty:
            summary.insert(0, "universe_name", spec.universe_name)
            review_summaries.append(summary)
        review_window_rows.append(
            {
                "universe_name": spec.universe_name,
                "review_start": review_dates[0].date().isoformat() if review_dates else "",
                "review_end": review_dates[-1].date().isoformat() if review_dates else "",
                "review_months": len(review_dates),
            }
        )

    metric_summary = pd.concat(metric_frames, ignore_index=True)
    review_summary = pd.concat(review_summaries, ignore_index=True)
    review_detail = pd.concat(review_details, ignore_index=True)
    review_window = pd.DataFrame(review_window_rows)
    summary = review_summary.merge(
        metric_summary[["universe_name", "candidate_name", "mean_policy_return_sign", "mean_oos_r2", "mean_ic"]],
        left_on=["universe_name", "path_name"],
        right_on=["universe_name", "candidate_name"],
        how="left",
    ).drop(columns=["candidate_name"])
    summary = summary.merge(review_window, on="universe_name", how="left")
    summary = summary.merge(manifest, on="universe_name", how="left")
    decision = choose_v28_decision(summary)
    decision_df = pd.DataFrame(
        [
            {
                "status": decision.status,
                "recommended_universe": decision.recommended_universe,
                "rationale": decision.rationale,
            }
        ]
    )
    return summary, review_detail, manifest.merge(review_window, on="universe_name", how="left"), decision_df


def write_outputs(
    summary: pd.DataFrame,
    review_detail: pd.DataFrame,
    manifest: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    summary.to_csv(output_dir / f"v28_universe_summary_{stamp}.csv", index=False)
    review_detail.to_csv(output_dir / f"v28_universe_review_detail_{stamp}.csv", index=False)
    manifest.to_csv(output_dir / f"v28_universe_manifest_{stamp}.csv", index=False)
    decision_df.to_csv(output_dir / f"v28_decision_{stamp}.csv", index=False)

    decision = decision_df.iloc[0]
    lines = [
        "# V28 Forecast Universe Review",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Recommended universe: `{decision['recommended_universe']}`",
        "",
        decision["rationale"],
    ]
    _write_text(output_dir / f"v28_decision_{stamp}.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v28 forecast-universe review.")
    parser.parse_args()

    summary, review_detail, manifest, decision_df = run_v28_review()
    write_outputs(summary, review_detail, manifest, decision_df)
    decision = decision_df.iloc[0]
    print("v28 forecast-universe review complete.")
    print(f"Decision: {decision['status']} / {decision['recommended_universe']}")


if __name__ == "__main__":
    main()
