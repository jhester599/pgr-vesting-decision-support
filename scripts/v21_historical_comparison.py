"""v21 historical comparison study."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, datetime
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
from src.research.v20 import V20_FORECAST_UNIVERSE, summarize_v20_review, v20_model_specs
from src.research.v21 import (
    V21_REVIEW_PATHS,
    choose_v21_decision,
    common_historical_dates,
    summarize_v21_slices,
    v21_review_ensemble_specs,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v21")
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
    return frame, {
        "benchmark": benchmark,
        "n_features": len(selected),
        "feature_columns": ",".join(selected),
        "mae": float(metrics["mae"]),
        "model_type": model_type,
    }


def _combine_member_frames(
    member_frames: list[tuple[pd.DataFrame, float]],
) -> pd.DataFrame:
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
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]:
    model_specs = v20_model_specs()
    ensemble_specs = v21_review_ensemble_specs()

    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    model_manifest_rows: list[dict[str, object]] = []
    for spec_name, spec in model_specs.items():
        model_frames[spec_name] = {}
        for benchmark in V20_FORECAST_UNIVERSE:
            if benchmark not in benchmark_data:
                continue
            frame, meta = _evaluate_model_prediction_series(
                benchmark_data=benchmark_data,
                model_type=spec.model_type,
                feature_columns=spec.features,
                benchmark=benchmark,
            )
            model_frames[spec_name][benchmark] = frame
            model_manifest_rows.append(
                {
                    "entry_name": spec_name,
                    "entry_type": spec.candidate_type,
                    "model_type": spec.model_type,
                    "benchmark": benchmark,
                    "n_features": meta["n_features"],
                    "feature_columns": meta["feature_columns"],
                    "mae": meta["mae"],
                    "notes": spec.notes,
                }
            )

    prediction_map: dict[str, dict[str, pd.DataFrame]] = {
        "shadow_baseline": {},
    }
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

    for ensemble_name, ensemble_spec in ensemble_specs.items():
        benchmark_frames: dict[str, pd.DataFrame] = {}
        for benchmark in V20_FORECAST_UNIVERSE:
            if benchmark not in benchmark_data:
                continue
            member_frames: list[tuple[pd.DataFrame, float]] = []
            for member_name in list(ensemble_spec["members"]):
                frame = model_frames[member_name][benchmark]
                manifest_match = next(
                    row for row in model_manifest_rows
                    if row["entry_name"] == member_name and row["benchmark"] == benchmark
                )
                member_frames.append((frame, float(manifest_match["mae"])))
            benchmark_frames[benchmark] = _combine_member_frames(member_frames)
        prediction_map[ensemble_name] = benchmark_frames

    return prediction_map, {
        "models": pd.DataFrame(model_manifest_rows),
    }


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
        summary = summarize_predictions(
            upto["y_hat"],
            upto["y_true"],
            target_horizon_months=DEFAULT_HORIZON,
        )
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
        "aggregate_nw_ic": float(aggregate_health["nw_ic"]) if aggregate_health else float("nan"),
    }


def _build_review_detail(
    prediction_map: dict[str, dict[str, pd.DataFrame]],
    review_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for as_of in review_dates:
        shadow = _build_path_snapshot(prediction_map, "shadow_baseline", as_of)
        live = _build_path_snapshot(prediction_map, "live_production_ensemble_reduced", as_of)
        shadow_row = {
            **shadow,
            "signal_agrees_with_shadow": True,
            "mode_agrees_with_shadow": True,
            "sell_agrees_with_shadow": True,
            "signal_agrees_with_live": shadow["consensus"] == live["consensus"],
            "mode_agrees_with_live": shadow["recommendation_mode"] == live["recommendation_mode"],
            "sell_agrees_with_live": abs(float(shadow["sell_pct"]) - float(live["sell_pct"])) < 1e-9,
        }
        live_row = {
            **live,
            "signal_agrees_with_shadow": live["consensus"] == shadow["consensus"],
            "mode_agrees_with_shadow": live["recommendation_mode"] == shadow["recommendation_mode"],
            "sell_agrees_with_shadow": abs(float(live["sell_pct"]) - float(shadow["sell_pct"])) < 1e-9,
            "signal_agrees_with_live": True,
            "mode_agrees_with_live": True,
            "sell_agrees_with_live": True,
        }
        rows.extend([shadow_row, live_row])
        for path_name in V21_REVIEW_PATHS:
            if path_name == "live_production_ensemble_reduced":
                continue
            snapshot = _build_path_snapshot(prediction_map, path_name, as_of)
            rows.append(
                {
                    **snapshot,
                    "signal_agrees_with_shadow": snapshot["consensus"] == shadow["consensus"],
                    "mode_agrees_with_shadow": snapshot["recommendation_mode"] == shadow["recommendation_mode"],
                    "sell_agrees_with_shadow": abs(float(snapshot["sell_pct"]) - float(shadow["sell_pct"])) < 1e-9,
                    "signal_agrees_with_live": snapshot["consensus"] == live["consensus"],
                    "mode_agrees_with_live": snapshot["recommendation_mode"] == live["recommendation_mode"],
                    "sell_agrees_with_live": abs(float(snapshot["sell_pct"]) - float(live["sell_pct"])) < 1e-9,
                }
            )
    return pd.DataFrame(rows)


def _summarize_candidate_metrics(prediction_map: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path_name, benchmark_map in prediction_map.items():
        path_type = "baseline" if path_name == "shadow_baseline" else "ensemble"
        model_type = "baseline" if path_name == "shadow_baseline" else "ensemble"
        metrics_rows: list[dict[str, float]] = []
        for benchmark, frame in benchmark_map.items():
            summary = summarize_predictions(
                frame["y_hat"],
                frame["y_true"],
                target_horizon_months=DEFAULT_HORIZON,
            )
            sign_policy = evaluate_policy_series(frame["y_hat"], frame["y_true"], "sign_hold_vs_sell")
            neutral_policy = evaluate_policy_series(frame["y_hat"], frame["y_true"], "neutral_band_3pct")
            metrics_rows.append(
                {
                    "benchmark": benchmark,
                    "current_predicted_return": float(frame["y_hat"].iloc[-1]),
                    "ic": float(summary.ic),
                    "hit_rate": float(summary.hit_rate),
                    "oos_r2": float(summary.oos_r2),
                    "nw_ic": float(summary.nw_ic),
                    "mae": float(summary.mae),
                    "policy_return_sign": float(sign_policy.mean_policy_return),
                    "policy_return_neutral_3pct": float(neutral_policy.mean_policy_return),
                }
            )
        group = pd.DataFrame(metrics_rows)
        rows.append(
            {
                "candidate_name": path_name if path_name != "shadow_baseline" else "baseline_historical_mean",
                "candidate_type": path_type,
                "model_type": model_type,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_current_predicted_return": float(group["current_predicted_return"].mean()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_nw_ic": float(group["nw_ic"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_neutral_3pct": float(group["policy_return_neutral_3pct"].mean()),
                "mean_mae": float(group["mae"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_v21_historical_comparison(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    horizon: int = DEFAULT_HORIZON,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)
    benchmark_data = _benchmark_dataset_map(conn, df, list(V20_FORECAST_UNIVERSE), horizon)
    prediction_map, manifests = _build_prediction_maps(benchmark_data)
    review_dates = common_historical_dates(prediction_map, list(V20_FORECAST_UNIVERSE))
    review_detail_df = _build_review_detail(prediction_map, review_dates)
    review_summary_df = summarize_v20_review(review_detail_df)
    slice_summary_df = summarize_v21_slices(
        review_detail_df,
        {
            "full_history": (None, None),
            "pre_2020": (None, "2019-12-31"),
            "post_2020": ("2020-01-01", None),
            "post_2022": ("2022-01-01", None),
        },
    )
    metric_summary_df = _summarize_candidate_metrics(prediction_map)
    decision = choose_v21_decision(metric_summary_df, review_summary_df, slice_summary_df)
    decision_df = pd.DataFrame(
        [{"status": decision.status, "recommended_path": decision.recommended_path, "rationale": decision.rationale}]
    )

    stamp = datetime.today().strftime("%Y%m%d")
    metric_summary_path = Path(output_dir) / f"v21_candidate_metric_summary_{stamp}.csv"
    review_detail_path = Path(output_dir) / f"v21_historical_review_detail_{stamp}.csv"
    review_summary_path = Path(output_dir) / f"v21_historical_review_summary_{stamp}.csv"
    slice_summary_path = Path(output_dir) / f"v21_slice_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v21_promotion_decision_{stamp}.csv"
    manifest_path = Path(output_dir) / f"v21_model_manifest_{stamp}.csv"
    metric_summary_df.to_csv(metric_summary_path, index=False)
    review_detail_df.to_csv(review_detail_path, index=False)
    review_summary_df.to_csv(review_summary_path, index=False)
    slice_summary_df.to_csv(slice_summary_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    manifests["models"].to_csv(manifest_path, index=False)

    top_row = metric_summary_df.iloc[0]
    live_review = review_summary_df[review_summary_df["path_name"] == "live_production_ensemble_reduced"].iloc[0]
    candidate_review = review_summary_df[review_summary_df["path_name"] == decision.recommended_path]
    candidate_lines: list[str] = []
    if not candidate_review.empty:
        row = candidate_review.iloc[0]
        candidate_lines = [
            "",
            "## Recommended-Path Historical Behavior",
            "",
            f"- Path: `{decision.recommended_path}`",
            f"- Signal agreement with shadow baseline: `{float(row['signal_agreement_with_shadow_rate']):.1%}`",
            f"- Signal agreement with live cross-check: `{float(row['signal_agreement_with_live_rate']):.1%}`",
            f"- Signal changes: `{int(row['signal_changes'])}`",
            f"- Outperform / neutral / underperform mix: `{float(row['outperform_rate']):.1%}` / `{float(row['neutral_rate']):.1%}` / `{float(row['underperform_rate']):.1%}`",
        ]

    result_lines = [
        "# V21 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v21 replaces the recent-window shadow gate with a point-in-time historical comparison over the full evaluable period.",
        "- It compares the current live reduced cross-check and the leading v16-v20 assembled candidates against the promoted simpler baseline.",
        "",
        "## Historical Window",
        "",
        f"- Common evaluable monthly dates: `{len(review_dates)}`",
        f"- First common date: `{review_dates[0].date().isoformat() if review_dates else 'n/a'}`",
        f"- Last common date: `{review_dates[-1].date().isoformat() if review_dates else 'n/a'}`",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended path: `{decision.recommended_path}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Top Metric Row",
        "",
        f"- Candidate: `{top_row['candidate_name']}`",
        f"- Mean sign-policy return: `{float(top_row['mean_policy_return_sign']):.4f}`",
        f"- Mean OOS R^2: `{float(top_row['mean_oos_r2']):.4f}`",
        f"- Mean IC: `{float(top_row['mean_ic']):.4f}`",
        f"- Mean hit rate: `{float(top_row['mean_hit_rate']):.4f}`",
        "",
        "## Current Live Historical Behavior",
        "",
        f"- Signal agreement with shadow baseline: `{float(live_review['signal_agreement_with_shadow_rate']):.1%}`",
        f"- Signal changes: `{int(live_review['signal_changes'])}`",
        f"- Outperform / neutral / underperform mix: `{float(live_review['outperform_rate']):.1%}` / `{float(live_review['neutral_rate']):.1%}` / `{float(live_review['underperform_rate']):.1%}`",
    ]
    result_lines.extend(candidate_lines)
    result_lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            f"- `results/v21/{metric_summary_path.name}`",
            f"- `results/v21/{review_detail_path.name}`",
            f"- `results/v21/{review_summary_path.name}`",
            f"- `results/v21/{slice_summary_path.name}`",
            f"- `results/v21/{decision_path.name}`",
            f"- `results/v21/{manifest_path.name}`",
        ]
    )
    _write_text(Path("docs") / "results" / "V21_RESULTS_SUMMARY.md", result_lines)

    closeout_lines = [
        "# V21 Closeout And V22 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v21 replaced the recent-window promotion gate with a point-in-time historical comparison over the full common evaluable period.",
        "",
        "## Result",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended path: `{decision.recommended_path}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Recommended V22 Scope",
        "",
        "- If the current live cross-check still wins historically, avoid another generic feature sweep.",
        "- Focus next on blocked-source expansion or narrow calibration / sign-bias diagnostics.",
    ]
    _write_text(Path("docs") / "closeouts" / "V21_CLOSEOUT_AND_V22_NEXT.md", closeout_lines)

    plan_lines = [
        "# codex-v21-plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Goal",
        "",
        "- Re-evaluate the leading post-v19 candidate stacks over the full historically evaluable period instead of the recent 12-month window.",
        "",
        "## Paths Compared",
        "",
        "- `shadow_baseline`",
        "- `live_production_ensemble_reduced`",
        "- `ensemble_ridge_gbt_v16`",
        "- `ensemble_ridge_gbt_v18`",
        "- `ensemble_ridge_gbt_v20_value`",
        "- `ensemble_ridge_gbt_v20_best`",
        "",
        "## Gate",
        "",
        "- Prefer a candidate only if it improves metrics and matches or exceeds the live path's historical agreement with the promoted simpler baseline.",
    ]
    _write_text(Path("docs") / "plans" / "codex-v21-plan.md", plan_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v21 historical comparison study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--horizon", default=str(DEFAULT_HORIZON), help="Target horizon in months.")
    args = parser.parse_args()
    run_v21_historical_comparison(output_dir=args.output_dir, horizon=int(args.horizon))


if __name__ == "__main__":
    main()
