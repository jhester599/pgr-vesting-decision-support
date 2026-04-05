"""v23 extended-history proxy study for the reduced forecast universe."""

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
from src.research.v21 import common_historical_dates, summarize_v21_slices
from src.research.v23 import (
    V23_REVIEW_PATHS,
    build_extended_relative_return_series,
    choose_v23_decision,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v23")
DEFAULT_HORIZON = 6


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_dataset_map(
    df: pd.DataFrame,
    relative_series_map: dict[str, pd.Series],
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for benchmark, rel_series in relative_series_map.items():
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
    ensemble_specs = {
        "live_production_ensemble_reduced": {
            "members": ["elasticnet_current", "ridge_current", "bayesian_ridge_current", "gbt_current"],
            "notes": "Current deployed 4-model stack on the reduced universe.",
        },
        "ensemble_ridge_gbt_v18": {
            "members": ["ridge_lean_v1__v18", "gbt_lean_plus_two__v18"],
            "notes": "v21-promoted reduced-universe candidate.",
        },
        "ensemble_ridge_gbt_v20_best": {
            "members": ["ridge_lean_v1__v20_value", "gbt_lean_plus_two__v18"],
            "notes": "Best-of-confirmed stack carried forward for v23 comparison.",
        },
    }

    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    manifest_rows: list[dict[str, object]] = []
    for spec_name, spec in model_specs.items():
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

    for ensemble_name, ensemble_spec in ensemble_specs.items():
        benchmark_frames: dict[str, pd.DataFrame] = {}
        for benchmark in benchmark_data:
            member_frames: list[tuple[pd.DataFrame, float]] = []
            for member_name in list(ensemble_spec["members"]):
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
        rows.append(
            {
                **shadow,
                "signal_agrees_with_shadow": True,
                "mode_agrees_with_shadow": True,
                "sell_agrees_with_shadow": True,
                "signal_agrees_with_live": shadow["consensus"] == live["consensus"],
                "mode_agrees_with_live": shadow["recommendation_mode"] == live["recommendation_mode"],
                "sell_agrees_with_live": abs(float(shadow["sell_pct"]) - float(live["sell_pct"])) < 1e-9,
            }
        )
        rows.append(
            {
                **live,
                "signal_agrees_with_shadow": live["consensus"] == shadow["consensus"],
                "mode_agrees_with_shadow": live["recommendation_mode"] == shadow["recommendation_mode"],
                "sell_agrees_with_shadow": abs(float(live["sell_pct"]) - float(shadow["sell_pct"])) < 1e-9,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
            }
        )
        for path_name in V23_REVIEW_PATHS:
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
        metrics_rows: list[dict[str, float]] = []
        for benchmark, frame in benchmark_map.items():
            summary = summarize_predictions(frame["y_hat"], frame["y_true"], target_horizon_months=DEFAULT_HORIZON)
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


def run_v23_extended_history_proxy_study(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    horizon: int = DEFAULT_HORIZON,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)

    relative_series_map, proxy_manifest = build_extended_relative_return_series(conn, horizon)
    benchmark_data = _benchmark_dataset_map(df, relative_series_map)
    prediction_map, model_manifest = _build_prediction_maps(benchmark_data)
    metric_summary = _summarize_candidate_metrics(prediction_map)

    common_dates = common_historical_dates(prediction_map, list(benchmark_data.keys()))
    detail_df = _build_review_detail(prediction_map, common_dates)
    review_summary = summarize_v20_review(detail_df)
    slice_summary = summarize_v21_slices(
        detail_df,
        {
            "full_history": (None, None),
            "pre_2020": (None, "2019-12-31"),
            "post_2020": ("2020-01-31", None),
            "post_2022": ("2022-01-31", None),
        },
    )
    decision = choose_v23_decision(metric_summary, review_summary)

    stamp = datetime.today().strftime("%Y%m%d")
    metric_path = Path(output_dir) / f"v23_candidate_metric_summary_{stamp}.csv"
    detail_path = Path(output_dir) / f"v23_historical_review_detail_{stamp}.csv"
    summary_path = Path(output_dir) / f"v23_historical_review_summary_{stamp}.csv"
    slice_path = Path(output_dir) / f"v23_slice_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v23_decision_{stamp}.csv"
    proxy_path = Path(output_dir) / f"v23_proxy_manifest_{stamp}.csv"
    model_manifest_path = Path(output_dir) / f"v23_model_manifest_{stamp}.csv"

    metric_summary.to_csv(metric_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    review_summary.to_csv(summary_path, index=False)
    slice_summary.to_csv(slice_path, index=False)
    proxy_manifest.to_csv(proxy_path, index=False)
    model_manifest.to_csv(model_manifest_path, index=False)
    pd.DataFrame(
        [
            {
                "status": decision.status,
                "recommended_path": decision.recommended_path,
                "rationale": decision.rationale,
                "common_start": common_dates[0].date().isoformat() if common_dates else "",
                "common_end": common_dates[-1].date().isoformat() if common_dates else "",
                "n_common_dates": len(common_dates),
            }
        ]
    ).to_csv(decision_path, index=False)

    result_lines = [
        "# V23 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v23 tests whether the v21 promotion result survives when the reduced forecast universe is extended backward with research-only pre-inception benchmark proxies.",
        "- Live benchmark definitions are unchanged; the proxy stitching is research-only.",
        "",
        "## Proxy Design",
        "",
        "- `VOO` pre-inception history: proxy with `VTI`.",
        "- `VXUS` pre-inception history: proxy with a fitted `VEA` + `VWO` blend.",
        "- `VMBS` pre-inception history: proxy with `BND`.",
        "",
        "## Extended Historical Window",
        "",
        f"- Common evaluable monthly dates: `{len(common_dates)}`",
        f"- First common date: `{common_dates[0].date().isoformat() if common_dates else 'n/a'}`",
        f"- Last common date: `{common_dates[-1].date().isoformat() if common_dates else 'n/a'}`",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended path: `{decision.recommended_path}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Review Summary",
        "",
    ]
    for row in review_summary.itertuples(index=False):
        result_lines.extend(
            [
                f"### {row.path_name}",
                "",
                f"- Signal agreement with shadow baseline: `{row.signal_agreement_with_shadow_rate:.1%}`",
                f"- Mean aggregate OOS R^2: `{row.mean_aggregate_oos_r2:.4f}`",
                f"- Signal changes: `{int(row.signal_changes)}`",
                f"- OUT / NEUTRAL / UNDER: `{row.outperform_rate:.1%}` / `{row.neutral_rate:.1%}` / `{row.underperform_rate:.1%}`",
                "",
            ]
        )
    _write_text(Path("docs") / "results" / "V23_RESULTS_SUMMARY.md", result_lines)

    closeout_lines = [
        "# V23 Closeout And V24 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v23 extended the reduced-universe benchmark histories backward with research-only proxies and re-ran the historical comparison.",
        f"- Result: `{decision.status}` with `{decision.recommended_path}`.",
        "",
        "## Recommended V24 Scope",
        "",
        "- If the extended-history result confirms the v21/v22 path, the next step is operational promotion cleanup rather than another large research loop.",
        "- If it does not confirm the v21/v22 path, revisit the benchmark-set definition before more model work.",
    ]
    _write_text(Path("docs") / "closeouts" / "V23_CLOSEOUT_AND_V24_NEXT.md", closeout_lines)

    plan_lines = [
        "# v23 Extended-History Proxy Study Plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Objective",
        "",
        "- Determine whether the v21 cross-check promotion result remains intact when the forecast universe is evaluated over the longest feasible common historical window.",
        "",
        "## Proxy Rules",
        "",
        "- Use actual benchmark history whenever it exists.",
        "- Use explicit pre-inception proxies only before the first valid actual observation.",
        "- Keep the study research-only; do not replace live benchmark definitions.",
    ]
    _write_text(Path("docs") / "plans" / "codex-v23-plan.md", plan_lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v23 extended-history proxy study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_v23_extended_history_proxy_study(
        output_dir=args.output_dir,
        horizon=args.horizon,
    )
