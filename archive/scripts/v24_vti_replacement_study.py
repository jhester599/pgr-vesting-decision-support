"""v24 narrow study: replace VOO with VTI in the reduced forecast universe."""

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
from src.processing.multi_total_return import build_etf_monthly_returns, load_relative_return_matrix
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
from src.research.v23 import fit_proxy_blend_weights
from src.research.v24 import (
    V24_CURRENT_UNIVERSE,
    V24_SCENARIOS,
    V24_VTI_UNIVERSE,
    choose_v24_decision,
    summarize_v24_scenarios,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v24")
DEFAULT_HORIZON = 6


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stitch_proxy_series(actual: pd.Series, proxy: pd.Series) -> pd.Series:
    actual_valid = actual.dropna()
    if actual_valid.empty:
        return proxy.sort_index()
    first_actual = actual_valid.index.min()
    combined = pd.concat([actual.rename("actual"), proxy.rename("proxy")], axis=1).sort_index()
    stitched = combined["actual"].copy()
    mask = stitched.isna() & (combined.index < first_actual)
    stitched.loc[mask] = combined.loc[mask, "proxy"]
    return stitched


def _build_relative_series_map(
    conn: Any,
    scenario_name: str,
    horizon: int,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    pgr_returns = build_etf_monthly_returns(conn, "PGR", horizon)
    manifest_rows: list[dict[str, object]] = []

    if scenario_name == "current_voo_actual":
        universe = V24_CURRENT_UNIVERSE
        benchmark_returns = {
            benchmark: build_etf_monthly_returns(conn, benchmark, horizon)
            for benchmark in universe
        }
        for benchmark, series in benchmark_returns.items():
            manifest_rows.append(
                {
                    "scenario_name": scenario_name,
                    "benchmark": benchmark,
                    "construction": "actual",
                    "weights": "",
                    "start_date": series.dropna().index.min().date().isoformat() if not series.dropna().empty else "",
                }
            )
    elif scenario_name == "vti_replacement_actual":
        universe = V24_VTI_UNIVERSE
        benchmark_returns = {
            benchmark: build_etf_monthly_returns(conn, benchmark, horizon)
            for benchmark in universe
        }
        for benchmark, series in benchmark_returns.items():
            manifest_rows.append(
                {
                    "scenario_name": scenario_name,
                    "benchmark": benchmark,
                    "construction": "actual",
                    "weights": "",
                    "start_date": series.dropna().index.min().date().isoformat() if not series.dropna().empty else "",
                }
            )
    elif scenario_name == "vti_replacement_stitched":
        universe = V24_VTI_UNIVERSE
        benchmark_returns = {benchmark: build_etf_monthly_returns(conn, benchmark, horizon) for benchmark in universe}

        vxus_actual = benchmark_returns["VXUS"]
        vea_proxy = build_etf_monthly_returns(conn, "VEA", horizon)
        vwo_proxy = build_etf_monthly_returns(conn, "VWO", horizon)
        vxus_weights = fit_proxy_blend_weights(
            vxus_actual,
            {"VEA": vea_proxy, "VWO": vwo_proxy},
            fallback_weights={"VEA": 0.75, "VWO": 0.25},
        )
        vxus_blend = sum(
            series * vxus_weights[name] for name, series in {"VEA": vea_proxy, "VWO": vwo_proxy}.items()
        )
        benchmark_returns["VXUS"] = _stitch_proxy_series(vxus_actual, vxus_blend)

        vmbs_actual = benchmark_returns["VMBS"]
        bnd_proxy = build_etf_monthly_returns(conn, "BND", horizon)
        benchmark_returns["VMBS"] = _stitch_proxy_series(vmbs_actual, bnd_proxy)

        for benchmark, series in benchmark_returns.items():
            if benchmark == "VXUS":
                construction = "stitched_blend"
                weights = ",".join(f"{name}={weight:.4f}" for name, weight in vxus_weights.items())
            elif benchmark == "VMBS":
                construction = "stitched_single"
                weights = "BND=1.0000"
            else:
                construction = "actual"
                weights = ""
            manifest_rows.append(
                {
                    "scenario_name": scenario_name,
                    "benchmark": benchmark,
                    "construction": construction,
                    "weights": weights,
                    "start_date": series.dropna().index.min().date().isoformat() if not series.dropna().empty else "",
                }
            )
    else:
        raise ValueError(f"Unknown v24 scenario: {scenario_name}")

    relative_map: dict[str, pd.Series] = {}
    for benchmark, benchmark_returns_series in benchmark_returns.items():
        aligned = pd.DataFrame({"pgr": pgr_returns, "benchmark": benchmark_returns_series}).dropna()
        if aligned.empty:
            continue
        relative_map[benchmark] = (aligned["pgr"] - aligned["benchmark"]).rename(f"{benchmark}_{horizon}m")
    return relative_map, pd.DataFrame(manifest_rows)


def _benchmark_dataset_map(
    df: pd.DataFrame,
    scenario_name: str,
    conn: Any,
    horizon: int,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], pd.DataFrame]:
    if scenario_name == "current_voo_actual":
        datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
        coverage_rows: list[dict[str, object]] = []
        for benchmark in V24_CURRENT_UNIVERSE:
            rel_series = load_relative_return_matrix(conn, benchmark, horizon)
            if rel_series.empty:
                continue
            try:
                X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
            except ValueError:
                continue
            datasets[benchmark] = (X_aligned, y_aligned)
            coverage_rows.append(
                {
                    "scenario_name": scenario_name,
                    "benchmark": benchmark,
                    "relative_start": y_aligned.index.min().date().isoformat(),
                    "relative_end": y_aligned.index.max().date().isoformat(),
                    "n_obs": int(y_aligned.shape[0]),
                }
            )
        return datasets, pd.DataFrame(coverage_rows)

    relative_map, manifest = _build_relative_series_map(conn, scenario_name, horizon)
    datasets = {}
    coverage_rows: list[dict[str, object]] = []
    for benchmark, rel_series in relative_map.items():
        try:
            X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue
        datasets[benchmark] = (X_aligned, y_aligned)
        coverage_rows.append(
            {
                "scenario_name": scenario_name,
                "benchmark": benchmark,
                "relative_start": y_aligned.index.min().date().isoformat(),
                "relative_end": y_aligned.index.max().date().isoformat(),
                "n_obs": int(y_aligned.shape[0]),
            }
        )
    return datasets, manifest.merge(pd.DataFrame(coverage_rows), on=["scenario_name", "benchmark"], how="left")


def _evaluate_model_prediction_series(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    model_type: str,
    feature_columns: list[str],
    benchmark: str,
) -> tuple[pd.DataFrame, float]:
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
    frame = pd.DataFrame(
        {
            "y_hat": pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all)),
            "y_true": pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all)),
        }
    ).sort_index()
    return frame, float(metrics["mae"])


def _combine_member_frames(member_frames: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    merged = member_frames[0][0].rename(columns={"y_hat": "pred_0"})
    for idx, (frame, _) in enumerate(member_frames[1:], start=1):
        merged = merged.join(frame[["y_hat"]].rename(columns={"y_hat": f"pred_{idx}"}), how="inner")
    weights = [1.0 / max(mae, 1e-9) ** 2 for _, mae in member_frames]
    total_weight = sum(weights)
    merged["y_hat"] = 0.0
    for idx, weight in enumerate(weights):
        merged["y_hat"] += merged[f"pred_{idx}"] * (weight / total_weight)
    return merged[["y_hat", "y_true"]]


def _build_prediction_maps(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, dict[str, pd.DataFrame]]:
    model_specs = v20_model_specs()
    ensemble_specs = {
        "live_production_ensemble_reduced": ["elasticnet_current", "ridge_current", "bayesian_ridge_current", "gbt_current"],
        "ensemble_ridge_gbt_v18": ["ridge_lean_v1__v18", "gbt_lean_plus_two__v18"],
    }
    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    model_maes: dict[str, dict[str, float]] = {}
    for spec_name, spec in model_specs.items():
        model_frames[spec_name] = {}
        model_maes[spec_name] = {}
        for benchmark in benchmark_data:
            frame, mae = _evaluate_model_prediction_series(
                benchmark_data,
                spec.model_type,
                spec.features,
                benchmark,
            )
            model_frames[spec_name][benchmark] = frame
            model_maes[spec_name][benchmark] = mae

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
        prediction_map["shadow_baseline"][benchmark] = pd.DataFrame({"y_hat": pred_series, "y_true": realized}).sort_index()

    for ensemble_name, members in ensemble_specs.items():
        prediction_map[ensemble_name] = {}
        for benchmark in benchmark_data:
            member_frames = [
                (model_frames[member_name][benchmark], model_maes[member_name][benchmark])
                for member_name in members
            ]
            prediction_map[ensemble_name][benchmark] = _combine_member_frames(member_frames)
    return prediction_map


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
    consensus = "NEUTRAL"
    if outperform_count > total / 2:
        consensus = "OUTPERFORM"
    elif underperform_count > total / 2:
        consensus = "UNDERPERFORM"
    recommendation_mode = determine_recommendation_mode(
        consensus, mean_pred, mean_ic, mean_hr, aggregate_health, representative_cpcv=None
    )
    return {
        "consensus": consensus,
        "recommendation_mode": str(recommendation_mode["label"]),
        "sell_pct": float(recommendation_mode["sell_pct"]),
        "mean_predicted": mean_pred,
        "mean_ic": mean_ic,
        "mean_hit_rate": mean_hr,
        "aggregate_oos_r2": float(aggregate_health["oos_r2"]) if aggregate_health else float("nan"),
        "aggregate_nw_ic": float(aggregate_health["nw_ic"]) if aggregate_health else float("nan"),
    }


def _summarize_metrics(prediction_map: dict[str, dict[str, pd.DataFrame]], scenario_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path_name, benchmark_map in prediction_map.items():
        path_type = "baseline" if path_name == "shadow_baseline" else "ensemble"
        metric_rows = []
        for benchmark, frame in benchmark_map.items():
            summary = summarize_predictions(frame["y_hat"], frame["y_true"], target_horizon_months=DEFAULT_HORIZON)
            sign_policy = evaluate_policy_series(frame["y_hat"], frame["y_true"], "sign_hold_vs_sell")
            metric_rows.append(
                {
                    "benchmark": benchmark,
                    "ic": float(summary.ic),
                    "hit_rate": float(summary.hit_rate),
                    "oos_r2": float(summary.oos_r2),
                    "mae": float(summary.mae),
                    "policy_return_sign": float(sign_policy.mean_policy_return),
                }
            )
        group = pd.DataFrame(metric_rows)
        rows.append(
            {
                "scenario_name": scenario_name,
                "candidate_name": path_name if path_name != "shadow_baseline" else "baseline_historical_mean",
                "candidate_type": path_type,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_mae": float(group["mae"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_review_detail(
    prediction_map: dict[str, dict[str, pd.DataFrame]],
    review_dates: list[pd.Timestamp],
    scenario_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for as_of in review_dates:
        shadow = _build_path_snapshot(prediction_map, "shadow_baseline", as_of)
        live = _build_path_snapshot(prediction_map, "live_production_ensemble_reduced", as_of)
        rows.append(
            {
                "scenario_name": scenario_name,
                "as_of": as_of.date().isoformat(),
                "path_name": "shadow_baseline",
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
                "scenario_name": scenario_name,
                "as_of": as_of.date().isoformat(),
                "path_name": "live_production_ensemble_reduced",
                **live,
                "signal_agrees_with_shadow": live["consensus"] == shadow["consensus"],
                "mode_agrees_with_shadow": live["recommendation_mode"] == shadow["recommendation_mode"],
                "sell_agrees_with_shadow": abs(float(live["sell_pct"]) - float(shadow["sell_pct"])) < 1e-9,
                "signal_agrees_with_live": True,
                "mode_agrees_with_live": True,
                "sell_agrees_with_live": True,
            }
        )
        candidate = _build_path_snapshot(prediction_map, "ensemble_ridge_gbt_v18", as_of)
        rows.append(
            {
                "scenario_name": scenario_name,
                "as_of": as_of.date().isoformat(),
                "path_name": "ensemble_ridge_gbt_v18",
                **candidate,
                "signal_agrees_with_shadow": candidate["consensus"] == shadow["consensus"],
                "mode_agrees_with_shadow": candidate["recommendation_mode"] == shadow["recommendation_mode"],
                "sell_agrees_with_shadow": abs(float(candidate["sell_pct"]) - float(shadow["sell_pct"])) < 1e-9,
                "signal_agrees_with_live": candidate["consensus"] == live["consensus"],
                "mode_agrees_with_live": candidate["recommendation_mode"] == live["recommendation_mode"],
                "sell_agrees_with_live": abs(float(candidate["sell_pct"]) - float(live["sell_pct"])) < 1e-9,
            }
        )
    return pd.DataFrame(rows)


def run_v24_vti_replacement_study(*, output_dir: str = DEFAULT_OUTPUT_DIR, horizon: int = DEFAULT_HORIZON) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)

    scenario_metric_frames: list[pd.DataFrame] = []
    scenario_review_frames: list[pd.DataFrame] = []
    scenario_window_rows: list[dict[str, object]] = []
    scenario_manifest_frames: list[pd.DataFrame] = []

    for scenario_name in V24_SCENARIOS:
        benchmark_data, manifest_df = _benchmark_dataset_map(df, scenario_name, conn, horizon)
        prediction_map = _build_prediction_maps(benchmark_data)
        review_dates = common_historical_dates(prediction_map, list(benchmark_data.keys()))
        detail_df = _build_review_detail(prediction_map, review_dates, scenario_name)
        summary_df = summarize_v20_review(detail_df)
        summary_df.insert(0, "scenario_name", scenario_name)
        scenario_review_frames.append(summary_df)
        scenario_metric_frames.append(_summarize_metrics(prediction_map, scenario_name))
        scenario_window_rows.append(
            {
                "scenario_name": scenario_name,
                "common_start": review_dates[0].date().isoformat() if review_dates else "",
                "common_end": review_dates[-1].date().isoformat() if review_dates else "",
                "n_common_dates": len(review_dates),
            }
        )
        scenario_manifest_frames.append(manifest_df)

    metric_df = pd.concat(scenario_metric_frames, ignore_index=True)
    review_df = pd.concat(scenario_review_frames, ignore_index=True)
    window_df = pd.DataFrame(scenario_window_rows)
    manifest_df = pd.concat(scenario_manifest_frames, ignore_index=True)
    scenario_summary = summarize_v24_scenarios(metric_df, review_df, window_df)
    decision = choose_v24_decision(scenario_summary)

    stamp = datetime.today().strftime("%Y%m%d")
    metric_path = Path(output_dir) / f"v24_candidate_metric_summary_{stamp}.csv"
    review_path = Path(output_dir) / f"v24_review_summary_{stamp}.csv"
    window_path = Path(output_dir) / f"v24_window_summary_{stamp}.csv"
    manifest_path = Path(output_dir) / f"v24_benchmark_manifest_{stamp}.csv"
    summary_path = Path(output_dir) / f"v24_scenario_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v24_decision_{stamp}.csv"
    metric_df.to_csv(metric_path, index=False)
    review_df.to_csv(review_path, index=False)
    window_df.to_csv(window_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)
    scenario_summary.to_csv(summary_path, index=False)
    pd.DataFrame([{
        "status": decision.status,
        "recommended_universe": decision.recommended_universe,
        "rationale": decision.rationale,
    }]).to_csv(decision_path, index=False)

    result_lines = [
        "# V24 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v24 tests whether replacing `VOO` with `VTI` improves the reduced forecast universe when everything else is held fixed.",
        "- It compares the current VOO-based universe, an actual VTI replacement universe, and a stitched-history VTI replacement universe.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended universe: `{decision.recommended_universe}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Scenario Summary",
        "",
    ]
    for row in scenario_summary.itertuples(index=False):
        result_lines.extend(
            [
                f"### {row.scenario_name}",
                "",
                f"- Common window: `{row.common_start}` to `{row.common_end}` ({int(row.n_common_dates)} monthly dates)",
                f"- Mean sign-policy return: `{row.mean_policy_return_sign:.4f}`",
                f"- Mean OOS R^2: `{row.mean_oos_r2:.4f}`",
                f"- Mean IC: `{row.mean_ic:.4f}`",
                f"- Signal agreement with simpler baseline: `{row.signal_agreement_with_shadow_rate:.1%}`",
                f"- Signal changes: `{int(row.signal_changes)}`",
                "",
            ]
        )
    _write_text(Path("docs") / "results" / "V24_RESULTS_SUMMARY.md", result_lines)

    closeout_lines = [
        "# V24 Closeout And V25 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v24 tested whether `VTI` should replace `VOO` in the reduced forecast universe.",
        f"- Result: `{decision.status}` with `{decision.recommended_universe}`.",
        "",
        "## Recommended V25 Scope",
        "",
        "- If VTI wins, carry it into future reduced-universe research and any later production-promotion study.",
        "- If VOO still wins, keep the current benchmark definition and focus future work elsewhere.",
    ]
    _write_text(Path("docs") / "closeouts" / "V24_CLOSEOUT_AND_V25_NEXT.md", closeout_lines)

    plan_lines = [
        "# v24 VTI Replacement Study Plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Objective",
        "",
        "- Isolate the effect of replacing `VOO` with `VTI` in the reduced forecast universe.",
        "",
        "## Scenarios",
        "",
        "- `current_voo_actual`",
        "- `vti_replacement_actual`",
        "- `vti_replacement_stitched`",
    ]
    _write_text(Path("docs") / "plans" / "codex-v24-plan.md", plan_lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v24 VTI replacement study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_v24_vti_replacement_study(output_dir=args.output_dir, horizon=args.horizon)
