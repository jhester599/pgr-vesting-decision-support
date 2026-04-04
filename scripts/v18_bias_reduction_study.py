"""v18 directional-bias study on the modified Ridge+GBT candidate stack."""

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
from sklearn.exceptions import ConvergenceWarning

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts import monthly_decision, v12_shadow_study
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import (
    evaluate_baseline_strategy,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    summarize_predictions,
)
from src.research.policy_metrics import evaluate_policy_series
from src.research.v12 import SnapshotSummary, aggregate_health_from_prediction_frames, recent_monthly_review_dates, signal_from_prediction
from src.research.v15 import apply_one_for_one_swap
from src.research.v16 import V16_FORECAST_UNIVERSE, v16_model_specs
from src.research.v17 import summarize_shadow_review
from src.research.v18 import (
    build_v18_candidate_specs,
    choose_best_v18_swaps,
    choose_v18_decision,
    v18_base_specs,
    v18_swap_candidates,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v18")
DEFAULT_HORIZON = 6
DEFAULT_REVIEW_MONTHS = 12


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    from src.models.regularized_models import build_gbt_pipeline, build_ridge_pipeline

    if model_type == "ridge":
        pipeline = build_ridge_pipeline()
    elif model_type == "gbt":
        pipeline = build_gbt_pipeline()
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        pipeline.fit(X_recent, y_recent)
    return float(pipeline.predict(X_curr)[0])


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


def _evaluate_model_candidate(
    spec_name: str,
    model_type: str,
    feature_columns: list[str],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        selected = [feature for feature in feature_columns if feature in X_aligned.columns]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            result, metrics = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=model_type,
                benchmark=benchmark,
                target_horizon_months=horizon,
                feature_columns=selected,
            )
        pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
        realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        summary = summarize_predictions(pred_series, realized, target_horizon_months=horizon)
        rows.append(
            {
                "candidate_name": spec_name,
                "model_type": model_type,
                "benchmark": benchmark,
                "n_features": len(selected),
                "feature_columns": ",".join(selected),
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "oos_r2": summary.oos_r2,
                "mae": summary.mae,
                "policy_return_sign": sign_policy.mean_policy_return,
            }
        )
    return pd.DataFrame(rows)


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
    weight_map = {col: 1.0 / max(float(col.split("__")[-1]), 1e-9) ** 2 for col in pred_cols}
    total_weight = sum(weight_map.values())
    merged["y_hat"] = sum(merged[col] * (weight_map[col] / total_weight) for col in pred_cols)
    return merged["y_hat"], merged["y_true"]


def _evaluate_ensemble_candidate(
    candidate_name: str,
    member_specs: dict[str, Any],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        prediction_frames: list[pd.DataFrame] = []
        current_predictions: list[tuple[float, float]] = []
        feature_columns: list[str] = []
        for member_name, spec in member_specs.items():
            selected = [feature for feature in spec.features if feature in X_aligned.columns]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                result, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=spec.model_type,
                    benchmark=benchmark,
                    target_horizon_months=horizon,
                    feature_columns=selected,
                )
            pred_series = pd.Series(
                result.y_hat_all,
                index=pd.DatetimeIndex(result.test_dates_all),
                name=f"pred_{member_name}__{max(float(metrics['mae']), 1e-9)}",
            )
            realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
            prediction_frames.append(pd.DataFrame({pred_series.name: pred_series, "y_true": realized}))
            current_predictions.append(
                (
                    _predict_current_custom(
                        X_full=X_aligned,
                        y_full=y_aligned,
                        X_current=X_aligned.iloc[[-1]],
                        model_type=spec.model_type,
                        selected_features=selected,
                    ),
                    max(float(metrics["mae"]), 1e-9),
                )
            )
            feature_columns.extend(selected)

        combined_pred, combined_true = _combine_prediction_frames(prediction_frames)
        summary = summarize_predictions(combined_pred, combined_true, target_horizon_months=horizon)
        sign_policy = evaluate_policy_series(combined_pred, combined_true, "sign_hold_vs_sell")
        total_weight = sum(1.0 / (mae**2) for _, mae in current_predictions)
        current_pred = sum(pred * (1.0 / (mae**2)) for pred, mae in current_predictions) / total_weight
        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_type": "ensemble",
                "model_type": "ensemble",
                "benchmark": benchmark,
                "n_features": len(feature_columns),
                "feature_columns": ",".join(feature_columns),
                "current_predicted_return": current_pred,
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "oos_r2": summary.oos_r2,
                "mae": summary.mae,
                "policy_return_sign": sign_policy.mean_policy_return,
            }
        )
    return pd.DataFrame(rows)


def _summarize_swap_phase(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in detail_df.groupby(["candidate_name", "candidate_feature", "replace_feature"], dropna=False):
        candidate_name, candidate_feature, replace_feature = keys
        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_feature": candidate_feature,
                "replace_feature": replace_feature,
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_sign_delta": float(group["policy_return_sign_delta"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_oos_r2_delta": float(group["oos_r2_delta"].mean()),
                "mean_ic": float(group["ic"].mean()),
                "mean_ic_delta": float(group["ic_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_candidates(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for candidate_name, group in detail_df.groupby("candidate_name", dropna=False):
        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_type": group["candidate_type"].iloc[0],
                "model_type": group["model_type"].iloc[0],
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_current_predicted_return": float(group["current_predicted_return"].mean()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_mae": float(group["mae"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_candidate_snapshot(
    conn: Any,
    as_of: date,
    member_specs: dict[str, Any],
    candidate_name: str,
    target_horizon_months: int = DEFAULT_HORIZON,
) -> SnapshotSummary | None:
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    X_event = df_full.loc[df_full.index <= pd.Timestamp(as_of)]
    if X_event.empty:
        return None

    signal_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for benchmark in V16_FORECAST_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, benchmark, target_horizon_months)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(X_event, rel_series, drop_na_target=True)
        except ValueError:
            continue
        if X_aligned.empty or y_aligned.empty:
            continue

        member_frames: list[pd.DataFrame] = []
        current_predictions: list[tuple[float, float]] = []
        for member_name, spec in member_specs.items():
            selected = [feature for feature in spec.features if feature in X_aligned.columns]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                result, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=spec.model_type,
                    benchmark=benchmark,
                    target_horizon_months=target_horizon_months,
                    feature_columns=selected,
                )
            pred_series = pd.Series(
                result.y_hat_all,
                index=pd.DatetimeIndex(result.test_dates_all),
                name=f"pred_{member_name}__{max(float(metrics['mae']), 1e-9)}",
            )
            realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
            member_frames.append(pd.DataFrame({pred_series.name: pred_series, "y_true": realized}))
            current_predictions.append(
                (
                    _predict_current_custom(
                        X_full=X_aligned,
                        y_full=y_aligned,
                        X_current=X_aligned.iloc[[-1]],
                        model_type=spec.model_type,
                        selected_features=selected,
                    ),
                    max(float(metrics["mae"]), 1e-9),
                )
            )

        combined_pred, combined_true = _combine_prediction_frames(member_frames)
        summary = summarize_predictions(combined_pred, combined_true, target_horizon_months=target_horizon_months)
        current_weight = sum(1.0 / (mae**2) for pred, mae in current_predictions)
        current_pred = sum(pred * (1.0 / (mae**2)) for pred, mae in current_predictions) / current_weight
        signal_rows.append(
            {
                "benchmark": benchmark,
                "predicted_relative_return": float(current_pred),
                "ic": float(summary.ic),
                "hit_rate": float(summary.hit_rate),
                "signal": signal_from_prediction(float(current_pred)),
                "confidence_tier": monthly_decision.confidence_from_hit_rate(float(summary.hit_rate)),
            }
        )
        prediction_frames.append(pd.DataFrame({"y_hat": combined_pred.values, "y_true": combined_true.values}))

    if not signal_rows:
        return None
    signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(prediction_frames, target_horizon_months)
    consensus, mean_pred, mean_ic, mean_hr, _, confidence_tier = monthly_decision._consensus_signal(signals)  # noqa: SLF001
    recommendation_mode = monthly_decision._determine_recommendation_mode(  # noqa: SLF001
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        representative_cpcv=None,
    )
    return SnapshotSummary(
        label=candidate_name,
        as_of=as_of,
        candidate_name=candidate_name,
        policy_name="current_production_mapping",
        consensus=consensus,
        confidence_tier=confidence_tier,
        recommendation_mode=str(recommendation_mode["label"]),
        sell_pct=float(recommendation_mode["sell_pct"]),
        mean_predicted=mean_pred,
        mean_ic=mean_ic,
        mean_hit_rate=mean_hr,
        aggregate_oos_r2=float(aggregate_health["oos_r2"]) if aggregate_health is not None else float("nan"),
        aggregate_nw_ic=float(aggregate_health["nw_ic"]) if aggregate_health is not None else float("nan"),
    )


def _snapshot_to_row(summary: SnapshotSummary, shadow_summary: SnapshotSummary, path_name: str) -> dict[str, object]:
    return {
        "as_of": summary.as_of.isoformat(),
        "path_name": path_name,
        "candidate_name": summary.candidate_name,
        "consensus": summary.consensus,
        "recommendation_mode": summary.recommendation_mode,
        "sell_pct": summary.sell_pct,
        "mean_predicted": summary.mean_predicted,
        "mean_ic": summary.mean_ic,
        "mean_hit_rate": summary.mean_hit_rate,
        "aggregate_oos_r2": summary.aggregate_oos_r2,
        "signal_agrees_with_shadow": summary.consensus == shadow_summary.consensus,
        "mode_agrees_with_shadow": summary.recommendation_mode == shadow_summary.recommendation_mode,
        "sell_agrees_with_shadow": abs(summary.sell_pct - shadow_summary.sell_pct) < 1e-9,
    }


def run_v18_bias_reduction_study(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    review_months: int = DEFAULT_REVIEW_MONTHS,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)
    benchmark_data = _benchmark_dataset_map(conn, df, list(V16_FORECAST_UNIVERSE), DEFAULT_HORIZON)
    stamp = datetime.today().strftime("%Y%m%d")

    # v18.0: narrow one-for-one swap screening on the v16 pair
    base_specs = v18_base_specs()
    detail_rows: list[dict[str, Any]] = []
    for swap in v18_swap_candidates():
        base_spec = base_specs[swap.candidate_name]
        base_detail = _evaluate_model_candidate(
            swap.candidate_name,
            base_spec.model_type,
            base_spec.features,
            benchmark_data,
            DEFAULT_HORIZON,
        )
        swapped_features = apply_one_for_one_swap(base_spec, swap.replace_feature, swap.candidate_feature)
        swapped_detail = _evaluate_model_candidate(
            f"{swap.candidate_name}__trial",
            base_spec.model_type,
            swapped_features,
            benchmark_data,
            DEFAULT_HORIZON,
        )
        merged = swapped_detail.merge(
            base_detail[["benchmark", "ic", "oos_r2", "policy_return_sign"]],
            on="benchmark",
            suffixes=("", "_baseline"),
        )
        for row in merged.itertuples(index=False):
            detail_rows.append(
                {
                    "candidate_name": swap.candidate_name,
                    "model_type": swap.model_type,
                    "candidate_feature": swap.candidate_feature,
                    "replace_feature": swap.replace_feature,
                    "benchmark": row.benchmark,
                    "ic": row.ic,
                    "ic_delta": row.ic - row.ic_baseline,
                    "oos_r2": row.oos_r2,
                    "oos_r2_delta": row.oos_r2 - row.oos_r2_baseline,
                    "policy_return_sign": row.policy_return_sign,
                    "policy_return_sign_delta": row.policy_return_sign - row.policy_return_sign_baseline,
                }
            )
    v18_core_detail = pd.DataFrame(detail_rows)
    v18_core_summary = _summarize_swap_phase(v18_core_detail)
    best_swaps = choose_best_v18_swaps(v18_core_summary)

    # v18.1: build the best v18 pair and compare against v16 and baseline
    candidate_specs = build_v18_candidate_specs(best_swaps)
    v18_members = {
        "ridge_lean_v1__v18": candidate_specs.get("ridge_lean_v1__v16__v18", candidate_specs["ridge_lean_v1__v16"]),
        "gbt_lean_plus_two__v18": candidate_specs.get("gbt_lean_plus_two__v16__v18", candidate_specs["gbt_lean_plus_two__v16"]),
    }
    v16_members = {
        "ridge_lean_v1__v16": v16_model_specs()["ridge_lean_v1__v16"],
        "gbt_lean_plus_two__v16": v16_model_specs()["gbt_lean_plus_two__v16"],
    }

    candidate_rows = [
        _evaluate_ensemble_candidate("ensemble_ridge_gbt_v16", v16_members, benchmark_data, DEFAULT_HORIZON),
        _evaluate_ensemble_candidate("ensemble_ridge_gbt_v18", v18_members, benchmark_data, DEFAULT_HORIZON),
    ]
    baseline_rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        metrics = evaluate_baseline_strategy(X_aligned, y_aligned, "historical_mean", DEFAULT_HORIZON)
        pred_series, realized = reconstruct_baseline_predictions(X_aligned, y_aligned, "historical_mean", DEFAULT_HORIZON)
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        summary = summarize_predictions(pred_series, realized, target_horizon_months=DEFAULT_HORIZON)
        baseline_rows.append(
            {
                "candidate_name": "baseline_historical_mean",
                "candidate_type": "baseline",
                "model_type": "baseline",
                "benchmark": benchmark,
                "n_features": 0,
                "feature_columns": "",
                "current_predicted_return": float(y_aligned.iloc[-min(len(y_aligned), config.WFO_TRAIN_WINDOW_MONTHS):].mean()),
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "oos_r2": summary.oos_r2,
                "mae": summary.mae,
                "policy_return_sign": sign_policy.mean_policy_return,
            }
        )
    candidate_metric_detail = pd.concat(candidate_rows + [pd.DataFrame(baseline_rows)], ignore_index=True)
    candidate_metric_summary = _summarize_candidates(candidate_metric_detail)

    # v18.2: shadow review for candidate_v16 vs candidate_v18
    review_dates = recent_monthly_review_dates(date.today(), review_months)
    shadow_rows: list[dict[str, object]] = []
    for as_of in review_dates:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            shadow_summary, _ = monthly_decision._build_shadow_baseline_summary(conn, as_of, DEFAULT_HORIZON)  # noqa: SLF001
            candidate_v16 = _build_candidate_snapshot(conn, as_of, v16_members, "ensemble_ridge_gbt_v16")
            candidate_v18 = _build_candidate_snapshot(conn, as_of, v18_members, "ensemble_ridge_gbt_v18")
        if shadow_summary is None or candidate_v16 is None or candidate_v18 is None:
            continue
        shadow_rows.append(_snapshot_to_row(shadow_summary, shadow_summary, "shadow_baseline"))
        shadow_rows.append(_snapshot_to_row(candidate_v16, shadow_summary, "candidate_v16"))
        shadow_rows.append(_snapshot_to_row(candidate_v18, shadow_summary, "candidate_v18"))
    shadow_detail = pd.DataFrame(shadow_rows)
    shadow_summary = summarize_shadow_review(shadow_detail)

    decision = choose_v18_decision(candidate_metric_summary, shadow_summary)
    decision_df = pd.DataFrame([{"status": decision.status, "recommended_candidate": decision.recommended_candidate, "rationale": decision.rationale}])

    core_detail_path = Path(output_dir) / f"v18_core_swap_detail_{stamp}.csv"
    core_summary_path = Path(output_dir) / f"v18_core_swap_summary_{stamp}.csv"
    best_swaps_path = Path(output_dir) / f"v18_best_swaps_{stamp}.csv"
    metric_detail_path = Path(output_dir) / f"v18_candidate_metric_detail_{stamp}.csv"
    metric_summary_path = Path(output_dir) / f"v18_candidate_metric_summary_{stamp}.csv"
    shadow_detail_path = Path(output_dir) / f"v18_shadow_review_detail_{stamp}.csv"
    shadow_summary_path = Path(output_dir) / f"v18_shadow_review_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v18_decision_{stamp}.csv"

    v18_core_detail.to_csv(core_detail_path, index=False)
    v18_core_summary.to_csv(core_summary_path, index=False)
    best_swaps.to_csv(best_swaps_path, index=False)
    candidate_metric_detail.to_csv(metric_detail_path, index=False)
    candidate_metric_summary.to_csv(metric_summary_path, index=False)
    shadow_detail.to_csv(shadow_detail_path, index=False)
    shadow_summary.to_csv(shadow_summary_path, index=False)
    decision_df.to_csv(decision_path, index=False)

    lines = [
        "# V18 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v18 focuses on reducing the modified candidate's directional bias against the promoted simpler baseline.",
        "- It only tests narrow benchmark-side and peer-relative one-for-one swaps on the v16 Ridge+GBT pair.",
        "",
        "## Best Swaps",
        "",
    ]
    for row in best_swaps.itertuples(index=False):
        lines.append(
            f"- `{row.candidate_name}`: `{row.candidate_feature}` for `{row.replace_feature}` "
            f"(policy delta `{float(row.mean_policy_return_sign_delta):+.4f}`, OOS R^2 delta `{float(row.mean_oos_r2_delta):+.4f}`)"
        )
    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            f"- Status: `{decision.status}`",
            f"- Recommended candidate: `{decision.recommended_candidate}`",
            f"- Rationale: {decision.rationale}",
            "",
            "## Output Artifacts",
            "",
            f"- `results/v18/{core_summary_path.name}`",
            f"- `results/v18/{best_swaps_path.name}`",
            f"- `results/v18/{metric_summary_path.name}`",
            f"- `results/v18/{shadow_summary_path.name}`",
            f"- `results/v18/{decision_path.name}`",
        ]
    )
    _write_text(Path("docs") / "results" / "V18_RESULTS_SUMMARY.md", lines)

    closeout_lines = [
        "# V18 Closeout And V19 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v18 tested narrow benchmark-side and peer-relative swaps on the modified Ridge+GBT candidate stack.",
        "",
        "## Result",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended candidate: `{decision.recommended_candidate}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Recommended V19 Scope",
        "",
        "- If v18 advances, run one more narrow promotion gate on the v18 candidate.",
        "- If v18 does not advance, keep the current production paths unchanged and return only to the highest-value deferred families that remain executable with existing data.",
    ]
    _write_text(Path("docs") / "closeouts" / "V18_CLOSEOUT_AND_V19_NEXT.md", closeout_lines)

    plan_lines = [
        "# codex-v18-plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Goal",
        "",
        "- Reduce the modified candidate's directional bias against the promoted simpler baseline without increasing model complexity.",
        "",
        "## Method",
        "",
        "- test only benchmark-side and peer-relative one-for-one swaps",
        "- preserve the lean v16 Ridge+GBT stack",
        "- require both metric stability and improved agreement with the simpler baseline",
    ]
    _write_text(Path("docs") / "plans" / "codex-v18-plan.md", plan_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v18 directional-bias reduction study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-months", default=str(DEFAULT_REVIEW_MONTHS))
    args = parser.parse_args()
    run_v18_bias_reduction_study(output_dir=args.output_dir, review_months=int(args.review_months))


if __name__ == "__main__":
    main()
