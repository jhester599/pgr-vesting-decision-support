"""v20 synthesis and promotion-readiness study."""

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
from src.reporting.decision_rendering import determine_recommendation_mode
from src.research.evaluation import (
    evaluate_baseline_strategy,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    summarize_predictions,
)
from src.research.policy_metrics import evaluate_policy_series
from src.research.v12 import (
    SnapshotSummary,
    aggregate_health_from_prediction_frames,
    recent_monthly_review_dates,
    signal_from_prediction,
)
from src.research.v20 import (
    V20_FORECAST_UNIVERSE,
    choose_v20_decision,
    summarize_v20_review,
    v20_ensemble_specs,
    v20_model_specs,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v20")
DEFAULT_HORIZON = 6
DEFAULT_REVIEW_MONTHS = 12


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
        build_bayesian_ridge_pipeline,
        build_elasticnet_pipeline,
        build_gbt_pipeline,
        build_ridge_pipeline,
    )

    if model_type == "elasticnet":
        pipeline = build_elasticnet_pipeline()
    elif model_type == "ridge":
        pipeline = build_ridge_pipeline()
    elif model_type == "bayesian_ridge":
        pipeline = build_bayesian_ridge_pipeline()
    elif model_type == "gbt":
        pipeline = build_gbt_pipeline()
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        pipeline.fit(X_recent, y_recent)
    return float(pipeline.predict(X_curr)[0])


def _evaluate_model_candidate(
    spec_name: str,
    model_type: str,
    feature_columns: list[str],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
    notes: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        selected = [feature for feature in feature_columns if feature in X_aligned.columns]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            result, _ = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=model_type,
                benchmark=benchmark,
                target_horizon_months=horizon,
                feature_columns=selected,
            )
        pred_series = pd.Series(result.y_hat_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_hat")
        realized = pd.Series(result.y_true_all, index=pd.DatetimeIndex(result.test_dates_all), name="y_true")
        current_pred = _predict_current_custom(
            X_full=X_aligned,
            y_full=y_aligned,
            X_current=X_aligned.iloc[[-1]],
            model_type=model_type,
            selected_features=selected,
        )
        summary = summarize_predictions(pred_series, realized, target_horizon_months=horizon)
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        neutral_policy = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
        rows.append(
            {
                "candidate_name": spec_name,
                "candidate_type": "model",
                "model_type": model_type,
                "benchmark": benchmark,
                "n_features": len(selected),
                "feature_columns": ",".join(selected),
                "notes": notes,
                "current_predicted_return": current_pred,
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_return_neutral_3pct": neutral_policy.mean_policy_return,
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "mae": summary.mae,
                "oos_r2": summary.oos_r2,
                "nw_ic": summary.nw_ic,
            }
        )
    return pd.DataFrame(rows)


def _ensemble_prediction_frame(
    X_aligned: pd.DataFrame,
    y_aligned: pd.Series,
    member_names: list[str],
    model_specs: dict[str, Any],
    benchmark: str,
    horizon: int,
) -> tuple[pd.Series, pd.Series, float, int, list[str]]:
    member_frames: list[pd.DataFrame] = []
    current_predictions: list[tuple[float, float]] = []
    feature_count = 0
    feature_labels: list[str] = []

    for member_name in member_names:
        spec = model_specs[member_name]
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
        feature_count += len(selected)
        feature_labels.append(f"{member_name}:{','.join(selected)}")

    merged = member_frames[0].copy()
    for frame in member_frames[1:]:
        pred_cols = [col for col in frame.columns if col.startswith("pred_")]
        merged = merged.join(frame[pred_cols], how="inner")
    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    weight_map = {col: 1.0 / max(float(col.split("__")[-1]), 1e-9) ** 2 for col in pred_cols}
    total_weight = sum(weight_map.values())
    merged["y_hat"] = sum(merged[col] * (weight_map[col] / total_weight) for col in pred_cols)
    current_weight = sum(1.0 / (mae**2) for _, mae in current_predictions)
    current_pred = sum(pred * (1.0 / (mae**2)) for pred, mae in current_predictions) / current_weight
    return merged["y_hat"], merged["y_true"], float(current_pred), feature_count, feature_labels


def _evaluate_ensemble_candidate(
    candidate_name: str,
    member_names: list[str],
    model_specs: dict[str, Any],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
    notes: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        pred_series, realized, current_pred, feature_count, feature_labels = _ensemble_prediction_frame(
            X_aligned=X_aligned,
            y_aligned=y_aligned,
            member_names=member_names,
            model_specs=model_specs,
            benchmark=benchmark,
            horizon=horizon,
        )
        summary = summarize_predictions(pred_series, realized, target_horizon_months=horizon)
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        neutral_policy = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_type": "ensemble",
                "model_type": "ensemble",
                "benchmark": benchmark,
                "n_features": feature_count,
                "feature_columns": " | ".join(feature_labels),
                "notes": notes,
                "current_predicted_return": current_pred,
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_return_neutral_3pct": neutral_policy.mean_policy_return,
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "mae": summary.mae,
                "oos_r2": summary.oos_r2,
                "nw_ic": summary.nw_ic,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_baseline_historical_mean(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        evaluate_baseline_strategy(
            X_aligned,
            y_aligned,
            strategy="historical_mean",
            target_horizon_months=horizon,
        )
        pred_series, realized = reconstruct_baseline_predictions(
            X_aligned,
            y_aligned,
            strategy="historical_mean",
            target_horizon_months=horizon,
        )
        current_pred = float(y_aligned.iloc[-min(len(y_aligned), config.WFO_TRAIN_WINDOW_MONTHS):].mean())
        summary = summarize_predictions(pred_series, realized, target_horizon_months=horizon)
        sign_policy = evaluate_policy_series(pred_series, realized, "sign_hold_vs_sell")
        neutral_policy = evaluate_policy_series(pred_series, realized, "neutral_band_3pct")
        rows.append(
            {
                "candidate_name": "baseline_historical_mean",
                "candidate_type": "baseline",
                "model_type": "baseline",
                "benchmark": benchmark,
                "n_features": 0,
                "feature_columns": "",
                "notes": "Historical-mean benchmark baseline.",
                "current_predicted_return": current_pred,
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_return_neutral_3pct": neutral_policy.mean_policy_return,
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "mae": summary.mae,
                "oos_r2": summary.oos_r2,
                "nw_ic": summary.nw_ic,
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
                "n_features": int(group["n_features"].iloc[0]),
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_current_predicted_return": float(group["current_predicted_return"].mean()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_nw_ic": float(group["nw_ic"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_neutral_3pct": float(group["policy_return_neutral_3pct"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "notes": str(group["notes"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_candidate_snapshot(
    conn: Any,
    as_of: date,
    model_specs: dict[str, Any],
    ensemble_name: str,
    member_names: list[str],
    target_horizon_months: int = DEFAULT_HORIZON,
) -> SnapshotSummary | None:
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    X_event = df_full.loc[df_full.index <= pd.Timestamp(as_of)]
    if X_event.empty:
        return None

    signal_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for benchmark in V20_FORECAST_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, benchmark, target_horizon_months)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(X_event, rel_series, drop_na_target=True)
        except ValueError:
            continue
        if X_aligned.empty or y_aligned.empty:
            continue

        pred_series, realized, current_pred, _, _ = _ensemble_prediction_frame(
            X_aligned=X_aligned,
            y_aligned=y_aligned,
            member_names=member_names,
            model_specs=model_specs,
            benchmark=benchmark,
            horizon=target_horizon_months,
        )
        summary = summarize_predictions(pred_series, realized, target_horizon_months=target_horizon_months)
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
        prediction_frames.append(pd.DataFrame({"y_hat": pred_series.values, "y_true": realized.values}))

    if not signal_rows:
        return None

    signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(prediction_frames, target_horizon_months)
    consensus, mean_pred, mean_ic, mean_hr, _, confidence_tier = monthly_decision._consensus_signal(signals)  # noqa: SLF001
    recommendation_mode = determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        representative_cpcv=None,
    )
    return SnapshotSummary(
        label=ensemble_name,
        as_of=as_of,
        candidate_name=ensemble_name,
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


def _snapshot_to_row(
    summary: SnapshotSummary,
    *,
    shadow_summary: SnapshotSummary,
    live_summary: SnapshotSummary,
    path_name: str,
) -> dict[str, object]:
    return {
        "as_of": summary.as_of.isoformat(),
        "path_name": path_name,
        "candidate_name": summary.candidate_name,
        "policy_name": summary.policy_name,
        "consensus": summary.consensus,
        "confidence_tier": summary.confidence_tier,
        "recommendation_mode": summary.recommendation_mode,
        "sell_pct": summary.sell_pct,
        "mean_predicted": summary.mean_predicted,
        "mean_ic": summary.mean_ic,
        "mean_hit_rate": summary.mean_hit_rate,
        "aggregate_oos_r2": summary.aggregate_oos_r2,
        "aggregate_nw_ic": summary.aggregate_nw_ic,
        "signal_agrees_with_shadow": summary.consensus == shadow_summary.consensus,
        "mode_agrees_with_shadow": summary.recommendation_mode == shadow_summary.recommendation_mode,
        "sell_agrees_with_shadow": abs(summary.sell_pct - shadow_summary.sell_pct) < 1e-9,
        "signal_agrees_with_live": summary.consensus == live_summary.consensus,
        "mode_agrees_with_live": summary.recommendation_mode == live_summary.recommendation_mode,
        "sell_agrees_with_live": abs(summary.sell_pct - live_summary.sell_pct) < 1e-9,
    }


def run_v20_synthesis_study(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    horizon: int = DEFAULT_HORIZON,
    review_months: int = DEFAULT_REVIEW_MONTHS,
    end_as_of: date | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)
    benchmark_data = _benchmark_dataset_map(conn, df, list(V20_FORECAST_UNIVERSE), horizon)
    stamp = datetime.today().strftime("%Y%m%d")

    model_specs = v20_model_specs()
    ensemble_specs = v20_ensemble_specs()

    detail_frames: list[pd.DataFrame] = []
    for spec_name, spec in model_specs.items():
        if spec.model_type not in {"ridge", "gbt", "elasticnet", "bayesian_ridge"}:
            continue
        detail_frames.append(
            _evaluate_model_candidate(
                spec_name=spec_name,
                model_type=spec.model_type,
                feature_columns=spec.features,
                benchmark_data=benchmark_data,
                horizon=horizon,
                notes=spec.notes,
            )
        )

    for ensemble_name, ensemble_spec in ensemble_specs.items():
        detail_frames.append(
            _evaluate_ensemble_candidate(
                candidate_name=ensemble_name,
                member_names=list(ensemble_spec["members"]),
                model_specs=model_specs,
                benchmark_data=benchmark_data,
                horizon=horizon,
                notes=str(ensemble_spec["notes"]),
            )
        )

    detail_frames.append(_evaluate_baseline_historical_mean(benchmark_data, horizon))
    metric_detail_df = pd.concat(detail_frames, ignore_index=True)
    metric_summary_df = _summarize_candidates(metric_detail_df)

    as_of_end = end_as_of or date.today()
    review_dates = recent_monthly_review_dates(as_of_end, review_months)
    review_rows: list[dict[str, object]] = []

    for as_of in review_dates:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            live_summary = v12_shadow_study._build_live_snapshot(conn, as_of)  # noqa: SLF001
            shadow_summary, _ = monthly_decision._build_shadow_baseline_summary(conn, as_of, horizon)  # noqa: SLF001
        if live_summary is None or shadow_summary is None:
            continue
        review_rows.append(_snapshot_to_row(shadow_summary, shadow_summary=shadow_summary, live_summary=live_summary, path_name="shadow_baseline"))
        review_rows.append(_snapshot_to_row(live_summary, shadow_summary=shadow_summary, live_summary=live_summary, path_name="live_production_ensemble_reduced"))
        for ensemble_name, ensemble_spec in ensemble_specs.items():
            if ensemble_name == "live_production_ensemble_reduced":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                candidate_summary = _build_candidate_snapshot(
                    conn=conn,
                    as_of=as_of,
                    model_specs=model_specs,
                    ensemble_name=ensemble_name,
                    member_names=list(ensemble_spec["members"]),
                    target_horizon_months=horizon,
                )
            if candidate_summary is None:
                continue
            review_rows.append(
                _snapshot_to_row(
                    candidate_summary,
                    shadow_summary=shadow_summary,
                    live_summary=live_summary,
                    path_name=ensemble_name,
                )
            )

    review_detail_df = pd.DataFrame(review_rows)
    review_summary_df = summarize_v20_review(review_detail_df)
    decision = choose_v20_decision(metric_summary_df, review_summary_df)
    decision_df = pd.DataFrame([{"status": decision.status, "recommended_candidate": decision.recommended_candidate, "rationale": decision.rationale}])

    manifest_rows: list[dict[str, object]] = []
    for spec_name, spec in model_specs.items():
        manifest_rows.append(
            {
                "entry_name": spec_name,
                "entry_type": spec.candidate_type,
                "model_type": spec.model_type,
                "members": "",
                "n_features": len(spec.features),
                "feature_columns": ",".join(spec.features),
                "notes": spec.notes,
            }
        )
    for ensemble_name, ensemble_spec in ensemble_specs.items():
        members = list(ensemble_spec["members"])
        manifest_rows.append(
            {
                "entry_name": ensemble_name,
                "entry_type": str(ensemble_spec["candidate_type"]),
                "model_type": "ensemble",
                "members": ",".join(members),
                "n_features": sum(len(model_specs[name].features) for name in members),
                "feature_columns": "",
                "notes": str(ensemble_spec["notes"]),
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)

    metric_detail_path = Path(output_dir) / f"v20_candidate_metric_detail_{stamp}.csv"
    metric_summary_path = Path(output_dir) / f"v20_candidate_metric_summary_{stamp}.csv"
    review_detail_path = Path(output_dir) / f"v20_shadow_review_detail_{stamp}.csv"
    review_summary_path = Path(output_dir) / f"v20_shadow_review_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v20_promotion_decision_{stamp}.csv"
    manifest_path = Path(output_dir) / f"v20_candidate_manifest_{stamp}.csv"
    metric_detail_df.to_csv(metric_detail_path, index=False)
    metric_summary_df.to_csv(metric_summary_path, index=False)
    review_detail_df.to_csv(review_detail_path, index=False)
    review_summary_df.to_csv(review_summary_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    top_row = metric_summary_df.iloc[0]
    review_row = review_summary_df[review_summary_df["path_name"] == decision.recommended_candidate]
    review_block: list[str] = []
    if not review_row.empty:
        row = review_row.iloc[0]
        review_block = [
            "",
            "## Best-Candidate Review Behavior",
            "",
            f"- Path: `{decision.recommended_candidate}`",
            f"- Signal agreement with shadow baseline: `{float(row['signal_agreement_with_shadow_rate']):.1%}`",
            f"- Mode agreement with shadow baseline: `{float(row['mode_agreement_with_shadow_rate']):.1%}`",
            f"- Signal changes: `{int(row['signal_changes'])}`",
            f"- Underperform share: `{float(row['underperform_rate']):.1%}`",
            f"- Outperform share: `{float(row['outperform_rate']):.1%}`",
            f"- Neutral share: `{float(row['neutral_rate']):.1%}`",
        ]

    result_lines = [
        "# V20 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v20 is a narrow synthesis and promotion-readiness study.",
        "- It assembles the strongest confirmed v16-v19 swaps into a small set of Ridge+GBT replacement stacks.",
        "- It compares those stacks against the reduced live production cross-check, the historical-mean baseline, and the promoted simpler baseline in a monthly shadow review.",
        "",
        "## Forecast Universe",
        "",
        f"- Reduced universe: `{', '.join(V20_FORECAST_UNIVERSE)}`",
        "",
        "## Promotion Decision",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended candidate: `{decision.recommended_candidate}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Top Metric Row",
        "",
        f"- Candidate: `{top_row['candidate_name']}`",
        f"- Mean sign-policy return: `{float(top_row['mean_policy_return_sign']):.4f}`",
        f"- Mean neutral-band return: `{float(top_row['mean_policy_return_neutral_3pct']):.4f}`",
        f"- Mean OOS R^2: `{float(top_row['mean_oos_r2']):.4f}`",
        f"- Mean IC: `{float(top_row['mean_ic']):.4f}`",
        f"- Mean hit rate: `{float(top_row['mean_hit_rate']):.4f}`",
    ]
    result_lines.extend(review_block)
    result_lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            f"- `results/v20/{metric_detail_path.name}`",
            f"- `results/v20/{metric_summary_path.name}`",
            f"- `results/v20/{review_detail_path.name}`",
            f"- `results/v20/{review_summary_path.name}`",
            f"- `results/v20/{decision_path.name}`",
            f"- `results/v20/{manifest_path.name}`",
        ]
    )
    _write_text(Path("docs") / "results" / "V20_RESULTS_SUMMARY.md", result_lines)

    closeout_lines = [
        "# V20 Closeout And V21 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v20 assembled the strongest confirmed v16-v19 swaps into a small set of replacement stacks and tested them against both research metrics and user-facing monthly behavior.",
        "",
        "## Result",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended candidate: `{decision.recommended_candidate}`",
        f"- Decision rationale: {decision.rationale}",
        "",
        "## Recommended V21 Scope",
        "",
        "- If v20 promotes, implement the selected cross-check candidate without changing the promoted v13.1 recommendation layer.",
        "- If v20 does not promote, keep the current cross-check and focus v21 on the blocked-source items or on narrower calibration diagnostics rather than another generic feature sweep.",
    ]
    _write_text(Path("docs") / "closeouts" / "V20_CLOSEOUT_AND_V21_NEXT.md", closeout_lines)

    plan_lines = [
        "# codex-v20-plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Goal",
        "",
        "- Build one best-of-v19 candidate stack from the strongest confirmed swaps and judge whether it is promotion-ready.",
        "",
        "## Candidate Stacks",
        "",
        "- `ensemble_ridge_gbt_v16`",
        "- `ensemble_ridge_gbt_v18`",
        "- `ensemble_ridge_gbt_v20_value`",
        "- `ensemble_ridge_gbt_v20_best`",
        "- `ensemble_ridge_gbt_v20_usd`",
        "- `ensemble_ridge_gbt_v20_pricing`",
        "",
        "## Comparators",
        "",
        "- `live_production_ensemble_reduced`",
        "- `baseline_historical_mean`",
        "- promoted simpler shadow baseline via monthly review",
        "",
        "## Gate",
        "",
        "- Promote only if the best assembled stack improves metrics and behaves at least as cleanly as the current cross-check versus the promoted simpler baseline.",
    ]
    _write_text(Path("docs") / "plans" / "codex-v20-plan.md", plan_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v20 synthesis study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--horizon", default=str(DEFAULT_HORIZON), help="Target horizon in months.")
    parser.add_argument("--review-months", default=str(DEFAULT_REVIEW_MONTHS), help="Number of recent monthly review snapshots.")
    parser.add_argument("--as-of", default="", help="Optional end as-of date (YYYY-MM-DD).")
    args = parser.parse_args()
    end_as_of = date.fromisoformat(args.as_of) if args.as_of else None
    run_v20_synthesis_study(
        output_dir=args.output_dir,
        horizon=int(args.horizon),
        review_months=int(args.review_months),
        end_as_of=end_as_of,
    )


if __name__ == "__main__":
    main()
