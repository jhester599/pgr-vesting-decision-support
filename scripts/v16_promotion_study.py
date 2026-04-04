"""v16 narrow promotion study for the best v15 feature-replacement candidate."""

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
from src.research.v16 import (
    V16_FORECAST_UNIVERSE,
    choose_v16_promotion,
    v16_ensemble_specs,
    v16_model_specs,
)


DEFAULT_OUTPUT_DIR = os.path.join("results", "v16")
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
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
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
        pred_series = pd.Series(
            result.y_hat_all,
            index=pd.DatetimeIndex(result.test_dates_all),
            name=f"pred_{benchmark}__{max(float(metrics['mae']), 1e-9)}",
        )
        realized = pd.Series(
            result.y_true_all,
            index=pd.DatetimeIndex(result.test_dates_all),
            name="y_true",
        )
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
        frames.append(pd.DataFrame({"y_hat": pred_series, "y_true": realized}))
    return pd.DataFrame(rows), frames


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


def _evaluate_ensemble_candidate(
    candidate_name: str,
    member_names: list[str],
    model_specs: dict[str, Any],
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
    notes: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []

    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
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
            realized = pd.Series(
                result.y_true_all,
                index=pd.DatetimeIndex(result.test_dates_all),
                name="y_true",
            )
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

        combined_pred, combined_true = _combine_prediction_frames(member_frames)
        summary = summarize_predictions(combined_pred, combined_true, target_horizon_months=horizon)
        sign_policy = evaluate_policy_series(combined_pred, combined_true, "sign_hold_vs_sell")
        neutral_policy = evaluate_policy_series(combined_pred, combined_true, "neutral_band_3pct")
        total_weight = sum(1.0 / (mae**2) for _, mae in current_predictions)
        current_pred = sum(pred * (1.0 / (mae**2)) for pred, mae in current_predictions) / total_weight

        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_type": "ensemble",
                "model_type": "ensemble",
                "benchmark": benchmark,
                "n_features": feature_count,
                "feature_columns": " | ".join(feature_labels),
                "notes": notes,
                "current_predicted_return": float(current_pred),
                "policy_return_sign": sign_policy.mean_policy_return,
                "policy_return_neutral_3pct": neutral_policy.mean_policy_return,
                "ic": summary.ic,
                "hit_rate": summary.hit_rate,
                "mae": summary.mae,
                "oos_r2": summary.oos_r2,
                "nw_ic": summary.nw_ic,
            }
        )
        all_frames.append(pd.DataFrame({"y_hat": combined_pred, "y_true": combined_true}))

    return pd.DataFrame(rows), all_frames


def _evaluate_baseline_historical_mean(
    benchmark_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    horizon: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for benchmark, (X_aligned, y_aligned) in benchmark_data.items():
        metrics = evaluate_baseline_strategy(
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
        frames.append(pd.DataFrame({"y_hat": pred_series, "y_true": realized}))
    return pd.DataFrame(rows), frames


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


def run_v16_promotion_study(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    horizon: int = DEFAULT_HORIZON,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    df = build_feature_matrix_from_db(conn, force_refresh=True)
    benchmark_data = _benchmark_dataset_map(conn, df, list(V16_FORECAST_UNIVERSE), horizon)
    stamp = datetime.today().strftime("%Y%m%d")

    model_specs = v16_model_specs()
    ensemble_specs = v16_ensemble_specs()

    detail_frames: list[pd.DataFrame] = []
    for spec_name, spec in model_specs.items():
        detail_df, _ = _evaluate_model_candidate(
            spec_name=spec_name,
            model_type=spec.model_type,
            feature_columns=spec.features,
            benchmark_data=benchmark_data,
            horizon=horizon,
            notes=spec.notes,
        )
        detail_frames.append(detail_df)

    for ensemble_name, ensemble_spec in ensemble_specs.items():
        detail_df, _ = _evaluate_ensemble_candidate(
            candidate_name=ensemble_name,
            member_names=list(ensemble_spec["members"]),
            model_specs=model_specs,
            benchmark_data=benchmark_data,
            horizon=horizon,
            notes=str(ensemble_spec["notes"]),
        )
        detail_frames.append(detail_df)

    baseline_detail, _ = _evaluate_baseline_historical_mean(benchmark_data, horizon)
    detail_frames.append(baseline_detail)

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = _summarize_candidates(detail_df)
    decision = choose_v16_promotion(summary_df)
    decision_df = pd.DataFrame(
        [
            {
                "status": decision.status,
                "recommended_candidate": decision.recommended_candidate,
                "rationale": decision.rationale,
            }
        ]
    )

    detail_path = Path(output_dir) / f"v16_candidate_bakeoff_detail_{stamp}.csv"
    summary_path = Path(output_dir) / f"v16_candidate_bakeoff_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v16_promotion_decision_{stamp}.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    decision_df.to_csv(decision_path, index=False)

    top_row = summary_df.iloc[0]
    lines = [
        "# V16 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v16 is a narrow promotion study, not a new feature search.",
        "- It tests the two confirmed v15 replacements inside the leading Ridge+GBT candidate stack.",
        "- The recommendation layer remains fixed at the promoted v13.1 path.",
        "",
        "## Forecast Universe",
        "",
        f"- Reduced universe: `{', '.join(V16_FORECAST_UNIVERSE)}`",
        "",
        "## Promotion Decision",
        "",
        f"- Status: `{decision.status}`",
        f"- Candidate: `{decision.recommended_candidate}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Top Row",
        "",
        f"- Candidate: `{top_row['candidate_name']}`",
        f"- Type: `{top_row['candidate_type']}`",
        f"- Mean sign-policy return: `{float(top_row['mean_policy_return_sign']):.4f}`",
        f"- Mean neutral-band return: `{float(top_row['mean_policy_return_neutral_3pct']):.4f}`",
        f"- Mean OOS R^2: `{float(top_row['mean_oos_r2']):.4f}`",
        f"- Mean IC: `{float(top_row['mean_ic']):.4f}`",
        "",
        "## Output Artifacts",
        "",
        f"- `results/v16/{detail_path.name}`",
        f"- `results/v16/{summary_path.name}`",
        f"- `results/v16/{decision_path.name}`",
    ]
    _write_text(Path("docs") / "results" / "V16_RESULTS_SUMMARY.md", lines)

    closeout_lines = [
        "# V16 Closeout And V17 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v16 completed the narrow promotion study recommended at the end of v15.",
        "- The study compared the modified Ridge+GBT pair against the reduced-universe live stack and the historical-mean baseline.",
        "",
        "## Result",
        "",
        f"- Promotion status: `{decision.status}`",
        f"- Candidate reviewed: `{decision.recommended_candidate}`",
        f"- Decision rationale: {decision.rationale}",
        "",
        "## Recommended V17 Scope",
        "",
        "- If v16 does not promote, keep the live prediction layer unchanged and continue with a narrow v17 feature phase focused only on the highest-value deferred families.",
        "- If v16 does promote, implement the Ridge and GBT feature swaps without changing the recommendation layer.",
        "- In either case, keep the v13.1 recommendation layer in place until a later study proves that a new prediction stack improves real usefulness as well as metrics.",
    ]
    _write_text(Path("docs") / "closeouts" / "V16_CLOSEOUT_AND_V17_NEXT.md", closeout_lines)

    plan_lines = [
        "# codex-v16-plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Goal",
        "",
        "- Run a narrow promotion study on the best v15 feature-replacement candidate rather than reopening the feature search.",
        "",
        "## Candidate Stack",
        "",
        "- Base candidate: `ensemble_ridge_gbt_v14`",
        "- Modified candidate: `ensemble_ridge_gbt_v16`",
        "- Modified swaps:",
        "  - Ridge: `book_value_per_share_growth_yoy` replacing `roe_net_income_ttm`",
        "  - GBT: `rate_adequacy_gap_yoy` replacing `vmt_yoy`",
        "",
        "## Comparators",
        "",
        "- `live_production_ensemble_reduced`",
        "- `baseline_historical_mean`",
        "- individual modified and unmodified Ridge / GBT rows",
        "",
        "## Promotion Gate",
        "",
        "- Promote only if the modified pair clearly beats the reduced live stack and also separates enough from the historical-mean baseline.",
        "- Otherwise keep the v13.1 recommendation layer and current live prediction layer in production.",
    ]
    _write_text(Path("docs") / "plans" / "codex-v16-plan.md", plan_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v16 narrow promotion study.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--horizon",
        default=str(DEFAULT_HORIZON),
        help="Target horizon in months.",
    )
    args = parser.parse_args()
    run_v16_promotion_study(
        output_dir=args.output_dir,
        horizon=int(args.horizon),
    )


if __name__ == "__main__":
    main()
