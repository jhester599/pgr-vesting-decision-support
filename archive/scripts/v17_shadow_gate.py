"""v17 shadow-gate study for promoting the v16 candidate as the live cross-check."""

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
from src.research.evaluation import evaluate_wfo_model, summarize_predictions
from src.research.v12 import (
    SnapshotSummary,
    aggregate_health_from_prediction_frames,
    recent_monthly_review_dates,
    signal_from_prediction,
)
from src.research.v16 import V16_FORECAST_UNIVERSE, v16_ensemble_specs, v16_model_specs
from src.research.v17 import choose_v17_promotion, summarize_shadow_review


DEFAULT_OUTPUT_DIR = os.path.join("results", "v17")
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


def _build_candidate_snapshot(
    conn: Any,
    as_of: date,
    target_horizon_months: int = DEFAULT_HORIZON,
) -> SnapshotSummary | None:
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    X_event = df_full.loc[df_full.index <= pd.Timestamp(as_of)]
    if X_event.empty:
        return None

    model_specs = v16_model_specs()
    ensemble_members = v16_ensemble_specs()["ensemble_ridge_gbt_v16"]["members"]
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
        for member_name in ensemble_members:
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
                    target_horizon_months=target_horizon_months,
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

        combined = member_frames[0].copy()
        for frame in member_frames[1:]:
            pred_cols = [col for col in frame.columns if col.startswith("pred_")]
            combined = combined.join(frame[pred_cols], how="inner")
        pred_cols = [col for col in combined.columns if col.startswith("pred_")]
        weight_map = {
            col: 1.0 / max(float(col.split("__")[-1]), 1e-9) ** 2
            for col in pred_cols
        }
        total_weight = sum(weight_map.values())
        combined["y_hat"] = sum(combined[col] * (weight_map[col] / total_weight) for col in pred_cols)
        pred_series = combined["y_hat"]
        realized = combined["y_true"]
        current_weight = sum(1.0 / (mae**2) for _, mae in current_predictions)
        current_pred = sum(pred * (1.0 / (mae**2)) for pred, mae in current_predictions) / current_weight
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
        label="candidate_v16",
        as_of=as_of,
        candidate_name="ensemble_ridge_gbt_v16",
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
    }


def run_v17_shadow_gate(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    review_months: int = DEFAULT_REVIEW_MONTHS,
    end_as_of: date | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = db_client.get_connection(config.DB_PATH)
    as_of_end = end_as_of or date.today()
    review_dates = recent_monthly_review_dates(as_of_end, review_months)

    detail_rows: list[dict[str, object]] = []
    for as_of in review_dates:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            live_summary = v12_shadow_study._build_live_snapshot(conn, as_of)  # noqa: SLF001
            shadow_summary, _ = monthly_decision._build_shadow_baseline_summary(conn, as_of, DEFAULT_HORIZON)  # noqa: SLF001
            candidate_summary = _build_candidate_snapshot(conn, as_of, DEFAULT_HORIZON)
        if shadow_summary is None or candidate_summary is None:
            continue
        detail_rows.append(_snapshot_to_row(shadow_summary, shadow_summary=shadow_summary, path_name="shadow_baseline"))
        detail_rows.append(_snapshot_to_row(live_summary, shadow_summary=shadow_summary, path_name="live_production"))
        detail_rows.append(_snapshot_to_row(candidate_summary, shadow_summary=shadow_summary, path_name="candidate_v16"))

    detail_df = pd.DataFrame(detail_rows)
    review_summary = summarize_shadow_review(detail_df)

    v16_summary = pd.read_csv(Path("results") / "v16" / "v16_candidate_bakeoff_summary_20260404.csv")
    decision = choose_v17_promotion(v16_summary, review_summary)
    decision_df = pd.DataFrame(
        [
            {
                "status": decision.status,
                "recommended_path": decision.recommended_path,
                "rationale": decision.rationale,
            }
        ]
    )

    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = Path(output_dir) / f"v17_shadow_review_detail_{stamp}.csv"
    summary_path = Path(output_dir) / f"v17_shadow_review_summary_{stamp}.csv"
    decision_path = Path(output_dir) / f"v17_promotion_decision_{stamp}.csv"
    detail_df.to_csv(detail_path, index=False)
    review_summary.to_csv(summary_path, index=False)
    decision_df.to_csv(decision_path, index=False)

    lines = [
        "# V17 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- v17 tests whether the modified Ridge+GBT pair should replace the current live production stack as the visible cross-check under the promoted v13.1 recommendation layer.",
        "- The active recommendation layer remains the simpler diversification-first baseline.",
        "",
        "## Review Window",
        "",
        f"- Monthly snapshots reviewed: `{len(review_dates)}`",
        f"- End as-of date: `{as_of_end.isoformat()}`",
        "",
        "## Promotion Decision",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended path: `{decision.recommended_path}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Review Summary",
        "",
    ]
    for row in review_summary.itertuples(index=False):
        lines.extend(
            [
                f"### {row.path_name}",
                "",
                f"- Signal agreement with shadow baseline: `{float(row.signal_agreement_with_shadow_rate):.1%}`",
                f"- Recommendation-mode agreement with shadow baseline: `{float(row.mode_agreement_with_shadow_rate):.1%}`",
                f"- Sell agreement with shadow baseline: `{float(row.sell_agreement_with_shadow_rate):.1%}`",
                f"- Signal changes: `{int(row.signal_changes)}`",
                f"- Mode changes: `{int(row.mode_changes)}`",
                f"- Mean aggregate OOS R^2: `{float(row.mean_aggregate_oos_r2):.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Output Artifacts",
            "",
            f"- `results/v17/{detail_path.name}`",
            f"- `results/v17/{summary_path.name}`",
            f"- `results/v17/{decision_path.name}`",
        ]
    )
    _write_text(Path("docs") / "results" / "V17_RESULTS_SUMMARY.md", lines)

    closeout_lines = [
        "# V17 Closeout And V18 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Closeout",
        "",
        "- v17 evaluated whether the modified Ridge+GBT pair is strong enough to replace the current live stack as the visible production cross-check.",
        "",
        "## Result",
        "",
        f"- Status: `{decision.status}`",
        f"- Recommended path: `{decision.recommended_path}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Recommended V18 Scope",
        "",
        "- If the candidate wins, consider a narrow production change that replaces the current live cross-check with the modified Ridge+GBT pair while leaving the recommendation layer untouched.",
        "- If it does not win, keep the current production cross-check and return to the deferred v15/v16 feature families only if they can be tested without broadening model complexity.",
    ]
    _write_text(Path("docs") / "closeouts" / "V17_CLOSEOUT_AND_V18_NEXT.md", closeout_lines)

    plan_lines = [
        "# codex-v17-plan",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Goal",
        "",
        "- Decide whether the modified Ridge+GBT pair should replace the current live stack as the cross-check path inside the promoted v13.1 recommendation layer.",
        "",
        "## Paths Compared",
        "",
        "- `shadow_baseline`",
        "- `live_production`",
        "- `candidate_v16`",
        "",
        "## Gate",
        "",
        "- Favor the candidate only if it agrees with the simpler baseline at least as well as the current live cross-check, while also carrying forward the v16 metric improvement over the reduced live stack.",
    ]
    _write_text(Path("docs") / "plans" / "codex-v17-plan.md", plan_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v17 shadow-gate study.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--review-months",
        default=str(DEFAULT_REVIEW_MONTHS),
        help="Number of recent monthly review snapshots.",
    )
    parser.add_argument(
        "--as-of",
        default="",
        help="Optional end as-of date (YYYY-MM-DD).",
    )
    args = parser.parse_args()
    end_as_of = date.fromisoformat(args.as_of) if args.as_of else None
    run_v17_shadow_gate(
        output_dir=args.output_dir,
        review_months=int(args.review_months),
        end_as_of=end_as_of,
    )


if __name__ == "__main__":
    main()
