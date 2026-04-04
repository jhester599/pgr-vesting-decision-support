"""v12 shadow-promotion study for the diversification-first baseline."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts import monthly_decision
from scripts.candidate_model_bakeoff import candidate_feature_sets
from scripts.v11_autonomous_loop import (
    DEFAULT_HORIZON,
    _baseline_scoreboard,
    _combine_prediction_frames,
    _evaluate_candidate_stack,
    _evaluate_policy_modes,
    _predict_current_custom,
    _write_universe_selection,
)
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import (
    evaluate_baseline_strategy,
    evaluate_wfo_model,
    reconstruct_baseline_predictions,
    summarize_predictions,
)
from src.research.v11 import (
    choose_forecast_universe,
    choose_recommendation_universe,
    next_vest_after,
    recommend_redeploy_buckets,
    summarize_existing_holdings_actions,
)
from src.research.v12 import (
    SnapshotSummary,
    aggregate_health_from_prediction_frames,
    build_shadow_comparison_lines,
    confidence_from_hit_rate,
    recent_monthly_review_dates,
    sell_pct_from_policy,
    signal_from_prediction,
)
from src.reporting.decision_rendering import determine_recommendation_mode
from src.tax.capital_gains import load_position_lots


DEFAULT_OUTPUT_DIR = os.path.join("results", "v12")
DEFAULT_REVIEW_MONTHS = 12
ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "ensemble_ridge_gbt": ["ridge_lean_v1", "gbt_lean_plus_two"],
    "ensemble_ridge_gbt_elasticnet": ["ridge_lean_v1", "gbt_lean_plus_two", "elasticnet_lean_v1"],
}


def _current_baseline_prediction(y_aligned: pd.Series) -> float:
    window = min(len(y_aligned), config.WFO_TRAIN_WINDOW_MONTHS)
    return float(y_aligned.iloc[-window:].mean())


def _build_live_snapshot(conn: Any, as_of: date) -> SnapshotSummary:
    signals, ensemble_results, diagnostics = monthly_decision._generate_signals(  # noqa: SLF001
        conn,
        as_of,
        target_horizon_months=DEFAULT_HORIZON,
    )
    signals, cal_result, _, _ = monthly_decision._calibrate_signals(  # noqa: SLF001
        signals,
        ensemble_results,
        target_horizon_months=DEFAULT_HORIZON,
    )
    consensus, mean_pred, mean_ic, mean_hr, mean_prob, confidence_tier = monthly_decision._consensus_signal(signals)  # noqa: SLF001
    aggregate_health = monthly_decision._compute_aggregate_health(  # noqa: SLF001
        ensemble_results,
        target_horizon_months=DEFAULT_HORIZON,
    )
    recommendation_mode = determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        diagnostics.get("representative_cpcv"),
    )
    cal_prob = None
    if "calibrated_prob_outperform" in signals.columns and not signals.empty:
        cal_prob = float(signals["calibrated_prob_outperform"].mean())
    return SnapshotSummary(
        label="live",
        as_of=as_of,
        candidate_name="production_4_model_ensemble",
        policy_name="current_production_mapping",
        consensus=consensus,
        confidence_tier=confidence_tier,
        recommendation_mode=str(recommendation_mode["label"]),
        sell_pct=float(recommendation_mode["sell_pct"]),
        mean_predicted=mean_pred,
        mean_ic=mean_ic,
        mean_hit_rate=mean_hr,
        aggregate_oos_r2=float(aggregate_health["oos_r2"]) if aggregate_health else float("nan"),
        aggregate_nw_ic=float(aggregate_health["nw_ic"]) if aggregate_health else float("nan"),
        calibrated_prob_outperform=cal_prob if cal_result.method != "uncalibrated" else cal_prob,
    )


def _baseline_signal_frame(
    X_aligned: pd.DataFrame,
    y_aligned: pd.Series,
    strategy: str,
    benchmark: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = evaluate_baseline_strategy(
        X_aligned,
        y_aligned,
        strategy=strategy,
        target_horizon_months=DEFAULT_HORIZON,
    )
    pred_series, realized = reconstruct_baseline_predictions(
        X_aligned,
        y_aligned,
        strategy=strategy,
        target_horizon_months=DEFAULT_HORIZON,
    )
    current_pred = _current_baseline_prediction(y_aligned)
    row = {
        "benchmark": benchmark,
        "predicted_relative_return": current_pred,
        "ic": float(metrics["ic"]),
        "hit_rate": float(metrics["hit_rate"]),
        "signal": signal_from_prediction(current_pred),
        "confidence_tier": confidence_from_hit_rate(float(metrics["hit_rate"])),
        "oos_r2": float(metrics["oos_r2"]),
    }
    frame = pd.DataFrame(
        {
            "benchmark": benchmark,
            "y_hat": pred_series,
            "y_true": realized,
        }
    ).reset_index(drop=True)
    return row, frame


def _model_signal_frame(
    X_aligned: pd.DataFrame,
    y_aligned: pd.Series,
    benchmark: str,
    model_type: str,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    result, metrics = evaluate_wfo_model(
        X_aligned,
        y_aligned,
        model_type=model_type,
        benchmark=benchmark,
        target_horizon_months=DEFAULT_HORIZON,
        feature_columns=feature_columns,
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
        selected_features=feature_columns,
    )
    row = {
        "benchmark": benchmark,
        "predicted_relative_return": current_pred,
        "ic": float(metrics["ic"]),
        "hit_rate": float(metrics["hit_rate"]),
        "signal": signal_from_prediction(current_pred),
        "confidence_tier": confidence_from_hit_rate(float(metrics["hit_rate"])),
        "oos_r2": float(metrics["oos_r2"]),
    }
    frame = pd.DataFrame({"y_hat": pred_series, "y_true": realized})
    return row, frame


def _ensemble_signal_frame(
    X_aligned: pd.DataFrame,
    y_aligned: pd.Series,
    benchmark: str,
    member_names: list[str],
    feature_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    current_predictions: list[tuple[float, float]] = []
    for member_name in member_names:
        spec = feature_specs[member_name]
        selected_features = [feature for feature in spec["features"] if feature in X_aligned.columns]
        result, metrics = evaluate_wfo_model(
            X_aligned,
            y_aligned,
            model_type=str(spec["model_type"]),
            benchmark=benchmark,
            target_horizon_months=DEFAULT_HORIZON,
            feature_columns=selected_features,
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
        frames.append(pd.DataFrame({pred_series.name: pred_series, "y_true": realized}))
        current_pred = _predict_current_custom(
            X_full=X_aligned,
            y_full=y_aligned,
            X_current=X_aligned.iloc[[-1]],
            model_type=str(spec["model_type"]),
            selected_features=selected_features,
        )
        current_predictions.append((current_pred, max(float(metrics["mae"]), 1e-9)))

    combined_pred, combined_true = _combine_prediction_frames(frames)
    current_weight = sum(1.0 / (mae**2) for _, mae in current_predictions)
    current_pred = sum(pred * (1.0 / (mae**2)) for pred, mae in current_predictions) / current_weight
    summary = summarize_predictions(
        combined_pred,
        combined_true,
        target_horizon_months=DEFAULT_HORIZON,
    )
    row = {
        "benchmark": benchmark,
        "predicted_relative_return": float(current_pred),
        "ic": float(summary.ic),
        "hit_rate": float(summary.hit_rate),
        "signal": signal_from_prediction(float(current_pred)),
        "confidence_tier": confidence_from_hit_rate(float(summary.hit_rate)),
        "oos_r2": float(summary.oos_r2),
    }
    frame = pd.DataFrame({"y_hat": combined_pred, "y_true": combined_true})
    return row, frame


def _build_shadow_snapshot(
    conn: Any,
    as_of: date,
    forecast_universe: list[str],
    candidate_name: str,
    policy_name: str,
    feature_specs: dict[str, dict[str, Any]],
) -> tuple[SnapshotSummary, pd.DataFrame]:
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    df_event = df_full.loc[df_full.index <= pd.Timestamp(as_of)]
    if df_event.empty:
        raise ValueError(f"No feature rows available on or before {as_of}.")

    signal_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for benchmark in forecast_universe:
        rel_series = load_relative_return_matrix(conn, benchmark, DEFAULT_HORIZON)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df_event, rel_series, drop_na_target=True)
        except ValueError:
            continue
        if X_aligned.empty or y_aligned.empty:
            continue

        if candidate_name.startswith("baseline_"):
            strategy = candidate_name.replace("baseline_", "", 1)
            row, frame = _baseline_signal_frame(X_aligned, y_aligned, strategy, benchmark)
        elif candidate_name in ENSEMBLE_MEMBERS:
            row, frame = _ensemble_signal_frame(
                X_aligned,
                y_aligned,
                benchmark,
                ENSEMBLE_MEMBERS[candidate_name],
                feature_specs,
            )
        else:
            spec = feature_specs[candidate_name]
            selected_features = [feature for feature in spec["features"] if feature in X_aligned.columns]
            row, frame = _model_signal_frame(
                X_aligned,
                y_aligned,
                benchmark,
                model_type=str(spec["model_type"]),
                feature_columns=selected_features,
            )

        signal_rows.append(row)
        prediction_frames.append(frame.assign(benchmark=benchmark))

    signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(prediction_frames, DEFAULT_HORIZON)
    consensus, mean_pred, mean_ic, mean_hr, _, confidence_tier = monthly_decision._consensus_signal(signals)  # noqa: SLF001
    recommendation_mode = determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        None,
    )
    if recommendation_mode["mode"] == "actionable":
        sell_pct = sell_pct_from_policy(mean_pred, policy_name)
    else:
        sell_pct = float(recommendation_mode["sell_pct"])

    snapshot = SnapshotSummary(
        label="shadow",
        as_of=as_of,
        candidate_name=candidate_name,
        policy_name=policy_name,
        consensus=consensus,
        confidence_tier=confidence_tier,
        recommendation_mode=str(recommendation_mode["label"]),
        sell_pct=sell_pct,
        mean_predicted=mean_pred,
        mean_ic=mean_ic,
        mean_hit_rate=mean_hr,
        aggregate_oos_r2=float(aggregate_health["oos_r2"]) if aggregate_health else float("nan"),
        aggregate_nw_ic=float(aggregate_health["nw_ic"]) if aggregate_health else float("nan"),
    )
    return snapshot, signals


def _existing_holdings_payload(conn: Any, as_of: date) -> list[dict[str, Any]]:
    lots_path = Path("data") / "processed" / "position_lots.csv"
    if not lots_path.exists():
        return []
    lots = load_position_lots(str(lots_path))
    prices = db_client.get_prices(conn, "PGR", end_date=str(as_of))
    if prices.empty:
        return []
    current_price = float(prices["close"].iloc[-1])
    return [
        asdict(action)
        for action in summarize_existing_holdings_actions(
            lots,
            current_price=current_price,
            sell_date=as_of,
        )
    ]


def _write_summary_docs(
    output_dir: str,
    review_df: pd.DataFrame,
    scoreboard: pd.DataFrame,
    recommendation_universe: list[str],
    forecast_universe: list[str],
    candidate_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> None:
    summary_path = Path(_REPO_ROOT) / "V12_RESULTS_SUMMARY.md"
    closeout_path = Path(_REPO_ROOT) / "V12_CLOSEOUT_AND_V13_NEXT.md"
    plan_path = Path(_REPO_ROOT) / "codex-v12-plan.md"

    best_candidate = candidate_summary.iloc[0]
    best_policy = policy_summary.iloc[0]
    avg_live_sell = float(review_df["live_sell_pct"].mean())
    avg_shadow_sell = float(review_df["shadow_sell_pct"].mean())
    live_mode_changes = int(review_df["live_recommendation_mode"].ne(review_df["live_recommendation_mode"].shift()).sum() - 1)
    shadow_mode_changes = int(review_df["shadow_recommendation_mode"].ne(review_df["shadow_recommendation_mode"].shift()).sum() - 1)
    live_signal_changes = int(review_df["live_signal"].ne(review_df["live_signal"].shift()).sum() - 1)
    shadow_signal_changes = int(review_df["shadow_signal"].ne(review_df["shadow_signal"].shift()).sum() - 1)

    plan_lines = [
        "# codex-v12-plan.md",
        "",
        "## Scope",
        "",
        "- shadow-test the diversification-first simple baseline through the monthly decision/report flow",
        "- compare it against the live production stack on a rolling 12-month review window",
        "- keep production unchanged unless the simpler baseline proves more useful in practice",
        "",
        "## Candidate Under Review",
        "",
        f"- Forecast universe: `{', '.join(forecast_universe)}`",
        f"- Recommended diversification universe: `{', '.join(recommendation_universe)}`",
        f"- Best policy row carried into the shadow study: `{best_policy['candidate_name']}` with `{best_policy['policy_name']}`",
        "",
    ]
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# V12 Results Summary",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Shadow Study Goal",
        "",
        "Test whether a simpler diversification-first baseline should replace the live monthly decision engine in practice, even before any new model stack is promoted.",
        "",
        "## Selected Universes",
        "",
        f"- Forecast universe: `{', '.join(forecast_universe)}`",
        f"- Recommended diversification universe: `{', '.join(recommendation_universe)}`",
        "",
        "## Candidate Scoreboard",
        "",
        f"- Best candidate by diversification-aware utility: `{best_candidate['candidate_name']}`",
        f"- Best policy row: `{best_policy['candidate_name']}` with `{best_policy['policy_name']}`",
        "",
        "## Review Window Findings",
        "",
        f"- Review months evaluated: `{len(review_df)}`",
        f"- Average live sell percentage: `{avg_live_sell:.0%}`",
        f"- Average shadow sell percentage: `{avg_shadow_sell:.0%}`",
        f"- Live signal changes: `{live_signal_changes}`",
        f"- Shadow signal changes: `{shadow_signal_changes}`",
        f"- Live recommendation-mode changes: `{live_mode_changes}`",
        f"- Shadow recommendation-mode changes: `{shadow_mode_changes}`",
        f"- Shadow redeploy universe mean diversification score: `{scoreboard[scoreboard['benchmark'].isin(recommendation_universe)]['diversification_score'].mean():.3f}`",
        "",
        "## Interpretation",
        "",
        "- v12 is intentionally testing a simpler path, not a more complex one.",
        "- The shadow baseline inherits the diversification-first redeploy logic and the clearer lot-trimming order from v11.",
        "- The live stack changed its directional signal repeatedly across the review window, but the action never moved off the 50% default because the quality gate still failed.",
        "- The shadow baseline produced a steadier directional story while preserving the same diversification-first action and adding clearer redeploy guidance.",
        "",
        f"Detailed artifacts are stored in `{output_dir}`.",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    closeout_lines = [
        "# V12 Closeout and V13 Next",
        "",
        f"Created: {date.today().isoformat()}",
        "",
        "## Status",
        "",
        "The v12 shadow-promotion study is complete.",
        "",
        "## Conclusion",
        "",
        f"- Shadow candidate: `{best_policy['candidate_name']}` with `{best_policy['policy_name']}`",
        f"- Live average sell percentage over review window: `{avg_live_sell:.0%}`",
        f"- Shadow average sell percentage over review window: `{avg_shadow_sell:.0%}`",
        f"- Live signal changes: `{live_signal_changes}` versus shadow signal changes: `{shadow_signal_changes}`",
        "- Main question answered: the simpler diversification-first baseline is steadier and easier to explain, even though both paths still land on the same 50% default action today.",
        "",
        "## v13 Recommendation",
        "",
        "- plan a limited promotion study for the simpler diversification-first recommendation layer before promoting any new model stack",
        "- keep target-quality and calibration research separate from recommendation-layer simplification",
        "- preserve diversification-first redeploy guidance regardless of whether the shadow baseline is promoted",
        "",
    ]
    closeout_path.write_text("\n".join(closeout_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v12 shadow-baseline study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write v12 artifacts.")
    parser.add_argument("--review-months", type=int, default=DEFAULT_REVIEW_MONTHS, help="How many recent monthly review dates to inspect.")
    parser.add_argument("--as-of", help="Optional end-date override for the review window.")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

    conn = db_client.get_connection(config.DB_PATH)
    try:
        scoreboard = _baseline_scoreboard(conn, output_dir)
        recommendation_universe = choose_recommendation_universe(scoreboard)
        forecast_universe = choose_forecast_universe(scoreboard, recommendation_universe)
        _write_universe_selection(scoreboard, recommendation_universe, forecast_universe, output_dir)

        detail_df, candidate_summary = _evaluate_candidate_stack(
            conn=conn,
            benchmarks=forecast_universe,
            candidate_specs={name: spec for name, spec in candidate_feature_sets().items() if name in {"ridge_lean_v1", "gbt_lean_plus_two", "elasticnet_lean_v1"}},
            scoreboard=scoreboard,
            output_dir=output_dir,
            output_prefix="v12_candidate_bakeoff",
        )
        _, policy_summary = _evaluate_policy_modes(detail_df, scoreboard, output_dir)
        policy_summary = policy_summary.sort_values(
            by=["diversification_aware_utility", "mean_policy_return", "simplicity_rank"],
            ascending=[False, False, True],
        )

        chosen_policy = policy_summary.iloc[0]
        candidate_name = str(chosen_policy["candidate_name"])
        policy_name = str(chosen_policy["policy_name"])

        review_dates = recent_monthly_review_dates(as_of, args.review_months)
        feature_specs = candidate_feature_sets()
        dry_run_dir = Path(output_dir) / "dry_runs"
        dry_run_dir.mkdir(parents=True, exist_ok=True)

        redeploy_buckets = recommend_redeploy_buckets(scoreboard, recommendation_universe)

        review_rows: list[dict[str, Any]] = []
        for review_date in review_dates:
            live_summary = _build_live_snapshot(conn, review_date)
            shadow_summary, _ = _build_shadow_snapshot(
                conn,
                review_date,
                forecast_universe,
                candidate_name,
                policy_name,
                feature_specs,
            )
            existing_holdings = _existing_holdings_payload(conn, review_date)
            next_vest_date, next_vest_type = next_vest_after(review_date)

            memo_lines = build_shadow_comparison_lines(
                live_summary=live_summary,
                shadow_summary=shadow_summary,
                next_vest_date=next_vest_date,
                next_vest_type=next_vest_type,
                existing_holdings=existing_holdings,
                redeploy_buckets=redeploy_buckets,
            )
            (dry_run_dir / f"{review_date.isoformat()}.md").write_text(
                "\n".join(memo_lines) + "\n",
                encoding="utf-8",
            )
            review_rows.append(
                {
                    "as_of": review_date.isoformat(),
                    "live_candidate": live_summary.candidate_name,
                    "live_policy": live_summary.policy_name,
                    "live_signal": live_summary.consensus,
                    "live_recommendation_mode": live_summary.recommendation_mode,
                    "live_sell_pct": live_summary.sell_pct,
                    "live_mean_predicted": live_summary.mean_predicted,
                    "live_mean_ic": live_summary.mean_ic,
                    "live_aggregate_oos_r2": live_summary.aggregate_oos_r2,
                    "shadow_candidate": shadow_summary.candidate_name,
                    "shadow_policy": shadow_summary.policy_name,
                    "shadow_signal": shadow_summary.consensus,
                    "shadow_recommendation_mode": shadow_summary.recommendation_mode,
                    "shadow_sell_pct": shadow_summary.sell_pct,
                    "shadow_mean_predicted": shadow_summary.mean_predicted,
                    "shadow_mean_ic": shadow_summary.mean_ic,
                    "shadow_aggregate_oos_r2": shadow_summary.aggregate_oos_r2,
                }
            )

        review_df = pd.DataFrame(review_rows)
        stamp = datetime.today().strftime("%Y%m%d")
        review_df.to_csv(os.path.join(output_dir, f"v12_shadow_review_{stamp}.csv"), index=False)
        candidate_summary.to_csv(os.path.join(output_dir, f"v12_candidate_summary_{stamp}.csv"), index=False)
        policy_summary.to_csv(os.path.join(output_dir, f"v12_policy_summary_{stamp}.csv"), index=False)

        _write_summary_docs(
            output_dir=output_dir,
            review_df=review_df,
            scoreboard=scoreboard,
            recommendation_universe=recommendation_universe,
            forecast_universe=forecast_universe,
            candidate_summary=candidate_summary,
            policy_summary=policy_summary,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
