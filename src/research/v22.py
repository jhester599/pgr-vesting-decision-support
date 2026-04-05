"""Helpers for the v22 promoted cross-check implementation step."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import config
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.reporting.decision_rendering import determine_recommendation_mode
from src.research.evaluation import evaluate_wfo_model, summarize_predictions
from src.research.v12 import (
    SnapshotSummary,
    aggregate_health_from_prediction_frames,
    confidence_from_hit_rate,
    signal_from_prediction,
)
from src.research.v20 import V20_FORECAST_UNIVERSE, v20_ensemble_specs, v20_model_specs


@dataclass(frozen=True)
class V22CrossCheckSpec:
    """Definition of the promoted v22 visible cross-check candidate."""

    candidate_name: str
    members: list[str]
    notes: str


def v22_promoted_cross_check_spec() -> V22CrossCheckSpec:
    """Return the single promoted cross-check candidate selected in v21."""
    candidate_name = config.V22_PROMOTED_CROSS_CHECK_CANDIDATE
    ensemble_specs = v20_ensemble_specs()
    if candidate_name not in ensemble_specs:
        raise KeyError(f"Unknown promoted v22 cross-check candidate: {candidate_name}")
    spec = ensemble_specs[candidate_name]
    return V22CrossCheckSpec(
        candidate_name=candidate_name,
        members=list(spec["members"]),
        notes=str(spec["notes"]),
    )


def _predict_current_custom(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_current: pd.DataFrame,
    model_type: str,
    selected_features: list[str],
    train_window_months: int = config.WFO_TRAIN_WINDOW_MONTHS,
) -> float:
    """Fit the selected model on the recent training window and score the latest row."""
    aligned = X_full[selected_features].join(y_full, how="inner").dropna(subset=[y_full.name])
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


def _consensus_from_signals(signals: pd.DataFrame) -> tuple[str, float, float, float, str]:
    """Rebuild the compact consensus summary used by the production memo."""
    if signals.empty:
        return "NEUTRAL", 0.0, 0.0, 0.0, "LOW"

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
    confidence_tier = confidence_from_hit_rate(mean_hr)
    return consensus, mean_pred, mean_ic, mean_hr, confidence_tier


def build_promoted_cross_check_summary(
    conn: Any,
    as_of: date,
    target_horizon_months: int = 6,
) -> SnapshotSummary | None:
    """Build the current snapshot for the promoted v22 visible cross-check candidate."""
    spec = v22_promoted_cross_check_spec()
    model_specs = v20_model_specs()

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

        member_frames: list[pd.DataFrame] = []
        current_predictions: list[tuple[float, float]] = []
        for member_name in spec.members:
            member_spec = model_specs[member_name]
            selected = [feature for feature in member_spec.features if feature in X_aligned.columns]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                result, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_aligned,
                    model_type=member_spec.model_type,
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
                        model_type=member_spec.model_type,
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
        weight_map = {col: 1.0 / max(float(col.split("__")[-1]), 1e-9) ** 2 for col in pred_cols}
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
            }
        )
        prediction_frames.append(pd.DataFrame({"y_hat": pred_series.values, "y_true": realized.values}))

    if not signal_rows:
        return None

    signals = pd.DataFrame(signal_rows).set_index("benchmark").sort_index()
    aggregate_health = aggregate_health_from_prediction_frames(prediction_frames, target_horizon_months)
    consensus, mean_pred, mean_ic, mean_hr, confidence_tier = _consensus_from_signals(signals)
    recommendation_mode = determine_recommendation_mode(
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        representative_cpcv=None,
    )
    return SnapshotSummary(
        label="cross-check",
        as_of=as_of,
        candidate_name=spec.candidate_name,
        policy_name="v21_promoted_cross_check",
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


__all__ = [
    "V22CrossCheckSpec",
    "build_promoted_cross_check_summary",
    "v22_promoted_cross_check_spec",
]
