"""Shared helpers for the v102-v117 post-review enhancement cycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.processing.feature_engineering import get_X_y_relative
from src.research.v87_utils import (
    ACTION_THRESHOLD,
    BENCHMARKS,
    aggregate_probability_sequences,
    build_quality_weighted_regression_consensus,
    evaluate_binary_time_series,
    feature_set_from_name,
    load_research_inputs,
    logistic_factory,
    prequential_logistic_calibration,
    probability_candidate_sequences,
    resolve_best_feature_set,
    resolve_primary_target,
    summarize_hold_series,
    summarize_monthly_policy_path,
)
from src.research.v37_utils import RESULTS_DIR


def regression_hold_fraction(consensus_frame: pd.DataFrame) -> pd.Series:
    """Return the current regression-style hold fraction path."""
    predicted = consensus_frame["predicted"]
    hold = np.where(
        predicted > ACTION_THRESHOLD,
        1.0,
        np.where(predicted < -ACTION_THRESHOLD, 0.0, 0.5),
    )
    return pd.Series(hold, index=consensus_frame.index, name="hold_fraction")


def calibrate_sequence_map(
    sequence_map: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply prequential logistic calibration to one probability sequence map."""
    calibrated: dict[str, pd.DataFrame] = {}
    for benchmark, frame in sequence_map.items():
        if frame.empty:
            continue
        result = frame.copy()
        result["y_prob"] = prequential_logistic_calibration(
            result["y_true"].to_numpy(dtype=int),
            result["y_prob"].to_numpy(dtype=float),
        )
        calibrated[benchmark] = result
    return calibrated


def build_overlay_alignment_frame(
    *,
    classifier_candidate: str = "pooled_fixed_effects_logistic_balanced",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned regression and classifier series for overlay research."""
    feature_df, rel_map = load_research_inputs()
    feature_set_name = resolve_best_feature_set()
    target_name = resolve_primary_target()
    consensus_frame, weights, _ = build_quality_weighted_regression_consensus(
        feature_df,
        rel_map,
    )
    sequence_candidates = probability_candidate_sequences(
        feature_df,
        rel_map,
        target_name,
        feature_set_name,
    )
    sequence_map = calibrate_sequence_map(sequence_candidates[classifier_candidate])
    prob_sell = aggregate_probability_sequences(sequence_map, weights=weights)
    aligned = pd.concat(
        [
            consensus_frame["predicted"].rename("predicted"),
            consensus_frame["realized"].rename("realized"),
            prob_sell.rename("prob_sell"),
        ],
        axis=1,
    ).dropna()
    return aligned, regression_hold_fraction(consensus_frame).reindex(aligned.index)


def evaluate_overlay_variant(
    aligned_df: pd.DataFrame,
    *,
    variant: str,
    hold_fraction: pd.Series,
    baseline_hold: pd.Series,
    gate_style: str,
    threshold: float,
) -> dict[str, Any]:
    """Summarize one overlay candidate against the regression baseline."""
    hold_fraction = hold_fraction.reindex(aligned_df.index).fillna(0.5)
    base = baseline_hold.reindex(aligned_df.index).fillna(0.5)
    policy = summarize_hold_series(
        variant,
        hold_fraction,
        aligned_df["realized"],
    )
    stability = summarize_monthly_policy_path(
        variant,
        hold_fraction,
        aligned_df["realized"],
    )
    action_mask = hold_fraction < 0.5
    unnecessary_action_rate = float(
        ((action_mask) & (aligned_df["realized"] > -ACTION_THRESHOLD)).mean()
    )
    return {
        **policy,
        **stability,
        "gate_style": gate_style,
        "threshold": float(threshold),
        "agreement_with_regression_rate": float((hold_fraction == base).mean()),
        "mean_abs_hold_diff_vs_regression": float(np.abs(hold_fraction - base).mean()),
        "unnecessary_action_rate": unnecessary_action_rate,
        "action_month_rate": float(action_mask.mean()),
    }


def veto_overlay_hold_fraction(
    aligned_df: pd.DataFrame,
    baseline_hold: pd.Series,
    *,
    threshold: float,
) -> pd.Series:
    """Apply a Gemini-style veto gate to regression sell months."""
    hold = baseline_hold.reindex(aligned_df.index).fillna(0.5).copy()
    veto_mask = (hold < 0.5) & (aligned_df["prob_sell"] < threshold)
    hold.loc[veto_mask] = 0.5
    hold.name = "hold_fraction"
    return hold


def permission_overlay_hold_fraction(
    aligned_df: pd.DataFrame,
    *,
    threshold: float,
) -> pd.Series:
    """Apply a ChatGPT-style permission overlay from the default 50% path."""
    hold = pd.Series(0.5, index=aligned_df.index, name="hold_fraction")
    allow_sell = (
        (aligned_df["prob_sell"] >= threshold)
        & (aligned_df["predicted"] < -ACTION_THRESHOLD)
    )
    hold.loc[allow_sell] = 0.0
    return hold


def target_series_for_name(
    rel_series: pd.Series,
    target_name: str,
) -> pd.Series:
    """Return an alternate target formulation for v112."""
    if target_name == "actionable_sell_3pct":
        target = (rel_series < -ACTION_THRESHOLD).astype(int)
    elif target_name == "actionable_hold_3pct":
        target = (rel_series > ACTION_THRESHOLD).astype(int)
    elif target_name == "deviate_from_default_50pct_sell":
        target = (rel_series.abs() > ACTION_THRESHOLD).astype(int)
    else:
        raise ValueError(f"Unsupported target '{target_name}'.")
    target.name = target_name
    return target


def build_target_probability_alignment(
    target_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned probability and regression series for one alternate target."""
    feature_df, rel_map = load_research_inputs()
    feature_set_name = resolve_best_feature_set()
    feature_columns = feature_set_from_name(feature_df, feature_set_name)
    consensus_frame, weights, _ = build_quality_weighted_regression_consensus(
        feature_df,
        rel_map,
    )
    sequence_map: dict[str, pd.DataFrame] = {}
    for benchmark in BENCHMARKS:
        rel_series = rel_map.get(benchmark)
        if rel_series is None or rel_series.empty:
            continue
        x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        usable_features = [feature for feature in feature_columns if feature in x_base.columns]
        if not usable_features:
            continue
        target = target_series_for_name(rel_series, target_name)
        pred_df = evaluate_binary_time_series(
            x_base[usable_features].copy(),
            target,
            logistic_factory(class_weight="balanced", c_value=0.5),
        )
        if pred_df.empty:
            continue
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df["y_prob"] = prequential_logistic_calibration(
            pred_df["y_true"].to_numpy(dtype=int),
            pred_df["y_prob"].to_numpy(dtype=float),
        )
        sequence_map[benchmark] = pred_df.set_index("date").sort_index()

    prob_series = aggregate_probability_sequences(sequence_map, weights=weights)
    aligned = pd.concat(
        [
            consensus_frame["predicted"].rename("predicted"),
            consensus_frame["realized"].rename("realized"),
            prob_series.rename("prob_target"),
        ],
        axis=1,
    ).dropna()
    return aligned, regression_hold_fraction(consensus_frame).reindex(aligned.index)


def write_results(
    filename: str,
    frame: pd.DataFrame,
) -> Path:
    """Write one results CSV into the research artifact directory."""
    path = RESULTS_DIR / filename
    frame.to_csv(path, index=False)
    return path


def write_summary(
    filename: str,
    title: str,
    lines: list[str],
) -> Path:
    """Write one short markdown summary into the research artifact directory."""
    path = RESULTS_DIR / filename
    path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="utf-8")
    return path
