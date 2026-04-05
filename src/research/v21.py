"""Helpers for the v21 historical comparison study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.v20 import summarize_v20_review, v20_ensemble_specs


V21_REVIEW_PATHS: tuple[str, ...] = (
    "live_production_ensemble_reduced",
    "ensemble_ridge_gbt_v16",
    "ensemble_ridge_gbt_v18",
    "ensemble_ridge_gbt_v20_value",
    "ensemble_ridge_gbt_v20_best",
)


@dataclass(frozen=True)
class V21Decision:
    """Decision record for v21."""

    status: str
    recommended_path: str
    rationale: str


def v21_review_ensemble_specs() -> dict[str, dict[str, object]]:
    """Return the narrowed ensemble set carried into the historical study."""
    specs = v20_ensemble_specs()
    return {name: specs[name] for name in V21_REVIEW_PATHS}


def common_historical_dates(
    prediction_map: dict[str, dict[str, pd.DataFrame]],
    benchmarks: list[str],
) -> list[pd.Timestamp]:
    """Return the common evaluable date set across all compared paths and benchmarks."""
    common: pd.Index | None = None
    for path_name, benchmark_map in prediction_map.items():
        for benchmark in benchmarks:
            frame = benchmark_map.get(benchmark)
            if frame is None or frame.empty:
                continue
            idx = pd.DatetimeIndex(frame.index)
            common = idx if common is None else common.intersection(idx)
    if common is None:
        return []
    return list(pd.DatetimeIndex(common).sort_values())


def summarize_v21_slices(
    detail_df: pd.DataFrame,
    slices: dict[str, tuple[str | None, str | None]],
) -> pd.DataFrame:
    """Aggregate the historical review by named time slices."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    detail = detail_df.copy()
    detail["as_of"] = pd.to_datetime(detail["as_of"])
    for slice_name, (start, end) in slices.items():
        subset = detail
        if start:
            subset = subset[subset["as_of"] >= pd.Timestamp(start)]
        if end:
            subset = subset[subset["as_of"] <= pd.Timestamp(end)]
        if subset.empty:
            continue
        summary = summarize_v20_review(subset)
        if summary.empty:
            continue
        summary.insert(0, "slice_name", slice_name)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def choose_v21_decision(
    metric_summary_df: pd.DataFrame,
    full_review_df: pd.DataFrame,
    slice_summary_df: pd.DataFrame,
) -> V21Decision:
    """Choose whether any candidate is cleaner than the current live cross-check across full history."""
    if metric_summary_df.empty or full_review_df.empty:
        return V21Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="Required v21 metric or review inputs were missing.",
        )

    def _metric(name: str) -> pd.Series | None:
        match = metric_summary_df[metric_summary_df["candidate_name"] == name]
        return None if match.empty else match.iloc[0]

    def _review(name: str) -> pd.Series | None:
        match = full_review_df[full_review_df["path_name"] == name]
        return None if match.empty else match.iloc[0]

    def _slice(name: str, slice_name: str) -> pd.Series | None:
        match = slice_summary_df[
            (slice_summary_df["path_name"] == name) & (slice_summary_df["slice_name"] == slice_name)
        ]
        return None if match.empty else match.iloc[0]

    live_metric = _metric("live_production_ensemble_reduced")
    baseline_metric = _metric("baseline_historical_mean")
    live_review = _review("live_production_ensemble_reduced")
    if live_metric is None or baseline_metric is None or live_review is None:
        return V21Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="A required live or baseline row was missing from the v21 inputs.",
        )

    candidate_rows = metric_summary_df[metric_summary_df["candidate_name"].isin(V21_REVIEW_PATHS)].copy()
    candidate_rows = candidate_rows[candidate_rows["candidate_name"] != "live_production_ensemble_reduced"]
    if candidate_rows.empty:
        return V21Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="No candidate rows were available for v21.",
        )
    candidate_rows = candidate_rows.sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    )
    best_candidate = candidate_rows.iloc[0]
    candidate_name = str(best_candidate["candidate_name"])
    candidate_review = _review(candidate_name)
    if candidate_review is None:
        return V21Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="The selected v21 candidate was missing from the historical review summary.",
        )

    post_2020_candidate = _slice(candidate_name, "post_2020")
    post_2020_live = _slice("live_production_ensemble_reduced", "post_2020")

    candidate_policy = float(best_candidate["mean_policy_return_sign"])
    candidate_oos = float(best_candidate["mean_oos_r2"])
    live_policy = float(live_metric["mean_policy_return_sign"])
    live_oos = float(live_metric["mean_oos_r2"])
    baseline_policy = float(baseline_metric["mean_policy_return_sign"])

    candidate_agree = float(candidate_review["signal_agreement_with_shadow_rate"])
    live_agree = float(live_review["signal_agreement_with_shadow_rate"])
    candidate_under = float(candidate_review["underperform_rate"])

    if (
        candidate_policy >= baseline_policy + 0.002
        and candidate_policy > live_policy
        and candidate_oos > live_oos
        and candidate_agree >= live_agree
        and candidate_under < 0.80
        and post_2020_candidate is not None
        and post_2020_live is not None
        and float(post_2020_candidate["signal_agreement_with_shadow_rate"])
        >= float(post_2020_live["signal_agreement_with_shadow_rate"])
    ):
        return V21Decision(
            status="promote_candidate_cross_check",
            recommended_path=candidate_name,
            rationale=(
                "The best v21 candidate improved on the reduced live stack and matched or exceeded the live "
                "cross-check's agreement with the promoted simpler baseline over the full historical window."
            ),
        )

    return V21Decision(
        status="keep_current_live_cross_check",
        recommended_path="live_production_ensemble_reduced",
        rationale=(
            "Across the full historically evaluable window, the leading candidate still does not behave cleanly "
            "enough versus the promoted simpler baseline to replace the current live cross-check."
        ),
    )


__all__ = [
    "V21Decision",
    "V21_REVIEW_PATHS",
    "choose_v21_decision",
    "common_historical_dates",
    "summarize_v21_slices",
    "v21_review_ensemble_specs",
]
