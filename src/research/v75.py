"""Helpers for the v75 quality-weighted promotion gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import timedelta

import pandas as pd

from src.research.v17 import count_label_changes


@dataclass(frozen=True)
class V75Decision:
    """Decision record for the v75 holdout-era shadow replay."""

    status: str
    recommended_path: str
    rationale: str


def holdout_monthly_review_dates(
    end_as_of: date,
    holdout_start: str,
) -> list[date]:
    """Return business month-end review dates from holdout start through end date."""
    start_period = pd.Period(holdout_start, freq="M")
    end_period = pd.Period(end_as_of, freq="M")
    if end_period.to_timestamp("M").date() > end_as_of:
        end_period -= 1

    dates: list[date] = []
    for period in pd.period_range(start=start_period, end=end_period, freq="M"):
        month_end = period.to_timestamp("M").date()
        current = month_end
        while current.weekday() >= 5:
            current = current - timedelta(days=1)
        dates.append(current)
    return dates


def summarize_v75_review(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the holdout replay into one summary row per path."""
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for path_name, group in detail_df.groupby("path_name", dropna=False):
        ordered = group.sort_values("as_of")
        signal_series = ordered["consensus"].astype(str).tolist()
        mode_series = ordered["recommendation_mode"].astype(str).tolist()
        rows.append(
            {
                "path_name": path_name,
                "review_months": int(len(ordered)),
                "mean_predicted": float(ordered["mean_predicted"].mean()),
                "mean_ic": float(ordered["mean_ic"].mean()),
                "mean_hit_rate": float(ordered["mean_hit_rate"].mean()),
                "mean_prob_outperform": float(ordered["mean_prob_outperform"].mean()),
                "mean_sell_pct": float(ordered["sell_pct"].mean()),
                "mean_top_benchmark_weight": float(ordered["top_benchmark_weight"].mean()),
                "max_top_benchmark_weight": float(ordered["top_benchmark_weight"].max()),
                "signal_changes": count_label_changes(signal_series),
                "mode_changes": count_label_changes(mode_series),
                "signal_agreement_with_live_rate": float(ordered["signal_agrees_with_live"].mean()),
                "mode_agreement_with_live_rate": float(ordered["mode_agrees_with_live"].mean()),
                "sell_agreement_with_live_rate": float(ordered["sell_agrees_with_live"].mean()),
                "mean_abs_sell_pct_diff_vs_live": float(ordered["abs_sell_pct_diff_vs_live"].mean()),
                "underperform_rate": float((ordered["consensus"] == "UNDERPERFORM").mean()),
                "neutral_rate": float((ordered["consensus"] == "NEUTRAL").mean()),
                "outperform_rate": float((ordered["consensus"] == "OUTPERFORM").mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=[
            "mode_agreement_with_live_rate",
            "sell_agreement_with_live_rate",
            "mean_ic",
            "mean_predicted",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def choose_v75_decision(review_summary_df: pd.DataFrame) -> V75Decision:
    """Choose whether the quality-weighted shadow path is ready for promotion."""
    if review_summary_df.empty:
        return V75Decision(
            status="insufficient_data",
            recommended_path="keep_v38_live",
            rationale="No v75 holdout replay rows were available.",
        )

    def _row(name: str) -> pd.Series | None:
        match = review_summary_df[review_summary_df["path_name"] == name]
        if match.empty:
            return None
        return match.iloc[0]

    live_row = _row("live_equal_weight")
    quality_row = _row("v74_quality_weighted")
    if live_row is None or quality_row is None:
        return V75Decision(
            status="insufficient_data",
            recommended_path="keep_v38_live",
            rationale="The live or quality-weighted replay summary row was missing.",
        )

    quality_mode_agree = float(quality_row["mode_agreement_with_live_rate"])
    quality_sell_agree = float(quality_row["sell_agreement_with_live_rate"])
    quality_mode_changes = int(quality_row["mode_changes"])
    live_mode_changes = int(live_row["mode_changes"])
    quality_mean_ic = float(quality_row["mean_ic"])
    live_mean_ic = float(live_row["mean_ic"])
    quality_mean_pred = float(quality_row["mean_predicted"])
    live_mean_pred = float(live_row["mean_predicted"])
    quality_max_weight = float(quality_row["max_top_benchmark_weight"])
    quality_mean_abs_sell_diff = float(quality_row["mean_abs_sell_pct_diff_vs_live"])

    if (
        quality_mode_agree >= 0.75
        and quality_sell_agree >= 0.75
        and quality_mode_changes <= live_mode_changes + 1
        and quality_mean_ic >= live_mean_ic
        and quality_mean_pred >= live_mean_pred
        and quality_max_weight <= 0.40
        and quality_mean_abs_sell_diff <= 0.10
    ):
        return V75Decision(
            status="advance_to_promotion_check",
            recommended_path="v74_quality_weighted",
            rationale=(
                "Across the holdout-era monthly replay, the quality-weighted consensus stayed "
                "operationally close to the live path while improving weighted signal strength "
                "without excessive benchmark concentration."
            ),
        )

    return V75Decision(
        status="keep_shadow_only",
        recommended_path="keep_v38_live",
        rationale=(
            "The quality-weighted path remains promising, but the holdout-era replay did not "
            "clear the stability and concentration gate strongly enough to justify immediate promotion."
        ),
    )
