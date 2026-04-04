"""Helpers for the v17 shadow-gate promotion study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class V17PromotionDecision:
    """Decision record for the v17 shadow gate."""

    status: str
    recommended_path: str
    rationale: str


def count_label_changes(labels: list[str]) -> int:
    """Count transitions in an ordered label sequence."""
    if len(labels) <= 1:
        return 0
    changes = 0
    previous = labels[0]
    for current in labels[1:]:
        if current != previous:
            changes += 1
            previous = current
    return changes


def summarize_shadow_review(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate v17 monthly shadow-review rows into one summary per path."""
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
                "mean_aggregate_oos_r2": float(ordered["aggregate_oos_r2"].mean()),
                "mean_sell_pct": float(ordered["sell_pct"].mean()),
                "signal_changes": count_label_changes(signal_series),
                "mode_changes": count_label_changes(mode_series),
                "signal_agreement_with_shadow_rate": float(ordered["signal_agrees_with_shadow"].mean()),
                "mode_agreement_with_shadow_rate": float(ordered["mode_agrees_with_shadow"].mean()),
                "sell_agreement_with_shadow_rate": float(ordered["sell_agrees_with_shadow"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=[
            "signal_agreement_with_shadow_rate",
            "sell_agreement_with_shadow_rate",
            "mean_aggregate_oos_r2",
            "mean_ic",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def choose_v17_promotion(
    v16_summary_df: pd.DataFrame,
    review_summary_df: pd.DataFrame,
) -> V17PromotionDecision:
    """Decide whether the v16 candidate should replace the current cross-check."""
    if v16_summary_df.empty or review_summary_df.empty:
        return V17PromotionDecision(
            status="insufficient_data",
            recommended_path="keep_current_live_cross_check",
            rationale="v16 or v17 review inputs were missing.",
        )

    def _row(frame: pd.DataFrame, key_col: str, key: str) -> pd.Series | None:
        match = frame[frame[key_col] == key]
        if match.empty:
            return None
        return match.iloc[0]

    v16_candidate = _row(v16_summary_df, "candidate_name", "ensemble_ridge_gbt_v16")
    v16_live = _row(v16_summary_df, "candidate_name", "live_production_ensemble_reduced")
    v17_candidate = _row(review_summary_df, "path_name", "candidate_v16")
    v17_live = _row(review_summary_df, "path_name", "live_production")

    if v16_candidate is None or v16_live is None or v17_candidate is None or v17_live is None:
        return V17PromotionDecision(
            status="insufficient_data",
            recommended_path="keep_current_live_cross_check",
            rationale="A required live or candidate row was missing from the v16/v17 summaries.",
        )

    candidate_policy = float(v16_candidate["mean_policy_return_sign"])
    live_policy = float(v16_live["mean_policy_return_sign"])
    candidate_oos = float(v16_candidate["mean_oos_r2"])
    live_oos = float(v16_live["mean_oos_r2"])

    candidate_signal_agree = float(v17_candidate["signal_agreement_with_shadow_rate"])
    live_signal_agree = float(v17_live["signal_agreement_with_shadow_rate"])
    candidate_mode_agree = float(v17_candidate["mode_agreement_with_shadow_rate"])
    live_mode_agree = float(v17_live["mode_agreement_with_shadow_rate"])
    candidate_sell_agree = float(v17_candidate["sell_agreement_with_shadow_rate"])
    live_sell_agree = float(v17_live["sell_agreement_with_shadow_rate"])
    candidate_signal_changes = int(v17_candidate["signal_changes"])
    live_signal_changes = int(v17_live["signal_changes"])
    candidate_mode_changes = int(v17_candidate["mode_changes"])
    live_mode_changes = int(v17_live["mode_changes"])

    if (
        candidate_policy > live_policy
        and candidate_oos > live_oos
        and candidate_signal_agree >= live_signal_agree
        and candidate_mode_agree >= live_mode_agree
        and candidate_sell_agree >= live_sell_agree
        and candidate_signal_changes <= live_signal_changes
        and candidate_mode_changes <= live_mode_changes
    ):
        return V17PromotionDecision(
            status="promote_cross_check_candidate",
            recommended_path="candidate_v16",
            rationale=(
                "The modified Ridge+GBT candidate improved on the reduced live stack in v16 and behaved "
                "at least as cleanly as the current live cross-check against the promoted simpler baseline "
                "across the v17 monthly review window."
            ),
        )

    return V17PromotionDecision(
        status="keep_current_live_cross_check",
        recommended_path="live_production",
        rationale=(
            "The modified Ridge+GBT candidate improved the reduced-universe metrics, but it did not "
            "behave clearly enough versus the current live cross-check over recent monthly snapshots to "
            "justify replacing the current cross-check path."
        ),
    )
