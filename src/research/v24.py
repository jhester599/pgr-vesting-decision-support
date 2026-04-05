"""Helpers for the v24 VTI-for-VOO benchmark replacement study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


V24_CURRENT_UNIVERSE: list[str] = [
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
]

V24_VTI_UNIVERSE: list[str] = [
    "VTI",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
]

V24_SCENARIOS: tuple[str, ...] = (
    "current_voo_actual",
    "vti_replacement_actual",
    "vti_replacement_stitched",
)


@dataclass(frozen=True)
class V24Decision:
    """Decision record for whether VTI should replace VOO."""

    status: str
    recommended_universe: str
    rationale: str


def summarize_v24_scenarios(
    metric_df: pd.DataFrame,
    review_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine scenario-level metric, review, and window summaries."""
    rows: list[dict[str, object]] = []
    for scenario in V24_SCENARIOS:
        metric_match = metric_df[
            (metric_df["scenario_name"] == scenario)
            & (metric_df["candidate_name"] == "ensemble_ridge_gbt_v18")
        ]
        review_match = review_df[
            (review_df["scenario_name"] == scenario)
            & (review_df["path_name"] == "ensemble_ridge_gbt_v18")
        ]
        window_match = window_df[window_df["scenario_name"] == scenario]
        if metric_match.empty or review_match.empty or window_match.empty:
            continue
        metric_row = metric_match.iloc[0]
        review_row = review_match.iloc[0]
        window_row = window_match.iloc[0]
        rows.append(
            {
                "scenario_name": scenario,
                "common_start": window_row["common_start"],
                "common_end": window_row["common_end"],
                "n_common_dates": int(window_row["n_common_dates"]),
                "mean_policy_return_sign": float(metric_row["mean_policy_return_sign"]),
                "mean_oos_r2": float(metric_row["mean_oos_r2"]),
                "mean_ic": float(metric_row["mean_ic"]),
                "signal_agreement_with_shadow_rate": float(review_row["signal_agreement_with_shadow_rate"]),
                "signal_changes": int(review_row["signal_changes"]),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def choose_v24_decision(summary_df: pd.DataFrame) -> V24Decision:
    """Choose whether VTI should replace VOO in the reduced forecast universe."""
    if summary_df.empty:
        return V24Decision(
            status="insufficient_data",
            recommended_universe="current_voo_actual",
            rationale="No v24 scenario summary rows were available.",
        )

    current = summary_df[summary_df["scenario_name"] == "current_voo_actual"]
    vti_actual = summary_df[summary_df["scenario_name"] == "vti_replacement_actual"]
    vti_stitched = summary_df[summary_df["scenario_name"] == "vti_replacement_stitched"]
    if current.empty:
        return V24Decision(
            status="insufficient_data",
            recommended_universe="current_voo_actual",
            rationale="The current VOO-based reference scenario was missing.",
        )
    current_row = current.iloc[0]
    best_row = current_row
    for candidate_df in (vti_actual, vti_stitched):
        if candidate_df.empty:
            continue
        candidate_row = candidate_df.iloc[0]
        better = (
            float(candidate_row["signal_agreement_with_shadow_rate"]) > float(best_row["signal_agreement_with_shadow_rate"])
            and float(candidate_row["mean_policy_return_sign"]) >= float(best_row["mean_policy_return_sign"]) - 0.002
            and float(candidate_row["mean_oos_r2"]) >= float(best_row["mean_oos_r2"]) - 0.05
        )
        if better:
            best_row = candidate_row

    if best_row["scenario_name"] == "current_voo_actual":
        return V24Decision(
            status="keep_voo",
            recommended_universe="current_voo_actual",
            rationale=(
                "Replacing VOO with VTI did not improve the leading candidate cleanly enough on agreement, "
                "policy utility, and OOS fit to justify changing the reduced forecast universe."
            ),
        )

    return V24Decision(
        status="prefer_vti",
        recommended_universe=str(best_row["scenario_name"]),
        rationale=(
            "A VTI-based reduced universe improved the leading candidate enough versus the current VOO setup "
            "to justify carrying VTI forward in future research and promotion work."
        ),
    )


__all__ = [
    "V24_CURRENT_UNIVERSE",
    "V24_VTI_UNIVERSE",
    "V24_SCENARIOS",
    "V24Decision",
    "choose_v24_decision",
    "summarize_v24_scenarios",
]
