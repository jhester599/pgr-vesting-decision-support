"""Helpers for the v20 synthesis and promotion-readiness study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.v15 import apply_one_for_one_swap
from src.research.v16 import V16CandidateSpec, V16_FORECAST_UNIVERSE, v16_model_specs
from src.research.v17 import count_label_changes

V20_FORECAST_UNIVERSE: list[str] = list(V16_FORECAST_UNIVERSE)


@dataclass(frozen=True)
class V20Decision:
    """Final v20 recommendation."""

    status: str
    recommended_candidate: str
    rationale: str


def v20_model_specs() -> dict[str, V16CandidateSpec]:
    """Return the assembled model-spec set evaluated in v20."""
    specs = v16_model_specs()

    ridge_v18 = V16CandidateSpec(
        candidate_name="ridge_lean_v1__v18",
        candidate_type="model",
        model_type="ridge",
        features=apply_one_for_one_swap(
            specs["ridge_lean_v1__v16"],
            "yield_curvature",
            "real_yield_change_6m",
        ),
        notes="v18 Ridge swap: real_yield_change_6m replacing yield_curvature on the v16 Ridge stack.",
    )
    ridge_v20_value = V16CandidateSpec(
        candidate_name="ridge_lean_v1__v20_value",
        candidate_type="model",
        model_type="ridge",
        features=apply_one_for_one_swap(
            specs["ridge_lean_v1"],
            "roe_net_income_ttm",
            "pgr_pe_vs_market_pe",
        ),
        notes="v20 Ridge valuation swap: pgr_pe_vs_market_pe replacing roe_net_income_ttm.",
    )
    gbt_v18 = V16CandidateSpec(
        candidate_name="gbt_lean_plus_two__v18",
        candidate_type="model",
        model_type="gbt",
        features=apply_one_for_one_swap(
            specs["gbt_lean_plus_two__v16"],
            "real_rate_10y",
            "vwo_vxus_spread_6m",
        ),
        notes="v18 GBT swap: vwo_vxus_spread_6m replacing real_rate_10y on the v16 GBT stack.",
    )
    gbt_v20_usd = V16CandidateSpec(
        candidate_name="gbt_lean_plus_two__v20_usd",
        candidate_type="model",
        model_type="gbt",
        features=apply_one_for_one_swap(
            specs["gbt_lean_plus_two"],
            "nfci",
            "usd_broad_return_3m",
        ),
        notes="v20 GBT macro swap: usd_broad_return_3m replacing nfci.",
    )
    gbt_v20_pricing = V16CandidateSpec(
        candidate_name="gbt_lean_plus_two__v20_pricing",
        candidate_type="model",
        model_type="gbt",
        features=apply_one_for_one_swap(
            specs["gbt_lean_plus_two"],
            "vmt_yoy",
            "auto_pricing_power_spread",
        ),
        notes="v20 GBT pricing swap: auto_pricing_power_spread replacing vmt_yoy.",
    )

    specs.update(
        {
            ridge_v18.candidate_name: ridge_v18,
            ridge_v20_value.candidate_name: ridge_v20_value,
            gbt_v18.candidate_name: gbt_v18,
            gbt_v20_usd.candidate_name: gbt_v20_usd,
            gbt_v20_pricing.candidate_name: gbt_v20_pricing,
        }
    )
    return specs


def v20_ensemble_specs() -> dict[str, dict[str, object]]:
    """Return the ensemble candidates carried through the v20 gate."""
    return {
        "live_production_ensemble_reduced": {
            "candidate_type": "ensemble",
            "members": [
                "elasticnet_current",
                "ridge_current",
                "bayesian_ridge_current",
                "gbt_current",
            ],
            "notes": "Current deployed 4-model stack on the reduced forecast universe.",
        },
        "ensemble_ridge_gbt_v16": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v16", "gbt_lean_plus_two__v16"],
            "notes": "v16 assembled stack from the first confirmed winning swaps.",
        },
        "ensemble_ridge_gbt_v18": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v18", "gbt_lean_plus_two__v18"],
            "notes": "v18 assembled stack from the benchmark-side bias-reduction swaps.",
        },
        "ensemble_ridge_gbt_v20_value": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v20_value", "gbt_lean_plus_two__v16"],
            "notes": "v20 valuation-focused stack: best Ridge valuation swap plus the strongest confirmed GBT v16 swap.",
        },
        "ensemble_ridge_gbt_v20_best": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v20_value", "gbt_lean_plus_two__v18"],
            "notes": "v20 best-of-confirmed stack: best Ridge valuation swap plus the strongest GBT benchmark-side stack.",
        },
        "ensemble_ridge_gbt_v20_usd": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v20_value", "gbt_lean_plus_two__v20_usd"],
            "notes": "v20 macro-focused stack: best Ridge valuation swap plus the best confirmed USD macro GBT swap.",
        },
        "ensemble_ridge_gbt_v20_pricing": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v20_value", "gbt_lean_plus_two__v20_pricing"],
            "notes": "v20 pricing-focused stack: best Ridge valuation swap plus the best confirmed GBT pricing-power swap.",
        },
    }


def summarize_v20_review(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate v20 monthly review rows into one summary per path."""
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
                "signal_agreement_with_live_rate": float(ordered["signal_agrees_with_live"].mean()),
                "mode_agreement_with_live_rate": float(ordered["mode_agrees_with_live"].mean()),
                "sell_agreement_with_live_rate": float(ordered["sell_agrees_with_live"].mean()),
                "outperform_rate": float((ordered["consensus"] == "OUTPERFORM").mean()),
                "neutral_rate": float((ordered["consensus"] == "NEUTRAL").mean()),
                "underperform_rate": float((ordered["consensus"] == "UNDERPERFORM").mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=[
            "signal_agreement_with_shadow_rate",
            "mean_aggregate_oos_r2",
            "mean_ic",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def choose_v20_decision(
    metric_summary_df: pd.DataFrame,
    review_summary_df: pd.DataFrame,
) -> V20Decision:
    """Choose whether any assembled v20 stack is ready for promotion."""
    if metric_summary_df.empty or review_summary_df.empty:
        return V20Decision(
            status="insufficient_data",
            recommended_candidate="none",
            rationale="Required v20 metric or shadow-review inputs were missing.",
        )

    candidate_names = [name for name in v20_ensemble_specs() if name != "live_production_ensemble_reduced"]
    candidate_frame = metric_summary_df[metric_summary_df["candidate_name"].isin(candidate_names)].copy()
    if candidate_frame.empty:
        return V20Decision(
            status="insufficient_data",
            recommended_candidate="none",
            rationale="No v20 ensemble candidate rows were available.",
        )
    candidate_frame = candidate_frame.sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    )
    best_candidate = candidate_frame.iloc[0]
    candidate_name = str(best_candidate["candidate_name"])

    def _metric(name: str) -> pd.Series | None:
        match = metric_summary_df[metric_summary_df["candidate_name"] == name]
        return None if match.empty else match.iloc[0]

    def _review(name: str) -> pd.Series | None:
        match = review_summary_df[review_summary_df["path_name"] == name]
        return None if match.empty else match.iloc[0]

    live_metric = _metric("live_production_ensemble_reduced")
    baseline_metric = _metric("baseline_historical_mean")
    candidate_review = _review(candidate_name)
    live_review = _review("live_production_ensemble_reduced")

    if live_metric is None or baseline_metric is None or candidate_review is None or live_review is None:
        return V20Decision(
            status="insufficient_data",
            recommended_candidate=candidate_name,
            rationale="A required live, baseline, or review row was missing from the v20 inputs.",
        )

    candidate_policy = float(best_candidate["mean_policy_return_sign"])
    candidate_oos = float(best_candidate["mean_oos_r2"])
    live_policy = float(live_metric["mean_policy_return_sign"])
    live_oos = float(live_metric["mean_oos_r2"])
    baseline_policy = float(baseline_metric["mean_policy_return_sign"])
    baseline_oos = float(baseline_metric["mean_oos_r2"])

    candidate_signal_agree = float(candidate_review["signal_agreement_with_shadow_rate"])
    live_signal_agree = float(live_review["signal_agreement_with_shadow_rate"])
    candidate_mode_agree = float(candidate_review["mode_agreement_with_shadow_rate"])
    live_mode_agree = float(live_review["mode_agreement_with_shadow_rate"])
    candidate_sell_agree = float(candidate_review["sell_agreement_with_shadow_rate"])
    live_sell_agree = float(live_review["sell_agreement_with_shadow_rate"])
    candidate_signal_changes = int(candidate_review["signal_changes"])
    live_signal_changes = int(live_review["signal_changes"])
    candidate_underperform_rate = float(candidate_review["underperform_rate"])

    if (
        candidate_policy >= baseline_policy + 0.002
        and candidate_policy > live_policy
        and candidate_oos >= baseline_oos
        and candidate_oos > live_oos
        and candidate_signal_agree >= live_signal_agree
        and candidate_mode_agree >= live_mode_agree
        and candidate_sell_agree >= live_sell_agree
        and candidate_signal_changes <= live_signal_changes
        and candidate_underperform_rate <= 0.50
    ):
        return V20Decision(
            status="promote_candidate_cross_check",
            recommended_candidate=candidate_name,
            rationale=(
                "The best assembled v20 stack beat the reduced live stack and the historical-mean baseline "
                "while also matching or improving the current cross-check's agreement with the promoted simpler baseline."
            ),
        )

    if (
        candidate_policy > live_policy
        and candidate_oos > live_oos
        and candidate_signal_agree >= 0.25
    ):
        return V20Decision(
            status="shadow_only_candidate",
            recommended_candidate=candidate_name,
            rationale=(
                "The best assembled v20 stack improved on the reduced live stack, but it still did not behave "
                "cleanly enough versus the promoted simpler baseline for direct promotion."
            ),
        )

    return V20Decision(
        status="continue_research_keep_current_cross_check",
        recommended_candidate=candidate_name,
        rationale=(
            "The best assembled v20 stack improved reduced-universe metrics, but it still diverged too much from "
            "the promoted simpler baseline to replace the current live cross-check."
        ),
    )


__all__ = [
    "V20Decision",
    "V20_FORECAST_UNIVERSE",
    "choose_v20_decision",
    "summarize_v20_review",
    "v20_ensemble_specs",
    "v20_model_specs",
]
