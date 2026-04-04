"""Helpers for the v18 directional-bias and benchmark-side feature study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.research.v15 import apply_one_for_one_swap
from src.research.v16 import V16CandidateSpec, v16_model_specs


@dataclass(frozen=True)
class V18SwapCandidate:
    """One narrow benchmark-side / peer-relative replacement candidate."""

    candidate_name: str
    model_type: str
    candidate_feature: str
    replace_feature: str
    rationale: str


@dataclass(frozen=True)
class V18Decision:
    """Final v18 recommendation."""

    status: str
    recommended_candidate: str
    rationale: str


def v18_swap_candidates() -> list[V18SwapCandidate]:
    """Return the narrow v18 swap queue."""
    return [
        V18SwapCandidate(
            candidate_name="ridge_lean_v1__v16",
            model_type="ridge",
            candidate_feature="pgr_vs_peers_6m",
            replace_feature="real_rate_10y",
            rationale="Peer-relative momentum may be a better directional anchor than generic real rates.",
        ),
        V18SwapCandidate(
            candidate_name="ridge_lean_v1__v16",
            model_type="ridge",
            candidate_feature="pgr_vs_vfh_6m",
            replace_feature="credit_spread_hy",
            rationale="Relative strength versus financials may better explain PGR's benchmark-relative drift.",
        ),
        V18SwapCandidate(
            candidate_name="ridge_lean_v1__v16",
            model_type="ridge",
            candidate_feature="vwo_vxus_spread_6m",
            replace_feature="yield_curvature",
            rationale="Emerging-vs-broad international leadership may help on the reduced global benchmark set.",
        ),
        V18SwapCandidate(
            candidate_name="ridge_lean_v1__v16",
            model_type="ridge",
            candidate_feature="gold_vs_treasury_6m",
            replace_feature="nfci",
            rationale="Gold-versus-duration risk appetite may be a cleaner cross-asset regime cue than NFCI.",
        ),
        V18SwapCandidate(
            candidate_name="ridge_lean_v1__v16",
            model_type="ridge",
            candidate_feature="breakeven_inflation_10y",
            replace_feature="real_rate_10y",
            rationale="Inflation expectations may map more directly to commodity / real-asset benchmarks.",
        ),
        V18SwapCandidate(
            candidate_name="ridge_lean_v1__v16",
            model_type="ridge",
            candidate_feature="real_yield_change_6m",
            replace_feature="yield_curvature",
            rationale="Real-yield momentum may carry more directional information than curve shape alone.",
        ),
        V18SwapCandidate(
            candidate_name="gbt_lean_plus_two__v16",
            model_type="gbt",
            candidate_feature="pgr_vs_peers_6m",
            replace_feature="mom_3m",
            rationale="Peer-relative strength may outperform generic short momentum in trees.",
        ),
        V18SwapCandidate(
            candidate_name="gbt_lean_plus_two__v16",
            model_type="gbt",
            candidate_feature="pgr_vs_vfh_6m",
            replace_feature="mom_6m",
            rationale="Financial-sector relative context may be more stable than generic medium momentum.",
        ),
        V18SwapCandidate(
            candidate_name="gbt_lean_plus_two__v16",
            model_type="gbt",
            candidate_feature="commodity_equity_momentum",
            replace_feature="yield_curvature",
            rationale="Commodity-vs-equity leadership may explain DBC / VDE / GLD behavior better than curve shape.",
        ),
        V18SwapCandidate(
            candidate_name="gbt_lean_plus_two__v16",
            model_type="gbt",
            candidate_feature="vwo_vxus_spread_6m",
            replace_feature="real_rate_10y",
            rationale="International leadership spread may help the reduced global benchmark set.",
        ),
        V18SwapCandidate(
            candidate_name="gbt_lean_plus_two__v16",
            model_type="gbt",
            candidate_feature="gold_vs_treasury_6m",
            replace_feature="credit_spread_hy",
            rationale="Gold-versus-bonds may act as a cleaner cross-asset defensive regime signal.",
        ),
        V18SwapCandidate(
            candidate_name="gbt_lean_plus_two__v16",
            model_type="gbt",
            candidate_feature="breakeven_inflation_10y",
            replace_feature="nfci",
            rationale="Inflation breakevens may better align with commodity and duration-sensitive benchmarks.",
        ),
    ]


def v18_base_specs() -> dict[str, V16CandidateSpec]:
    """Return the v16 model specs used as the v18 starting point."""
    specs = v16_model_specs()
    return {
        "ridge_lean_v1__v16": specs["ridge_lean_v1__v16"],
        "gbt_lean_plus_two__v16": specs["gbt_lean_plus_two__v16"],
    }


def build_v18_candidate_specs(best_swaps_df: pd.DataFrame) -> dict[str, V16CandidateSpec]:
    """Build v18 model specs by applying the selected swap per model."""
    specs = v16_model_specs()
    result = {
        "ridge_lean_v1__v16": specs["ridge_lean_v1__v16"],
        "gbt_lean_plus_two__v16": specs["gbt_lean_plus_two__v16"],
    }

    for row in best_swaps_df.itertuples(index=False):
        base_spec = specs[str(row.candidate_name)]
        result[f"{row.candidate_name}__v18"] = V16CandidateSpec(
            candidate_name=f"{row.candidate_name}__v18",
            candidate_type="model",
            model_type=base_spec.model_type,
            features=apply_one_for_one_swap(
                base_spec,
                str(row.replace_feature),
                str(row.candidate_feature),
            ),
            notes=f"v18 swap: {row.candidate_feature} for {row.replace_feature}.",
        )
    return result


def choose_best_v18_swaps(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pick one swap per model using policy utility, then OOS fit, then IC."""
    if summary_df.empty:
        return pd.DataFrame()
    ordered = summary_df.sort_values(
        by=[
            "candidate_name",
            "mean_policy_return_sign_delta",
            "mean_oos_r2_delta",
            "mean_ic_delta",
        ],
        ascending=[True, False, False, False],
    )
    return ordered.groupby("candidate_name", group_keys=False).head(1).reset_index(drop=True)


def choose_v18_decision(metric_summary_df: pd.DataFrame, shadow_summary_df: pd.DataFrame) -> V18Decision:
    """Choose the final v18 recommendation."""
    if metric_summary_df.empty or shadow_summary_df.empty:
        return V18Decision(
            status="insufficient_data",
            recommended_candidate="none",
            rationale="Required v18 metric or shadow-review inputs were missing.",
        )

    def _metric(name: str) -> pd.Series | None:
        match = metric_summary_df[metric_summary_df["candidate_name"] == name]
        return None if match.empty else match.iloc[0]

    def _shadow(name: str) -> pd.Series | None:
        match = shadow_summary_df[shadow_summary_df["path_name"] == name]
        return None if match.empty else match.iloc[0]

    candidate = _metric("ensemble_ridge_gbt_v18")
    prior_candidate = _metric("ensemble_ridge_gbt_v16")
    baseline = _metric("baseline_historical_mean")
    candidate_shadow = _shadow("candidate_v18")
    prior_shadow = _shadow("candidate_v16")

    if any(item is None for item in (candidate, prior_candidate, baseline, candidate_shadow, prior_shadow)):
        return V18Decision(
            status="insufficient_data",
            recommended_candidate="none",
            rationale="A required v18 candidate or shadow-review row was missing.",
        )

    if (
        float(candidate["mean_policy_return_sign"]) >= float(prior_candidate["mean_policy_return_sign"])
        and float(candidate["mean_oos_r2"]) >= float(prior_candidate["mean_oos_r2"])
        and float(candidate_shadow["signal_agreement_with_shadow_rate"]) > float(prior_shadow["signal_agreement_with_shadow_rate"])
        and float(candidate_shadow["signal_agreement_with_shadow_rate"]) >= 0.25
    ):
        return V18Decision(
            status="advance_to_v19",
            recommended_candidate="ensemble_ridge_gbt_v18",
            rationale=(
                "The v18 candidate improved the v16 pair on reduced-universe metrics and reduced the directional "
                "bias against the promoted simpler baseline enough to justify one more narrow promotion gate."
            ),
        )

    return V18Decision(
        status="keep_v16_as_research_only",
        recommended_candidate="ensemble_ridge_gbt_v16",
        rationale=(
            "The benchmark-side and peer-relative swaps did not reduce the candidate's directional bias against "
            "the promoted simpler baseline enough to justify another promotion attempt."
        ),
    )
