"""Helpers for the v16 narrow prediction-stack promotion study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import config
from scripts.candidate_model_bakeoff import candidate_feature_sets
from src.research.v15 import apply_one_for_one_swap


V16_FORECAST_UNIVERSE: list[str] = [
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
]


@dataclass(frozen=True)
class V16CandidateSpec:
    """Canonical candidate definition for the v16 promotion study."""

    candidate_name: str
    candidate_type: str
    model_type: str
    features: list[str]
    notes: str


@dataclass(frozen=True)
class PromotionDecision:
    """Simple recommendation output for the v16 promotion gate."""

    status: str
    recommended_candidate: str
    rationale: str


def v16_model_specs() -> dict[str, V16CandidateSpec]:
    """Return the individual-model candidates used in v16."""
    v9_specs = candidate_feature_sets()

    elasticnet_current = list(config.MODEL_FEATURE_OVERRIDES["elasticnet"])
    ridge_current = list(config.MODEL_FEATURE_OVERRIDES["ridge"])
    bayesian_ridge_current = list(config.MODEL_FEATURE_OVERRIDES["bayesian_ridge"])
    gbt_current = list(config.MODEL_FEATURE_OVERRIDES["gbt"])

    ridge_lean = list(v9_specs["ridge_lean_v1"]["features"])
    gbt_lean = list(v9_specs["gbt_lean_plus_two"]["features"])

    ridge_v16 = apply_one_for_one_swap(
        V16CandidateSpec(
            candidate_name="ridge_lean_v1",
            candidate_type="model",
            model_type="ridge",
            features=ridge_lean,
            notes="Base v14/v15 Ridge candidate.",
        ),
        "roe_net_income_ttm",
        "book_value_per_share_growth_yoy",
    )
    gbt_v16 = apply_one_for_one_swap(
        V16CandidateSpec(
            candidate_name="gbt_lean_plus_two",
            candidate_type="model",
            model_type="gbt",
            features=gbt_lean,
            notes="Base v14/v15 GBT candidate.",
        ),
        "vmt_yoy",
        "rate_adequacy_gap_yoy",
    )

    return {
        "elasticnet_current": V16CandidateSpec(
            candidate_name="elasticnet_current",
            candidate_type="model",
            model_type="elasticnet",
            features=elasticnet_current,
            notes="Current deployed ElasticNet feature set.",
        ),
        "ridge_current": V16CandidateSpec(
            candidate_name="ridge_current",
            candidate_type="model",
            model_type="ridge",
            features=ridge_current,
            notes="Current deployed Ridge feature set.",
        ),
        "bayesian_ridge_current": V16CandidateSpec(
            candidate_name="bayesian_ridge_current",
            candidate_type="model",
            model_type="bayesian_ridge",
            features=bayesian_ridge_current,
            notes="Current deployed BayesianRidge feature set.",
        ),
        "gbt_current": V16CandidateSpec(
            candidate_name="gbt_current",
            candidate_type="model",
            model_type="gbt",
            features=gbt_current,
            notes="Current deployed GBT feature set.",
        ),
        "ridge_lean_v1": V16CandidateSpec(
            candidate_name="ridge_lean_v1",
            candidate_type="model",
            model_type="ridge",
            features=ridge_lean,
            notes="Base v14/v15 Ridge replacement candidate.",
        ),
        "gbt_lean_plus_two": V16CandidateSpec(
            candidate_name="gbt_lean_plus_two",
            candidate_type="model",
            model_type="gbt",
            features=gbt_lean,
            notes="Base v14/v15 GBT replacement candidate.",
        ),
        "ridge_lean_v1__v16": V16CandidateSpec(
            candidate_name="ridge_lean_v1__v16",
            candidate_type="model",
            model_type="ridge",
            features=ridge_v16,
            notes="v16 Ridge with book value growth replacing ROE net income TTM.",
        ),
        "gbt_lean_plus_two__v16": V16CandidateSpec(
            candidate_name="gbt_lean_plus_two__v16",
            candidate_type="model",
            model_type="gbt",
            features=gbt_v16,
            notes="v16 GBT with rate adequacy gap replacing vehicle miles traveled.",
        ),
    }


def v16_ensemble_specs() -> dict[str, dict[str, Any]]:
    """Return the ensemble candidates carried through the promotion study."""
    return {
        "live_production_ensemble_reduced": {
            "candidate_type": "ensemble",
            "members": [
                "elasticnet_current",
                "ridge_current",
                "bayesian_ridge_current",
                "gbt_current",
            ],
            "notes": "Current deployed 4-model stack evaluated on the reduced v14/v16 forecast universe.",
        },
        "ensemble_ridge_gbt_v14": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1", "gbt_lean_plus_two"],
            "notes": "Base v14/v15 replacement pair before the v15 winning feature swaps.",
        },
        "ensemble_ridge_gbt_v16": {
            "candidate_type": "ensemble",
            "members": ["ridge_lean_v1__v16", "gbt_lean_plus_two__v16"],
            "notes": "v16 modified Ridge+GBT pair carrying the confirmed v15 feature swaps.",
        },
    }


def choose_v16_promotion(summary_df: pd.DataFrame) -> PromotionDecision:
    """Apply the narrow v16 promotion gate."""
    if summary_df.empty:
        return PromotionDecision(
            status="insufficient_data",
            recommended_candidate="none",
            rationale="No v16 summary rows were available.",
        )

    def _row(name: str) -> pd.Series | None:
        match = summary_df[summary_df["candidate_name"] == name]
        if match.empty:
            return None
        return match.iloc[0]

    live_row = _row("live_production_ensemble_reduced")
    candidate_row = _row("ensemble_ridge_gbt_v16")
    baseline_row = _row("baseline_historical_mean")

    if candidate_row is None:
        return PromotionDecision(
            status="insufficient_data",
            recommended_candidate="none",
            rationale="The v16 modified Ridge+GBT candidate was not evaluated.",
        )

    candidate_name = str(candidate_row["candidate_name"])

    if live_row is None or baseline_row is None:
        return PromotionDecision(
            status="insufficient_data",
            recommended_candidate=candidate_name,
            rationale="A required comparison row was missing from the v16 summary.",
        )

    candidate_policy = float(candidate_row["mean_policy_return_sign"])
    candidate_oos_r2 = float(candidate_row["mean_oos_r2"])
    live_policy = float(live_row["mean_policy_return_sign"])
    live_oos_r2 = float(live_row["mean_oos_r2"])
    baseline_policy = float(baseline_row["mean_policy_return_sign"])

    if (
        candidate_policy >= baseline_policy + 0.005
        and candidate_policy > live_policy
        and candidate_oos_r2 > live_oos_r2
    ):
        return PromotionDecision(
            status="promote_candidate",
            recommended_candidate=candidate_name,
            rationale=(
                "The modified Ridge+GBT pair cleared the v16 gate by beating both the reduced live stack "
                "and the historical-mean baseline on sign-policy return while also improving OOS R^2."
            ),
        )

    if (
        candidate_policy > live_policy
        and candidate_policy >= baseline_policy - 0.002
        and candidate_oos_r2 > live_oos_r2
    ):
        return PromotionDecision(
            status="shadow_for_v17",
            recommended_candidate=candidate_name,
            rationale=(
                "The modified Ridge+GBT pair improved on the reduced live stack and stayed close to the "
                "historical-mean baseline, but it did not clear a strong enough edge for direct promotion."
            ),
        )

    return PromotionDecision(
        status="do_not_promote",
        recommended_candidate=candidate_name,
        rationale=(
            "The modified Ridge+GBT pair did not separate enough from the historical-mean baseline and "
            "should remain a research candidate rather than a production replacement."
        ),
    )
