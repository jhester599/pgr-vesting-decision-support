"""Helpers for the v28 forecast-universe review."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.v20 import V20_FORECAST_UNIVERSE
from src.research.v27 import v27_benchmark_pruning_review, v27_investable_redeploy_universe


@dataclass(frozen=True)
class V28UniverseSpec:
    """One forecast-universe candidate for the v28 review."""

    universe_name: str
    benchmarks: list[str]
    description: str


@dataclass(frozen=True)
class V28Decision:
    """Final v28 recommendation."""

    status: str
    recommended_universe: str
    rationale: str


def v28_universe_specs() -> dict[str, V28UniverseSpec]:
    """Return the forecast-universe candidates evaluated in v28."""
    buyable = list(v27_investable_redeploy_universe())
    buyable_plus_context = buyable + ["VFH", "KIE"]
    return {
        "current_reduced": V28UniverseSpec(
            universe_name="current_reduced",
            benchmarks=list(V20_FORECAST_UNIVERSE),
            description="Current reduced forecast universe promoted through v21-v24.",
        ),
        "buyable_only": V28UniverseSpec(
            universe_name="buyable_only",
            benchmarks=buyable,
            description="Pure buyable redeploy universe using only realistic default sell-proceeds destinations.",
        ),
        "buyable_plus_context": V28UniverseSpec(
            universe_name="buyable_plus_context",
            benchmarks=buyable_plus_context,
            description="Buyable redeploy universe plus the smallest contextual financial-sector sleeve (VFH, KIE).",
        ),
    }


def v28_universe_manifest() -> pd.DataFrame:
    """Summarize each v28 universe with buyable/context labels from v27."""
    review = pd.DataFrame(v27_benchmark_pruning_review())
    if review.empty:
        return pd.DataFrame()
    status_map = review.set_index("benchmark")["status"].to_dict()

    rows: list[dict[str, object]] = []
    for spec in v28_universe_specs().values():
        statuses = [status_map.get(ticker, "unknown") for ticker in spec.benchmarks]
        rows.append(
            {
                "universe_name": spec.universe_name,
                "description": spec.description,
                "benchmarks": ",".join(spec.benchmarks),
                "n_benchmarks": len(spec.benchmarks),
                "n_keep_for_redeploy": sum(status == "keep_for_redeploy" for status in statuses),
                "n_contextual_only": sum(status == "contextual_only" for status in statuses),
                "n_optional_substitute": sum(status == "optional_substitute" for status in statuses),
                "n_not_preferred_for_redeploy": sum(
                    status == "not_preferred_for_redeploy" for status in statuses
                ),
                "buyable_share": sum(status == "keep_for_redeploy" for status in statuses) / len(statuses),
            }
        )
    return pd.DataFrame(rows).sort_values("universe_name").reset_index(drop=True)


def choose_v28_decision(summary_df: pd.DataFrame) -> V28Decision:
    """Choose whether the forecast universe should be pruned."""
    if summary_df.empty:
        return V28Decision(
            status="insufficient_data",
            recommended_universe="current_reduced",
            rationale="No v28 universe comparison rows were available.",
        )

    def _row(universe_name: str, path_name: str) -> pd.Series | None:
        match = summary_df[
            (summary_df["universe_name"] == universe_name)
            & (summary_df["path_name"] == path_name)
        ]
        return None if match.empty else match.iloc[0]

    current_candidate = _row("current_reduced", "ensemble_ridge_gbt_v18")
    if current_candidate is None:
        return V28Decision(
            status="insufficient_data",
            recommended_universe="current_reduced",
            rationale="The current reduced-universe candidate row was missing from v28.",
        )

    candidate_order = ("buyable_only", "buyable_plus_context")
    for universe_name in candidate_order:
        candidate = _row(universe_name, "ensemble_ridge_gbt_v18")
        live_row = _row(universe_name, "live_production_ensemble_reduced")
        baseline_row = _row(universe_name, "shadow_baseline")
        if candidate is None or live_row is None or baseline_row is None:
            continue

        if (
            float(candidate["mean_policy_return_sign"]) >= float(live_row["mean_policy_return_sign"])
            and float(candidate["mean_oos_r2"]) >= float(live_row["mean_oos_r2"])
            and float(candidate["signal_agreement_with_shadow_rate"])
            >= float(live_row["signal_agreement_with_shadow_rate"])
            and float(candidate["mean_policy_return_sign"])
            >= float(baseline_row["mean_policy_return_sign"]) - 0.002
            and float(candidate["mean_policy_return_sign"])
            >= float(current_candidate["mean_policy_return_sign"]) - 0.002
            and float(candidate["mean_oos_r2"])
            >= float(current_candidate["mean_oos_r2"]) - 0.03
            and float(candidate["signal_agreement_with_shadow_rate"])
            >= float(current_candidate["signal_agreement_with_shadow_rate"]) - 0.05
        ):
            return V28Decision(
                status="prune_forecast_universe",
                recommended_universe=universe_name,
                rationale=(
                    "The narrower buyable-first universe preserved the promoted candidate's policy utility, "
                    "historical agreement, and OOS fit closely enough to justify pruning the broader "
                    "forecast list."
                ),
            )

    return V28Decision(
        status="keep_current_forecast_universe",
        recommended_universe="current_reduced",
        rationale=(
            "The narrower buyable-first universes weakened the promoted candidate too much relative to the "
            "current reduced forecast universe, so the broader benchmark list should stay in place for now."
        ),
    )


__all__ = [
    "V28Decision",
    "V28UniverseSpec",
    "choose_v28_decision",
    "v28_universe_manifest",
    "v28_universe_specs",
]
