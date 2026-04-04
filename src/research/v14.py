"""Shared helpers for the v14 reduced-universe prediction-layer study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


V14_BALANCED_CORE7: list[str] = [
    "VXUS",
    "VEA",
    "VHT",
    "VPU",
    "BNDX",
    "BND",
    "VNQ",
]


@dataclass(frozen=True)
class UniverseSelection:
    """One selected forecast universe for v14."""

    universe_name: str
    benchmarks: list[str]


def count_signal_changes(signals: list[str]) -> int:
    """Count how many times the ordered signal series changes label."""
    if len(signals) <= 1:
        return 0
    changes = 0
    previous = signals[0]
    for current in signals[1:]:
        if current != previous:
            changes += 1
            previous = current
    return changes


def select_best_universe(universe_summary: pd.DataFrame) -> UniverseSelection:
    """Choose the v14 forecast universe using policy, then simplicity, then OOS fit."""
    if universe_summary.empty:
        raise ValueError("universe_summary must not be empty.")

    ordered = universe_summary.sort_values(
        by=[
            "best_nonbaseline_policy_return",
            "n_benchmarks",
            "best_nonbaseline_oos_r2",
        ],
        ascending=[False, True, False],
    )
    row = ordered.iloc[0]
    benchmarks = [
        value.strip()
        for value in str(row["benchmarks"]).split(",")
        if value.strip()
    ]
    return UniverseSelection(
        universe_name=str(row["universe_name"]),
        benchmarks=benchmarks,
    )


def choose_feature_surgery_candidates(summary_df: pd.DataFrame) -> list[str]:
    """Select model candidates for minimal add/drop surgery."""
    if summary_df.empty:
        return []

    non_baseline = summary_df[
        ~summary_df["candidate_name"].astype(str).str.startswith("baseline_")
    ].copy()
    model_only = non_baseline[non_baseline["candidate_type"] == "model"].copy()
    if model_only.empty:
        return []

    live_row = model_only[model_only["candidate_name"] == "live_production_ensemble"]
    live_utility = float(live_row["diversification_aware_utility"].iloc[0]) if not live_row.empty else float("-inf")

    baseline_row = summary_df[summary_df["candidate_name"] == "baseline_historical_mean"]
    baseline_utility = (
        float(baseline_row["diversification_aware_utility"].iloc[0])
        if not baseline_row.empty
        else float("inf")
    )

    survivors = model_only[
        (model_only["candidate_name"] != "live_production_ensemble")
        & (model_only["diversification_aware_utility"] >= live_utility)
        & (model_only["diversification_aware_utility"] >= baseline_utility - 0.002)
    ].copy()
    if survivors.empty:
        fallback = model_only[model_only["candidate_name"] != "live_production_ensemble"].sort_values(
            by=["diversification_aware_utility", "mean_policy_return_neutral_3pct", "mean_oos_r2"],
            ascending=[False, False, False],
        )
        return fallback.head(1)["candidate_name"].astype(str).tolist()

    survivors = survivors.sort_values(
        by=["diversification_aware_utility", "mean_policy_return_neutral_3pct", "mean_oos_r2"],
        ascending=[False, False, False],
    )
    return survivors.head(2)["candidate_name"].astype(str).tolist()


def choose_final_candidate(summary_df: pd.DataFrame) -> str:
    """Pick the final v14 candidate for shadow review."""
    if summary_df.empty:
        raise ValueError("summary_df must not be empty.")

    non_baseline = summary_df[
        ~summary_df["candidate_name"].astype(str).str.startswith("baseline_")
    ].copy()
    if non_baseline.empty:
        return "baseline_historical_mean"

    ordered = non_baseline.sort_values(
        by=["diversification_aware_utility", "mean_policy_return_neutral_3pct", "mean_oos_r2"],
        ascending=[False, False, False],
    )
    return str(ordered.iloc[0]["candidate_name"])
