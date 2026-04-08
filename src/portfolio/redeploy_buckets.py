"""Shared helpers for diversification-first redeploy bucket recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

import config
from src.portfolio.benchmark_sets import BENCHMARK_FAMILIES
from src.tax.capital_gains import TaxLot


CONTEXTUAL_ONLY_TICKERS: set[str] = {"VFH", "KIE"}
RECOMMENDATION_BUCKET_PRIORITY: tuple[str, ...] = (
    "broad_us_equity",
    "international_equity",
    "fixed_income",
    "real_assets",
    "sector_context",
)
RIDGE_CLASSIFIER_FEATURES: list[str] = [
    "investment_book_yield",
    "roe",
    "npw_growth_yoy",
    "buyback_yield",
    "credit_spread_hy",
    "gainshare_est",
    "unearned_premium_growth_yoy",
    "roe_net_income_ttm",
    "nfci",
    "combined_ratio_ttm",
    "investment_income_growth_yoy",
    "yield_slope",
]


@dataclass(frozen=True)
class LotAction:
    """One suggested action for an existing PGR tax lot."""

    vest_date: date
    shares: float
    cost_basis_per_share: float
    tax_bucket: str
    unrealized_gain: float
    unrealized_return: float
    action_priority: int
    rationale: str


def recommendation_bucket_for_ticker(ticker: str) -> str:
    """Map a benchmark ticker to a diversification-oriented redeploy bucket."""
    family = BENCHMARK_FAMILIES.get(ticker, "other")
    if ticker in {"VTI", "VOO"}:
        return "broad_us_equity"
    if family == "broad_equity":
        return "international_equity"
    if family == "fixed_income":
        return "fixed_income"
    if family == "real_asset":
        return "real_assets"
    return "sector_context"


def assign_destination_role(
    ticker: str,
    family: str,
    corr_bucket: str,
) -> str:
    """Assign a diversification-aware recommendation role to one fund."""
    if ticker in CONTEXTUAL_ONLY_TICKERS:
        return "contextual_only"
    if corr_bucket == "highly_correlated":
        return "unsuitable_due_to_correlation"
    if family in {"fixed_income", "real_asset"}:
        return "diversification_candidate"
    if ticker in {"VTI", "VOO", "VXUS", "VEA", "VWO"}:
        return "diversification_candidate"
    if family == "sector":
        return "diversification_candidate" if corr_bucket == "diversifying" else "contextual_only"
    return "diversification_candidate"


def add_destination_roles(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """Attach recommendation roles and redeploy buckets to a benchmark scoreboard."""
    enriched = scoreboard.copy()
    enriched["family"] = enriched["benchmark"].map(BENCHMARK_FAMILIES).fillna("other")
    enriched["recommendation_role"] = enriched.apply(
        lambda row: assign_destination_role(
            ticker=str(row["benchmark"]),
            family=str(row["family"]),
            corr_bucket=str(row["corr_bucket"]),
        ),
        axis=1,
    )
    enriched["recommendation_bucket"] = enriched["benchmark"].map(recommendation_bucket_for_ticker)
    return enriched


def choose_recommendation_universe(scoreboard: pd.DataFrame) -> list[str]:
    """Pick a diversification-first redeploy universe from the scored benchmarks."""
    eligible = scoreboard[scoreboard["recommendation_role"] == "diversification_candidate"].copy()
    if eligible.empty:
        return []

    selected: list[str] = []
    for bucket in RECOMMENDATION_BUCKET_PRIORITY:
        bucket_df = eligible[eligible["recommendation_bucket"] == bucket]
        if bucket_df.empty:
            continue
        bucket_df = bucket_df.sort_values(
            by=["diversification_score", "composite_score"],
            ascending=[False, True],
        )
        if bucket in {"international_equity", "fixed_income", "real_assets"}:
            picks = bucket_df.head(2)["benchmark"].tolist()
        else:
            picks = bucket_df.head(1)["benchmark"].tolist()
        for ticker in picks:
            if ticker not in selected:
                selected.append(str(ticker))
    return selected


def choose_forecast_universe(scoreboard: pd.DataFrame, recommendation_universe: list[str]) -> list[str]:
    """Add at most one contextual benchmark to the redeploy universe for forecasting."""
    forecast = list(recommendation_universe)
    context_df = scoreboard[scoreboard["recommendation_role"] == "contextual_only"].copy()
    if context_df.empty:
        return forecast

    context_df = context_df.sort_values(
        by=["composite_score", "ensemble_ic"],
        ascending=[True, False],
    )
    best = str(context_df.iloc[0]["benchmark"])
    if best not in forecast:
        forecast.append(best)
    return forecast


def mean_diversification_score(scoreboard: pd.DataFrame, benchmarks: list[str]) -> float:
    """Average diversification score for a selected set of benchmarks."""
    if not benchmarks:
        return float("nan")
    subset = scoreboard[scoreboard["benchmark"].isin(benchmarks)]
    if subset.empty:
        return float("nan")
    return float(subset["diversification_score"].mean())


def diversification_adjusted_policy_utility(
    candidate_rows: pd.DataFrame,
    scoreboard: pd.DataFrame,
    policy_column: str = "policy_return_sign",
) -> dict[str, float]:
    """Compute a diversification-aware utility score for one candidate."""
    merged = candidate_rows.merge(
        scoreboard[
            [
                "benchmark",
                "diversification_score",
                "recommendation_role",
            ]
        ],
        on="benchmark",
        how="left",
    )
    if merged.empty:
        return {
            "weighted_policy_return": float("nan"),
            "contextual_penalty": float("nan"),
            "diversification_aware_utility": float("nan"),
            "mean_diversification_score": float("nan"),
        }

    weights = 0.5 + merged["diversification_score"].fillna(0.0)
    weighted_return = float((merged[policy_column] * weights).sum() / weights.sum())
    contextual_mask = merged["recommendation_role"] != "diversification_candidate"
    contextual_penalty = float(
        merged.loc[contextual_mask, policy_column].clip(lower=0.0).mean()
        if contextual_mask.any()
        else 0.0
    )
    return {
        "weighted_policy_return": weighted_return,
        "contextual_penalty": contextual_penalty,
        "diversification_aware_utility": weighted_return - (0.25 * contextual_penalty),
        "mean_diversification_score": float(merged["diversification_score"].mean()),
    }


def recommend_redeploy_buckets(
    scoreboard: pd.DataFrame,
    recommendation_universe: list[str],
) -> list[dict[str, Any]]:
    """Summarize the preferred diversification buckets for user-facing guidance."""
    subset = scoreboard[scoreboard["benchmark"].isin(recommendation_universe)].copy()
    if subset.empty:
        return []

    bucket_rows: list[dict[str, Any]] = []
    for bucket in RECOMMENDATION_BUCKET_PRIORITY:
        bucket_df = subset[subset["recommendation_bucket"] == bucket]
        if bucket_df.empty:
            continue
        ordered = bucket_df.sort_values(
            by=["diversification_score", "composite_score"],
            ascending=[False, True],
        )
        tickers = ordered["benchmark"].tolist()
        bucket_rows.append(
            {
                "bucket": bucket,
                "example_funds": ", ".join(tickers[:2]),
                "mean_diversification_score": float(ordered["diversification_score"].mean()),
                "note": _bucket_note(bucket),
            }
        )
    return bucket_rows


def _bucket_note(bucket: str) -> str:
    """Human-readable note for one recommendation bucket."""
    notes = {
        "broad_us_equity": "Broad US equity diversifies away from single-stock risk without concentrating further in insurance.",
        "international_equity": "International equity lowers home-market and insurance concentration.",
        "fixed_income": "Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.",
        "real_assets": "Real assets add inflation and non-equity diversification.",
        "sector_context": "Sector funds are context-only unless no stronger diversifying destination is available.",
    }
    return notes.get(bucket, "Diversification-focused redeploy bucket.")


def summarize_existing_holdings_actions(
    lots: list[TaxLot],
    current_price: float,
    sell_date: date,
) -> list[LotAction]:
    """Rank existing held lots by tax-aware trimming priority."""
    actions: list[LotAction] = []
    for lot in lots:
        shares = float(lot.shares_remaining or 0.0)
        if shares <= 0:
            continue
        unrealized_gain = shares * (current_price - lot.cost_basis_per_share)
        unrealized_return = (
            (current_price / lot.cost_basis_per_share) - 1.0
            if lot.cost_basis_per_share > 0
            else 0.0
        )
        if unrealized_gain < 0:
            actions.append(
                LotAction(
                    vest_date=lot.vest_date,
                    shares=shares,
                    cost_basis_per_share=lot.cost_basis_per_share,
                    tax_bucket="LOSS",
                    unrealized_gain=unrealized_gain,
                    unrealized_return=unrealized_return,
                    action_priority=1,
                    rationale="Trim loss lots first when reducing concentration; they create tax assets instead of tax drag.",
                )
            )
            continue
        if lot.is_ltcg_eligible(sell_date):
            actions.append(
                LotAction(
                    vest_date=lot.vest_date,
                    shares=shares,
                    cost_basis_per_share=lot.cost_basis_per_share,
                    tax_bucket="LTCG",
                    unrealized_gain=unrealized_gain,
                    unrealized_return=unrealized_return,
                    action_priority=2,
                    rationale="After losses, trim LTCG gain lots next; concentration falls with lower tax friction.",
                )
            )
            continue
        actions.append(
            LotAction(
                vest_date=lot.vest_date,
                shares=shares,
                cost_basis_per_share=lot.cost_basis_per_share,
                tax_bucket="STCG",
                unrealized_gain=unrealized_gain,
                unrealized_return=unrealized_return,
                action_priority=3,
                rationale="Avoid STCG gain lots unless the signal is unusually strong or concentration risk is urgent.",
            )
        )

    return sorted(
        actions,
        key=lambda action: (
            action.action_priority,
            action.unrealized_gain,
            -action.cost_basis_per_share,
        ),
    )


def next_vest_after(as_of: date) -> tuple[date, str]:
    """Return the next vest date and RSU type after the given date."""
    candidates = [
        (date(as_of.year, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY), "time"),
        (date(as_of.year, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY), "performance"),
        (date(as_of.year + 1, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY), "time"),
        (date(as_of.year + 1, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY), "performance"),
    ]
    future = [(vest_date, rsu_type) for vest_date, rsu_type in candidates if vest_date > as_of]
    return sorted(future, key=lambda item: item[0])[0]
