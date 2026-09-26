"""Backtest of the live ACTIONABLE sell-% mapping on the realised-only OOS record.

Review 2026-09-25, F20. The report's "Decision Policy Backtest" scored a
different policy (``tiered_25_50_100``) from the one the monthly decision
uses. This module replays the live path itself at every historical OOS date
``t`` of a prequential panel (``build_prequential_panel``):

1. per benchmark, the issued forecast ``y_hat`` at ``t``;
2. per benchmark, the OOS quality (``nw_ic``, hit rate) of the rows realised
   by ``t`` only (``summarize_panel_diagnostics``), which stands in for the
   fold-mean IC and the benchmark-quality table the live run computes on its
   whole (by then realised) OOS history;
3. the live per-benchmark signal (``classify_benchmark_signal``), the live
   consensus variant (``build_shadow_consensus_table`` with the configured
   score column, lambda and weighting mode), the equal-weight IC, and
4. the live mapping (``decision_rendering.sell_pct_from_consensus``).

The mapping is scored as if the quality gate passed at every date, because it
is only used in ACTIONABLE months. A decision's outcome is the equal-weight
mean relative return of PGR against the benchmarks at ``t``; the policy keeps
``1 - sell_pct`` of the vesting tranche in PGR, so its value is
``(1 - sell_pct) * realised`` (``policy_metrics.evaluate_hold_fraction_series``)
and its uplift over the always-50 % default is ``(0.5 - sell_pct) * realised``.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

import config
from src.models.consensus_shadow import build_shadow_consensus_table
from src.models.forecast_diagnostics import summarize_panel_diagnostics
from src.models.multi_benchmark_wfo import classify_benchmark_signal
from src.models.policy_metrics import PolicySummary, evaluate_hold_fraction_series
from src.models.prequential import month_ordinals

SellPctMapping = Callable[[str, float, float], float]

# Policy key of the live mapping in the monthly "Decision Policy Backtest".
LIVE_MAPPING_POLICY: str = "live_actionable_mapping"

# Realised OOS rows a benchmark needs before its IC counts; below this its
# signal is NEUTRAL and it has no quality weight.
MIN_REALISED_ROWS_FOR_IC: int = 12


def _live_variant_name() -> str:
    return (
        "quality_weighted"
        if config.CONSENSUS_WEIGHTING_MODE == "quality_weighted"
        else "equal_weight"
    )


def historical_live_decisions(
    panel: pd.DataFrame,
    mapping: SellPctMapping,
    horizon_months: int | None = None,
    min_realised_rows: int = MIN_REALISED_ROWS_FOR_IC,
) -> pd.DataFrame:
    """Replay the live consensus and ``mapping`` at every OOS date of ``panel``.

    Args:
        panel: Prequential panel with ``benchmark``, ``date``, ``y_true``,
            ``y_hat``, ``naive`` and optionally ``z``.
        mapping: ``(consensus, mean_predicted, mean_ic) -> sell_pct``, e.g.
            ``decision_rendering.sell_pct_from_consensus``.
        horizon_months: Target horizon; defaults to ``panel.attrs`` or 6.
        min_realised_rows: Realised rows a benchmark needs for its IC.

    Returns:
        One row per date: ``date``, ``consensus``, ``mean_predicted``,
        ``mean_ic`` (equal weight), ``n_benchmarks``, ``sell_pct`` and
        ``realized`` (equal-weight mean relative return at that date).
    """
    if horizon_months is None:
        horizon_months = int(panel.attrs.get("horizon_months", 6))
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "benchmark"], kind="mergesort").reset_index(drop=True)
    months = month_ordinals(frame["date"])
    frame["_month"] = months

    rows: list[dict[str, object]] = []
    for as_of, current in frame.groupby("date", sort=True):
        as_of_month = int(current["_month"].iloc[0])
        realised = frame[frame["_month"] + horizon_months <= as_of_month]
        quality = _realised_quality(realised, horizon_months, min_realised_rows)

        signals = pd.DataFrame(
            {
                "predicted_relative_return": current["y_hat"].to_numpy(dtype=float),
                "ic": current["benchmark"].map(quality["nw_ic"]).to_numpy(dtype=float),
                "hit_rate": current["benchmark"].map(quality["hit_rate"]).to_numpy(dtype=float),
            },
            index=pd.Index(current["benchmark"].astype(str), name="benchmark"),
        )
        signals["signal"] = [
            classify_benchmark_signal(pred, ic)
            for pred, ic in zip(signals["predicted_relative_return"], signals["ic"])
        ]
        # Missing ICs weigh and count as zero skill, as build_quality_weights does.
        signals["ic"] = signals["ic"].fillna(0.0)
        signals["hit_rate"] = signals["hit_rate"].fillna(0.5)

        quality_df = quality.reset_index().rename(columns={"index": "benchmark"})
        table = build_shadow_consensus_table(
            signals=signals,
            benchmark_quality_df=quality_df,
            score_col=config.V74_SHADOW_CONSENSUS_SCORE_COL,
            lambda_mix=config.V74_SHADOW_CONSENSUS_LAMBDA_MIX,
        )
        live = table[table["variant"] == _live_variant_name()].iloc[0]
        mean_ic = float(signals["ic"].mean())
        consensus = str(live["consensus"])
        mean_predicted = float(live["mean_predicted_return"])
        rows.append(
            {
                "date": as_of,
                "consensus": consensus,
                "mean_predicted": mean_predicted,
                "mean_ic": mean_ic,
                "n_benchmarks": int(len(signals)),
                "sell_pct": float(mapping(consensus, mean_predicted, mean_ic)),
                "realized": float(current["y_true"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def _realised_quality(
    realised: pd.DataFrame,
    horizon_months: int,
    min_realised_rows: int,
) -> pd.DataFrame:
    """Per-benchmark ``nw_ic`` and ``hit_rate`` of the realised rows (index = benchmark)."""
    empty = pd.DataFrame(columns=["nw_ic", "hit_rate"], dtype=float)
    if realised.empty:
        return empty
    _, per_benchmark = summarize_panel_diagnostics(
        realised.drop(columns=["_month"]), target_horizon_months=horizon_months
    )
    if per_benchmark.empty:
        return empty
    per_benchmark = per_benchmark.set_index("benchmark")
    per_benchmark = per_benchmark[per_benchmark["n_obs"] >= min_realised_rows]
    return per_benchmark[["nw_ic", "hit_rate"]].astype(float)


def evaluate_live_mapping(decisions: pd.DataFrame) -> PolicySummary:
    """Score a decision table from ``historical_live_decisions``."""
    index = pd.DatetimeIndex(decisions["date"])
    hold = pd.Series(1.0 - decisions["sell_pct"].to_numpy(dtype=float), index=index)
    realized = pd.Series(decisions["realized"].to_numpy(dtype=float), index=index)
    return evaluate_hold_fraction_series(hold, realized)


def uplift_vs_default(decisions: pd.DataFrame, default_sell_pct: float = 0.50) -> pd.Series:
    """Per-decision uplift over always selling ``default_sell_pct``: (default - sell) * realised."""
    return pd.Series(
        (default_sell_pct - decisions["sell_pct"].to_numpy(dtype=float))
        * decisions["realized"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(decisions["date"]),
        name="uplift_vs_default",
    )


def summarize_by_consensus(decisions: pd.DataFrame) -> pd.DataFrame:
    """Count, mean sell %, mean realised return and mean uplift per consensus/sell cell."""
    work = decisions.assign(uplift=uplift_vs_default(decisions).to_numpy())
    grouped = work.groupby(["consensus", "sell_pct"], sort=True)
    return grouped.agg(
        n=("realized", "size"),
        mean_realized=("realized", "mean"),
        mean_uplift=("uplift", "mean"),
    ).reset_index()


__all__ = [
    "LIVE_MAPPING_POLICY",
    "MIN_REALISED_ROWS_FOR_IC",
    "evaluate_live_mapping",
    "historical_live_decisions",
    "summarize_by_consensus",
    "uplift_vs_default",
]
