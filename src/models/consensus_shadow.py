"""Shadow consensus helpers for promotion-oriented monthly monitoring."""

from __future__ import annotations

import pandas as pd


def build_quality_weights(
    benchmarks: pd.Index,
    benchmark_quality_df: pd.DataFrame | None,
    score_col: str = "nw_ic",
    lambda_mix: float = 0.25,
) -> pd.Series:
    """Build conservative benchmark weights shrunk toward equal weight."""
    benchmark_index = pd.Index([str(benchmark) for benchmark in benchmarks], name="benchmark")
    if benchmark_index.empty:
        return pd.Series(dtype=float, name="weight")

    equal_weight = pd.Series(
        1.0 / len(benchmark_index),
        index=benchmark_index,
        dtype=float,
        name="weight",
    )
    if (
        benchmark_quality_df is None
        or benchmark_quality_df.empty
        or score_col not in benchmark_quality_df.columns
    ):
        return equal_weight

    lambda_mix = min(max(float(lambda_mix), 0.0), 1.0)
    score_series = (
        benchmark_quality_df.set_index("benchmark")[score_col]
        .reindex(benchmark_index)
        .astype(float)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    score_sum = float(score_series.sum())
    if score_sum <= 1e-12:
        return equal_weight

    normalized = score_series / score_sum
    return ((1.0 - lambda_mix) * equal_weight + lambda_mix * normalized).rename("weight")


def summarize_consensus_variant(
    signals: pd.DataFrame,
    *,
    variant: str,
    weights: pd.Series | None = None,
    weight_mode: str,
    score_col: str | None = None,
    lambda_mix: float = 0.0,
) -> dict[str, object]:
    """Summarize one live cross-benchmark consensus variant."""
    if signals.empty:
        return {
            "variant": variant,
            "n_benchmarks": 0,
            "consensus": "NEUTRAL",
            "mean_predicted_return": 0.0,
            "mean_ic": 0.0,
            "mean_hit_rate": 0.0,
            "mean_prob_outperform": 0.5,
            "confidence_tier": "LOW",
            "weight_mode": weight_mode,
            "score_col": score_col or "",
            "lambda_mix": float(lambda_mix),
            "top_benchmark": "",
            "top_benchmark_weight": 0.0,
        }

    working = signals.copy()
    working.index = pd.Index([str(idx) for idx in working.index], name="benchmark")
    if weights is None or weights.empty:
        aligned_weights = pd.Series(
            1.0 / len(working),
            index=working.index,
            dtype=float,
            name="weight",
        )
    else:
        aligned_weights = weights.reindex(working.index).astype(float).fillna(0.0)
        weight_sum = float(aligned_weights.sum())
        if weight_sum <= 1e-12:
            aligned_weights = pd.Series(
                1.0 / len(working),
                index=working.index,
                dtype=float,
                name="weight",
            )
        else:
            aligned_weights = aligned_weights / weight_sum

    mean_predicted_return = float(
        (working["predicted_relative_return"].astype(float) * aligned_weights).sum()
    )
    mean_ic = float((working["ic"].astype(float) * aligned_weights).sum())
    mean_hit_rate = float((working["hit_rate"].astype(float) * aligned_weights).sum())
    if "prob_outperform" in working.columns:
        mean_prob_outperform = float(
            (working["prob_outperform"].astype(float) * aligned_weights).sum()
        )
    else:
        mean_prob_outperform = 0.5

    outperform_weight = float(aligned_weights[working["signal"] == "OUTPERFORM"].sum())
    underperform_weight = float(aligned_weights[working["signal"] == "UNDERPERFORM"].sum())
    if outperform_weight > 0.5:
        consensus = "OUTPERFORM"
    elif underperform_weight > 0.5:
        consensus = "UNDERPERFORM"
    else:
        consensus = "NEUTRAL"

    if mean_prob_outperform >= 0.70 or mean_prob_outperform <= 0.30:
        confidence_tier = "HIGH"
    elif mean_prob_outperform >= 0.60 or mean_prob_outperform <= 0.40:
        confidence_tier = "MODERATE"
    else:
        confidence_tier = "LOW"

    top_benchmark = str(aligned_weights.idxmax()) if not aligned_weights.empty else ""
    top_benchmark_weight = float(aligned_weights.max()) if not aligned_weights.empty else 0.0
    return {
        "variant": variant,
        "n_benchmarks": int(len(working)),
        "consensus": consensus,
        "mean_predicted_return": mean_predicted_return,
        "mean_ic": mean_ic,
        "mean_hit_rate": mean_hit_rate,
        "mean_prob_outperform": mean_prob_outperform,
        "confidence_tier": confidence_tier,
        "weight_mode": weight_mode,
        "score_col": score_col or "",
        "lambda_mix": float(lambda_mix),
        "top_benchmark": top_benchmark,
        "top_benchmark_weight": top_benchmark_weight,
    }


def build_shadow_consensus_table(
    signals: pd.DataFrame,
    benchmark_quality_df: pd.DataFrame | None,
    score_col: str = "nw_ic",
    lambda_mix: float = 0.25,
) -> pd.DataFrame:
    """Build equal-weight and quality-weighted live consensus snapshots."""
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "n_benchmarks",
                "consensus",
                "mean_predicted_return",
                "mean_ic",
                "mean_hit_rate",
                "mean_prob_outperform",
                "confidence_tier",
                "weight_mode",
                "score_col",
                "lambda_mix",
                "top_benchmark",
                "top_benchmark_weight",
            ]
        )

    equal_weights = pd.Series(
        1.0 / len(signals),
        index=pd.Index([str(idx) for idx in signals.index], name="benchmark"),
        dtype=float,
        name="weight",
    )
    quality_weights = build_quality_weights(
        equal_weights.index,
        benchmark_quality_df,
        score_col=score_col,
        lambda_mix=lambda_mix,
    )
    rows = [
        summarize_consensus_variant(
            signals,
            variant="equal_weight",
            weights=equal_weights,
            weight_mode="equal",
            score_col="",
            lambda_mix=0.0,
        ),
        summarize_consensus_variant(
            signals,
            variant="quality_weighted",
            weights=quality_weights,
            weight_mode="shrink_to_equal",
            score_col=score_col,
            lambda_mix=lambda_mix,
        ),
    ]
    return pd.DataFrame(rows)
