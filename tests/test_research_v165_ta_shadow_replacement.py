"""Tests for the v165 TA classification replacement shadow harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_apply_feature_swaps_preserves_order_and_feature_count() -> None:
    from results.research.v165_ta_shadow_replacement_eval import apply_feature_swaps

    baseline = ["mom_12m", "vol_63d", "yield_slope", "vix"]
    swapped = apply_feature_swaps(
        baseline,
        {
            "mom_12m": "ta_pgr_obv_detrended",
            "vol_63d": "ta_pgr_natr_63d",
        },
    )

    assert swapped == [
        "ta_pgr_obv_detrended",
        "ta_pgr_natr_63d",
        "yield_slope",
        "vix",
    ]
    assert len(swapped) == len(baseline)


def test_candidate_variants_are_replacement_only() -> None:
    from results.research.v165_ta_shadow_replacement_eval import (
        build_candidate_variants,
    )

    variants = build_candidate_variants()
    by_name = {variant["variant"]: variant for variant in variants}

    assert "lean_baseline" in by_name
    assert "ta_minimal_replacement" in by_name
    assert by_name["ta_minimal_replacement"]["feature_swaps"] == {
        "mom_12m": "ta_pgr_obv_detrended",
        "vol_63d": "ta_pgr_natr_63d",
    }
    assert all(not str(variant["variant"]).endswith("all_ta") for variant in variants)


def test_attach_variant_deltas_uses_balanced_accuracy_and_brier() -> None:
    from results.research.v165_ta_shadow_replacement_eval import attach_variant_deltas

    detail = pd.DataFrame(
        [
            {
                "benchmark": "VOO",
                "variant": "lean_baseline",
                "balanced_accuracy": 0.50,
                "brier_score": 0.24,
            },
            {
                "benchmark": "VOO",
                "variant": "ta_minimal_replacement",
                "balanced_accuracy": 0.55,
                "brier_score": 0.20,
            },
        ]
    )

    with_deltas = attach_variant_deltas(detail)
    candidate = with_deltas.loc[
        with_deltas["variant"].eq("ta_minimal_replacement")
    ].iloc[0]

    assert abs(float(candidate["delta_balanced_accuracy"]) - 0.05) < 1e-12
    assert abs(float(candidate["delta_brier_score"]) + 0.04) < 1e-12
    assert bool(candidate["improved_vs_baseline"]) is True


def test_summarize_variants_counts_positive_benchmarks() -> None:
    from results.research.v165_ta_shadow_replacement_eval import summarize_variants

    detail = pd.DataFrame(
        [
            {
                "benchmark": "VOO",
                "variant": "ta_minimal_replacement",
                "delta_balanced_accuracy": 0.03,
                "delta_brier_score": 0.01,
                "improved_vs_baseline": True,
            },
            {
                "benchmark": "BND",
                "variant": "ta_minimal_replacement",
                "delta_balanced_accuracy": -0.01,
                "delta_brier_score": -0.03,
                "improved_vs_baseline": True,
            },
            {
                "benchmark": "GLD",
                "variant": "ta_minimal_replacement",
                "delta_balanced_accuracy": -0.01,
                "delta_brier_score": 0.02,
                "improved_vs_baseline": False,
            },
        ]
    )

    summary = summarize_variants(detail).iloc[0]

    assert summary["positive_benchmark_count"] == 2
    assert summary["degraded_benchmark_count"] == 1


def test_summarize_current_shadow_builds_variant_stance() -> None:
    from results.research.v165_ta_shadow_replacement_eval import (
        summarize_current_shadow,
    )

    current = pd.DataFrame(
        [
            {
                "variant": "ta_minimal_replacement",
                "benchmark": "VOO",
                "probability_actionable_sell": 0.80,
            },
            {
                "variant": "ta_minimal_replacement",
                "benchmark": "BND",
                "probability_actionable_sell": 0.70,
            },
        ]
    )

    summary = summarize_current_shadow(current).iloc[0]

    assert summary["benchmark_count"] == 2
    assert summary["probability_actionable_sell"] == 0.75
    assert summary["stance"] == "ACTIONABLE-SELL"
