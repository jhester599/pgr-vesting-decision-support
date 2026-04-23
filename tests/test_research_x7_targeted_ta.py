"""Tests for x7 targeted TA replacement utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_apply_feature_swaps_preserves_order_and_feature_count() -> None:
    from src.research.x7_targeted_ta import apply_feature_swaps

    baseline = ["mom_12m", "vol_63d", "vix", "pb_ratio"]
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
        "vix",
        "pb_ratio",
    ]
    assert len(swapped) == len(baseline)


def test_x7_variants_are_bounded_replacement_only() -> None:
    from src.research.x7_targeted_ta import build_x7_ta_variants

    variants = build_x7_ta_variants()
    names = [variant.variant for variant in variants]

    assert names[0] == "x2_core_baseline"
    assert "ta_minimal_replacement" in names
    assert len(variants) <= 4
    assert all(variant.experiment_mode == "replacement" for variant in variants)
    assert all("all_ta" not in variant.variant for variant in variants)


def test_attach_baseline_deltas_uses_horizon_specific_baseline() -> None:
    from src.research.x7_targeted_ta import attach_baseline_deltas

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 1,
                "variant": "x2_core_baseline",
                "balanced_accuracy": 0.50,
                "brier_score": 0.25,
            },
            {
                "horizon_months": 1,
                "variant": "ta_minimal_replacement",
                "balanced_accuracy": 0.56,
                "brier_score": 0.22,
            },
        ]
    )

    result = attach_baseline_deltas(detail)
    candidate = result[result["variant"] == "ta_minimal_replacement"].iloc[0]

    assert candidate["delta_balanced_accuracy"] == 0.06
    assert candidate["delta_brier_score"] == -0.03
    assert bool(candidate["clears_x7_gate"]) is True


def test_summarize_ta_variants_counts_cleared_horizons() -> None:
    from src.research.x7_targeted_ta import summarize_ta_variants

    detail = pd.DataFrame(
        [
            {
                "horizon_months": 1,
                "variant": "ta_minimal_replacement",
                "delta_balanced_accuracy": 0.02,
                "delta_brier_score": -0.01,
                "clears_x7_gate": True,
            },
            {
                "horizon_months": 3,
                "variant": "ta_minimal_replacement",
                "delta_balanced_accuracy": -0.01,
                "delta_brier_score": -0.02,
                "clears_x7_gate": False,
            },
        ]
    )

    summary = summarize_ta_variants(detail)

    assert summary.iloc[0]["variant"] == "ta_minimal_replacement"
    assert summary.iloc[0]["cleared_horizon_count"] == 1
    assert summary.iloc[0]["tested_horizon_count"] == 2
