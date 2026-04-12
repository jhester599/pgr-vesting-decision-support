from __future__ import annotations

from src.models.classification_shadow import (
    agreement_with_live_recommendation,
    classification_confidence_tier,
    classification_interpretation,
    classification_stance,
)


def test_classification_confidence_tier_thresholds() -> None:
    assert classification_confidence_tier(0.75) == "HIGH"
    assert classification_confidence_tier(0.62) == "MODERATE"
    assert classification_confidence_tier(0.50) == "LOW"
    assert classification_confidence_tier(0.28) == "HIGH"


def test_classification_stance_thresholds() -> None:
    assert classification_stance(0.74) == "ACTIONABLE-SELL"
    assert classification_stance(0.28) == "NON-ACTIONABLE"
    assert classification_stance(0.52) == "NEUTRAL"


def test_agreement_with_live_recommendation() -> None:
    assert agreement_with_live_recommendation("ACTIONABLE-SELL", "ACTIONABLE") is True
    assert agreement_with_live_recommendation("NON-ACTIONABLE", "DEFER-TO-TAX-DEFAULT") is True
    assert agreement_with_live_recommendation("ACTIONABLE-SELL", "DEFER-TO-TAX-DEFAULT") is False


def test_classification_interpretation_mentions_probability() -> None:
    interpretation = classification_interpretation(0.28, "NON-ACTIONABLE", "HIGH")
    assert "28.0%" in interpretation
    assert "hold/defer" in interpretation


# ---------------------------------------------------------------------------
# Tests for _portfolio_weighted_aggregate (v123)
# ---------------------------------------------------------------------------

import pandas as pd
from src.models.classification_shadow import _portfolio_weighted_aggregate


def _make_detail_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal detail_df matching classification_shadow schema."""
    return pd.DataFrame(rows)


def test_portfolio_weighted_aggregate_all_benchmarks_present() -> None:
    detail_df = _make_detail_df([
        {"benchmark": "VOO", "classifier_prob_actionable_sell": 0.60},
        {"benchmark": "VXUS", "classifier_prob_actionable_sell": 0.40},
        {"benchmark": "VWO", "classifier_prob_actionable_sell": 0.50},
        {"benchmark": "BND", "classifier_prob_actionable_sell": 0.20},
    ])
    result = _portfolio_weighted_aggregate(
        detail_df,
        investable_benchmarks=["VOO", "VXUS", "VWO", "BND"],
        base_weights={"VOO": 0.6154, "VXUS": 0.1538, "VWO": 0.1538, "BND": 0.0769},
    )
    assert result is not None
    expected = 0.6154 * 0.60 + 0.1538 * 0.40 + 0.1538 * 0.50 + 0.0769 * 0.20
    assert abs(result - expected) < 1e-4


def test_portfolio_weighted_aggregate_missing_benchmark_renormalizes() -> None:
    # VOO missing — weights over VXUS, VWO, BND should be renormalized
    detail_df = _make_detail_df([
        {"benchmark": "VXUS", "classifier_prob_actionable_sell": 0.40},
        {"benchmark": "VWO", "classifier_prob_actionable_sell": 0.50},
        {"benchmark": "BND", "classifier_prob_actionable_sell": 0.20},
    ])
    result = _portfolio_weighted_aggregate(
        detail_df,
        investable_benchmarks=["VOO", "VXUS", "VWO", "BND"],
        base_weights={"VOO": 0.6154, "VXUS": 0.1538, "VWO": 0.1538, "BND": 0.0769},
    )
    assert result is not None
    total = 0.1538 + 0.1538 + 0.0769
    expected = (0.1538 * 0.40 + 0.1538 * 0.50 + 0.0769 * 0.20) / total
    assert abs(result - expected) < 1e-3


def test_portfolio_weighted_aggregate_no_investable_benchmarks_returns_none() -> None:
    detail_df = _make_detail_df([
        {"benchmark": "GLD", "classifier_prob_actionable_sell": 0.60},
        {"benchmark": "DBC", "classifier_prob_actionable_sell": 0.40},
    ])
    result = _portfolio_weighted_aggregate(
        detail_df,
        investable_benchmarks=["VOO", "VXUS", "VWO", "BND"],
        base_weights={"VOO": 0.6154, "VXUS": 0.1538, "VWO": 0.1538, "BND": 0.0769},
    )
    assert result is None


def test_portfolio_weighted_aggregate_excludes_nan_probs() -> None:
    detail_df = _make_detail_df([
        {"benchmark": "VOO", "classifier_prob_actionable_sell": 0.60},
        {"benchmark": "VXUS", "classifier_prob_actionable_sell": float("nan")},
        {"benchmark": "VWO", "classifier_prob_actionable_sell": 0.50},
        {"benchmark": "BND", "classifier_prob_actionable_sell": 0.20},
    ])
    result = _portfolio_weighted_aggregate(
        detail_df,
        investable_benchmarks=["VOO", "VXUS", "VWO", "BND"],
        base_weights={"VOO": 0.6154, "VXUS": 0.1538, "VWO": 0.1538, "BND": 0.0769},
    )
    assert result is not None
    total = 0.6154 + 0.1538 + 0.0769
    expected = (0.6154 * 0.60 + 0.1538 * 0.50 + 0.0769 * 0.20) / total
    assert abs(result - expected) < 1e-3


def test_portfolio_weighted_aggregate_empty_df_returns_none() -> None:
    result = _portfolio_weighted_aggregate(
        pd.DataFrame(columns=["benchmark", "classifier_prob_actionable_sell"]),
        investable_benchmarks=["VOO", "VXUS", "VWO", "BND"],
        base_weights={"VOO": 0.6154, "VXUS": 0.1538, "VWO": 0.1538, "BND": 0.0769},
    )
    assert result is None
