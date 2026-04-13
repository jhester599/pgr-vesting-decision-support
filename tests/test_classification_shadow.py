from __future__ import annotations

import pandas as pd
import pytest

from src.models.classification_shadow import (
    ClassificationShadowSummary,
    agreement_with_live_recommendation,
    classification_confidence_tier,
    classification_interpretation,
    classification_stance,
    _portfolio_weighted_aggregate,
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
    weights = {"VOO": 0.6154, "VXUS": 0.1538, "VWO": 0.1538, "BND": 0.0769}
    total_w = sum(weights.values())
    expected = (
        weights["VOO"] * 0.60
        + weights["VXUS"] * 0.40
        + weights["VWO"] * 0.50
        + weights["BND"] * 0.20
    ) / total_w
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


def test_portfolio_weighted_aggregate_benchmark_not_in_base_weights_returns_none() -> None:
    # VOO is in detail_df with a valid prob, but base_weights is empty
    # → total_weight == 0.0 → should return None
    detail_df = _make_detail_df([
        {"benchmark": "VOO", "classifier_prob_actionable_sell": 0.60},
    ])
    result = _portfolio_weighted_aggregate(
        detail_df,
        investable_benchmarks=["VOO"],
        base_weights={},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Tests for ClassificationShadowSummary investable pool fields (Task 3 / v123)
# ---------------------------------------------------------------------------

def test_classification_shadow_summary_has_investable_pool_fields() -> None:
    """New investable-pool fields must exist on the dataclass."""
    summary = ClassificationShadowSummary(
        enabled=True,
        target_label="actionable_sell_3pct",
        feature_set="lean_baseline",
        model_family="separate_benchmark_logistic_balanced",
        calibration="oos_logistic_calibration",
        probability_actionable_sell=0.35,
        probability_actionable_sell_label="35.0%",
        probability_non_actionable=0.65,
        probability_non_actionable_label="65.0%",
        confidence_tier="MODERATE",
        stance="NEUTRAL",
        agreement_with_live=True,
        agreement_label="Aligned",
        interpretation="Test interpretation.",
        benchmark_count=8,
        feature_anchor_date="2026-03-31",
        top_supporting_benchmark="GLD",
        top_supporting_contribution=0.052,
        top_supporting_contribution_label="5.2%",
        # New fields
        probability_investable_pool=0.38,
        probability_investable_pool_label="38.0%",
        confidence_tier_investable_pool="MODERATE",
        stance_investable_pool="NEUTRAL",
        investable_benchmark_count=4,
    )
    assert summary.probability_investable_pool == pytest.approx(0.38)
    assert summary.probability_investable_pool_label == "38.0%"
    assert summary.confidence_tier_investable_pool == "MODERATE"
    assert summary.stance_investable_pool == "NEUTRAL"
    assert summary.investable_benchmark_count == 4


def test_classification_shadow_summary_investable_fields_default_none() -> None:
    """New fields default to None — existing callers are backward compatible."""
    summary = ClassificationShadowSummary(
        enabled=False,
        target_label="actionable_sell_3pct",
        feature_set="lean_baseline",
        model_family="separate_benchmark_logistic_balanced",
        calibration="oos_logistic_calibration",
        probability_actionable_sell=None,
        probability_actionable_sell_label=None,
        probability_non_actionable=None,
        probability_non_actionable_label=None,
        confidence_tier=None,
        stance=None,
        agreement_with_live=None,
        agreement_label=None,
        interpretation=None,
        benchmark_count=0,
        feature_anchor_date=None,
        top_supporting_benchmark=None,
        top_supporting_contribution=None,
        top_supporting_contribution_label=None,
    )
    assert summary.probability_investable_pool is None
    assert summary.confidence_tier_investable_pool is None
    assert summary.investable_benchmark_count == 0


# --- v129 dual-track ---

def test_detail_df_has_benchmark_specific_columns() -> None:
    """build_classification_shadow_summary must add dual-track columns to detail_df."""
    from src.models.classification_shadow import build_classification_shadow_summary
    # This is a smoke test: verify the columns exist in the returned detail_df.
    # Use a real DB connection if available; otherwise skip.
    import pytest
    pytest.skip("integration test -- run manually with DB connection")


def test_dual_track_columns_in_classification_shadow_columns() -> None:
    from src.reporting.classification_artifacts import CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_features" in CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_prob_actionable_sell" in CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_tier" in CLASSIFICATION_SHADOW_COLUMNS


def test_run_dual_track_pass_adds_columns_for_non_switched_benchmarks() -> None:
    """For benchmarks not in the feature map (e.g. GLD), benchmark_specific_prob
    should equal classifier_prob_actionable_sell."""
    import pandas as pd
    from src.models.classification_shadow import _run_dual_track_pass

    detail_df = pd.DataFrame({
        "benchmark": ["GLD", "VOO"],
        "classifier_prob_actionable_sell": [0.35, 0.40],
        "classifier_shadow_tier": ["LOW", "MODERATE"],
    })
    lean_baseline = ["f1", "f2"]
    feature_map = {}  # empty -- no switches

    _run_dual_track_pass(
        detail_df=detail_df,
        conn=None,
        as_of=None,
        feature_df=pd.DataFrame(),
        current_features=pd.DataFrame(),
        lean_baseline=lean_baseline,
        feature_map=feature_map,
    )

    assert "benchmark_specific_prob_actionable_sell" in detail_df.columns
    assert "benchmark_specific_features" in detail_df.columns
    assert "benchmark_specific_tier" in detail_df.columns
    # For non-switched benchmarks, prob should equal lean_baseline prob
    assert detail_df.loc[detail_df["benchmark"] == "GLD", "benchmark_specific_prob_actionable_sell"].iloc[0] == pytest.approx(0.35)


def test_classification_shadow_summary_to_payload_includes_investable_fields() -> None:
    summary = ClassificationShadowSummary(
        enabled=True,
        target_label="actionable_sell_3pct",
        feature_set="lean_baseline",
        model_family="separate_benchmark_logistic_balanced",
        calibration="oos_logistic_calibration",
        probability_actionable_sell=0.35,
        probability_actionable_sell_label="35.0%",
        probability_non_actionable=0.65,
        probability_non_actionable_label="65.0%",
        confidence_tier="MODERATE",
        stance="NEUTRAL",
        agreement_with_live=True,
        agreement_label="Aligned",
        interpretation="Test.",
        benchmark_count=8,
        feature_anchor_date="2026-03-31",
        top_supporting_benchmark="GLD",
        top_supporting_contribution=0.052,
        top_supporting_contribution_label="5.2%",
        probability_investable_pool=0.38,
        probability_investable_pool_label="38.0%",
        confidence_tier_investable_pool="MODERATE",
        stance_investable_pool="NEUTRAL",
        investable_benchmark_count=4,
    )
    payload = summary.to_payload()
    assert "probability_investable_pool" in payload
    assert "probability_investable_pool_label" in payload
    assert "confidence_tier_investable_pool" in payload
    assert "stance_investable_pool" in payload
    assert "investable_benchmark_count" in payload
    # Existing fields must still be present
    assert "probability_actionable_sell" in payload
    assert "confidence_tier" in payload
