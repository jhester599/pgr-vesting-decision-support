from __future__ import annotations

import pandas as pd
import pytest

from src.models.consensus_shadow import build_quality_weights, build_shadow_consensus_table


def test_build_quality_weights_shrinks_toward_equal() -> None:
    benchmarks = pd.Index(["VOO", "BND", "GLD"], name="benchmark")
    quality_df = pd.DataFrame(
        {
            "benchmark": ["VOO", "BND", "GLD"],
            "nw_ic": [0.30, 0.10, 0.00],
        }
    )

    weights = build_quality_weights(
        benchmarks,
        quality_df,
        score_col="nw_ic",
        lambda_mix=0.25,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert weights["VOO"] > weights["BND"] > weights["GLD"]
    assert weights["GLD"] > 0.0


def test_build_shadow_consensus_table_returns_live_and_quality_rows() -> None:
    signals = pd.DataFrame(
        {
            "predicted_relative_return": [0.06, -0.01, 0.02],
            "ic": [0.11, 0.03, 0.08],
            "hit_rate": [0.61, 0.53, 0.57],
            "signal": ["OUTPERFORM", "UNDERPERFORM", "OUTPERFORM"],
            "prob_outperform": [0.70, 0.46, 0.62],
        },
        index=pd.Index(["VOO", "BND", "GLD"], name="benchmark"),
    )
    quality_df = pd.DataFrame(
        {
            "benchmark": ["VOO", "BND", "GLD"],
            "nw_ic": [0.30, 0.02, 0.12],
        }
    )

    result = build_shadow_consensus_table(
        signals,
        quality_df,
        score_col="nw_ic",
        lambda_mix=0.25,
    )

    assert list(result["variant"]) == ["equal_weight", "quality_weighted"]
    assert set(result["consensus"]) == {"OUTPERFORM"}
    assert "top_benchmark" in result.columns
    assert result.loc[result["variant"] == "quality_weighted", "top_benchmark"].iloc[0] == "VOO"


def test_build_shadow_consensus_table_falls_back_to_equal_weights() -> None:
    signals = pd.DataFrame(
        {
            "predicted_relative_return": [0.01, -0.01],
            "ic": [0.04, 0.04],
            "hit_rate": [0.55, 0.55],
            "signal": ["NEUTRAL", "NEUTRAL"],
        },
        index=pd.Index(["VOO", "BND"], name="benchmark"),
    )

    result = build_shadow_consensus_table(signals, benchmark_quality_df=None)

    equal_row = result[result["variant"] == "equal_weight"].iloc[0]
    shadow_row = result[result["variant"] == "quality_weighted"].iloc[0]
    assert shadow_row["mean_predicted_return"] == pytest.approx(equal_row["mean_predicted_return"])
    assert shadow_row["mean_ic"] == pytest.approx(equal_row["mean_ic"])
