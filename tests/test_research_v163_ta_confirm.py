"""Tests for the v163 TA survivor confirmation helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_survivor_cap_is_enforced_for_features_and_groups() -> None:
    from results.research.v163_ta_survivor_confirm import select_survivors

    screen = pd.DataFrame(
        {
            "feature": [f"ta_feature_{idx}" for idx in range(8)],
            "feature_group": ["momentum", "trend", "volatility", "regime"] * 2,
            "positive_benchmark_count": [8, 7, 6, 5, 4, 3, 2, 1],
            "mean_delta_score": [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        }
    )
    selected = select_survivors(screen, max_features=6, max_groups=3)

    assert len(selected) <= 6
    assert selected["feature_group"].nunique() <= 3
    assert selected.iloc[0]["feature"] == "ta_feature_0"


def test_correlation_pruning_removes_redundant_ta_features() -> None:
    from results.research.v163_ta_survivor_confirm import prune_correlated_features

    frame = pd.DataFrame(
        {
            "mom_6m": [1.0, 2.0, 3.0, 4.0],
            "ta_ratio_roc_6m_voo": [1.01, 2.01, 3.01, 4.01],
            "ta_ratio_rsi_6m_voo": [4.0, 1.0, 3.0, 2.0],
        }
    )
    kept = prune_correlated_features(
        frame,
        candidate_features=["ta_ratio_roc_6m_voo", "ta_ratio_rsi_6m_voo"],
        baseline_features=["mom_6m"],
        threshold=0.95,
    )

    assert "ta_ratio_roc_6m_voo" not in kept
    assert "ta_ratio_rsi_6m_voo" in kept


def test_regime_slices_are_stably_labeled() -> None:
    from results.research.v163_ta_survivor_confirm import assign_regime_slice

    dates = pd.to_datetime(["2019-12-31", "2020-06-30", "2022-01-31"])
    labels = [assign_regime_slice(ts) for ts in dates]

    assert labels == ["pre_2020", "covid_2020_2021", "post_2022"]


def test_candidate_json_is_deterministic() -> None:
    from results.research.v163_ta_survivor_confirm import build_candidate_payload

    survivors = pd.DataFrame(
        [
            {"feature": "b", "feature_group": "trend", "mean_delta_score": 0.02},
            {"feature": "a", "feature_group": "momentum", "mean_delta_score": 0.03},
        ]
    )
    payload_one = build_candidate_payload(survivors, recommendation="monitor_only")
    payload_two = build_candidate_payload(survivors, recommendation="monitor_only")

    assert json.dumps(payload_one, sort_keys=True) == json.dumps(payload_two, sort_keys=True)
    assert payload_one["features"] == ["a", "b"]
    assert payload_one["recommendation"] == "monitor_only"
