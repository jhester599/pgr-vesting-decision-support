from __future__ import annotations

import pandas as pd
import pytest

from scripts.candidate_model_bakeoff import _dedupe, summarize_candidate_bakeoff


def test_dedupe_preserves_first_occurrence_order():
    assert _dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_summarize_candidate_bakeoff_groups_candidates():
    detail = pd.DataFrame(
        [
            {
                "candidate_name": "elasticnet_lean_v1",
                "model_type": "elasticnet",
                "benchmark": "VXUS",
                "n_features": 12,
                "ic": 0.10,
                "hit_rate": 0.56,
                "oos_r2": -0.20,
                "mae": 0.11,
                "policy_return_sign": 0.04,
                "policy_return_tiered": 0.02,
                "policy_uplift_vs_sell_50_sign": 0.01,
                "policy_uplift_vs_sell_50_tiered": -0.01,
                "notes": "lean",
            },
            {
                "candidate_name": "elasticnet_lean_v1",
                "model_type": "elasticnet",
                "benchmark": "BND",
                "n_features": 12,
                "ic": 0.14,
                "hit_rate": 0.60,
                "oos_r2": -0.10,
                "mae": 0.10,
                "policy_return_sign": 0.05,
                "policy_return_tiered": 0.03,
                "policy_uplift_vs_sell_50_sign": 0.02,
                "policy_uplift_vs_sell_50_tiered": 0.00,
                "notes": "lean",
            },
        ]
    )

    summary = summarize_candidate_bakeoff(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["model_type"] == "elasticnet"
    assert row["n_benchmarks"] == 2
    assert row["mean_ic"] == pytest.approx(0.12)
    assert row["mean_policy_return_sign"] == pytest.approx(0.045)
