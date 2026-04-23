"""Tests for x8 synthesis utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_extract_horizon_leaders_selects_lowest_rank_per_horizon() -> None:
    from src.research.x8_synthesis import extract_horizon_leaders

    rows = [
        {
            "horizon_months": 1,
            "model_name": "weaker",
            "rank": 2,
            "implied_price_mae": 5.0,
        },
        {
            "horizon_months": 1,
            "model_name": "stronger",
            "rank": 1,
            "implied_price_mae": 4.0,
        },
        {
            "horizon_months": 3,
            "model_name": "only",
            "rank": 1,
            "implied_price_mae": 7.0,
        },
    ]

    leaders = extract_horizon_leaders(
        rows,
        lane="direct_return",
        primary_metric="implied_price_mae",
    )

    assert leaders[0]["horizon_months"] == 1
    assert leaders[0]["model_name"] == "stronger"
    assert leaders[0]["lane"] == "direct_return"
    assert leaders[0]["primary_metric_value"] == 4.0
    assert leaders[1]["horizon_months"] == 3


def test_count_gate_successes_counts_true_values_and_horizons() -> None:
    from src.research.x8_synthesis import count_gate_successes

    rows = [
        {"horizon_months": 1, "beats_no_change": True},
        {"horizon_months": 1, "beats_no_change": False},
        {"horizon_months": 3, "beats_no_change": True},
    ]

    result = count_gate_successes(rows, gate_column="beats_no_change")

    assert result == {
        "gate_column": "beats_no_change",
        "true_count": 2,
        "true_horizon_count": 2,
        "observed_row_count": 3,
        "observed_horizon_count": 2,
    }


def test_build_shadow_readiness_stays_not_ready_for_mixed_research() -> None:
    from src.research.x8_synthesis import build_shadow_readiness

    readiness = build_shadow_readiness(
        classification_gate_horizons=2,
        direct_return_gate_horizons=1,
        decomposition_path_consistent=False,
        special_dividend_n_obs=18,
    )

    assert readiness["status"] == "not_ready"
    assert readiness["production_changes"] is False
    assert readiness["shadow_changes"] is False
    assert "mixed" in readiness["rationale"].lower()


def test_json_records_converts_missing_and_nonfinite_values_to_null() -> None:
    from src.research.x8_synthesis import json_records

    frame = pd.DataFrame(
        [
            {
                "model_name": "demo",
                "finite": 1.5,
                "missing": np.nan,
                "infinite": np.inf,
            }
        ]
    )

    records = json_records(frame)

    assert records == [
        {
            "model_name": "demo",
            "finite": 1.5,
            "missing": None,
            "infinite": None,
        }
    ]
