"""Tests for x19 post-policy dividend helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _annual_frame() -> pd.DataFrame:
    dates = pd.to_datetime([f"{year}-11-30" for year in range(2018, 2026)])
    return pd.DataFrame(
        {
            "current_bvps": np.linspace(30.0, 40.0, len(dates)),
            "persistent_bvps": np.linspace(31.0, 43.0, len(dates)),
            "bvps_growth_ytd": np.linspace(0.01, 0.12, len(dates)),
            "bvps_growth_3m": np.linspace(0.0, 0.05, len(dates)),
            "bvps_yoy_dollar_change": np.linspace(1.0, 4.0, len(dates)),
            "capital_generation_proxy": np.linspace(0.0, 0.2, len(dates)),
            "excess_capital_proxy": np.linspace(0.0, 0.15, len(dates)),
            "buyback_vs_capital_generation": np.linspace(0.0, 0.4, len(dates)),
            "premium_growth_x_margin_for_dividend": np.linspace(0.0, 0.03, len(dates)),
            "special_dividend_occurred": [0, 1, 1, 0, 1, 0, 1, 1],
            "special_dividend_excess": [0.0, 2.0, 4.6, 0.0, 1.4, 0.0, 4.5, 13.5],
        },
        index=dates,
    )


def test_filter_post_policy_annual_frame_keeps_eligible_snapshots() -> None:
    from src.research.x19_post_policy_dividend_model import filter_post_policy_annual_frame

    frame = _annual_frame()
    frame["model_eligible_post_policy"] = [0, 1, 1, 1, 1, 1, 1, 1]

    result = filter_post_policy_annual_frame(frame)

    assert result.index.min() == pd.Timestamp("2019-11-30")
    assert result["model_eligible_post_policy"].eq(1).all()


def test_build_x19_feature_sets_include_persistent_capital_block() -> None:
    from src.research.x19_post_policy_dividend_model import build_x19_feature_sets

    sets = build_x19_feature_sets(_annual_frame())

    assert "persistent_capital_generation" in sets
    assert "persistent_bvps" in sets["persistent_capital_generation"]
    assert len(sets["persistent_capital_generation"]) <= 12
