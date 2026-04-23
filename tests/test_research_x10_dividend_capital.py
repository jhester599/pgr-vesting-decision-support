"""Tests for x10 dividend capital feature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _monthly_features() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=36, freq="BME")
    return pd.DataFrame(
        {
            "current_bvps": np.linspace(20.0, 30.0, 36),
            "bvps_growth_ytd": np.linspace(0.0, 0.12, 36),
            "bvps_growth_3m": np.linspace(0.0, 0.04, 36),
            "bvps_yoy_dollar_change": np.linspace(1.0, 3.0, 36),
            "underwriting_margin_ttm": np.linspace(0.01, 0.08, 36),
            "npw_growth_yoy": np.linspace(0.03, 0.15, 36),
            "pgr_premium_to_surplus": np.linspace(2.0, 2.5, 36),
            "unrealized_gain_pct_equity": np.linspace(-0.05, 0.03, 36),
            "buyback_yield": np.linspace(0.0, 0.02, 36),
            "premium_to_surplus_x_cr_delta": np.linspace(-1.0, 1.0, 36),
            "buyback_yield_x_pb_ratio": np.linspace(0.0, 0.06, 36),
            "premium_growth_x_underwriting_margin": np.linspace(0.0, 0.01, 36),
        },
        index=dates,
    )


def test_build_dividend_capital_features_adds_capital_proxies() -> None:
    from src.research.x10_dividend_capital import build_dividend_capital_features

    result = build_dividend_capital_features(_monthly_features())

    assert "capital_generation_proxy" in result.columns
    assert "excess_capital_proxy" in result.columns
    assert "buyback_vs_capital_generation" in result.columns
    assert result["capital_generation_proxy"].notna().any()


def test_november_capital_snapshots_are_november_only() -> None:
    from src.research.x10_dividend_capital import november_capital_snapshots

    result = november_capital_snapshots(_monthly_features())

    assert not result.empty
    assert set(result.index.month) == {11}


def test_x10_feature_sets_are_low_count_and_include_x9_capital() -> None:
    from src.research.x10_dividend_capital import build_x10_feature_sets

    features = build_dividend_features_for_test()
    sets = build_x10_feature_sets(features)

    assert "x9_capital_generation" in sets
    assert len(sets["x9_capital_generation"]) <= 12
    assert "capital_generation_proxy" in sets["x9_capital_generation"]


def build_dividend_features_for_test() -> pd.DataFrame:
    from src.research.x10_dividend_capital import build_dividend_capital_features

    return build_dividend_capital_features(_monthly_features())
