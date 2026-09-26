"""Tests for scripts/feature_cost_report.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.feature_cost_report import build_feature_cost_report, build_feature_set_health


def _sample_feature_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "mom_3m": np.arange(12, dtype=float),
            "combined_ratio_ttm": [np.nan, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "buyback_yield": [np.nan] * 6 + [0.01] * 6,
            "target_6m_return": np.random.randn(12),
        },
        index=idx,
    )


def test_build_feature_cost_report_has_expected_columns():
    df = _sample_feature_df()
    report = build_feature_cost_report(df)
    assert {"feature", "family", "n_non_null", "completeness_pct", "production_models"}.issubset(report.columns)
    assert "target_6m_return" not in set(report["feature"])


def test_build_feature_set_health_summarizes_models():
    df = _sample_feature_df()
    health = build_feature_set_health(df)
    assert "model_type" in health.columns
    assert "n_features" in health.columns
    assert len(health) >= 1
