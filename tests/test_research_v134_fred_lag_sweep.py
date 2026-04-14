"""Tests for the v134 FRED publication lag sweep harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v134_fred_lag_sweep import (
    DEFAULT_CANDIDATE_PATH,
    _parse_lag_overrides,
    run_lag_sweep,
)


def test_default_no_override_recovers_v38_style_baseline() -> None:
    """The default config should reproduce the current research-frame baseline."""
    metrics = run_lag_sweep({})
    assert metrics["pooled_oos_r2"] == pytest.approx(-0.1578, abs=0.01)
    assert metrics["pooled_ic"] == pytest.approx(0.1261, abs=0.03)
    assert metrics["pooled_hit_rate"] == pytest.approx(0.7002, abs=0.03)


def test_negative_lag_raises_value_error() -> None:
    """Lag values below zero are structurally invalid."""
    with pytest.raises(ValueError, match="must be in \\[0, 3\\]"):
        run_lag_sweep({"GS10": -1})


def test_unknown_series_raises_key_error() -> None:
    """Unknown FRED series must be rejected."""
    with pytest.raises(KeyError, match="unknown FRED series"):
        run_lag_sweep({"NONEXISTENT_SERIES": 1})


def test_candidate_json_parses_to_expected_object() -> None:
    """The candidate file should be parseable through the CLI helper."""
    payload = _parse_lag_overrides(None, str(DEFAULT_CANDIDATE_PATH))
    assert set(payload) == {
        "GS10",
        "GS5",
        "GS2",
        "T10Y2Y",
        "T10YIE",
        "VIXCLS",
        "BAA10Y",
        "BAMLH0A0HYM2",
        "MORTGAGE30US",
    }
