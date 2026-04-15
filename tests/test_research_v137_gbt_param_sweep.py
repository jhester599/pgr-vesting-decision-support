"""Tests for the v137 standalone GBT parameter sweep harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v137_gbt_param_sweep import run_gbt_sweep


def test_default_params_produce_reasonable_r2() -> None:
    """The baseline GBT config should land in a broad sanity range."""
    metrics = run_gbt_sweep(2, 50, 0.1, 0.8)
    assert -1.0 <= metrics["pooled_oos_r2"] <= 0.10
    assert 0.0 <= metrics["pooled_hit_rate"] <= 1.0


def test_max_depth_out_of_range_raises() -> None:
    """Tree depth above the approved range should fail."""
    with pytest.raises(ValueError, match="max_depth must be one of"):
        run_gbt_sweep(5, 50, 0.1, 0.8)


def test_n_estimators_out_of_range_raises() -> None:
    """Too many trees should be rejected."""
    with pytest.raises(ValueError, match="n_estimators must be in"):
        run_gbt_sweep(2, 201, 0.1, 0.8)


def test_subsample_below_floor_raises() -> None:
    """Subsample below the floor should be rejected."""
    with pytest.raises(ValueError, match="subsample must be in"):
        run_gbt_sweep(2, 50, 0.1, 0.4)
