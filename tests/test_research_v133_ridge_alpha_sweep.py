"""Tests for the v133 Ridge alpha-grid sweep harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v133_ridge_alpha_sweep import run_ridge_alpha_sweep


@pytest.mark.slow
def test_default_grid_produces_reasonable_r2() -> None:
    """The default grid should recover the current ridge-only baseline band."""
    metrics = run_ridge_alpha_sweep(1e-4, 1e2, 50, benchmarks=["VOO", "BND"])
    assert -5.0 <= metrics["pooled_oos_r2"] <= 0.50
    assert 0.0 <= metrics["pooled_hit_rate"] <= 1.0


def test_alpha_max_less_than_alpha_min_raises() -> None:
    """Invalid alpha bounds should fail fast."""
    with pytest.raises(ValueError, match="alpha_max must be >= alpha_min"):
        run_ridge_alpha_sweep(1e2, 1e-4, 50)


def test_small_grid_size_raises() -> None:
    """Too-small alpha grids should be rejected."""
    with pytest.raises(ValueError, match="n_alpha must be >= 10"):
        run_ridge_alpha_sweep(1e-4, 1e2, 9)
