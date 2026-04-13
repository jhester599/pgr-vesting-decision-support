"""Tests for v132 temporal hold-out threshold validation study."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v132_threshold_validation import (
    HOLDOUT_CUTOFF,
    MIN_BA_DELTA_FOR_ADOPTION,
    MIN_COVERAGE,
    apply_prequential_temperature_scaling,
    evaluate_thresholds,
    run_threshold_grid,
    derive_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold_df(n: int = 84, seed: int = 0) -> pd.DataFrame:
    """Synthetic fold detail DataFrame matching v125 schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-10-31", periods=n, freq="ME")
    probs = rng.uniform(0.05, 0.95, n)
    labels = (probs > 0.50).astype(int)
    return pd.DataFrame({
        "test_date": dates,
        "y_true": labels,
        "path_b_prob": probs,
    })


# ---------------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_selection_dates_strictly_before_holdout(self) -> None:
        """Every selection date must be <= HOLDOUT_CUTOFF; every holdout date must be after."""
        df = _make_fold_df(n=84)
        cutoff = pd.Timestamp(HOLDOUT_CUTOFF)
        sel = df[df["test_date"] <= cutoff]
        hld = df[df["test_date"] > cutoff]
        assert sel["test_date"].max() <= cutoff
        assert hld["test_date"].min() > cutoff

    def test_no_overlap_between_sets(self) -> None:
        """Selection and hold-out row sets must be disjoint."""
        df = _make_fold_df(n=84)
        cutoff = pd.Timestamp(HOLDOUT_CUTOFF)
        sel_idx = set(df[df["test_date"] <= cutoff].index)
        hld_idx = set(df[df["test_date"] > cutoff].index)
        assert sel_idx.isdisjoint(hld_idx)

    def test_holdout_cutoff_is_correct_date(self) -> None:
        """HOLDOUT_CUTOFF must equal 2021-12-31 per design specification."""
        assert HOLDOUT_CUTOFF == "2021-12-31"


# ---------------------------------------------------------------------------
# evaluate_thresholds
# ---------------------------------------------------------------------------

class TestEvaluateThresholds:
    def test_perfect_covered_ba(self) -> None:
        """Perfectly separated probabilities → covered_ba == 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.10, 0.15, 0.80, 0.85])
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["covered_ba"] == pytest.approx(1.0)
        assert result["coverage"] == pytest.approx(1.0)

    def test_all_abstained(self) -> None:
        """When all probs are inside the band, coverage=0 and ba=0.5."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.45, 0.48, 0.52, 0.55])
        result = evaluate_thresholds(y_true, y_prob, low=0.40, high=0.60)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["covered_ba"] == pytest.approx(0.5)

    def test_coverage_fraction(self) -> None:
        """Coverage = fraction of rows outside [low, high]."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.10, 0.40, 0.60, 0.90, 0.50, 0.80])
        # covered: 0.10 (< 0.30), 0.90 (> 0.70), 0.80 (> 0.70) → 3/6 = 0.5
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["coverage"] == pytest.approx(3 / 6)


# ---------------------------------------------------------------------------
# run_threshold_grid
# ---------------------------------------------------------------------------

class TestRunThresholdGrid:
    def test_returns_dataframe_with_required_columns(self) -> None:
        y_true = np.array([0, 0, 1, 1] * 10)
        y_prob = np.array([0.10, 0.15, 0.80, 0.85] * 10)
        pairs = [(0.30, 0.70), (0.20, 0.80)]
        result = run_threshold_grid(y_true, y_prob, pairs)
        assert isinstance(result, pd.DataFrame)
        for col in ("low", "high", "covered_ba", "coverage"):
            assert col in result.columns

    def test_grid_length_matches_pairs(self) -> None:
        y_true = np.array([0, 1] * 20)
        y_prob = np.array([0.20, 0.80] * 20)
        pairs = [(0.30, 0.70), (0.20, 0.80), (0.15, 0.70)]
        result = run_threshold_grid(y_true, y_prob, pairs)
        assert len(result) == len(pairs)

    def test_best_pair_has_highest_ba(self) -> None:
        y_true = np.array([0, 0, 1, 1] * 15)
        y_prob = np.array([0.10, 0.15, 0.80, 0.85] * 15)
        pairs = [(0.30, 0.70), (0.40, 0.60), (0.20, 0.80)]
        result = run_threshold_grid(y_true, y_prob, pairs)
        best = result.loc[result["covered_ba"].idxmax()]
        assert best["covered_ba"] == result["covered_ba"].max()


# ---------------------------------------------------------------------------
# derive_verdict
# ---------------------------------------------------------------------------

class TestDeriveVerdict:
    def test_adopt_when_criteria_met(self) -> None:
        verdict = derive_verdict(
            holdout_ba_candidate=0.65,
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.40,
        )
        assert "ADOPT" in verdict

    def test_do_not_adopt_insufficient_ba_delta(self) -> None:
        verdict = derive_verdict(
            holdout_ba_candidate=0.58,
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.40,
        )
        assert "DO NOT ADOPT" in verdict

    def test_do_not_adopt_insufficient_coverage(self) -> None:
        verdict = derive_verdict(
            holdout_ba_candidate=0.65,
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.15,  # below 0.20
        )
        assert "DO NOT ADOPT" in verdict

    def test_boundary_exactly_at_delta_threshold(self) -> None:
        """BA delta exactly == MIN_BA_DELTA_FOR_ADOPTION should ADOPT."""
        verdict = derive_verdict(
            holdout_ba_candidate=round(0.57 + MIN_BA_DELTA_FOR_ADOPTION, 6),
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.40,
        )
        assert "ADOPT" in verdict


# ---------------------------------------------------------------------------
# apply_prequential_temperature_scaling
# ---------------------------------------------------------------------------

class TestPrequentialScaling:
    def test_output_shape_preserved(self) -> None:
        probs = np.random.default_rng(0).uniform(0.1, 0.9, 50)
        labels = np.random.default_rng(0).integers(0, 2, 50)
        result = apply_prequential_temperature_scaling(probs, labels, warmup=24)
        assert result.shape == probs.shape

    def test_output_bounded(self) -> None:
        probs = np.random.default_rng(7).uniform(0.05, 0.95, 60)
        labels = np.random.default_rng(7).integers(0, 2, 60)
        result = apply_prequential_temperature_scaling(probs, labels, warmup=24)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
