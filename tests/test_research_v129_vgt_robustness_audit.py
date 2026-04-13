"""Tests for v129 VGT robustness audit logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v129_vgt_robustness_audit import (
    RESULTS_CSV,
    SUMMARY_MD,
    determine_verdict,
)


# ---------------------------------------------------------------------------
# Test: audit CSV has expected columns
# ---------------------------------------------------------------------------


class TestAuditCSVSchema:
    """Verify the output CSV artifact has the expected structure."""

    @pytest.fixture()
    def results_df(self) -> pd.DataFrame:
        assert RESULTS_CSV.exists(), f"Audit results CSV not found: {RESULTS_CSV}"
        return pd.read_csv(RESULTS_CSV)

    def test_expected_columns_present(self, results_df: pd.DataFrame) -> None:
        required = [
            "as_of_date",
            "model",
            "n_covered",
            "balanced_accuracy_covered",
            "brier_score",
        ]
        for col in required:
            assert col in results_df.columns, f"Missing column: {col}"

    def test_has_both_models(self, results_df: pd.DataFrame) -> None:
        models = set(results_df["model"].unique())
        assert "vgt_2feature" in models
        assert "lean_baseline" in models

    def test_has_all_as_of_dates(self, results_df: pd.DataFrame) -> None:
        dates = set(results_df["as_of_date"].unique())
        assert "2022-03-31" in dates
        assert "2023-03-31" in dates
        assert "2024-03-31" in dates

    def test_n_covered_non_negative(self, results_df: pd.DataFrame) -> None:
        assert (results_df["n_covered"] >= 0).all()


# ---------------------------------------------------------------------------
# Test: verdict logic with synthetic data
# ---------------------------------------------------------------------------


class TestVerdictLogic:
    """Verify the determine_verdict function with synthetic inputs."""

    @staticmethod
    def _make_results(
        ba_pairs: list[tuple[float, float]],
        n_covered_values: list[int] | None = None,
    ) -> pd.DataFrame:
        """Build a synthetic results DataFrame.

        Parameters
        ----------
        ba_pairs : list of (ba_2feature, ba_baseline) per as-of date
        n_covered_values : optional n_covered for the 2-feature model per date
        """
        dates = ["2022-03-31", "2023-03-31", "2024-03-31"]
        if n_covered_values is None:
            n_covered_values = [15, 15, 15]
        rows: list[dict[str, object]] = []
        for i, (ba2, bab) in enumerate(ba_pairs):
            rows.append({
                "as_of_date": dates[i],
                "model": "vgt_2feature",
                "n_covered": n_covered_values[i],
                "balanced_accuracy_covered": ba2,
                "brier_score": 0.20,
                "n_obs": 100,
                "features": "a|b",
                "n_features": 2,
            })
            rows.append({
                "as_of_date": dates[i],
                "model": "lean_baseline",
                "n_covered": 30,
                "balanced_accuracy_covered": bab,
                "brier_score": 0.25,
                "n_obs": 100,
                "features": "c|d|e",
                "n_features": 3,
            })
        return pd.DataFrame(rows)

    def test_stable_when_all_criteria_met(self) -> None:
        """BA advantage >= 0.05 at all dates, n_covered >= 10."""
        df = self._make_results([
            (0.80, 0.60),
            (0.75, 0.65),
            (0.90, 0.55),
        ])
        assert determine_verdict(df) == "STABLE"

    def test_unstable_when_ba_advantage_too_small(self) -> None:
        """One date has delta_BA < 0.05."""
        df = self._make_results([
            (0.80, 0.60),
            (0.62, 0.60),  # delta = 0.02 < 0.05
            (0.90, 0.55),
        ])
        assert determine_verdict(df) == "UNSTABLE"

    def test_unstable_when_n_covered_too_low(self) -> None:
        """One date has n_covered < 10."""
        df = self._make_results(
            [
                (0.80, 0.60),
                (0.75, 0.65),
                (0.90, 0.55),
            ],
            n_covered_values=[5, 15, 15],  # first date < 10
        )
        assert determine_verdict(df) == "UNSTABLE"

    def test_unstable_when_ba_is_nan(self) -> None:
        """NaN BA should produce UNSTABLE."""
        df = self._make_results([
            (float("nan"), 0.60),
            (0.75, 0.65),
            (0.90, 0.55),
        ])
        assert determine_verdict(df) == "UNSTABLE"

    def test_unstable_when_no_data(self) -> None:
        """Empty DataFrame should produce UNSTABLE."""
        df = pd.DataFrame(columns=[
            "as_of_date", "model", "n_covered",
            "balanced_accuracy_covered", "brier_score",
        ])
        assert determine_verdict(df) == "UNSTABLE"

    def test_stable_at_boundary(self) -> None:
        """delta_BA just above 0.05 and n_covered = 10 should be STABLE."""
        df = self._make_results(
            [
                (0.66, 0.60),  # delta = 0.06
                (0.71, 0.65),  # delta = 0.06
                (0.61, 0.55),  # delta = 0.06
            ],
            n_covered_values=[10, 10, 10],
        )
        assert determine_verdict(df) == "STABLE"


# ---------------------------------------------------------------------------
# Test: summary markdown exists
# ---------------------------------------------------------------------------


class TestSummaryArtifact:
    """Verify the summary markdown was generated."""

    def test_summary_exists(self) -> None:
        assert SUMMARY_MD.exists(), f"Summary MD not found: {SUMMARY_MD}"

    def test_summary_contains_verdict(self) -> None:
        text = SUMMARY_MD.read_text(encoding="utf-8")
        assert "STABLE" in text or "UNSTABLE" in text
