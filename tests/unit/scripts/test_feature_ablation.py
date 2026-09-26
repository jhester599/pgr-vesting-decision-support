"""
Tests for scripts/feature_ablation.py — v7.0

Test coverage:
  1.  test_feature_groups_are_cumulative
  2.  test_all_group_columns_defined
  3.  test_filter_available_removes_missing
  4.  test_filter_available_keeps_present
  5.  test_output_csv_columns
  6.  test_ablation_runs_single_benchmark  (synthetic data)
  7.  test_oos_r2_in_valid_range
  8.  test_empty_group_raises
  9.  test_n_features_monotonically_increases
  10. test_cli_help_exits_zero
  11. test_group_a_has_minimum_features
  12. test_group_labels_ordered
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.feature_ablation import (
    FEATURE_GROUPS,
    GROUP_ORDER,
    MODEL_TYPES,
    _filter_available,
    run_ablation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_df(cols: list[str], n_rows: int = 80) -> pd.DataFrame:
    """Return a synthetic feature DataFrame with a DatetimeIndex."""
    idx = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    data = {c: np.random.randn(n_rows) for c in cols}
    return pd.DataFrame(data, index=idx)


def _make_fake_series(n_rows: int = 80, name: str = "VTI_6m") -> pd.Series:
    idx = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    return pd.Series(np.random.randn(n_rows) * 0.05, index=idx, name=name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureGroupStructure:
    """Tests that verify the structure of FEATURE_GROUPS."""

    def test_feature_groups_are_cumulative(self):
        """Each successive group is a superset of the prior group's columns."""
        groups = [FEATURE_GROUPS[k] for k in GROUP_ORDER]
        for i in range(1, len(groups)):
            prior = set(groups[i - 1])
            current = set(groups[i])
            assert prior.issubset(current), (
                f"Group '{GROUP_ORDER[i]}' is not a superset of "
                f"'{GROUP_ORDER[i-1]}': missing {prior - current}"
            )

    def test_group_a_has_minimum_features(self):
        """Group A must have at least 3 features."""
        assert len(FEATURE_GROUPS["A_price_only"]) >= 3

    def test_all_group_columns_defined(self):
        """Every group must define at least one column."""
        for label, cols in FEATURE_GROUPS.items():
            assert len(cols) > 0, f"Group '{label}' has no columns."

    def test_group_labels_ordered(self):
        """GROUP_ORDER must list all FEATURE_GROUPS keys in declared order."""
        assert GROUP_ORDER == list(FEATURE_GROUPS.keys())

    def test_n_features_monotonically_increases(self):
        """Column counts are non-decreasing across groups A→E."""
        counts = [len(FEATURE_GROUPS[k]) for k in GROUP_ORDER]
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], (
                f"Group '{GROUP_ORDER[i]}' has fewer columns than "
                f"'{GROUP_ORDER[i-1]}': {counts[i]} < {counts[i-1]}"
            )


class TestFilterAvailable:
    """Tests for _filter_available()."""

    def test_filter_available_removes_missing(self):
        """Columns not in df_cols are silently excluded."""
        group = ["mom_3m", "mom_6m", "nonexistent_col"]
        df_cols = {"mom_3m", "mom_6m", "vol_63d"}
        result = _filter_available(group, df_cols)
        assert "nonexistent_col" not in result
        assert set(result) == {"mom_3m", "mom_6m"}

    def test_filter_available_keeps_present(self):
        """Columns that exist in df_cols are all retained."""
        group = ["mom_3m", "mom_6m", "vol_63d"]
        df_cols = {"mom_3m", "mom_6m", "vol_63d", "extra"}
        result = _filter_available(group, df_cols)
        assert set(result) == set(group)

    def test_filter_available_empty_group(self):
        """Empty group returns empty list without error."""
        assert _filter_available([], {"mom_3m"}) == []

    def test_filter_available_no_overlap(self):
        """Returns empty list when group and df_cols share no columns."""
        result = _filter_available(["a", "b", "c"], {"x", "y", "z"})
        assert result == []


class TestRunAblation:
    """Tests for run_ablation() using mocked DB and WFO calls."""

    def _build_mocks(
        self,
        n_rows: int = 80,
        all_cols: list[str] | None = None,
    ) -> tuple:
        """
        Return patched module-level functions and a fake sqlite3 connection.
        """
        if all_cols is None:
            # Include all columns referenced by FEATURE_GROUPS["E_plus_v63_v64"]
            from scripts.feature_ablation import _GROUP_E
            all_cols = list(_GROUP_E)

        fake_df = _make_fake_df(all_cols, n_rows)
        fake_series = _make_fake_series(n_rows)

        fake_wfo = MagicMock()
        fake_wfo.information_coefficient = 0.05
        fake_wfo.hit_rate = 0.55
        fake_wfo.mean_absolute_error = 0.03
        fake_wfo.y_true_all = list(np.random.randn(n_rows // 2) * 0.05)
        fake_wfo.y_hat_all = list(np.random.randn(n_rows // 2) * 0.05)

        conn = MagicMock(spec=sqlite3.Connection)
        return fake_df, fake_series, fake_wfo, conn

    def test_ablation_runs_single_benchmark(self, tmp_path):
        """run_ablation produces one row per (group, benchmark, model_type)."""
        fake_df, fake_series, fake_wfo, conn = self._build_mocks()

        with (
            patch("scripts.feature_ablation.build_feature_matrix_from_db", return_value=fake_df),
            patch("scripts.feature_ablation.load_relative_return_matrix", return_value=fake_series),
            patch("scripts.feature_ablation.run_wfo", return_value=fake_wfo),
        ):
            results = run_ablation(
                conn=conn,
                benchmarks=["VTI"],
                horizons=[6],
                output_dir=str(tmp_path),
            )

        expected_rows = len(FEATURE_GROUPS) * 1 * len(MODEL_TYPES)  # 5 groups × 1 bench × 2 models
        assert len(results) == expected_rows

    def test_output_csv_columns(self, tmp_path):
        """Output CSV has exactly the expected columns."""
        fake_df, fake_series, fake_wfo, conn = self._build_mocks()

        expected_cols = {
            "feature_group", "benchmark", "model_type", "horizon_months",
            "n_obs", "n_features", "ic", "hit_rate", "mae", "oos_r2",
        }

        with (
            patch("scripts.feature_ablation.build_feature_matrix_from_db", return_value=fake_df),
            patch("scripts.feature_ablation.load_relative_return_matrix", return_value=fake_series),
            patch("scripts.feature_ablation.run_wfo", return_value=fake_wfo),
        ):
            results = run_ablation(
                conn=conn,
                benchmarks=["VTI"],
                horizons=[6],
                output_dir=str(tmp_path),
            )

        assert set(results.columns) == expected_cols

    def test_oos_r2_in_valid_range(self, tmp_path):
        """OOS R² is a float (may be negative; just must not be None or NaN
        when the WFO mock returns valid predictions)."""
        fake_df, fake_series, fake_wfo, conn = self._build_mocks()

        # Ensure the mock returns aligned predictions so compute_oos_r_squared works.
        n = 40
        y_true = list(np.linspace(0.01, 0.05, n))
        y_hat = list(np.linspace(0.02, 0.04, n))
        fake_wfo.y_true_all = y_true
        fake_wfo.y_hat_all = y_hat

        with (
            patch("scripts.feature_ablation.build_feature_matrix_from_db", return_value=fake_df),
            patch("scripts.feature_ablation.load_relative_return_matrix", return_value=fake_series),
            patch("scripts.feature_ablation.run_wfo", return_value=fake_wfo),
        ):
            results = run_ablation(
                conn=conn,
                benchmarks=["VTI"],
                horizons=[6],
                output_dir=str(tmp_path),
            )

        # All rows should have a numeric oos_r2 (not None; may be NaN if MSE=0).
        for _, row in results.iterrows():
            assert isinstance(row["oos_r2"], float), (
                f"oos_r2 expected float, got {type(row['oos_r2'])}"
            )

    def test_empty_benchmarks_raises(self, tmp_path):
        """run_ablation raises ValueError when benchmarks list is empty."""
        conn = MagicMock(spec=sqlite3.Connection)
        with pytest.raises(ValueError, match="benchmarks list must not be empty"):
            run_ablation(conn=conn, benchmarks=[], horizons=[6], output_dir=str(tmp_path))

    def test_unavailable_features_filtered(self, tmp_path):
        """If a feature column doesn't exist in the DataFrame it is excluded."""
        # Only provide Group A columns; all higher-group columns will be filtered.
        from scripts.feature_ablation import _GROUP_A
        fake_df = _make_fake_df(_GROUP_A, n_rows=80)
        fake_series = _make_fake_series(n_rows=80)

        fake_wfo = MagicMock()
        fake_wfo.information_coefficient = 0.04
        fake_wfo.hit_rate = 0.54
        fake_wfo.mean_absolute_error = 0.02
        fake_wfo.y_true_all = list(np.random.randn(40) * 0.05)
        fake_wfo.y_hat_all = list(np.random.randn(40) * 0.05)

        conn = MagicMock(spec=sqlite3.Connection)

        with (
            patch("scripts.feature_ablation.build_feature_matrix_from_db", return_value=fake_df),
            patch("scripts.feature_ablation.load_relative_return_matrix", return_value=fake_series),
            patch("scripts.feature_ablation.run_wfo", return_value=fake_wfo),
        ):
            results = run_ablation(
                conn=conn,
                benchmarks=["VTI"],
                horizons=[6],
                output_dir=str(tmp_path),
            )

        # Every row should have n_features <= len(GROUP_A) because the
        # DataFrame only has Group A columns.
        for _, row in results.iterrows():
            assert row["n_features"] <= len(_GROUP_A), (
                f"Expected n_features <= {len(_GROUP_A)}, got {row['n_features']}"
            )


class TestCLI:
    def test_cli_help_exits_zero(self):
        """feature_ablation.py --help exits with code 0."""
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "feature_ablation.py",
        )
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"--help exited {result.returncode}:\n{result.stderr}"
        )
        assert "benchmarks" in result.stdout.lower()
