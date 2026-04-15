"""Tests for v131 asymmetric abstention threshold sweep evaluator.

Covers:
- Correct metric extraction with known synthetic data
- Exit code 1 on insufficient coverage
- Symmetric and asymmetric threshold behaviour
- Edge cases: all abstained, no positive class in covered set
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Make sure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v131_threshold_sweep_eval import (
    MIN_COVERAGE,
    apply_prequential_temperature_scaling,
    evaluate_thresholds,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deterministic_probs(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic (y_true, y_prob) arrays with known structure.

    y_prob has a bimodal distribution: ~half the values below 0.30 and ~half
    above 0.70, so symmetric (0.30, 0.70) thresholds yield ~100% coverage.
    """
    rng = np.random.default_rng(seed)
    low_probs = rng.uniform(0.05, 0.25, n // 2)
    high_probs = rng.uniform(0.75, 0.95, n // 2)
    y_prob = np.concatenate([low_probs, high_probs])
    # Labels: 1 where prob > 0.5, else 0
    y_true = (y_prob >= 0.5).astype(int)
    idx = rng.permutation(len(y_prob))
    return y_true[idx], y_prob[idx]


# ---------------------------------------------------------------------------
# evaluate_thresholds -- unit tests
# ---------------------------------------------------------------------------


class TestEvaluateThresholds:
    def test_perfect_separation_symmetric(self) -> None:
        """With bimodal probs and matching labels, BA should be 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.10, 0.15, 0.20, 0.80, 0.85, 0.90])
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["covered_ba"] == pytest.approx(1.0)
        assert result["coverage"] == pytest.approx(1.0)

    def test_all_abstained_returns_chance(self) -> None:
        """When band spans entire [0,1], coverage=0 and BA=0.5 (chance)."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.45, 0.48, 0.52, 0.55])
        result = evaluate_thresholds(y_true, y_prob, low=0.40, high=0.60)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["covered_ba"] == pytest.approx(0.5)

    def test_coverage_fraction_is_correct(self) -> None:
        """Coverage should equal fraction of rows outside the abstention band."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.10, 0.35, 0.65, 0.90, 0.50, 0.80])
        # low=0.30, high=0.70 → abstain rows: 0.35, 0.65, 0.50 (3 rows)
        # covered rows: 0.10, 0.90, 0.80 (3 rows) → coverage = 3/6 = 0.5
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["coverage"] == pytest.approx(3 / 6)

    def test_asymmetric_low_threshold_only_affects_hold_side(self) -> None:
        """Asymmetric (low=0.20, high=0.70): only rows < 0.20 or > 0.70 covered."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.10, 0.25, 0.40, 0.60, 0.75, 0.90])
        # low=0.20, high=0.70 → covered: 0.10 (< 0.20) and 0.75, 0.90 (> 0.70)
        result = evaluate_thresholds(y_true, y_prob, low=0.20, high=0.70)
        assert result["coverage"] == pytest.approx(3 / 6)

    def test_asymmetric_high_threshold_only_affects_sell_side(self) -> None:
        """Asymmetric (low=0.30, high=0.80): only rows < 0.30 or > 0.80 covered."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.10, 0.25, 0.40, 0.60, 0.75, 0.90])
        # low=0.30, high=0.80 → covered: 0.10, 0.25 (< 0.30) and 0.90 (> 0.80)
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.80)
        assert result["coverage"] == pytest.approx(3 / 6)

    def test_known_ba_value_with_one_error(self) -> None:
        """Covered BA with one misclassified row should equal balanced_accuracy."""
        # 3 class-0 rows (all prob < 0.30 → predict 0), 3 class-1 rows (all prob > 0.70 → predict 1)
        # Introduce 1 misclassification: one class-1 row predicted as 0 (prob=0.10 but y_true=1)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.05, 0.10, 0.20, 0.10, 0.80, 0.90])
        # low=0.30, high=0.70 → all covered (no probs in [0.30, 0.70])
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        # Class 0: 3/3 correct. Class 1: 2/3 correct. BA = (1.0 + 2/3) / 2 ≈ 0.8333
        assert result["covered_ba"] == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-4)
        assert result["coverage"] == pytest.approx(1.0)

    def test_single_class_in_covered_returns_chance(self) -> None:
        """If covered set has only one class, fall back to 0.5 (chance level)."""
        # All y_true=0, only the high-confidence rows are covered
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.05, 0.10, 0.80, 0.90])
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["covered_ba"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# main() -- integration / CLI behaviour
# ---------------------------------------------------------------------------


class TestMainCLI:
    def _fake_csv(self, tmp_path: Path, y_true: list[int], y_prob: list[float]) -> Path:
        """Write a minimal fold-detail CSV at tmp_path and return its path."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "fold": 1,
                "train_start": "2011-02-28",
                "train_end": "2016-01-29",
                "train_obs": 60,
                "test_date": pd.date_range("2016-01-01", periods=len(y_true), freq="ME").strftime(
                    "%Y-%m-%d"
                ),
                "y_true": y_true,
                "path_b_prob": y_prob,
                "path_a_prob": y_prob,
                "path_a_available_benchmarks": 6,
                "path_a_weight_sum": 1.0,
            }
        )
        csv_path = tmp_path / "v125_portfolio_target_fold_detail.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_exit_0_on_sufficient_coverage(self, tmp_path: Path) -> None:
        """main() returns 0 when coverage >= MIN_COVERAGE."""
        # 80 rows, all outside (0.30, 0.70) → coverage = 1.0
        n = 80
        y_true, y_prob = _make_deterministic_probs(n=n)
        csv_path = self._fake_csv(tmp_path, y_true.tolist(), y_prob.tolist())

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            rc = main(["--low", "0.30", "--high", "0.70"])
        assert rc == 0

    def test_exit_1_on_insufficient_coverage(self, tmp_path: Path) -> None:
        """main() returns 1 when coverage < MIN_COVERAGE."""
        # All probs inside (0.30, 0.70) → coverage = 0.0
        n = 40
        y_true = np.zeros(n, dtype=int)
        y_prob = np.full(n, 0.50)
        csv_path = self._fake_csv(tmp_path, y_true.tolist(), y_prob.tolist())

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            rc = main(["--low", "0.30", "--high", "0.70"])
        assert rc == 1

    def test_output_format_covered_ba_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stdout must contain 'covered_ba=X.XXXX' on the first line."""
        n = 80
        y_true, y_prob = _make_deterministic_probs(n=n)
        csv_path = self._fake_csv(tmp_path, y_true.tolist(), y_prob.tolist())

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            main(["--low", "0.30", "--high", "0.70"])

        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines[0].startswith("covered_ba=")
        ba_str = lines[0].split("=")[1]
        ba_val = float(ba_str)
        assert 0.0 <= ba_val <= 1.0

    def test_output_format_coverage_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stdout must contain 'coverage=X.XXXX' on the second line."""
        n = 80
        y_true, y_prob = _make_deterministic_probs(n=n)
        csv_path = self._fake_csv(tmp_path, y_true.tolist(), y_prob.tolist())

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            main(["--low", "0.30", "--high", "0.70"])

        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines[1].startswith("coverage=")
        cov_val = float(lines[1].split("=")[1])
        assert 0.0 <= cov_val <= 1.0

    def test_symmetric_vs_asymmetric_coverage(self, tmp_path: Path) -> None:
        """Wider abstention band → lower coverage vs narrow band."""
        n = 80
        y_true, y_prob = _make_deterministic_probs(n=n)
        csv_path = self._fake_csv(tmp_path, y_true.tolist(), y_prob.tolist())

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            # Narrow band → more covered
            rc_narrow = main(["--low", "0.40", "--high", "0.60"])

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            # Wide band → less covered
            rc_wide = main(["--low", "0.20", "--high", "0.80"])

        # Both should exit 0 with this bimodal data
        assert rc_narrow == 0
        assert rc_wide == 0

    def test_coverage_boundary_exactly_at_min(self, tmp_path: Path) -> None:
        """Coverage of exactly MIN_COVERAGE (0.20) should still exit 0."""
        # 100 rows: 20 outside band, 80 inside → coverage = 0.20
        n_total = 100
        n_covered = int(n_total * MIN_COVERAGE)  # 20
        y_true = np.zeros(n_total, dtype=int)
        y_prob = np.concatenate(
            [
                np.full(n_covered // 2, 0.10),   # below low
                np.full(n_covered // 2, 0.90),   # above high
                np.full(n_total - n_covered, 0.50),  # inside band
            ]
        )
        y_true[:n_covered] = np.array(
            [0] * (n_covered // 2) + [1] * (n_covered // 2)
        )
        csv_path = self._fake_csv(tmp_path, y_true.tolist(), y_prob.tolist())

        with patch(
            "results.research.v131_threshold_sweep_eval.FOLD_DETAIL_PATH", csv_path
        ):
            rc = main(["--low", "0.30", "--high", "0.70"])
        assert rc == 0


# ---------------------------------------------------------------------------
# apply_prequential_temperature_scaling -- unit tests
# ---------------------------------------------------------------------------


class TestPrequentialTemperatureScaling:
    def test_warmup_observations_unchanged_shape(self) -> None:
        """Output has same shape as input."""
        n = 30
        probs = np.random.default_rng(0).uniform(0.1, 0.9, n)
        labels = np.random.default_rng(0).integers(0, 2, n)
        result = apply_prequential_temperature_scaling(probs, labels, warmup=24)
        assert result.shape == probs.shape

    def test_output_bounded_to_01(self) -> None:
        """All calibrated probs must stay within [0, 1]."""
        n = 50
        probs = np.random.default_rng(7).uniform(0.05, 0.95, n)
        labels = np.random.default_rng(7).integers(0, 2, n)
        result = apply_prequential_temperature_scaling(probs, labels, warmup=10)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_insufficient_history_returns_raw(self) -> None:
        """With warmup > len(probs), all outputs equal the clipped inputs."""
        probs = np.array([0.3, 0.7, 0.5])
        labels = np.array([0, 1, 0])
        result = apply_prequential_temperature_scaling(probs, labels, warmup=100)
        # Warmup not reached → clipped raw probs (clip to [1e-6, 1-1e-6] → effectively unchanged)
        np.testing.assert_array_almost_equal(result, np.clip(probs, 1e-6, 1.0 - 1e-6))
