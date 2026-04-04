"""Tests for scripts/benchmark_reduction.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.benchmark_reduction import (
    build_benchmark_scorecard,
    build_candidate_universes,
    run_benchmark_reduction,
)


def _sample_detail() -> pd.DataFrame:
    rows = []
    for benchmark, ic, hr, r2 in [
        ("VTI", 0.10, 0.56, 0.02),
        ("VOO", 0.08, 0.55, 0.01),
        ("VFH", 0.12, 0.57, 0.03),
        ("BND", 0.06, 0.54, 0.00),
        ("GLD", 0.04, 0.53, -0.01),
    ]:
        rows.append({"item_type": "model", "benchmark": benchmark, "ic": ic, "hit_rate": hr, "oos_r2": r2, "mae": 0.10})
        rows.append({"item_type": "ensemble", "benchmark": benchmark, "ic": ic + 0.01, "hit_rate": hr, "oos_r2": r2, "mae": 0.09})
    return pd.DataFrame(rows)


def test_build_benchmark_scorecard_has_composite_score():
    scorecard = build_benchmark_scorecard(_sample_detail())
    assert "composite_score" in scorecard.columns
    assert scorecard.iloc[0]["benchmark"] == "VFH"


def test_build_candidate_universes_returns_named_sets():
    scorecard = build_benchmark_scorecard(_sample_detail())
    candidates = build_candidate_universes(scorecard)
    assert {"top8_composite", "diversified_top8", "balanced_core7"}.issubset(set(candidates["universe_name"]))


def test_run_benchmark_reduction_writes_outputs(tmp_path):
    detail = _sample_detail()
    detail_path = tmp_path / "detail.csv"
    detail.to_csv(detail_path, index=False)

    fake_summary = pd.DataFrame(
        [
            {"item_type": "ensemble", "item_name": "ensemble", "horizon_months": 6, "n_benchmarks": 3, "mean_ic": 0.08, "median_ic": 0.08, "mean_hit_rate": 0.55, "mean_oos_r2": 0.01, "mean_mae": 0.10, "pass_rate": 0.0, "gate_status": "MARGINAL"}
        ]
    )

    with patch("scripts.benchmark_reduction.run_benchmark_suite", return_value=(pd.DataFrame(), fake_summary)):
        scorecard, candidates, candidate_summary = run_benchmark_reduction(
            conn=MagicMock(),
            detail_csv_path=str(detail_path),
            output_dir=str(tmp_path),
        )

    assert not scorecard.empty
    assert not candidates.empty
    assert not candidate_summary.empty
