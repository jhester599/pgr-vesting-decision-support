"""Tests for the v157 term premium 3M differential evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_compute_term_premium_diff_basic() -> None:
    from results.research.v157_term_premium_eval import compute_term_premium_diff
    raw = pd.Series([1.0, 1.1, 1.2, 1.5, 1.4, 1.6], name="term_premium_10y")
    result = compute_term_premium_diff(raw, periods=3)
    assert result.name == "term_premium_diff_3m"
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[2])
    assert abs(result.iloc[3] - 0.5) < 1e-9


def test_compute_term_premium_diff_wrong_periods() -> None:
    from results.research.v157_term_premium_eval import compute_term_premium_diff
    with pytest.raises(ValueError, match="periods"):
        compute_term_premium_diff(pd.Series([1.0, 2.0]), periods=0)


def test_augment_feature_df_adds_column() -> None:
    from results.research.v157_term_premium_eval import augment_with_term_diff
    df = pd.DataFrame({"term_premium_10y": [1.0, 1.1, 1.2, 1.5, 1.4]})
    result = augment_with_term_diff(df)
    assert "term_premium_diff_3m" in result.columns
    assert "term_premium_10y" in result.columns


def test_augment_feature_df_missing_column() -> None:
    from results.research.v157_term_premium_eval import augment_with_term_diff
    df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
    result = augment_with_term_diff(df)
    assert "term_premium_diff_3m" not in result.columns


@pytest.mark.slow
def test_run_term_premium_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v157_term_premium_eval import run_term_premium_evaluation
    candidate_path = tmp_path / "v157_candidate.json"
    result = run_term_premium_evaluation(benchmarks=["BND"], candidate_path=candidate_path)
    assert candidate_path.exists()
    assert "term_premium_winners" in result


def test_candidate_file_schema() -> None:
    import json
    from results.research.v157_term_premium_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "term_premium_winners" in data
    assert "recommendation" in data
