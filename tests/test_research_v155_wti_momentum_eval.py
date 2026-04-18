"""Tests for the v155 WTI momentum evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_wti_feature_is_in_feature_family() -> None:
    """wti_return_3m must exist in the macro_rates_spreads family."""
    from src.research.v87_utils import available_feature_families
    import pandas as pd
    stub_cols = ["wti_return_3m", "yield_slope", "vix", "nfci", "mom_12m"]
    stub_df = pd.DataFrame({c: [0.0] for c in stub_cols})
    families = available_feature_families(stub_df)
    assert "wti_return_3m" in families.get("macro_rates_spreads", [])


def test_build_augmented_features_adds_wti() -> None:
    from results.research.v155_wti_momentum_eval import build_augmented_feature_cols
    from src.research.v37_utils import RIDGE_FEATURES_12
    base = list(RIDGE_FEATURES_12)
    augmented = build_augmented_feature_cols(base, extra=["wti_return_3m"])
    assert "wti_return_3m" in augmented
    assert len(augmented) == len(set(augmented))


def test_build_augmented_features_preserves_order() -> None:
    from results.research.v155_wti_momentum_eval import build_augmented_feature_cols
    base = ["a", "b", "c"]
    result = build_augmented_feature_cols(base, extra=["d", "b"])
    assert result == ["a", "b", "c", "d"]


@pytest.mark.slow
def test_run_wti_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v155_wti_momentum_eval import run_wti_evaluation
    candidate_path = tmp_path / "v155_wti_candidate.json"
    result = run_wti_evaluation(
        benchmarks=["DBC"],
        candidate_path=candidate_path,
    )
    assert candidate_path.exists()
    assert "rows" in result
    assert len(result["rows"]) >= 1


def test_candidate_file_schema() -> None:
    import json
    from results.research.v155_wti_momentum_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "wti_winners" in data
    assert "recommendation" in data
