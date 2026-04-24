"""Tests for x22 dividend size baseline helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_prior_positive_year_baseline_uses_only_earlier_history() -> None:
    from src.research.x22_dividend_size_baselines import baseline_from_history

    history = pd.Series([0.02, 0.05, 0.09], index=[2019, 2021, 2022], dtype=float)

    result = baseline_from_history(history, mode="prior_positive_year")

    assert result == 0.09


def test_x22_candidate_target_scales_keep_raw_anchor_and_best_normalized_scales() -> None:
    from src.research.x22_dividend_size_baselines import candidate_target_scales

    result = candidate_target_scales()

    assert "raw_dollars" in result
    assert "to_current_bvps" in result
    assert "to_persistent_bvps" in result
    assert "to_price" not in result


def test_summarize_x22_results_ranks_by_dollar_mae() -> None:
    from src.research.x22_dividend_size_baselines import summarize_x22_results

    detail = pd.DataFrame(
        [
            {"target_scale": "raw_dollars", "model_name": "a", "dollar_mae": 5.0},
            {"target_scale": "to_current_bvps", "model_name": "b", "dollar_mae": 3.0},
        ]
    )

    summary = summarize_x22_results(detail)

    assert list(summary["target_scale"]) == ["to_current_bvps", "raw_dollars"]
    assert list(summary["rank"]) == [1, 2]
