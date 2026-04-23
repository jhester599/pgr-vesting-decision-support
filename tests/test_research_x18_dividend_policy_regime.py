"""Tests for x18 dividend policy regime utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_classify_dividend_regime_marks_pre_and_post_policy_change() -> None:
    from src.research.x18_dividend_policy_regime import classify_dividend_regime

    dates = pd.to_datetime(["2018-02-01", "2019-04-04"])
    dividends = pd.DataFrame({"amount": [1.1247, 0.10]}, index=dates)

    result = classify_dividend_regime(dividends)

    assert result.loc[dates[0], "policy_regime"] == "quantitative_annual"
    assert result.loc[dates[1], "policy_regime"] == "regular_plus_special"


def test_build_regime_aware_targets_uses_december_to_february_window() -> None:
    from src.research.x18_dividend_policy_regime import build_regime_aware_dividend_targets

    snapshots = pd.DataFrame(
        {"book_value_per_share": [25.0]},
        index=pd.to_datetime(["2021-11-30"]),
    )
    dividends = pd.DataFrame(
        {"amount": [0.10, 1.50, 0.10]},
        index=pd.to_datetime(["2021-10-06", "2021-12-17", "2022-03-01"]),
    )

    result = build_regime_aware_dividend_targets(snapshots, dividends)
    row = result.loc[pd.Timestamp("2021-11-30")]

    assert row["target_window_start"] == pd.Timestamp("2021-12-01")
    assert row["target_window_end"] == pd.Timestamp("2022-02-28")
    assert row["window_dividend_total"] == pytest.approx(1.50)


def test_build_regime_aware_targets_sets_post_policy_special_excess() -> None:
    from src.research.x18_dividend_policy_regime import build_regime_aware_dividend_targets

    snapshots = pd.DataFrame(
        {"book_value_per_share": [25.0]},
        index=pd.to_datetime(["2022-11-30"]),
    )
    dividends = pd.DataFrame(
        {"amount": [0.10, 4.60, 0.10]},
        index=pd.to_datetime(["2022-10-06", "2024-01-18", "2024-03-01"]),
    )

    # Use 2023-11 snapshot / 2023-12 to 2024-02 window logic.
    snapshots = pd.DataFrame(
        {"book_value_per_share": [30.0]},
        index=pd.to_datetime(["2023-11-30"]),
    )
    result = build_regime_aware_dividend_targets(snapshots, dividends)
    row = result.loc[pd.Timestamp("2023-11-30")]

    assert row["policy_regime"] == "regular_plus_special"
    assert row["regular_baseline_dividend"] == pytest.approx(0.10)
    assert row["window_dividend_total"] == pytest.approx(4.60)
    assert row["special_dividend_excess_regime"] == pytest.approx(4.50)
