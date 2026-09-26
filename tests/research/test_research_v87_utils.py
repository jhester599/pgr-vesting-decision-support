from __future__ import annotations

import pandas as pd

from src.research.v87_utils import (
    build_basket_target_series,
    build_target_series,
    classifier_hold_fraction,
    hybrid_hold_fraction,
)


def test_build_target_series_underperform_and_actionable() -> None:
    rel = pd.Series(
        [-0.05, -0.01, 0.02, -0.04],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
        name="rel",
    )

    under = build_target_series(rel, "benchmark_underperform_0pct")
    actionable = build_target_series(rel, "actionable_sell_3pct")

    assert under.tolist() == [1, 1, 0, 1]
    assert actionable.tolist() == [1, 0, 0, 1]


def test_build_basket_target_series_majority() -> None:
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    rel_map = {
        "A": pd.Series([-0.01, 0.01, -0.02], index=idx),
        "B": pd.Series([-0.02, 0.02, -0.03], index=idx),
        "C": pd.Series([0.01, 0.03, -0.01], index=idx),
    }

    basket = build_basket_target_series(rel_map, "breadth_underperform_majority")

    assert basket.tolist() == [1, 0, 1]


def test_classifier_hold_fraction_respects_neutral_band() -> None:
    probs = pd.Series([0.80, 0.50, 0.10], index=pd.date_range("2020-01-31", periods=3, freq="ME"))

    hold = classifier_hold_fraction(probs, lower=0.35, upper=0.65)

    assert hold.tolist() == [0.0, 0.5, 1.0]


def test_hybrid_hold_fraction_uses_both_probability_and_regression_strength() -> None:
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    predicted = pd.Series([-0.05, -0.01, 0.06, 0.01], index=idx)
    prob_sell = pd.Series([0.80, 0.80, 0.20, 0.20], index=idx)

    hold = hybrid_hold_fraction(predicted, prob_sell, lower=0.35, upper=0.65)

    assert hold.tolist() == [0.0, 0.5, 1.0, 0.5]
