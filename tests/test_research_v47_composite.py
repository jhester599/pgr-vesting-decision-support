"""Validation checks for the v47 composite-target outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v47_composite_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_three_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {
        "A_equal_weighted",
        "B_inv_vol_weighted",
        "C_equity_only",
    }


def test_all_rows_are_composite() -> None:
    df = pd.read_csv(CSV)
    assert set(df["benchmark"].unique()) == {"COMPOSITE"}


def test_equal_weighted_best_on_r2() -> None:
    df = pd.read_csv(CSV).set_index("variant")
    assert df["r2"].idxmax() == "A_equal_weighted"


def test_inv_vol_has_highest_hit_rate() -> None:
    df = pd.read_csv(CSV).set_index("variant")
    assert df["hit_rate"].idxmax() == "B_inv_vol_weighted"
