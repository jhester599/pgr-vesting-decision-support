"""Validation checks for the v42 expanding-window variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v42_expanding_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_three_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {
        "A_pure_expanding",
        "B_expanding_decay",
        "C_expanding_cap120",
    }


def test_each_variant_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 3


def test_decay_variant_is_best_pooled_r2() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled["r2"].idxmax() == "B_expanding_decay"


def test_decay_variant_has_higher_ic_than_pure_expanding() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled.loc["B_expanding_decay", "ic"] > pooled.loc["A_pure_expanding", "ic"]
