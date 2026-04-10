"""Validation checks for the v43 feature-reduction outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v43_feature_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_three_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {
        "A_ridge7",
        "B_gbt7",
        "C_shared7_ridge",
    }


def test_each_variant_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 3


def test_shared7_ridge_is_best_pooled_r2() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled["r2"].idxmax() == "C_shared7_ridge"


def test_ridge7_has_positive_information_gain_vs_gbt7() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled.loc["A_ridge7", "ic"] > pooled.loc["B_gbt7", "ic"]
