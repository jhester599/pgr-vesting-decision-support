"""Validation checks for the v44 blockwise PCA outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v44_pca_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_two_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {"A_2comp_per_block", "B_1comp_per_block"}


def test_each_variant_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 2


def test_no_nan_r2() -> None:
    df = pd.read_csv(CSV)
    assert df["r2"].notna().all()


def test_two_component_variant_beats_one_component_variant() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled.loc["A_2comp_per_block", "r2"] > pooled.loc["B_1comp_per_block", "r2"]
