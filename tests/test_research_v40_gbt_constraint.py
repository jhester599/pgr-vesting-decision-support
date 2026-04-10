"""Validation checks for the v40 GBT constraint/removal outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v40_gbt_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_three_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {
        "ridge_only",
        "constrained_gbt",
        "reweight_80_20",
    }


def test_each_variant_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 3


def test_constrained_gbt_is_best_variant_by_pooled_r2() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    best_variant = pooled["r2"].idxmax()
    assert best_variant == "constrained_gbt"


def test_constrained_gbt_beats_ridge_only() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled.loc["constrained_gbt", "r2"] > pooled.loc["ridge_only", "r2"]
