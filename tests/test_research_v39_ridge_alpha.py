"""Validation checks for the v39 Ridge alpha-grid comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v39_ridge_alpha_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_all_grids_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["grid"].unique()) == {
        "current_logspace(-4,4)",
        "extended_logspace(0,6)",
        "aggressive_logspace(2,6)",
    }


def test_each_grid_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 3


def test_aggressive_grid_is_best_pooled_variant() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("grid")
    best_grid = pooled["r2"].idxmax()
    assert best_grid == "aggressive_logspace(2,6)"


def test_aggressive_grid_beats_current_grid() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("grid")
    assert pooled.loc["aggressive_logspace(2,6)", "r2"] > pooled.loc["current_logspace(-4,4)", "r2"]
