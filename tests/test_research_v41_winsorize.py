"""Validation checks for the v41 target-winsorization outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v41_winsorize_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_two_clip_levels_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["clip_level"].unique()) == {"p5_p95", "p10_p90"}


def test_each_clip_level_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 2


def test_p10_p90_beats_p5_p95_on_pooled_r2() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("clip_level")
    assert pooled.loc["p10_p90", "r2"] > pooled.loc["p5_p95", "r2"]
