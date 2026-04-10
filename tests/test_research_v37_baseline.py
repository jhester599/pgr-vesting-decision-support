"""Validation checks for the v37 baseline research output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v37_baseline_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists(), "Run results/research/v37_baseline.py first."


def test_row_count() -> None:
    df = pd.read_csv(CSV)
    assert len(df) == 9


def test_pooled_row_present() -> None:
    df = pd.read_csv(CSV)
    assert "POOLED" in df["benchmark"].values


def test_pooled_ic_positive() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["ic"] > 0.10


def test_pooled_hit_rate_above_gate() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["hit_rate"] > 0.65


def test_pooled_sigma_ratio_below_one() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].iloc[0]
    assert 0.0 < pooled["sigma_ratio"] < 1.0
