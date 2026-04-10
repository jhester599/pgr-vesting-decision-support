"""Validation checks for the v46 binary-classification outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v46_classification_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_pooled_row_present() -> None:
    df = pd.read_csv(CSV)
    assert "POOLED" in df["benchmark"].values


def test_pooled_accuracy_above_chance() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["accuracy"] > 0.60


def test_pooled_balanced_accuracy_above_chance() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["balanced_accuracy"] > 0.52


def test_pooled_brier_score_reasonable() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].iloc[0]
    assert 0.0 < pooled["brier_score"] < 0.30
