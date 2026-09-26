"""Validation checks for the v49 regime-feature outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


CSV = Path("results/research/v49_regime_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


@pytest.mark.artifact
def test_three_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {"A", "B", "C"}


@pytest.mark.artifact
def test_each_variant_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 3


@pytest.mark.artifact
def test_no_nan_metrics() -> None:
    df = pd.read_csv(CSV)
    assert df["r2"].notna().all()
    assert df["ic"].notna().all()


@pytest.mark.artifact
def test_variant_c_is_best_pooled_r2() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled["r2"].idxmax() == "C"
