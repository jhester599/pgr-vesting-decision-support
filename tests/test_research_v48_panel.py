"""Validation checks for the v48 panel-pooling outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v48_panel_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_two_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {
        "A_panel_fixed_effects",
        "B_panel_shared_only",
    }


def test_panel_observation_count_large() -> None:
    df = pd.read_csv(CSV)
    assert (df["n_total_obs"] > 1000).all()


def test_shared_only_beats_fixed_effects_here() -> None:
    df = pd.read_csv(CSV).set_index("variant")
    assert df.loc["B_panel_shared_only", "r2"] > df.loc["A_panel_fixed_effects", "r2"]


def test_panel_ic_remains_positive() -> None:
    df = pd.read_csv(CSV)
    assert (df["ic"] > 0.15).all()
