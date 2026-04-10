"""Validation checks for the v45 BayesianRidge outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV = Path("results/research/v45_bayesian_ridge_results.csv")


def test_csv_exists() -> None:
    assert CSV.exists()


def test_three_variants_present() -> None:
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {
        "A_default_bayesian_ridge",
        "B_tight_prior",
        "C_bayesian_ridge_gbt",
    }


def test_each_variant_has_pooled_row() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"]
    assert len(pooled) == 3


def test_blended_variant_is_best_on_pooled_r2() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled["r2"].idxmax() == "C_bayesian_ridge_gbt"


def test_tight_prior_not_better_than_default_here() -> None:
    df = pd.read_csv(CSV)
    pooled = df.loc[df["benchmark"] == "POOLED"].set_index("variant")
    assert pooled.loc["A_default_bayesian_ridge", "r2"] >= pooled.loc["B_tight_prior", "r2"]
