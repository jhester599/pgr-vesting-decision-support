"""Validation checks for the v38 shrinkage sweep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SWEEP = Path("results/research/v38_shrinkage_results.csv")
BEST = Path("results/research/v38_shrinkage_best_results.csv")


def test_sweep_csv_exists() -> None:
    assert SWEEP.exists()


def test_best_csv_exists() -> None:
    assert BEST.exists()


def test_sweep_has_all_alphas() -> None:
    df = pd.read_csv(SWEEP)
    assert len(df) == 10


def test_ic_invariant_across_alphas() -> None:
    df = pd.read_csv(SWEEP)
    assert df["ic"].max() - df["ic"].min() < 1e-10


def test_hit_rate_invariant_across_alphas() -> None:
    df = pd.read_csv(SWEEP)
    assert df["hit_rate"].max() - df["hit_rate"].min() < 1e-10


def test_optimal_alpha_is_half() -> None:
    df = pd.read_csv(SWEEP)
    best = df.loc[df["r2"].idxmax()]
    assert abs(best["alpha"] - 0.50) < 1e-12


def test_best_r2_improves_over_raw() -> None:
    df = pd.read_csv(SWEEP)
    raw_r2 = df.loc[df["alpha"] == 1.0, "r2"].iloc[0]
    best_r2 = df["r2"].max()
    assert best_r2 > raw_r2
