from pathlib import Path

import pandas as pd


def test_three_variants_present() -> None:
    df = pd.read_csv(Path("results/research/v54_gpr_results.csv"))
    assert len(df["variant"].unique()) == 3


def test_gpr_sigma_ratio_bounded() -> None:
    df = pd.read_csv(Path("results/research/v54_gpr_results.csv"))
    matern = df[
        (df["variant"] == "A_gpr_matern52") & (df["benchmark"] != "POOLED")
    ]
    if not matern.empty:
        assert float(matern["sigma_ratio"].mean()) < 3.0
