from pathlib import Path

import pandas as pd


def test_three_variants_present() -> None:
    df = pd.read_csv(Path("results/research/v52_test_window_results.csv"))
    assert len(df["variant"].unique()) == 3


def test_monthly_window_has_more_obs_than_quarterly() -> None:
    df = pd.read_csv(Path("results/research/v52_test_window_results.csv"))
    monthly = df[
        (df["variant"] == "A_test1_rolling60") & (df["benchmark"] != "POOLED")
    ]
    quarterly = df[
        (df["variant"] == "B_test3_rolling60") & (df["benchmark"] != "POOLED")
    ]
    assert monthly["n"].sum() > quarterly["n"].sum()
