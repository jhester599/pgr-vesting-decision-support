from pathlib import Path

import pandas as pd


def test_three_variants_present() -> None:
    df = pd.read_csv(Path("results/research/v50_pred_winsorize_results.csv"))
    assert len(df["variant"].unique()) == 3


def test_pooled_row_present_for_each_variant() -> None:
    df = pd.read_csv(Path("results/research/v50_pred_winsorize_results.csv"))
    pooled_count = int((df["benchmark"] == "POOLED").sum())
    assert pooled_count == 3
