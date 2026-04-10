from pathlib import Path

import pandas as pd


def test_three_variants_present() -> None:
    df = pd.read_csv(Path("results/research/v57_transforms_results.csv"))
    assert len(df["variant"].unique()) == 3
