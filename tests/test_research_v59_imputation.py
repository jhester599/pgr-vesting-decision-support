from pathlib import Path

import pandas as pd


def test_three_strategies_present() -> None:
    df = pd.read_csv(Path("results/research/v59_imputation_results.csv"))
    assert len(df["variant"].unique()) == 3
