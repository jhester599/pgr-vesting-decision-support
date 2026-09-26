from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("results/research/v72_quality_weighted_consensus_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_both_consensus_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"equal_weight", "quality_weighted"} == set(df["variant"])


@pytest.mark.artifact
def test_policy_columns_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"mean_policy_return", "uplift_vs_sell_50", "capture_ratio"} <= set(df.columns)
