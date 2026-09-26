from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("research/studies/v70_benchmark_shrinkage/outputs/v70_benchmark_shrinkage_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_expected_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"A_prior12", "B_prior24"} <= set(df["variant"])


@pytest.mark.artifact
def test_pooled_rows_present() -> None:
    df = pd.read_csv(CSV_PATH)
    pooled = df[df["benchmark"] == "POOLED"]
    assert len(pooled) == 2


@pytest.mark.artifact
def test_mean_alpha_column_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert "mean_alpha" in df.columns
