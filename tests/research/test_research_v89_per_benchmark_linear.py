from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("research/studies/v89_per_benchmark_linear/outputs/v89_per_benchmark_linear_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_linear_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "logistic_l2",
        "logistic_balanced",
        "logistic_l1_balanced",
    } <= set(df["model_name"])


@pytest.mark.artifact
def test_pooled_row_per_model() -> None:
    df = pd.read_csv(CSV_PATH)
    pooled = df[df["benchmark"] == "POOLED"]
    assert set(pooled["model_name"]) == {
        "logistic_l2",
        "logistic_balanced",
        "logistic_l1_balanced",
    }
