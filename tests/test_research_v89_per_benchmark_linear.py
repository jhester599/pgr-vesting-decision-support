from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v89_per_benchmark_linear_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_linear_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "logistic_l2",
        "logistic_balanced",
        "logistic_l1_balanced",
    } <= set(df["model_name"])


def test_pooled_row_per_model() -> None:
    df = pd.read_csv(CSV_PATH)
    pooled = df[df["benchmark"] == "POOLED"]
    assert set(pooled["model_name"]) == {
        "logistic_l2",
        "logistic_balanced",
        "logistic_l1_balanced",
    }
