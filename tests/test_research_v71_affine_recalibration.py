from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v71_affine_recalibration_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_expected_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"A_ridge8_prior24", "B_ridge16_prior24"} <= set(df["variant"])


def test_pooled_rows_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert len(df[df["benchmark"] == "POOLED"]) == 2


def test_affine_columns_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"mean_intercept", "mean_slope"} <= set(df.columns)
