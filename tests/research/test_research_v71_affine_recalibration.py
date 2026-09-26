from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("research/studies/v71_affine_recalibration/outputs/v71_affine_recalibration_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_expected_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"A_ridge8_prior24", "B_ridge16_prior24"} <= set(df["variant"])


@pytest.mark.artifact
def test_pooled_rows_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert len(df[df["benchmark"] == "POOLED"]) == 2


@pytest.mark.artifact
def test_affine_columns_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"mean_intercept", "mean_slope"} <= set(df.columns)
