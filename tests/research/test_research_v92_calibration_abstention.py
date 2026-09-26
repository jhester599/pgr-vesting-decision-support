from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("research/studies/v92_calibration_and_abstention/outputs/v92_calibration_and_abstention_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_calibration_modes_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"raw", "prequential_logistic"} <= set(df["calibration"])


@pytest.mark.artifact
def test_selected_variant_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert df["selected_next"].sum() >= 1
