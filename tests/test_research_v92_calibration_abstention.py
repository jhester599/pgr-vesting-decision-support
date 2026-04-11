from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v92_calibration_and_abstention_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_calibration_modes_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"raw", "prequential_logistic"} <= set(df["calibration"])


def test_selected_variant_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert df["selected_next"].sum() >= 1
