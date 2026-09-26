from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("results/research/v91_nonlinear_classifier_sweep_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_expected_model_families_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "logistic_fixed_effects_balanced",
        "histgb_depth2",
        "histgb_depth3",
    } <= set(df["model_name"])


@pytest.mark.artifact
def test_selected_model_unique() -> None:
    df = pd.read_csv(CSV_PATH)
    pooled = df[df["benchmark"] == "POOLED"]
    assert pooled["selected_model"].nunique() == 1
