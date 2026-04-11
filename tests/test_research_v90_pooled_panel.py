from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v90_pooled_panel_classifiers_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_expected_panel_models_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "separate_logistic_balanced",
        "pooled_shared_logistic_balanced",
        "pooled_fixed_effects_logistic_balanced",
    } <= set(df["model_name"])


def test_selected_model_is_unique() -> None:
    df = pd.read_csv(CSV_PATH)
    pooled = df[df["benchmark"] == "POOLED"]
    assert pooled["selected_model"].nunique() == 1
