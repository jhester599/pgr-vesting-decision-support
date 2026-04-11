from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v94_hybrid_gate_results.csv")
DETAIL_PATH = Path("results/research/v94_hybrid_gate_detail.csv")


def test_csvs_exist() -> None:
    assert CSV_PATH.exists()
    assert DETAIL_PATH.exists()


def test_expected_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "regression_only_quality_weighted",
        "classifier_only_benchmark_panel",
        "hybrid_benchmark_panel_35_65",
    } <= set(df["variant"])


def test_selected_variant_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert df["selected_next"].sum() == 1
