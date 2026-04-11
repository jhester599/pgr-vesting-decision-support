from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v87_target_taxonomy_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_expected_targets_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "benchmark_underperform_0pct",
        "actionable_sell_3pct",
        "basket_underperform_0pct",
        "breadth_underperform_majority",
    } <= set(df["target"])


def test_single_recommended_target_value() -> None:
    df = pd.read_csv(CSV_PATH)
    assert len(set(df["recommended_target"])) == 1
