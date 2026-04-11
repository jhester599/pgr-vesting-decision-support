from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v93_basket_targets_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_candidate_names_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "benchmark_panel_primary",
        "basket_underperform_0pct",
        "basket_actionable_sell_3pct",
        "breadth_underperform_majority",
    } <= set(df["candidate_name"])


def test_selected_target_candidate_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert df["selected_next"].sum() == 1
