from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v95_policy_replay_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_regression_reference_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert "regression_only_quality_weighted" in set(df["variant"])


def test_agreement_columns_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {
        "agreement_with_regression_rate",
        "mean_abs_hold_diff_vs_regression",
    } <= set(df.columns)
