from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v73_hybrid_decision_gate_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


def test_gate_variants_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"A_gate_40_60", "B_gate_35_65"} == set(df["variant"])


def test_policy_metrics_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"policy_mean_return", "uplift_vs_sell_50", "capture_ratio"} <= set(df.columns)
