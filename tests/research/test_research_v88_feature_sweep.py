from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("research/studies/v88_feature_sweep/outputs/v88_feature_sweep_results.csv")


def test_csv_exists() -> None:
    assert CSV_PATH.exists()


@pytest.mark.artifact
def test_core_feature_sets_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"lean_baseline", "lean_plus_benchmark_context", "lean_plus_all_curated"} <= set(df["feature_set"])


@pytest.mark.artifact
def test_single_selected_feature_set() -> None:
    df = pd.read_csv(CSV_PATH)
    pooled = df[df["benchmark"] == "POOLED"]
    assert pooled["selected_feature_set"].nunique() == 1
