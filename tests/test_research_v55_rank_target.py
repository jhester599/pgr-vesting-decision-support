from pathlib import Path

import pandas as pd
import pytest

# Stored-artifact tests: they read committed data (review F28, step 9).
pytestmark = pytest.mark.artifact


def test_two_variants_present() -> None:
    df = pd.read_csv(Path("results/research/v55_rank_target_results.csv"))
    assert len(df["variant"].unique()) == 2


def test_pooled_rows_exist() -> None:
    df = pd.read_csv(Path("results/research/v55_rank_target_results.csv"))
    pooled_count = int((df["benchmark"] == "POOLED").sum())
    assert pooled_count == 2
