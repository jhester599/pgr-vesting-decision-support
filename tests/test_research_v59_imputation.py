from pathlib import Path

import pandas as pd
import pytest

# Stored-artifact tests: they read committed data (review F28, step 9).
pytestmark = pytest.mark.artifact


def test_three_strategies_present() -> None:
    df = pd.read_csv(Path("results/research/v59_imputation_results.csv"))
    assert len(df["variant"].unique()) == 3
