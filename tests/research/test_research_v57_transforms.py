from pathlib import Path

import pandas as pd
import pytest

# Stored-artifact tests: they read committed data (review F28, step 9).
pytestmark = pytest.mark.artifact


def test_three_variants_present() -> None:
    df = pd.read_csv(Path("research/studies/v57_transforms/outputs/v57_transforms_results.csv"))
    assert len(df["variant"].unique()) == 3
