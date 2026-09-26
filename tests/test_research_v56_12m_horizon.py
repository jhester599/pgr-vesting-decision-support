from pathlib import Path

import pandas as pd
import pytest


def test_csv_exists() -> None:
    result_path = Path("results/research/v56_12m_results.csv")
    assert result_path.exists()


@pytest.mark.artifact
def test_horizon_recorded() -> None:
    df = pd.read_csv(Path("results/research/v56_12m_results.csv"))
    assert (df["horizon"] == 12).all()
