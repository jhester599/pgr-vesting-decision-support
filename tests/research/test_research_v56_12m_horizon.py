from pathlib import Path

import pandas as pd
import pytest


def test_csv_exists() -> None:
    result_path = Path("research/studies/v56_12m_horizon/outputs/v56_12m_results.csv")
    assert result_path.exists()


@pytest.mark.artifact
def test_horizon_recorded() -> None:
    df = pd.read_csv(Path("research/studies/v56_12m_horizon/outputs/v56_12m_results.csv"))
    assert (df["horizon"] == 12).all()
