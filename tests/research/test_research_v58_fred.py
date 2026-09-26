from pathlib import Path


def test_csv_exists() -> None:
    assert Path("research/studies/v58_fred_features/outputs/v58_fred_results.csv").exists()
