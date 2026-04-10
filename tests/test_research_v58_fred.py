from pathlib import Path


def test_csv_exists() -> None:
    assert Path("results/research/v58_fred_results.csv").exists()
