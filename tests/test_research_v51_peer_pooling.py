from pathlib import Path

import pandas as pd


def test_csv_exists() -> None:
    assert Path("results/research/v51_peer_pooling_results.csv").exists()


def test_at_least_one_variant_present() -> None:
    df = pd.read_csv(Path("results/research/v51_peer_pooling_results.csv"))
    assert len(df["variant"].unique()) >= 1


def test_pooled_rows_exist() -> None:
    df = pd.read_csv(Path("results/research/v51_peer_pooling_results.csv"))
    assert "POOLED" in df["benchmark"].values
