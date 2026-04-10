from pathlib import Path

import pandas as pd


def test_csv_exists() -> None:
    assert Path("results/research/v60_diagnostics_results.csv").exists()


def test_pooled_row_present() -> None:
    df = pd.read_csv(Path("results/research/v60_diagnostics_results.csv"))
    assert "POOLED" in df["benchmark"].values


def test_cw_stat_present() -> None:
    df = pd.read_csv(Path("results/research/v60_diagnostics_results.csv"))
    assert "cw_stat" in df.columns
    assert df["cw_stat"].notna().all()


def test_mse_columns_non_negative() -> None:
    df = pd.read_csv(Path("results/research/v60_diagnostics_results.csv"))
    pooled = df[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["mse_mse"] >= 0.0
    assert pooled["mse_var_pred"] >= 0.0
    assert pooled["mse_bias_sq"] >= 0.0
