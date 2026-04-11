from pathlib import Path

import pandas as pd


CSV_PATH = Path("results/research/v96_decision_summary_results.csv")
MD_PATH = Path("results/research/v96_decision_summary.md")


def test_outputs_exist() -> None:
    assert CSV_PATH.exists()
    assert MD_PATH.exists()


def test_key_stages_present() -> None:
    df = pd.read_csv(CSV_PATH)
    assert {"v87", "v88", "v94", "v95"} <= set(df["stage"])
