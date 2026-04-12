"""v115 - Horizon-aware classifier monitoring summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.classification_monitoring import summarize_matured_classifier_history
from src.reporting.classification_artifacts import classification_history_path
from src.research.v102_utils import write_results, write_summary


def main() -> None:
    history_path = classification_history_path()
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    summary = summarize_matured_classifier_history(history_df).to_payload()
    summary_df = pd.DataFrame([summary])
    write_results("v115_classifier_monitoring_summary_results.csv", summary_df)
    write_summary(
        "v115_classifier_monitoring_summary.md",
        "v115 Classifier Monitoring Summary",
        [
            f"- matured observations: {summary['matured_n']}",
            f"- Brier score: {summary['brier_score'] if summary['brier_score'] is not None else 'n/a'}",
            f"- log loss: {summary['log_loss'] if summary['log_loss'] is not None else 'n/a'}",
            f"- ECE (10-bin): {summary['ece_10'] if summary['ece_10'] is not None else 'n/a'}",
        ],
    )


if __name__ == "__main__":
    main()
