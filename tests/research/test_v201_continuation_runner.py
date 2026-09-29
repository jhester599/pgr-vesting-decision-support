"""Synthetic orchestration checks before development execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def runner() -> object:
    path = Path(__file__).resolve().parents[2] / (
        "research/studies/v201_price_macro/run.py"
    )
    spec = importlib.util.spec_from_file_location("v201_attempt2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_preserves_six_slots_and_production_grids() -> None:
    module = runner()
    catalog = module.procedure()
    assert catalog["candidate_order"] == ["P1", "P2", "P3", "M1", "M2", "M3"]
    assert len(catalog["ridge_grid"]) == 50
    assert catalog["shrinkage_grid"] == [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.75,
        1.0,
    ]
    assert catalog["candidate_slots"] == [1, 2, 3, 4, 5, 6]
    assert catalog["publication_lags"]["NFCI"] == 2


def test_matched_support_checks_answers_naive_and_target_dates() -> None:
    module = runner()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"]),
            "benchmark": ["VOO"],
            "horizon": [6],
            "fold": [2],
            "target_end": pd.to_datetime(["2020-07-31"]),
            "available": pd.to_datetime(["2020-07-31"]),
            "y_true": [0.1],
            "naive": [0.05],
            "base_prediction": [1],
        }
    )
    module.verify_matched_support(frame, frame.copy())
    for column, value in [
        ("naive", 0.1),
        ("y_true", 0.2),
        ("fold", 3),
        ("target_end", pd.Timestamp("2020-08-31")),
    ]:
        changed = frame.copy()
        changed.loc[0, column] = value
        with pytest.raises(ValueError, match="matched v200"):
            module.verify_matched_support(changed, frame)


def test_outer_label_end_check_rejects_late_ends_even_if_available_early() -> (
    None
):
    module = runner()
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-31", "2020-01-31"]),
            "horizon": [6, 6],
            "benchmark": ["VOO", "VOO"],
            "target_end": pd.to_datetime(["2020-02-28", "2020-07-31"]),
            "available": pd.to_datetime(["2019-07-31", "2020-07-31"]),
        }
    )
    ledger = pd.DataFrame(
        [
            {
                "kind": "outer",
                "status": "scorable",
                "horizon": 6,
                "benchmark": "VOO",
                "train_start": "2019-01-31",
                "train_end": "2019-01-31",
                "test_start": "2020-01-31",
                "test_end": "2020-06-30",
            }
        ]
    )
    with pytest.raises(ValueError, match="label end"):
        module.verify_label_ends(targets, ledger)
