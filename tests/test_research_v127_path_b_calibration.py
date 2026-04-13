from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "research"
        / "v127_path_b_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v127_path_b_calibration",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_apply_temperature_softens_extreme_probabilities() -> None:
    softened = MODULE._apply_temperature(0.90, 2.0)
    sharpened = MODULE._apply_temperature(0.90, 0.75)
    assert 0.50 < softened < 0.90
    assert sharpened > 0.90


def test_prequential_temperature_calibration_waits_for_history() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y_prob = np.array([0.85, 0.80, 0.20, 0.25, 0.70, 0.75], dtype=float)
    calibrated, temperatures = MODULE.prequential_temperature_calibration(
        y_true,
        y_prob,
        min_history=4,
    )
    np.testing.assert_allclose(calibrated[:4], y_prob[:4])
    np.testing.assert_allclose(temperatures[:4], np.ones(4))
    assert temperatures[4] > 0.0


def test_prequential_platt_calibration_waits_for_history() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y_prob = np.array([0.40, 0.60, 0.30, 0.70, 0.35, 0.65], dtype=float)
    calibrated = MODULE.prequential_platt_calibration(
        y_true,
        y_prob,
        min_history=4,
    )
    np.testing.assert_allclose(calibrated[:4], y_prob[:4])


def test_rank_candidates_prefers_calibration_improvement() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "path_b_raw_v126",
                "balanced_accuracy_covered": 0.64,
                "brier_score": 0.22,
                "log_loss": 0.82,
                "ece": 0.22,
            },
            {
                "model": "path_b_platt_v127",
                "balanced_accuracy_covered": 0.63,
                "brier_score": 0.19,
                "log_loss": 0.58,
                "ece": 0.11,
            },
            {
                "model": "path_b_temp_v127",
                "balanced_accuracy_covered": 0.61,
                "brier_score": 0.20,
                "log_loss": 0.62,
                "ece": 0.13,
            },
        ]
    )
    ranked = MODULE._rank_candidates(df)
    platt_row = ranked.loc[ranked["model"] == "path_b_platt_v127"].iloc[0]
    raw_row = ranked.loc[ranked["model"] == "path_b_raw_v126"].iloc[0]
    assert bool(platt_row["best_calibrated_candidate"]) is True
    assert bool(platt_row["selected_next"]) is True
    assert bool(raw_row["best_calibrated_candidate"]) is False
