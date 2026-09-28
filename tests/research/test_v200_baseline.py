"""Expected output checks for the fixed baseline/control procedure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pgr_vds.research_lib.baseline import (
    calibration_summary,
    temperature_stream,
    strict_predictions,
)


def test_calibration_scores_known_coin_and_interval_outputs() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [-1.0, 1.0, -1.0, 1.0],
            "probability": [0.5] * 4,
            "lower": [-2.0] * 4,
            "upper": [2.0] * 4,
        }
    )
    summary = calibration_summary(frame)
    assert summary["brier"] == pytest.approx(0.25)
    assert summary["log_loss"] == pytest.approx(np.log(2))
    assert summary["ece"] == pytest.approx(0.0)
    assert summary["coverage_80"] == 1.0


def test_temperature_warmup_and_mature_balanced_coin() -> None:
    dates = pd.date_range("2000-01-31", periods=40, freq="BME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "available": dates + pd.offsets.BMonthEnd(6),
            "raw_probability": 0.5,
            "y_true": np.tile([0, 1], 20),
        }
    )
    result = temperature_stream(frame)
    assert result.iloc[:29]["calibration_warmup"].all()
    assert result.iloc[29]["probability"] == pytest.approx(0.5)
    changed = frame.copy()
    changed.loc[30:, "y_true"] = 0
    later = temperature_stream(changed)
    np.testing.assert_array_equal(
        result.iloc[:35]["probability"], later.iloc[:35]["probability"]
    )


def test_strict_constant_targets_have_expected_oos_components() -> None:
    dates = pd.date_range("2000-01-31", periods=84, freq="BME")
    features = pd.DataFrame({"constant": np.zeros(84)}, index=dates)
    targets = pd.DataFrame(
        {
            "date": dates,
            "benchmark": "A",
            "y_true": 2.0,
            "available": dates + pd.offsets.BMonthEnd(6),
        }
    )
    parts, ledger = strict_predictions(
        features, targets, 6, {"ridge": ["constant"], "gbt": ["constant"]}
    )
    assert len(parts) == 12
    assert parts["ridge"].tolist() == pytest.approx([2.0] * 12)
    assert parts["gbt"].tolist() == pytest.approx([2.0] * 12)
    outer = [row for row in ledger if row["kind"] == "outer"]
    assert [row["n_train"] for row in outer] == [60, 60]
    assert (parts["date"] >= dates[72]).all()


def test_empty_target_history_is_explicitly_unscorable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_fit(*args: object, **kwargs: object) -> object:
        pytest.fail("An unsupported history must not fit a model")

    monkeypatch.setattr(
        "pgr_vds.research_lib.baseline.nested_ridge", forbidden_fit
    )
    dates = pd.date_range("2000-01-31", periods=84, freq="BME")
    features = pd.DataFrame({"constant": np.zeros(84)}, index=dates)
    targets = pd.DataFrame(
        columns=["date", "benchmark", "y_true", "available"]
    )
    parts, ledger = strict_predictions(
        features, targets, 6, {"ridge": ["constant"], "gbt": ["constant"]}
    )
    assert parts.empty
    assert {"date", "benchmark", "ridge", "gbt"}.issubset(parts.columns)
    assert len(ledger) == 1
    assert ledger[0]["status"] == "unscorable"
    assert ledger[0]["n_train"] == ledger[0]["n_test"] == 0
    assert ledger[0]["n_origins"] == 0
    assert ledger[0]["gap"] == 12
    assert "empty" in ledger[0]["reason"].lower()


@pytest.mark.parametrize("horizon,months,gap", [(6, 78, 12), (12, 150, 24)])
def test_insufficient_outer_history_is_unscorable_without_fit(
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
    months: int,
    gap: int,
) -> None:
    def forbidden_fit(*args: object, **kwargs: object) -> object:
        pytest.fail("An unsupported history must not fit a model")

    monkeypatch.setattr(
        "pgr_vds.research_lib.baseline.nested_ridge", forbidden_fit
    )
    dates = pd.date_range("2000-01-31", periods=months, freq="BME")
    features = pd.DataFrame({"constant": np.zeros(months)}, index=dates)
    targets = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "benchmark": benchmark,
                    "y_true": 2.0,
                    "available": dates + pd.offsets.BMonthEnd(horizon),
                }
            )
            for benchmark in ("A", "B")
        ],
        ignore_index=True,
    )
    parts, ledger = strict_predictions(
        features,
        targets,
        horizon,
        {"ridge": ["constant"], "gbt": ["constant"]},
    )
    assert parts.empty
    assert len(ledger) == 2
    assert {entry["benchmark"] for entry in ledger} == {"A", "B"}
    assert all(entry["status"] == "unscorable" for entry in ledger)
    assert all(entry["n_origins"] == months for entry in ledger)
    assert all(entry["gap"] == gap for entry in ledger)
    assert all(entry["n_train"] == entry["n_test"] == 0 for entry in ledger)
    assert all("insufficient" in entry["reason"].lower() for entry in ledger)
