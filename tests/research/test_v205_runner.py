"""Synthetic end-to-end check of the v205 fold, basis and scoring path.

The fixture has a 4-for-1 split inside every training window and an exact
linear relation that holds only on one common share basis. A runner that
mixed share bases, dropped the restatement back to each test origin's
share, or crashed on v200's missing pre-output warmup flag would fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pgr_vds.research_lib.temporal import label_end

RUNNER = (
    Path(__file__).resolve().parents[2]
    / "research/studies/v205_dividend_bvps/run.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("v205_run", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen() -> dict:
    dates = pd.date_range("2000-01-31", periods=170, freq="BME")
    split = pd.Timestamp("2004-06-15")
    rng = np.random.default_rng(11)
    step = np.arange(len(dates))
    # Economic values per post-split share.
    past = 0.1 + 0.001 * step
    ratio = rng.uniform(90.0, 100.0, len(dates))
    truth = past + 0.02 * (ratio - 95.0)
    # Stored per origin-date share: four times larger before the split.
    scale = np.where(dates < split, 4.0, 1.0)
    features = pd.DataFrame(
        {
            "date": dates,
            "past12_cash": past * scale,
            "prior_past12_cash": (past - 0.01) * scale,
            "cr_ttm": ratio,
        }
    )
    ends = [label_end(date, 12) for date in dates]
    targets = pd.DataFrame(
        {
            "date": dates,
            "endpoint": "cash_dividend_12m",
            "y_true": truth * scale,
            "target_end": ends,
            "available": ends,
            "v200_y_hat": past * scale,
            "v200_naive": np.r_[np.nan, np.cumsum(truth)[:-1] / step[1:]],
            "v200_warmup": [np.nan] + [False] * (len(dates) - 1),
            "in_v200_output": True,
        }
    )
    targets["v200_warmup"] = targets["v200_warmup"].astype(object)
    return {
        "targets": targets,
        "features": features,
        "training": pd.DataFrame(),
        "sources": {"splits": pd.Series([4.0], index=[split])},
    }


def test_runner_restates_share_basis_and_scores_matched_rows() -> None:
    run = _runner()
    frozen = _frozen()
    candidate = next(
        item for item in run.candidate_register() if item["id"] == "D1"
    )
    predictions, ledger = run.run_candidate(candidate, frozen)
    outer = [entry for entry in ledger if entry["kind"] == "outer"]
    assert len(outer) == 4
    assert all(entry["status"] == "scorable" for entry in outer)
    assert all(entry["gap"] == 24 for entry in outer)
    assert len(predictions) == 24
    # Test origins are after the split, so y_true is per post-split share.
    error = (predictions["y_hat"] - predictions["y_true"]).abs()
    assert error.max() < 0.01
    summary, stream = run.evaluate(candidate, predictions, frozen)
    assert summary["status"] == "scored"
    assert summary["n_scored"] == 24
    assert summary["relative_mae_reduction"] > 0.5
    assert len(stream) == 24


def test_d3_closure_is_enforced_from_the_rule_outcome() -> None:
    run = _runner()
    assert run.d3_closure({"positive_annual_events": 3})["closed"] is True
    with pytest.raises(ValueError):
        run.d3_closure({"positive_annual_events": 5})
