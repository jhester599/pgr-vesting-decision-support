"""Regression tests for review 2026-09-25 finding F07.

The PGR-specific FRED series (``config.FRED_SERIES_PGR``) were fetched only by
the yearly bootstrap, so they went stale and the live GBT feature
``rate_adequacy_gap_yoy`` was silently median-imputed from 2026-04 onward.

Fixes under test:
* the weekly and monthly jobs fetch ``FRED_SERIES_PGR`` as well as the macro
  series;
* ``monthly_decision`` logs a WARNING (and records the features for the run
  manifest) whenever a live-model feature is NaN in the decision row.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import config
from scripts import monthly_decision, weekly_fetch


def test_production_fred_series_includes_macro_and_pgr_series() -> None:
    from src.ingestion.fred_loader import production_fred_series

    series = production_fred_series()
    assert set(config.FRED_SERIES_PGR) <= set(series)
    assert set(config.FRED_SERIES_MACRO) <= set(series)
    assert len(series) == len(set(series))


@patch("src.ingestion.fred_loader.upsert_fred_to_db", return_value=0)
@patch("src.ingestion.fred_loader.fetch_all_fred_macro", return_value=pd.DataFrame())
def test_weekly_fetch_requests_pgr_fred_series(
    mock_fetch: MagicMock,
    mock_upsert: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
    weekly_fetch._fetch_fred_step(MagicMock(), dry_run=False)
    mock_fetch.assert_called_once()
    requested = list(mock_fetch.call_args.args[0])
    missing = [sid for sid in config.FRED_SERIES_PGR if sid not in requested]
    assert missing == [], f"weekly fetch omits PGR FRED series: {missing}"
    assert set(config.FRED_SERIES_MACRO) <= set(requested)
    mock_upsert.assert_called_once()


@patch("src.ingestion.fred_loader.upsert_fred_to_db", return_value=0)
@patch("src.ingestion.fred_loader.fetch_all_fred_macro", return_value=pd.DataFrame())
def test_monthly_decision_requests_pgr_fred_series(
    mock_fetch: MagicMock,
    mock_upsert: MagicMock,
) -> None:
    monthly_decision._fetch_fred_step(MagicMock(), dry_run=False, skip_fred=False)
    mock_fetch.assert_called_once()
    requested = list(mock_fetch.call_args.args[0])
    missing = [sid for sid in config.FRED_SERIES_PGR if sid not in requested]
    assert missing == [], f"monthly decision omits PGR FRED series: {missing}"
    assert set(config.FRED_SERIES_MACRO) <= set(requested)
    mock_upsert.assert_called_once()


def _live_feature_matrix(nan_feature: str | None) -> pd.DataFrame:
    """Small monthly feature matrix containing every live-model feature."""
    live_cols: list[str] = []
    for model_type in config.ENSEMBLE_MODELS:
        for col in config.MODEL_FEATURE_OVERRIDES.get(model_type, []):
            if col not in live_cols:
                live_cols.append(col)
    index = pd.date_range("2024-01-31", periods=30, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(len(index), len(live_cols))), index=index, columns=live_cols)
    df["target_6m_return"] = rng.normal(size=len(index))
    if nan_feature is not None:
        df.loc[df.index[-1], nan_feature] = np.nan
    return df


def test_find_nan_live_features_flags_only_nan_live_columns() -> None:
    df = _live_feature_matrix("rate_adequacy_gap_yoy")
    df["not_a_live_feature"] = np.nan
    x_current = df.drop(columns=["target_6m_return"]).iloc[[-1]]
    assert monthly_decision._find_nan_live_features(x_current) == ["rate_adequacy_gap_yoy"]

    clean = _live_feature_matrix(None).drop(columns=["target_6m_return"]).iloc[[-1]]
    assert monthly_decision._find_nan_live_features(clean) == []


@pytest.mark.parametrize("nan_feature", ["rate_adequacy_gap_yoy", None])
def test_generate_signals_warns_on_nan_live_feature(
    nan_feature: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert "rate_adequacy_gap_yoy" in config.MODEL_FEATURE_OVERRIDES["gbt"]
    df = _live_feature_matrix(nan_feature)
    monkeypatch.setattr(monthly_decision, "build_feature_matrix_from_db", lambda conn, force_refresh: df)
    monkeypatch.setattr(monthly_decision, "compute_vif", lambda *a, **k: pd.Series(dtype=float))
    # No targets: _generate_signals returns right after building diagnostics.
    monkeypatch.setattr(
        monthly_decision,
        "load_relative_return_matrix",
        lambda *a, **k: pd.Series(dtype=float),
    )

    with caplog.at_level(logging.WARNING, logger=monthly_decision.logger.name):
        signals, ensemble_results, diagnostics = monthly_decision._generate_signals(
            MagicMock(), df.index[-1].date()
        )

    assert signals.empty and ensemble_results == {}
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "[Live features]" in r.getMessage()
    ]
    if nan_feature is None:
        assert diagnostics["nan_live_features"] == []
        assert warnings == []
    else:
        assert diagnostics["nan_live_features"] == [nan_feature]
        assert len(warnings) == 1
        assert nan_feature in warnings[0].getMessage()
