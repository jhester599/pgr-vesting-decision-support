from __future__ import annotations

import math

import pandas as pd
import pytest

import config
from src.models.drift_monitor import add_rolling_model_health, summarize_latest_model_drift


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_end": [
                "2025-10-31",
                "2025-11-30",
                "2025-12-31",
                "2026-01-31",
                "2026-02-28",
                "2026-03-31",
            ],
            "aggregate_nw_ic": [0.11, 0.06, 0.05, 0.04, 0.03, 0.02],
            "aggregate_hit_rate": [0.60, 0.58, 0.56, 0.55, 0.54, 0.53],
            "ece": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )


def test_add_rolling_model_health_computes_rolling_metrics() -> None:
    history = _history_frame()

    enriched = add_rolling_model_health(history, window_months=3, ic_threshold=0.07)

    assert list(enriched.columns)[-4:] == [
        "rolling_12_ic",
        "rolling_12_hit_rate",
        "rolling_12_ece",
        "rolling_ic_below_threshold",
    ]
    assert enriched["rolling_12_ic"].iloc[-1] == pytest.approx((0.04 + 0.03 + 0.02) / 3)
    assert enriched["rolling_12_hit_rate"].iloc[-1] == pytest.approx((0.55 + 0.54 + 0.53) / 3)
    assert enriched["rolling_12_ece"].iloc[-1] == pytest.approx((0.06 + 0.07 + 0.08) / 3)
    assert bool(enriched["rolling_ic_below_threshold"].iloc[-1]) is True


def test_add_rolling_model_health_sorts_by_month() -> None:
    history = _history_frame().iloc[::-1].reset_index(drop=True)

    enriched = add_rolling_model_health(history, window_months=2)

    assert [ts.strftime("%Y-%m-%d") for ts in enriched["month_end"]] == [
        "2025-10-31",
        "2025-11-30",
        "2025-12-31",
        "2026-01-31",
        "2026-02-28",
        "2026-03-31",
    ]


def test_add_rolling_model_health_validates_required_columns() -> None:
    history = _history_frame().drop(columns=["ece"])

    with pytest.raises(ValueError, match="missing required columns: ece"):
        add_rolling_model_health(history)


def test_summarize_latest_model_drift_reports_breach_streak() -> None:
    history = _history_frame()

    summary = summarize_latest_model_drift(
        history,
        window_months=3,
        ic_threshold=0.07,
        min_consecutive_breaches=3,
    )

    assert summary is not None
    assert summary.as_of_month == "2026-03-31"
    assert summary.rolling_ic == pytest.approx((0.04 + 0.03 + 0.02) / 3)
    assert summary.rolling_hit_rate == pytest.approx((0.55 + 0.54 + 0.53) / 3)
    assert summary.rolling_ece == pytest.approx((0.06 + 0.07 + 0.08) / 3)
    assert summary.ic_below_threshold_streak == 3
    assert summary.drift_flag is True


def test_summarize_latest_model_drift_returns_none_for_empty_history() -> None:
    empty = pd.DataFrame(columns=["month_end", "aggregate_nw_ic", "aggregate_hit_rate", "ece"])

    assert summarize_latest_model_drift(empty) is None


def test_summarize_latest_model_drift_default_threshold_uses_config() -> None:
    history = pd.DataFrame(
        {
            "month_end": ["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31"],
            "aggregate_nw_ic": [
                config.DIAG_MIN_IC + 0.02,
                config.DIAG_MIN_IC - 0.04,
                config.DIAG_MIN_IC - 0.03,
                config.DIAG_MIN_IC - 0.02,
            ],
            "aggregate_hit_rate": [0.57, 0.56, 0.55, 0.54],
            "ece": [0.03, 0.04, 0.05, 0.06],
        }
    )

    summary = summarize_latest_model_drift(history, window_months=2, min_consecutive_breaches=2)

    assert summary is not None
    assert math.isfinite(summary.rolling_ic)
    assert summary.ic_below_threshold_streak == 3
    assert summary.drift_flag is True
