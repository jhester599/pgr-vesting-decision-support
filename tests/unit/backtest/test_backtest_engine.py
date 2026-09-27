"""
Tests for src/backtest/vesting_events.py and src/backtest/backtest_engine.py

Covers:
  - Vesting event date enumeration (month/day proximity, business-day snap,
    rsu_type assignment, forward-window computation)
  - BacktestEventResult correctness (direction logic, sell_pct mapping)
  - run_historical_backtest temporal integrity (no future leakage)
  - run_full_backtest output structure
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config

from src.backtest.vesting_events import (
    VestingEvent,
    enumerate_vesting_events,
    get_nearest_month_end,
    _nearest_business_day,
    _add_months,
)
from src.backtest.backtest_engine import (
    BacktestEventResult,
    _signal_from_prediction,
    _realized_direction,
    _sell_pct_from_signal,
    run_historical_backtest,
    run_monthly_stability_backtest,
    compute_event_outcome,
)
from src.database import db_client
from src.reporting.backtest_report import export_backtest_to_csv


@pytest.fixture
def event_window_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Synthetic raw bars/actions; never read the tracked database."""
    conn = sqlite3.connect(tmp_path / "event_window.db")
    db_client.initialize_schema(conn)
    bars = [
        ("2020-01-10", 100.0, 100.0),
        ("2020-01-17", 120.0, 100.0),
        ("2020-01-24", 60.0, 100.0),
        ("2020-01-31", 70.0, 100.0),
        ("2020-07-10", 80.0, 110.0),
        ("2020-07-17", 200.0, 115.0),
        ("2020-07-31", 210.0, 120.0),
        ("2021-01-15", 90.0, 110.0),
    ]
    for bar_date, pgr, vti in bars:
        conn.executemany(
            "INSERT INTO daily_prices (ticker, date, close, source) "
            "VALUES (?, ?, ?, 'synthetic')",
            [("PGR", bar_date, pgr), ("VTI", bar_date, vti)],
        )
    conn.execute(
        "INSERT INTO split_history "
        "(ticker, split_date, split_ratio, numerator, denominator) "
        "VALUES ('PGR', '2020-01-21', 2, 2, 1)"
    )
    conn.execute(
        "INSERT INTO daily_dividends (ticker, ex_date, amount, source) "
        "VALUES ('PGR', '2020-01-24', 3, 'synthetic')"
    )
    conn.execute(
        "INSERT INTO monthly_relative_returns "
        "(date, benchmark, target_horizon, pgr_return, benchmark_return, "
        "relative_return) VALUES ('2020-01-31', 'VTI', 6, 2, 0.2, 1.8)"
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _synthetic_event_backtest(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    as_of: date = date(2021, 1, 15),
    horizon: int = 6,
    monthly: bool = False,
    expect_complete: bool = True,
) -> BacktestEventResult | None:
    """Keep forecasts fixed; exercise actual raw-bar outcome calculation."""
    import src.backtest.backtest_engine as engine

    class EvaluationDate(date):
        @classmethod
        def today(cls) -> date:
            return as_of

    feature_index = pd.date_range("2005-01-31", "2019-12-31", freq="BME")
    features = pd.DataFrame({"feature_a": 1.0}, index=feature_index)
    targets = pd.Series(0.05, index=feature_index, name=f"VTI_{horizon}m")
    # The independently calculated next-month target is deliberately different.
    targets.loc[pd.Timestamp("2020-01-31")] = 1.8

    def load_targets(
        db: sqlite3.Connection,
        benchmark: str,
        months: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        return targets.loc[start_date:end_date].copy()

    def forecast(**kwargs: object) -> dict[str, float]:
        assert kwargs["X_current"].index[-1] == pd.Timestamp("2019-12-31")
        return {"predicted_return": 0.20, "ic": 0.15, "hit_rate": 0.65}

    monkeypatch.setattr(engine, "date", EvaluationDate)
    monkeypatch.setattr(
        engine, "build_feature_matrix_from_db", lambda *a, **k: features
    )
    monkeypatch.setattr(engine, "load_relative_return_matrix", load_targets)
    monkeypatch.setattr(engine, "run_wfo", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(engine, "predict_current", forecast)
    monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
    if monthly:
        event = VestingEvent(
            date(2020, 1, 31), "time", date(2020, 7, 31), date(2021, 1, 31)
        )
        monkeypatch.setattr(
            engine, "enumerate_monthly_evaluation_dates", lambda **k: [event]
        )
        results = run_monthly_stability_backtest(conn, as_of=as_of)
    else:
        event = VestingEvent(
            date(2020, 1, 15), "time", date(2020, 7, 15), date(2021, 1, 15)
        )
        results = run_historical_backtest(
            conn, events_override=[event], target_horizon_months=horizon
        )
    assert len(results) == int(expect_complete)
    return results[0] if results else None


def test_vest_event_return_uses_event_window_not_next_month_target(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_event_backtest(event_window_db, monkeypatch)
    assert result is not None
    assert result.realized_relative_return == pytest.approx(0.58)
    assert result.predicted_relative_return == 0.20
    assert result.predicted_sell_pct == 0.25
    assert result.forecast_anchor == date(2019, 12, 31)
    assert result.outcome_convention == "event"


def test_event_outcome_requires_complete_horizon(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_event_backtest(
        event_window_db, monkeypatch, as_of=date(2020, 7, 14),
        expect_complete=False,
    )
    assert result is None


def test_event_return_uses_only_observable_start_price(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This later close is unavailable on January 15 and is not a DRIP bar.
    event_window_db.execute(
        "UPDATE daily_prices SET close = 1000 "
        "WHERE ticker = 'PGR' AND date = '2020-01-17'"
    )
    result = _synthetic_event_backtest(event_window_db, monkeypatch)
    assert result is not None
    assert result.realized_relative_return == pytest.approx(0.58)
    assert result.outcome is not None
    assert result.outcome.pgr_total_return == pytest.approx(0.68)
    assert result.outcome.benchmark_total_return == pytest.approx(0.10)
    assert result.outcome.start_date == date(2020, 1, 15)
    assert result.outcome.end_date == date(2020, 7, 15)
    assert result.outcome.pgr_start_bar == date(2020, 1, 10)
    assert result.outcome.pgr_end_bar == date(2020, 7, 10)


def test_event_outcome_requires_both_feeds_through_endpoint(
    event_window_db: sqlite3.Connection,
) -> None:
    event_window_db.execute(
        "DELETE FROM daily_prices WHERE ticker = 'VTI' AND date > '2020-07-10'"
    )
    assert compute_event_outcome(
        event_window_db, "VTI", date(2020, 1, 15), date(2020, 7, 15),
        date(2021, 1, 15),
    ) is None


@pytest.mark.parametrize("invalid_close", [0.0, float("inf")])
def test_event_outcome_requires_valid_covering_price(
    event_window_db: sqlite3.Connection, invalid_close: float,
) -> None:
    event_window_db.execute(
        "UPDATE daily_prices SET close = ? "
        "WHERE ticker = 'VTI' AND date > '2020-07-10'",
        (invalid_close,),
    )
    assert compute_event_outcome(
        event_window_db, "VTI", date(2020, 1, 15), date(2020, 7, 15),
        date(2021, 1, 15),
    ) is None


def test_event_outcome_without_observable_start_is_unavailable(
    event_window_db: sqlite3.Connection,
) -> None:
    event_window_db.execute(
        "DELETE FROM daily_prices WHERE date <= '2020-01-15'"
    )
    assert compute_event_outcome(
        event_window_db, "VTI", date(2020, 1, 15), date(2020, 7, 15),
        date(2021, 1, 15),
    ) is None


def test_event_twelve_month_outcome_uses_stated_end(
    event_window_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_event_backtest(
        event_window_db, monkeypatch, horizon=12
    )
    # 2.1 * 90 / 100 - 1 = 89%; VTI 10%; relative 79%.
    assert result is not None
    assert result.realized_relative_return == pytest.approx(0.79)
    assert result.outcome.end_date == date(2021, 1, 15)


def test_monthly_stability_keeps_monthly_model_target(
    event_window_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_event_backtest(
        event_window_db, monkeypatch, monthly=True
    )
    # BME window: PGR 210/70 - 1 = 200%; VTI 120/100 - 1 = 20%.
    assert result is not None
    assert result.realized_relative_return == pytest.approx(1.8)
    assert result.outcome_convention == "monthly"
    assert result.predicted_relative_return == 0.20
    assert result.predicted_sell_pct == 0.25


def test_policy_comparisons_share_event_outcome(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _synthetic_event_backtest(event_window_db, monkeypatch)
    outcome = result.outcome
    assert outcome is not None
    # Untaxed $100 allocation smoke fixture, not a performance test.
    baseline = 100 * (
        1 + 0.5 * outcome.pgr_total_return
        + 0.5 * outcome.benchmark_total_return
    )
    candidate = 100 * (
        1 + 0.75 * outcome.pgr_total_return
        + 0.25 * outcome.benchmark_total_return
    )
    assert baseline == pytest.approx(139.0)
    assert candidate == pytest.approx(153.5)


def test_event_window_split_dividend_matches_hand_calculation(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1 share -> 2; $6 / $60 buys 0.1; 2.1 * $80 = $168.
    # VTI $100 -> $110; 68% - 10% = 58%.
    result = _synthetic_event_backtest(event_window_db, monkeypatch)
    assert result is not None
    assert result.realized_relative_return == pytest.approx(0.58)
    assert result.outcome is not None
    assert result.outcome.pgr_total_return == pytest.approx(0.68)
    assert result.outcome.benchmark_total_return == pytest.approx(0.10)


def test_event_outcome_leaves_monthly_targets_unchanged(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = event_window_db.execute(
        "SELECT * FROM monthly_relative_returns"
    ).fetchall()
    result = _synthetic_event_backtest(event_window_db, monkeypatch)
    assert result is not None
    assert result.realized_relative_return == pytest.approx(0.58)
    after = event_window_db.execute(
        "SELECT * FROM monthly_relative_returns"
    ).fetchall()
    assert before == after
    assert after[0][3:6] == (2.0, 0.2, 1.8)


def test_event_csv_labels_model_anchor_and_holding_period(
    event_window_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _synthetic_event_backtest(event_window_db, monkeypatch)
    assert result is not None
    output = tmp_path / "event.csv"
    export_backtest_to_csv([result], str(output))
    row = pd.read_csv(output).iloc[0]
    assert row["forecast_anchor"] == "2019-12-31"
    assert row["outcome_convention"] == "event"
    assert row["outcome_start"] == "2020-01-15"
    assert row["outcome_end"] == "2020-07-15"
    assert row["pgr_start_bar"] == "2020-01-10"
    assert row["pgr_end_bar"] == "2020-07-10"
    assert row["pgr_total_return"] == pytest.approx(0.68)
    assert row["benchmark_total_return"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# VestingEvent / enumeration tests
# ---------------------------------------------------------------------------

class TestNearestBusinessDay:
    def test_weekday_unchanged(self):
        # 2024-01-19 is a Friday
        assert _nearest_business_day(date(2024, 1, 19)) == date(2024, 1, 19)

    def test_saturday_becomes_friday(self):
        # 2025-07-19 is a Saturday â†’ Friday 2025-07-18
        d = date(2025, 7, 19)
        assert d.weekday() == 5  # confirm Saturday
        result = _nearest_business_day(d)
        assert result.weekday() < 5   # must be a weekday
        assert result == date(2025, 7, 18)

    def test_sunday_becomes_monday(self):
        # Find a Sunday: 2025-01-19 is a Sunday
        d = date(2025, 1, 19)
        assert d.weekday() == 6  # confirm Sunday
        result = _nearest_business_day(d)
        assert result.weekday() < 5
        assert result == date(2025, 1, 20)


class TestAddMonths:
    def test_simple_addition(self):
        assert _add_months(date(2024, 1, 19), 6) == date(2024, 7, 19)

    def test_year_rollover(self):
        assert _add_months(date(2024, 9, 30), 6) == date(2025, 3, 30)

    def test_month_end_clamping(self):
        # Jan 31 + 1 month = Feb 28 (non-leap) or Feb 29 (leap)
        result = _add_months(date(2023, 1, 31), 1)
        assert result == date(2023, 2, 28)

    def test_12_months(self):
        assert _add_months(date(2024, 1, 19), 12) == date(2025, 1, 19)


class TestEnumerateVestingEvents:
    def test_returns_sorted_by_date(self):
        events = enumerate_vesting_events(start_year=2020, end_year=2023)
        dates = [e.event_date for e in events]
        assert dates == sorted(dates)

    def test_january_events_within_5_business_days_of_jan19(self):
        events = enumerate_vesting_events(start_year=2014, end_year=2024)
        jan_events = [e for e in events if e.rsu_type == "time"]
        for e in jan_events:
            # Must be within 5 calendar days of Jan 19 in the event's year
            target = date(e.event_date.year, 1, 19)
            delta = abs((e.event_date - target).days)
            assert delta <= 5, (
                f"Jan event {e.event_date} is {delta} days from Jan 19 "
                f"{e.event_date.year}"
            )

    def test_july_events_within_5_business_days_of_jul17(self):
        events = enumerate_vesting_events(start_year=2014, end_year=2024)
        jul_events = [e for e in events if e.rsu_type == "performance"]
        for e in jul_events:
            target = date(e.event_date.year, 7, 17)
            delta = abs((e.event_date - target).days)
            assert delta <= 5, (
                f"Jul event {e.event_date} is {delta} days from Jul 17 "
                f"{e.event_date.year}"
            )

    def test_all_event_dates_are_weekdays(self):
        events = enumerate_vesting_events(start_year=2014, end_year=2024)
        for e in events:
            assert e.event_date.weekday() < 5, (
                f"{e.event_date} ({e.rsu_type}) falls on a weekend."
            )

    def test_rsu_types_correct(self):
        events = enumerate_vesting_events(start_year=2020, end_year=2020)
        types = {e.rsu_type for e in events}
        assert "time" in types
        assert "performance" in types
        jan = [e for e in events if e.rsu_type == "time"]
        jul = [e for e in events if e.rsu_type == "performance"]
        assert all(e.event_date.month == 1 for e in jan)
        assert all(e.event_date.month == 7 for e in jul)

    def test_two_events_per_year(self):
        events = enumerate_vesting_events(start_year=2018, end_year=2022)
        for year in range(2018, 2023):
            year_events = [e for e in events if e.event_date.year == year]
            assert len(year_events) == 2, (
                f"Expected 2 events for {year}, got {len(year_events)}."
            )

    def test_forward_window_6m(self):
        events = enumerate_vesting_events(start_year=2020, end_year=2020)
        for e in events:
            expected_6m = _add_months(e.event_date, 6)
            assert e.horizon_6m_end == expected_6m

    def test_forward_window_12m(self):
        events = enumerate_vesting_events(start_year=2020, end_year=2020)
        for e in events:
            expected_12m = _add_months(e.event_date, 12)
            assert e.horizon_12m_end == expected_12m

    def test_default_end_year_excludes_recent_events(self):
        from datetime import date as date_cls
        events_default = enumerate_vesting_events(start_year=2014)
        events_all = enumerate_vesting_events(
            start_year=2014, end_year=date_cls.today().year
        )
        assert len(events_default) <= len(events_all)

    def test_event_count_matches_expected(self):
        # 2014â€“2023 inclusive = 10 years Ã— 2 events = 20 events
        events = enumerate_vesting_events(start_year=2014, end_year=2023)
        assert len(events) == 20


class TestGetNearestMonthEnd:
    def test_returns_timestamp(self):
        result = get_nearest_month_end(date(2024, 1, 19))
        assert isinstance(result, pd.Timestamp)

    def test_on_or_before_input_date(self):
        d = date(2024, 1, 19)
        result = get_nearest_month_end(d)
        assert result <= pd.Timestamp(d)

    def test_last_business_day_of_month(self):
        # Jan 31 2024 is a Wednesday â€” should return Jan 31 2024
        result = get_nearest_month_end(date(2024, 1, 31))
        assert result == pd.Timestamp("2024-01-31")


# ---------------------------------------------------------------------------
# Signal direction and sell-pct logic tests
# ---------------------------------------------------------------------------

class TestSignalDirection:
    def test_positive_prediction_is_outperform(self):
        assert _signal_from_prediction(0.05) == "OUTPERFORM"

    def test_zero_prediction_is_outperform(self):
        assert _signal_from_prediction(0.0) == "OUTPERFORM"

    def test_negative_prediction_is_underperform(self):
        assert _signal_from_prediction(-0.01) == "UNDERPERFORM"

    def test_realized_positive_is_outperform(self):
        assert _realized_direction(0.03) == "OUTPERFORM"

    def test_realized_negative_is_underperform(self):
        assert _realized_direction(-0.02) == "UNDERPERFORM"


class TestSellPct:
    def test_low_ic_returns_50pct(self):
        assert _sell_pct_from_signal(0.20, ic=0.03, hit_rate=0.6) == 0.50

    def test_high_predicted_return_high_ic_returns_25pct(self):
        assert _sell_pct_from_signal(0.20, ic=0.15, hit_rate=0.65) == 0.25

    def test_modest_return_returns_50pct(self):
        assert _sell_pct_from_signal(0.08, ic=0.10, hit_rate=0.60) == 0.50

    def test_negative_return_returns_100pct(self):
        assert _sell_pct_from_signal(-0.05, ic=0.10, hit_rate=0.55) == 1.00


# ---------------------------------------------------------------------------
# BacktestEventResult correctness tests (synthetic data, no DB required)
# ---------------------------------------------------------------------------

class TestBacktestEventResultLogic:
    def _make_event(self, year: int = 2020) -> VestingEvent:
        return VestingEvent(
            event_date=date(year, 1, 19),
            rsu_type="time",
            horizon_6m_end=date(year, 7, 19),
            horizon_12m_end=date(year + 1, 1, 19),
        )

    def test_correct_direction_when_signals_agree(self):
        event = self._make_event()
        r = BacktestEventResult(
            event=event,
            benchmark="VTI",
            target_horizon=6,
            predicted_relative_return=0.05,
            realized_relative_return=0.03,
            signal_direction="OUTPERFORM",
            correct_direction=True,
            predicted_sell_pct=0.25,
            ic_at_event=0.12,
            hit_rate_at_event=0.60,
            n_train_observations=120,
            proxy_fill_fraction=0.0,
        )
        assert r.correct_direction is True

    def test_incorrect_direction_when_signals_disagree(self):
        event = self._make_event()
        r = BacktestEventResult(
            event=event,
            benchmark="BND",
            target_horizon=6,
            predicted_relative_return=0.05,
            realized_relative_return=-0.03,
            signal_direction="OUTPERFORM",
            correct_direction=False,
            predicted_sell_pct=0.25,
            ic_at_event=0.12,
            hit_rate_at_event=0.60,
            n_train_observations=120,
            proxy_fill_fraction=0.0,
        )
        assert r.correct_direction is False

    def test_proxy_fill_fraction_in_range(self):
        event = self._make_event()
        r = BacktestEventResult(
            event=event,
            benchmark="VWO",
            target_horizon=12,
            predicted_relative_return=-0.02,
            realized_relative_return=-0.04,
            signal_direction="UNDERPERFORM",
            correct_direction=True,
            predicted_sell_pct=1.0,
            ic_at_event=0.08,
            hit_rate_at_event=0.55,
            n_train_observations=100,
            proxy_fill_fraction=0.15,
        )
        assert 0.0 <= r.proxy_fill_fraction <= 1.0


class TestBacktestLoggingFallbacks:
    @staticmethod
    def _make_feature_df() -> pd.DataFrame:
        idx = pd.bdate_range("2014-01-31", periods=140, freq="BME")
        return pd.DataFrame(
            {
                "feature_a": np.linspace(0.0, 1.0, len(idx)),
                "target_6m_return": np.linspace(0.0, 0.1, len(idx)),
            },
            index=idx,
        )

    @staticmethod
    def _make_event() -> VestingEvent:
        return VestingEvent(
            event_date=date(2024, 1, 31),
            rsu_type="time",
            horizon_6m_end=date(2024, 7, 31),
            horizon_12m_end=date(2025, 1, 31),
        )

    def test_logs_prediction_failure_and_skips_result(self, caplog):
        feature_df = self._make_feature_df()
        embargo_cutoff = pd.Timestamp("2023-07-31")
        aligned_index = feature_df.index[feature_df.index <= embargo_cutoff]
        X_aligned = feature_df.loc[aligned_index, ["feature_a"]]
        y_aligned = pd.Series(
            np.linspace(-0.02, 0.03, len(aligned_index)),
            index=aligned_index,
            name="VTI_6m",
        )
        rel_series = y_aligned.copy()

        with caplog.at_level("ERROR"), patch(
            "src.backtest.backtest_engine.build_feature_matrix_from_db",
            return_value=feature_df,
        ), patch(
            "src.backtest.backtest_engine.load_relative_return_matrix",
            return_value=rel_series,
        ), patch(
            "src.backtest.backtest_engine.get_X_y_relative",
            return_value=(X_aligned, y_aligned),
        ), patch(
            "src.backtest.backtest_engine.run_wfo",
            return_value=SimpleNamespace(),
        ), patch(
            "src.backtest.backtest_engine.predict_current",
            side_effect=RuntimeError("synthetic backtest prediction failure"),
        ), patch(
            "src.backtest.backtest_engine.config.ETF_BENCHMARK_UNIVERSE",
            ["VTI"],
        ):
            results = run_historical_backtest(
                sqlite3.connect(":memory:"),
                events_override=[self._make_event()],
            )

        assert results == []
        assert "Skipping backtest prediction for benchmark VTI" in caplog.text
        assert "synthetic backtest prediction failure" in caplog.text

    def test_logs_proxy_fill_failure_and_defaults_to_zero(
        self, caplog: pytest.LogCaptureFixture,
        event_window_db: sqlite3.Connection,
    ) -> None:
        feature_df = self._make_feature_df()
        embargo_cutoff = pd.Timestamp("2019-07-31")
        aligned_index = feature_df.index[feature_df.index <= embargo_cutoff]
        X_aligned = feature_df.loc[aligned_index, ["feature_a"]]
        y_aligned = pd.Series(
            np.linspace(-0.02, 0.03, len(aligned_index)),
            index=aligned_index,
            name="VTI_6m",
        )
        train_rel = y_aligned.copy()
        event = VestingEvent(
            date(2020, 1, 31), "time", date(2020, 7, 31), date(2021, 1, 31)
        )

        with caplog.at_level("ERROR"), patch(
            "src.backtest.backtest_engine.build_feature_matrix_from_db",
            return_value=feature_df,
        ), patch(
            "src.backtest.backtest_engine.load_relative_return_matrix",
            return_value=train_rel,
        ), patch(
            "src.backtest.backtest_engine.get_X_y_relative",
            return_value=(X_aligned, y_aligned),
        ), patch(
            "src.backtest.backtest_engine.run_wfo",
            return_value=SimpleNamespace(),
        ), patch(
            "src.backtest.backtest_engine.predict_current",
            return_value={
                "predicted_return": 0.05,
                "ic": 0.10,
                "hit_rate": 0.60,
            },
        ), patch(
            "src.backtest.backtest_engine._compute_proxy_fill_fraction",
            side_effect=RuntimeError("synthetic proxy-fill failure"),
        ), patch(
            "src.backtest.backtest_engine.config.ETF_BENCHMARK_UNIVERSE",
            ["VTI"],
        ):
            results = run_historical_backtest(
                event_window_db,
                events_override=[event],
            )

        assert len(results) == 1
        assert results[0].proxy_fill_fraction == 0.0
        assert "Could not compute proxy fill fraction for benchmark VTI" in caplog.text
        assert "synthetic proxy-fill failure" in caplog.text
