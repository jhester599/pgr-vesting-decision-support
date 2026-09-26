"""
Tests for src/models/retrain_trigger.py (Tier 5.4, v35.1).

Coverage:
  - evaluate_retrain_trigger: all decision branches
      * no drift summary → not triggered
      * streak below threshold → not triggered
      * streak at threshold, no prior trigger → triggered
      * streak at threshold, cooldown active → suppressed
      * streak at threshold, cooldown elapsed → triggered
  - RetainTriggerResult field correctness
  - db_client helpers: record_retrain_event, get_last_retrain_trigger_date
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta

import pytest

from src.models.drift_monitor import ModelDriftSummary
from src.models.retrain_trigger import RetainTriggerResult, evaluate_retrain_trigger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_drift(streak: int, drift_flag: bool | None = None) -> ModelDriftSummary:
    """Build a minimal ModelDriftSummary with a given breach streak."""
    if drift_flag is None:
        drift_flag = streak >= 3
    return ModelDriftSummary(
        as_of_month="2026-03-31",
        window_months=12,
        history_months=max(streak, 1),
        rolling_ic=0.04 if drift_flag else 0.09,
        rolling_hit_rate=0.55,
        rolling_ece=0.05,
        ic_below_threshold_streak=streak,
        drift_flag=drift_flag,
    )


NOW = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)


def _eval(drift=None, last_date=None, streak_threshold=3, cooldown=30, **kwargs):
    return evaluate_retrain_trigger(
        drift_summary=drift,
        last_trigger_date=last_date,
        breach_streak_threshold=streak_threshold,
        cooldown_days=cooldown,
        now=NOW,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# No drift summary
# ---------------------------------------------------------------------------

class TestNoDriftSummary:
    def test_not_triggered_when_no_summary(self):
        result = _eval(drift=None)
        assert result.triggered is False
        assert result.cooldown_active is False
        assert result.breach_streak == 0

    def test_notes_explain_reason(self):
        result = _eval(drift=None)
        assert "No drift summary" in result.notes

    def test_returns_correct_type(self):
        result = _eval(drift=None)
        assert isinstance(result, RetainTriggerResult)

    def test_evaluated_at_is_set(self):
        result = _eval(drift=None)
        assert result.evaluated_at.startswith("2026-04-07")


# ---------------------------------------------------------------------------
# Streak below threshold
# ---------------------------------------------------------------------------

class TestStreakBelowThreshold:
    def test_streak_zero_not_triggered(self):
        result = _eval(drift=_make_drift(0))
        assert result.triggered is False

    def test_streak_one_not_triggered(self):
        result = _eval(drift=_make_drift(1))
        assert result.triggered is False

    def test_streak_two_not_triggered_at_threshold_three(self):
        result = _eval(drift=_make_drift(2), streak_threshold=3)
        assert result.triggered is False

    def test_streak_stored_correctly(self):
        result = _eval(drift=_make_drift(2))
        assert result.breach_streak == 2

    def test_notes_mention_streak_and_threshold(self):
        result = _eval(drift=_make_drift(1), streak_threshold=3)
        assert "1" in result.notes
        assert "3" in result.notes


# ---------------------------------------------------------------------------
# Streak at or above threshold — no prior trigger (never triggered)
# ---------------------------------------------------------------------------

class TestTriggerFirstTime:
    def test_fires_when_streak_meets_threshold(self):
        result = _eval(drift=_make_drift(3), last_date=None, streak_threshold=3)
        assert result.triggered is True

    def test_fires_when_streak_exceeds_threshold(self):
        result = _eval(drift=_make_drift(5), last_date=None, streak_threshold=3)
        assert result.triggered is True

    def test_no_cooldown_active(self):
        result = _eval(drift=_make_drift(3), last_date=None)
        assert result.cooldown_active is False

    def test_last_trigger_date_is_none(self):
        result = _eval(drift=_make_drift(3), last_date=None)
        assert result.last_trigger_date is None

    def test_notes_say_dispatched(self):
        result = _eval(drift=_make_drift(3), last_date=None)
        assert "dispatched" in result.notes.lower() or "triggered" in result.notes.lower()


# ---------------------------------------------------------------------------
# Cooldown active (triggered recently)
# ---------------------------------------------------------------------------

class TestCooldownActive:
    def test_suppressed_when_within_cooldown(self):
        # Last trigger was 10 days ago; cooldown = 30 days → still in window
        last = (NOW.date() - timedelta(days=10)).isoformat()
        result = _eval(drift=_make_drift(4), last_date=last, cooldown=30)
        assert result.triggered is False
        assert result.cooldown_active is True

    def test_suppressed_at_one_day_before_cooldown_expiry(self):
        last = (NOW.date() - timedelta(days=29)).isoformat()
        result = _eval(drift=_make_drift(3), last_date=last, cooldown=30)
        assert result.triggered is False
        assert result.cooldown_active is True

    def test_notes_mention_cooldown(self):
        last = (NOW.date() - timedelta(days=5)).isoformat()
        result = _eval(drift=_make_drift(3), last_date=last, cooldown=30)
        assert "cooldown" in result.notes.lower()

    def test_cooldown_stores_days(self):
        last = (NOW.date() - timedelta(days=5)).isoformat()
        result = _eval(drift=_make_drift(3), last_date=last, cooldown=30)
        assert result.cooldown_days == 30


# ---------------------------------------------------------------------------
# Cooldown elapsed (prior trigger, but old enough)
# ---------------------------------------------------------------------------

class TestCooldownElapsed:
    def test_fires_when_cooldown_elapsed_exactly(self):
        last = (NOW.date() - timedelta(days=30)).isoformat()
        result = _eval(drift=_make_drift(3), last_date=last, cooldown=30)
        assert result.triggered is True

    def test_fires_when_cooldown_more_than_elapsed(self):
        last = (NOW.date() - timedelta(days=60)).isoformat()
        result = _eval(drift=_make_drift(3), last_date=last, cooldown=30)
        assert result.triggered is True
        assert result.cooldown_active is False

    def test_last_trigger_date_preserved(self):
        last = (NOW.date() - timedelta(days=31)).isoformat()
        result = _eval(drift=_make_drift(3), last_date=last, cooldown=30)
        assert result.last_trigger_date == last


# ---------------------------------------------------------------------------
# db_client integration: record and retrieve retrain events
# ---------------------------------------------------------------------------

def _make_in_memory_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the model_retrain_log table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE model_retrain_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at    TEXT NOT NULL,
            breach_streak   INTEGER NOT NULL,
            triggered       INTEGER NOT NULL,
            cooldown_active INTEGER NOT NULL,
            last_trigger_date TEXT,
            notes           TEXT,
            created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


class TestDbClientRetrain:
    def test_get_last_retrain_returns_none_on_empty(self):
        from src.database.db_client import get_last_retrain_trigger_date
        conn = _make_in_memory_conn()
        assert get_last_retrain_trigger_date(conn) is None

    def test_record_and_retrieve_triggered_event(self):
        from src.database.db_client import (
            get_last_retrain_trigger_date,
            record_retrain_event,
        )
        conn = _make_in_memory_conn()
        record_retrain_event(
            conn,
            triggered_at="2026-03-15T10:00:00+00:00",
            breach_streak=3,
            triggered=True,
            cooldown_active=False,
            last_trigger_date=None,
            notes="Test trigger.",
        )
        last = get_last_retrain_trigger_date(conn)
        assert last == "2026-03-15"

    def test_suppressed_event_not_returned_as_last_trigger(self):
        from src.database.db_client import (
            get_last_retrain_trigger_date,
            record_retrain_event,
        )
        conn = _make_in_memory_conn()
        # Record suppressed event (triggered=False)
        record_retrain_event(
            conn,
            triggered_at="2026-04-01T10:00:00+00:00",
            breach_streak=3,
            triggered=False,
            cooldown_active=True,
            last_trigger_date="2026-03-01",
            notes="Suppressed by cooldown.",
        )
        # get_last_retrain_trigger_date returns only triggered=True rows
        last = get_last_retrain_trigger_date(conn)
        assert last is None

    def test_most_recent_triggered_returned(self):
        from src.database.db_client import (
            get_last_retrain_trigger_date,
            record_retrain_event,
        )
        conn = _make_in_memory_conn()
        for ts in [
            "2026-01-15T10:00:00+00:00",
            "2026-02-15T10:00:00+00:00",
            "2026-03-15T10:00:00+00:00",
        ]:
            record_retrain_event(
                conn,
                triggered_at=ts,
                breach_streak=3,
                triggered=True,
                cooldown_active=False,
                last_trigger_date=None,
                notes="Test.",
            )
        assert get_last_retrain_trigger_date(conn) == "2026-03-15"
