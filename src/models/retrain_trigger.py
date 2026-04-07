"""
Automated model retraining trigger logic (Tier 5.4, v35.1).

Evaluates whether the current drift state warrants an out-of-cycle retrain
dispatch and enforces a cooldown guard to avoid redundant re-runs.

Design intent
-------------
The monthly pipeline already re-fits WFO models from scratch on every run.
The "retrain trigger" adds orchestration on top of that:

  1. After each weekly data fetch, drift is re-evaluated from the
     model_performance_log table.
  2. If rolling-IC has been below DIAG_MIN_IC for >= RETRAIN_TRIGGER_BREACH_STREAK
     consecutive months AND the last triggered retrain was >= RETRAIN_COOLDOWN_DAYS
     ago, the trigger fires.
  3. The GitHub Actions drift_retrain_trigger workflow calls this evaluation and,
     when triggered=True, dispatches the monthly_decision workflow out-of-cycle via
     the GitHub REST API (workflow_dispatch event).
  4. Every evaluation (triggered or suppressed) is recorded in model_retrain_log
     for governance and audit.

The trigger intentionally does NOT modify model hyperparameters or feature sets —
it simply re-runs the existing monthly pipeline with fresh data.  A human
governance review is required before promoting any model configuration change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import config
from src.models.drift_monitor import ModelDriftSummary

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetainTriggerResult:
    """Outcome of a single retrain-trigger evaluation."""

    triggered: bool                  # True → out-of-cycle retrain should be dispatched
    cooldown_active: bool            # True → trigger suppressed by cooldown guard
    breach_streak: int               # Consecutive IC-breach months at evaluation time
    last_trigger_date: str | None    # ISO date of most-recent prior trigger, or None
    cooldown_days: int               # Cooldown window configured
    notes: str                       # Human-readable decision rationale
    evaluated_at: str                # ISO 8601 datetime of this evaluation


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_retrain_trigger(
    drift_summary: ModelDriftSummary | None,
    last_trigger_date: str | None,
    breach_streak_threshold: int | None = None,
    cooldown_days: int | None = None,
    now: datetime | None = None,
) -> RetainTriggerResult:
    """
    Decide whether an out-of-cycle retrain should be dispatched.

    Args:
        drift_summary:            Latest ModelDriftSummary from drift_monitor.
                                  If None, trigger is suppressed (no drift data).
        last_trigger_date:        ISO date (YYYY-MM-DD) of the most-recent prior
                                  triggered retrain, or None if never triggered.
        breach_streak_threshold:  Minimum consecutive IC-breach months to fire.
                                  Default: config.RETRAIN_TRIGGER_BREACH_STREAK.
        cooldown_days:            Minimum days between consecutive dispatches.
                                  Default: config.RETRAIN_COOLDOWN_DAYS.
        now:                      Override for the current datetime (testing).

    Returns:
        RetainTriggerResult with triggered=True when a dispatch is warranted.
    """
    if breach_streak_threshold is None:
        breach_streak_threshold = config.RETRAIN_TRIGGER_BREACH_STREAK
    if cooldown_days is None:
        cooldown_days = config.RETRAIN_COOLDOWN_DAYS
    if now is None:
        now = datetime.now(tz=timezone.utc)

    evaluated_at = now.isoformat()

    # --- Guard: no drift data available ---
    if drift_summary is None:
        return RetainTriggerResult(
            triggered=False,
            cooldown_active=False,
            breach_streak=0,
            last_trigger_date=last_trigger_date,
            cooldown_days=cooldown_days,
            notes="No drift summary available — retrain trigger skipped.",
            evaluated_at=evaluated_at,
        )

    streak = drift_summary.ic_below_threshold_streak

    # --- Guard: streak below threshold ---
    if streak < breach_streak_threshold:
        return RetainTriggerResult(
            triggered=False,
            cooldown_active=False,
            breach_streak=streak,
            last_trigger_date=last_trigger_date,
            cooldown_days=cooldown_days,
            notes=(
                f"IC breach streak {streak} < threshold {breach_streak_threshold} — "
                "no retrain needed."
            ),
            evaluated_at=evaluated_at,
        )

    # --- Guard: cooldown active ---
    if last_trigger_date is not None:
        last_dt = date.fromisoformat(last_trigger_date)
        days_since = (now.date() - last_dt).days
        if days_since < cooldown_days:
            return RetainTriggerResult(
                triggered=False,
                cooldown_active=True,
                breach_streak=streak,
                last_trigger_date=last_trigger_date,
                cooldown_days=cooldown_days,
                notes=(
                    f"IC breach streak {streak} >= threshold {breach_streak_threshold} but "
                    f"cooldown active ({days_since} days since last trigger, "
                    f"cooldown={cooldown_days} days)."
                ),
                evaluated_at=evaluated_at,
            )

    # --- Trigger fires ---
    return RetainTriggerResult(
        triggered=True,
        cooldown_active=False,
        breach_streak=streak,
        last_trigger_date=last_trigger_date,
        cooldown_days=cooldown_days,
        notes=(
            f"IC breach streak {streak} >= threshold {breach_streak_threshold} and "
            f"cooldown elapsed (last={last_trigger_date or 'never'}). "
            "Out-of-cycle retrain dispatched."
        ),
        evaluated_at=evaluated_at,
    )
