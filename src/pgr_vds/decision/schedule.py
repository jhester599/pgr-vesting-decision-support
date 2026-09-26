"""As-of date and run-mode resolution for the monthly decision run."""

from __future__ import annotations

from datetime import date, timedelta

import config


def is_business_day(d: date) -> bool:
    """Return True if ``d`` is a weekday (Mon–Fri)."""
    return d.weekday() < 5


def last_business_day_on_or_before(d: date) -> date:
    """Move ``d`` back to the previous business day if it falls on a weekend."""
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def resolve_as_of_date(as_of_arg: str | None) -> date:
    """
    Determine the as-of date for this run. It is never later than today.

    If ``as_of_arg`` is provided, parse it; a date after today raises
    ``ValueError``. Otherwise resolve it from today's date.

    On or after the 20th of the month, the target is anchored to the 20th
    of *this* month, moved back to the previous business day if the 20th
    falls on a weekend. The monthly workflow schedules fallback runs on the
    21st and 22nd; anchoring makes those runs resolve to the *same* as-of
    date as the 20th's run, so ``artifacts.already_ran`` treats them as no-ops
    instead of regenerating the report (and re-sending the email).

    Review 2026-09-25, F26: a weekend 20th used to move *forward* to Monday,
    so a Saturday run (June and September 2026) had an as-of date later than
    the run date. Moving back keeps the 20th/21st/22nd runs in agreement
    without dating a run in the future.

    Before the 20th (e.g. a manual/testing run), there is no monthly
    target yet to anchor to, so this just resolves to today.
    """
    today = date.today()
    if as_of_arg:
        as_of = date.fromisoformat(as_of_arg)
        if as_of > today:
            raise ValueError(f"--as-of {as_of} is later than today ({today}).")
        return as_of
    if today.day < 20:
        return today
    target = date(today.year, today.month, 20)
    return last_business_day_on_or_before(target)


def validate_layer_mode(mode: str) -> str:
    """Return ``mode`` if it is a known recommendation-layer mode, else raise.

    Review 2026-09-25, F26: an unknown value (e.g. the typo ``live-only``)
    used to fall back silently to the retired ``shadow_promoted`` mode.
    """
    if mode not in config.RECOMMENDATION_LAYER_VALID_MODES:
        raise ValueError(
            f"Unknown RECOMMENDATION_LAYER_MODE {mode!r}; expected one of "
            f"{', '.join(config.RECOMMENDATION_LAYER_VALID_MODES)}."
        )
    return mode
