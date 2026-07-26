"""Regression tests for the monthly-decision as-of date resolution.

Covers the bug where the 21st/22nd fallback cron runs (scheduled in case the
20th falls on a weekend) resolved a *different* as-of date than the 20th's
run whenever the 20th was already a business day. That mismatch made
``_already_ran`` think each fallback run was a brand-new report, so the
report — and the monthly email — was regenerated and re-sent up to three
times in a single month.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts import monthly_decision


class _FrozenDate(date):
    """A ``datetime.date`` subclass whose ``today()`` returns a fixed value."""

    _frozen: date

    @classmethod
    def today(cls) -> date:
        return cls._frozen


def _freeze(monkeypatch: pytest.MonkeyPatch, frozen: date) -> None:
    frozen_cls = type("_FrozenDate", (_FrozenDate,), {"_frozen": frozen})
    monkeypatch.setattr(monthly_decision, "date", frozen_cls)


def test_resolve_as_of_date_matches_across_20_21_22_when_20th_is_weekday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the 20th is a business day, all three fallback runs must agree."""
    # 2026-07-20 is a Monday.
    as_of_dates = set()
    for day in (20, 21, 22):
        _freeze(monkeypatch, date(2026, 7, day))
        as_of_dates.add(monthly_decision._resolve_as_of_date(None))
    assert as_of_dates == {date(2026, 7, 20)}


def test_resolve_as_of_date_advances_when_20th_is_saturday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the 20th falls on a weekend, all fallback runs should agree on Monday."""
    # Find a month where the 20th is a Saturday.
    saturday_20th = date(2026, 6, 20)
    assert saturday_20th.weekday() == 5

    expected = date(2026, 6, 22)  # Monday
    as_of_dates = set()
    for day in (20, 21, 22):
        _freeze(monkeypatch, date(2026, 6, day))
        as_of_dates.add(monthly_decision._resolve_as_of_date(None))
    assert as_of_dates == {expected}


def test_resolve_as_of_date_before_20th_uses_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual/testing runs earlier in the month aren't anchored to the 20th."""
    _freeze(monkeypatch, date(2026, 7, 5))
    assert monthly_decision._resolve_as_of_date(None) == date(2026, 7, 5)


def test_resolve_as_of_date_explicit_override_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze(monkeypatch, date(2026, 7, 21))
    assert monthly_decision._resolve_as_of_date("2026-07-15") == date(2026, 7, 15)


def test_already_ran_skips_fallback_runs_once_report_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """End-to-end check that the fallback runs no-op once a report exists."""
    out_dir = tmp_path / "2026-07"
    out_dir.mkdir()
    monkeypatch.setattr(monthly_decision, "_output_dir", lambda as_of: out_dir)

    # First (20th) run: no manifest yet.
    _freeze(monkeypatch, date(2026, 7, 20))
    as_of_20 = monthly_decision._resolve_as_of_date(None)
    assert monthly_decision._already_ran(as_of_20) is False

    # Simulate the report having been written for the 20th.
    (out_dir / "run_manifest.json").write_text(
        '{"as_of_date": "%s"}' % as_of_20.isoformat(), encoding="utf-8"
    )

    # 21st and 22nd fallback runs must now see the report as already done.
    for day in (21, 22):
        _freeze(monkeypatch, date(2026, 7, day))
        as_of = monthly_decision._resolve_as_of_date(None)
        assert monthly_decision._already_ran(as_of) is True
