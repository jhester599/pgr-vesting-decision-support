"""Plotted data of ``scripts/capital_return_charts.py`` (step 4b).

Checks the frames the annual charts are drawn from, not pixels, on the
fixture DB in ``tests/capital_return_fixture.py``.  The annual charts have no
split marker; the split checks here are that the split year's buybacks and
dividends are on one share basis per date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import capital_return_charts as charts
from scripts import repurchase_timeseries_charts as timeseries_charts
from src.database import db_client
from tests.capital_return_fixture import build_fixture_db, edgar_rows, last_weekly_close

_WORKFLOW = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "monthly_decision.yml"
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return build_fixture_db(tmp_path / "fixture.db")


@pytest.fixture()
def frames(db_path: Path) -> dict:
    conn = db_client.get_connection(str(db_path), read_only=True)
    try:
        return charts.build_chart_frames(conn)
    finally:
        conn.close()


def _row(month: str) -> dict:
    return next(r for r in edgar_rows() if r["month_end"].startswith(month))


def test_year_end_market_cap_is_price_times_shares(frames) -> None:
    annual = frames["annual"]
    for year, month in ((2006, "2006-12"), (2007, "2007-08")):
        _, close = last_weekly_close(month)
        expected = close * _row(month)["common_shares_outstanding"]
        assert annual.loc[year, "market_cap"] == pytest.approx(expected, rel=1e-9), year
        assert annual.loc[year, "year_end_month"] == pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    # 2005-12 has no usable share count: use the nearest earlier month's
    # shares (same basis) with the December close, never equity / BVPS = 0.97M.
    _, close = last_weekly_close("2005-12")
    expected = close * _row("2005-11")["common_shares_outstanding"]
    assert annual.loc[2005, "market_cap"] == pytest.approx(expected, rel=1e-9)


def test_monthly_market_cap_is_price_times_shares(frames) -> None:
    monthly = frames["monthly"].dropna(subset=["market_cap"])
    assert len(monthly) == len(edgar_rows()) - 1
    pd.testing.assert_series_equal(
        monthly["market_cap"],
        monthly["price"] * monthly["shares_outstanding"],
        check_names=False,
    )


def test_no_nan_in_split_month(frames) -> None:
    may = frames["monthly"].loc[pd.Timestamp("2006-05-31")]
    for column in ("repurchase_dollars", "market_cap", "shares_outstanding"):
        assert not pd.isna(may[column]), column


def test_annual_buybacks_are_the_sum_of_monthly_totals(frames) -> None:
    monthly = frames["monthly"]
    annual = frames["annual"]
    edgar = monthly[monthly["edgar_month"]]
    for year in (2005, 2006, 2007):
        in_year = edgar[edgar.index.year == year]["repurchase_dollars"]
        assert annual.loc[year, "repurchase_dollars"] == pytest.approx(in_year.sum()), year
    # 2006 includes the split month (2.3M shares at the estimated cost).
    reported_2006 = sum(
        r["shares_repurchased"] * r["avg_cost_per_share"]
        for r in edgar_rows()
        if r["month_end"].startswith("2006") and r["avg_cost_per_share"] is not None
    )
    may = monthly.loc[pd.Timestamp("2006-05-31"), "repurchase_dollars"]
    assert may > 0
    assert annual.loc[2006, "repurchase_dollars"] == pytest.approx(reported_2006 + may)


def test_dividends_use_the_share_basis_of_the_ex_date(frames) -> None:
    dividends = frames["dividends"].set_index("ex_date")
    # Pre-split ex-date: pre-split shares; post-split ex-date: post-split shares.
    march = dividends.loc[pd.Timestamp("2006-03-08")]
    june = dividends.loc[pd.Timestamp("2006-06-07")]
    assert march["dividend_dollars"] == pytest.approx(
        0.03 * _row("2006-03")["common_shares_outstanding"])
    assert june["dividend_dollars"] == pytest.approx(
        0.0075 * _row("2006-06")["common_shares_outstanding"])
    # Q1 ex-dates are attributed to the prior year.
    assert march["year"] == 2005
    # No share count before the first EDGAR month: excluded, not guessed.
    assert pd.Timestamp("2004-12-08") not in dividends.index


def test_partial_year_labels_come_from_the_data(frames) -> None:
    annual = frames["annual"]
    assert annual.loc[2007, "partial_label"] == "Jan–Aug 2007"
    assert pd.isna(annual.loc[2005, "partial_label"])
    assert pd.isna(annual.loc[2006, "partial_label"])


def test_main_writes_every_named_chart(db_path: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "charts"
    assert charts.main(["--db-path", str(db_path), "--out-dir", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(charts.CHART_FILES)


def _workflow_chart_list(text: str) -> list[str]:
    """Names in the job-level ``RESEARCH_CHARTS: >-`` block (PyYAML is not a
    dependency, so the workflow is read as text)."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "RESEARCH_CHARTS: >-" in line)
    indent = len(lines[start]) - len(lines[start].lstrip())
    names = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        names.extend(line.split())
    return names


def _workflow_step(text: str, name: str) -> str:
    """Text of the step named ``name``, up to the next step."""
    start = text.index(f"- name: {name}\n")
    end = text.find("\n      - name: ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def test_workflow_regenerates_and_commits_the_named_charts() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    named = _workflow_chart_list(text)
    assert sorted(named) == sorted(set(charts.CHART_FILES) | set(timeseries_charts.CHART_FILES))

    chart_step = _workflow_step(text, "Regenerate research charts")
    assert "continue-on-error" not in chart_step
    assert "set -euo pipefail" in chart_step
    assert "for chart in $RESEARCH_CHARTS" in chart_step
    assert "exit 1" in chart_step

    commit = _workflow_step(text, "Commit results")
    assert "pgr_*.png" not in commit
    assert "for chart in $RESEARCH_CHARTS" in commit
    # A chart failure must not stop the decision/DB commit or the email.
    commit_if = commit[commit.index("if:"):commit.index("env:")]
    assert "!cancelled()" in commit_if and "steps.charts" not in commit_if
    assert "!cancelled()" in _workflow_step(text, "Send monthly decision email")


def test_cr_scatter_axes_show_every_year() -> None:
    """Year labels are drawn with ``ax.text``, which does not widen the axes:
    a year far from the regression line (2025: $8.3B) used to fall outside
    the plot."""
    import matplotlib.pyplot as plt

    annual = pd.DataFrame(
        {
            "combined_ratio": [95.0, 94.0, 90.0, 87.4],
            "total_dollars": [500.0, 600.0, 1500.0, 8300.0],
            "total_pct_market_cap": [2.0, 9.0, 3.0, 6.3],
            "market_cap": [25000.0, 6700.0, 50000.0, 133000.0],
            "cr_months": [12, 12, 12, 12],
            "partial_label": [None, None, None, None],
        },
        index=pd.Index([2010, 2015, 2020, 2025], name="year"),
    )
    fig = charts.plot_cr_scatter(annual)
    try:
        panels = charts.scatter_panels(annual)
        for ax, (_, xs, ys) in zip(fig.axes[:2], panels):
            x_low, x_high = sorted(ax.get_xlim())
            y_low, y_high = sorted(ax.get_ylim())
            for x, y in zip(xs, ys):
                assert x_low < x < x_high and y_low < y < y_high, (x, y)
    finally:
        plt.close(fig)
