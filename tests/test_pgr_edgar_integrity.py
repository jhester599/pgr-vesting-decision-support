"""Row-level integrity of the committed PGR EDGAR tables (review 2026-09-25, WP5/WP6).

Every row of ``pgr_edgar_monthly`` must satisfy the accounting identities
the monthly releases print, reconcile to XBRL quarterly net income, and
trace each value to the filing named in its ``accession_number``.

The committed DB is opened read-only.  Set ``PGR_EDGAR_INTEGRITY_DB`` to run
the same checks against another copy (e.g. the pre-repair DB).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.edgar_8k_fetcher import DERIVED_FIELDS, _compute_derived_fields
from src.database import db_client
from src.processing import pgr_edgar_validation as v

_DB_PATH = Path(
    os.environ.get(
        "PGR_EDGAR_INTEGRITY_DB",
        Path(__file__).resolve().parents[1] / "data" / "pgr_financials.db",
    )
)

pytestmark = pytest.mark.skipif(not _DB_PATH.exists(), reason="committed DB not present")


@pytest.fixture(scope="module")
def ro_conn():
    conn = db_client.get_connection(str(_DB_PATH), read_only=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def monthly(ro_conn) -> pd.DataFrame:
    return db_client.get_pgr_edgar_monthly(ro_conn)


@pytest.fixture(scope="module")
def quarterly(ro_conn) -> pd.DataFrame:
    return db_client.get_pgr_fundamentals(ro_conn)


def _fmt(df: pd.DataFrame) -> str:
    return "\n" + df.to_string()


# ---------------------------------------------------------------------------
# Identities on every row (F12, F18, F34)
# ---------------------------------------------------------------------------


def test_revenue_minus_expenses_equals_pretax(monthly) -> None:
    bad = v.income_identity_violations(monthly)
    assert bad.empty, _fmt(bad)


def test_combined_ratio_equals_loss_plus_expense_ratio(monthly) -> None:
    bad = v.combined_ratio_violations(monthly)
    assert bad.empty, _fmt(bad)


def test_equity_matches_book_value_times_shares(monthly) -> None:
    bad = v.equity_violations(monthly)
    assert bad.empty, _fmt(bad)


def test_monthly_net_income_reconciles_to_xbrl_quarters(monthly, quarterly) -> None:
    assert v.quarters_compared(monthly, quarterly) >= 60
    bad = v.quarterly_net_income_violations(monthly, quarterly)
    assert bad.empty, _fmt(bad)


def test_no_missing_months_since_2004_08(monthly) -> None:
    assert v.missing_months(monthly) == []
    assert str(monthly.index.min().date()) == "2004-08-31"


def test_known_filed_values(monthly) -> None:
    """Values checked against the filings cited in the review (Appendix A)."""
    row = monthly.loc
    assert row["2025-09-30", "combined_ratio"] == pytest.approx(100.4)  # F34
    assert row["2017-08-31", "net_income"] == pytest.approx(-16.8)  # F12
    assert row["2018-10-31", "income_before_income_taxes"] == pytest.approx(-69.4)
    assert row["2018-12-31", "total_net_realized_gains"] == pytest.approx(-330.8)
    assert row["2015-05-31", "net_premiums_written"] == pytest.approx(1581.4)  # F17
    assert row["2015-05-31", "combined_ratio"] == pytest.approx(94.3)
    assert row["2019-04-30", "net_income"] == pytest.approx(487.8)
    assert row["2019-04-30", "book_value_per_share"] == pytest.approx(20.74)
    assert row["2004-12-31", "shareholders_equity"] == pytest.approx(5155.4, rel=0.01)  # F18


# ---------------------------------------------------------------------------
# Derived fields (F10, F11, F16)
# ---------------------------------------------------------------------------


def test_pif_total_has_no_unannotated_jumps(monthly) -> None:
    bad = v.pif_jump_violations(monthly)
    assert bad.empty, _fmt(bad)


def test_pif_growth_defined_after_leap_years(monthly) -> None:
    for month in ("2009-02-28", "2013-02-28", "2017-02-28", "2021-02-28", "2025-02-28"):
        assert not math.isnan(monthly.loc[month, "pif_growth_yoy"]), month


def test_derived_fields_match_a_full_recompute(monthly) -> None:
    """Stored derived fields equal a calendar-month recompute of the whole table."""
    frame = monthly.reset_index()
    frame["month_end"] = frame["month_end"].dt.strftime("%Y-%m-%d")
    records = [
        {k: (None if isinstance(val, float) and math.isnan(val) else val) for k, val in r.items()}
        for r in frame.to_dict("records")
    ]
    expected = pd.DataFrame(_compute_derived_fields(records)).set_index("month_end")
    for field in DERIVED_FIELDS:
        stored = monthly[field].to_numpy(dtype=float)
        fresh = expected[field].astype(float).to_numpy()
        both_nan = pd.isna(stored) & pd.isna(fresh)
        close = (abs(stored - fresh) <= 1e-9 * (1 + abs(fresh))) | both_nan
        assert close.all(), f"{field}: stored values differ from a full recompute"


def test_investment_book_yield_is_percent(monthly) -> None:
    yields = monthly["investment_book_yield"].dropna()
    assert yields.between(0.5, 10.0).all(), yields[~yields.between(0.5, 10.0)]


# ---------------------------------------------------------------------------
# Provenance (F33)
# ---------------------------------------------------------------------------


def test_accession_numbers_are_dashed(monthly) -> None:
    pattern = r"^\d{10}-\d{2}-\d{6}$"
    assert monthly["accession_number"].str.match(pattern).all()


def test_every_value_traces_to_its_row_accession(ro_conn, monthly) -> None:
    """Each parsed value equals the first-reported raw value of the row's filing."""
    raw = db_client.get_pgr_edgar_first_reported(ro_conn)
    raw["month_end"] = pd.to_datetime(raw["month_end"])
    raw = raw.set_index(["month_end", "field"])
    skip = set(DERIVED_FIELDS) | set(db_client.PGR_EDGAR_MONTHLY_DERIVED_ONLY_COLUMNS)
    problems: list[str] = []
    for month_end, row in monthly.iterrows():
        for field in db_client.PGR_EDGAR_MONTHLY_VALUE_COLUMNS:
            if field in skip:
                continue
            value = row[field]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            key = (month_end, field)
            if key not in raw.index:
                problems.append(f"{month_end.date()} {field}: no raw value")
                continue
            src = raw.loc[key]
            if src["accession_number"] != row["accession_number"]:
                problems.append(f"{month_end.date()} {field}: from {src['accession_number']}")
            stored = src["value_text"] if field in db_client.PGR_EDGAR_MONTHLY_TEXT_COLUMNS else src["value_real"]
            if stored != value and not (
                isinstance(value, float) and abs(float(stored) - value) < 1e-9
            ):
                problems.append(f"{month_end.date()} {field}: {value} != raw {stored}")
    assert not problems, "\n".join(problems[:40])


# ---------------------------------------------------------------------------
# Quarterly fundamentals (F09)
# ---------------------------------------------------------------------------


def test_quarterly_fundamentals_are_discrete_quarters(ro_conn, quarterly) -> None:
    cols = {r[1] for r in ro_conn.execute("PRAGMA table_info(pgr_fundamentals_quarterly)")}
    assert "pe_ratio" not in cols and "pb_ratio" not in cols
    roe = quarterly["roe"].dropna()
    assert roe.between(-0.5, 0.6).all(), roe[~roe.between(-0.5, 0.6)]
    # Q4 is one quarter: its EPS is not the full year's.
    q4 = quarterly[quarterly.index.month == 12]["eps"].dropna()
    q3 = quarterly[quarterly.index.month == 9]["eps"].dropna()
    assert q4.median() < 2.0 * q3.median()
    assert quarterly["filing_date"].notna().all()
