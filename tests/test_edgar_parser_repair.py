"""Parser, derivation and provenance tests for review 2026-09-25 step 3b.

Findings covered (see docs/reviews/REPO_REVIEW_2026-09-25.md):

* F12 — parenthesised negatives, split-cell parentheses, Unicode minus.
* F17 — flat pagination files; 9.01-only 8-Ks whose EX-99 is the release.
* F18 — anchored equity/debt labels; BVPS x shares when no equity line.
* F11 — leap-year prior-year key; one PIF definition (property excluded).
* F16 — calendar YoY: NaN when the base month is missing.
* F33 — upsert never mixes filings; append-only raw table; idempotent CSV load.
* one Gainshare formula shared by every producer.

All tests are offline; HTML fixtures are hand-built from the cited filings.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import edgar_8k_fetcher as fetcher
from scripts.edgar_8k_fetcher import (
    _compute_derived_fields,
    _extract_balance_sheet,
    _parse_html_exhibit,
    _parse_number,
    _prior_year_key,
    _row_numbers,
    load_from_csv,
    recompute_derived_fields,
)
from src.database import db_client
from src.processing import pgr_edgar_derived


def _fresh_db(tmp_path: Path) -> sqlite3.Connection:
    conn = db_client.get_connection(str(tmp_path / "t.db"))
    db_client.initialize_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# F12: number parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(56.7)", -56.7),
        ("$(56.7)", -56.7),
        ("$ (1,234.5)", -1234.5),
        ("(56.7", -56.7),  # split cell: ")" is in the next cell
        ("−56.7", -56.7),  # U+2212 MINUS SIGN
        ("– 0.03", -0.03),  # en dash used as a minus
        ("(4.2)%", -4.2),
        ("89.2%", 89.2),
        ("$1,234.5", 1234.5),
        ("2.9% 1", 2.9),  # trailing footnote marker (2022-09 book yield)
        ("(2 )%", -2.0),  # percent change printed "(2 )%"
        ("—", None),  # em dash alone means "nil", not a number
        ("", None),
    ],
)
def test_parse_number_signs(text: str, expected: float | None) -> None:
    assert _parse_number(text) == (pytest.approx(expected) if expected is not None else None)


def test_row_numbers_carry_sign_across_split_cells() -> None:
    assert _row_numbers(["(", "16.8", ")"]) == [pytest.approx(-16.8)]
    assert _row_numbers(["$", "(16.8", ")", "$", "41.5"]) == [
        pytest.approx(-16.8), pytest.approx(41.5),
    ]
    assert _row_numbers(["$(", "0.03", ")", "0.07"]) == [
        pytest.approx(-0.03), pytest.approx(0.07),
    ]


def _aug_2017_income_statement() -> str:
    """Current-month income statement of the Aug-2017 release (0000080661-17-000060).

    EDGAR shows a net loss: NI $(16.8), EPS $(0.03), pretax (49.5), tax
    (29.8), realized gains (11.7).  The negative cells are split the way the
    filing's HTML splits them: "(49.5" in one cell and ")" in the next.
    """
    rows = [
        ("Net premiums written", "$", "2,592.3", ""),
        ("Revenues:", "", "", ""),
        ("Net premiums earned", "$", "2,436.6", ""),
        ("Investment income", "", "46.9", ""),
        ("Total net realized gains (losses) on securities", "", "(11.7", ")"),
        ("Fees and other revenues", "", "37.4", ""),
        ("Service revenues", "", "9.4", ""),
        ("Total revenues", "", "2,518.6", ""),
        ("Expenses:", "", "", ""),
        ("Losses and loss adjustment expenses", "", "2,025.5", ""),
        ("Policy acquisition costs", "", "196.8", ""),
        ("Other underwriting expenses", "", "328.8", ""),
        ("Investment expenses", "", "1.8", ""),
        ("Service expenses", "", "9.4", ""),
        ("Interest expense", "", "5.8", ""),
        ("Total expenses", "", "2,568.1", ""),
        ("Income (loss) before income taxes", "", "(49.5", ")"),
        ("Provision (benefit) for income taxes", "", "(29.8", ")"),
        ("Net income (loss)", "$", "(16.8", ")"),
    ]
    body = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows
    )
    summary = (
        "<table><tr><td>Net premiums written</td><td>2,592.3</td></tr>"
        "<tr><td>Combined ratio</td><td>104.3</td></tr>"
        "<tr><td>Net income (loss)</td><td>(16.8</td><td>)</td></tr></table>"
    )
    return f"<html><body>{summary}<table>{body}</table></body></html>"


def test_net_loss_month_parses_negative() -> None:
    parsed = _parse_html_exhibit(_aug_2017_income_statement(), "2017-09-20")
    assert parsed is not None
    assert parsed["month_end"] == "2017-08-31"
    assert parsed["net_income"] == pytest.approx(-16.8)
    assert parsed["income_before_income_taxes"] == pytest.approx(-49.5)
    assert parsed["provision_for_income_taxes"] == pytest.approx(-29.8)
    assert parsed["total_net_realized_gains"] == pytest.approx(-11.7)
    # Identity: total revenues - total expenses = pretax income.
    assert parsed["total_revenues"] - parsed["total_expenses"] == pytest.approx(
        parsed["income_before_income_taxes"], abs=0.11
    )


# ---------------------------------------------------------------------------
# F17: pagination and item filter
# ---------------------------------------------------------------------------

_PRIMARY = {
    "filings": {
        "recent": {
            "accessionNumber": ["0000080661-26-000322"],
            "filingDate": ["2026-09-17"],
            "form": ["8-K"],
            "items": ["7.01,9.01"],
        },
        "files": [
            {"name": "CIK0000080661-submissions-001.json",
             "filingFrom": "2009-02-19", "filingTo": "2022-05-16"},
        ],
    }
}

# Pagination files are flat: the parallel arrays sit at the top level.
_FLAT_PAGE = {
    "accessionNumber": [
        "0000080661-19-000027", "0000080661-15-000034", "0000080661-15-000033",
    ],
    "filingDate": ["2019-05-15", "2015-06-17", "2015-06-10"],
    "form": ["8-K", "8-K", "8-K"],
    "items": ["5.07,7.01,9.01", "9.01", "5.02"],
}


def test_flat_pagination_file_is_read(monkeypatch) -> None:
    pages = {None: _PRIMARY, "001": _FLAT_PAGE}
    monkeypatch.setattr(fetcher, "_fetch_submissions_page", lambda page_id=None: pages[page_id])
    filings = fetcher.fetch_all_8k_filings("2010-01-01")
    by_acc = {f["accession_dashed"]: f for f in filings}
    assert set(by_acc) == {
        "0000080661-26-000322", "0000080661-19-000027", "0000080661-15-000034",
    }
    assert by_acc["0000080661-19-000027"]["item_code"] == "7.01"
    assert fetcher._match_item_code("2.02,7.01,9.01") == "2.02"
    assert by_acc["0000080661-15-000034"]["item_code"] == "9.01"


_INDEX_HTML = """
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/80661/000008066115000034/a8-kmay2015earningsrelease.htm">a</a></td><td>8-K</td><td>1</td></tr>
{ex99}
</table>
"""
_EX99_ROW = (
    '<tr><td>2</td><td>EXHIBIT 99</td><td><a href="/Archives/edgar/data/80661/'
    '000008066115000034/exhibit99may2015earningsre.htm">e</a></td><td>EX-99</td><td>1</td></tr>'
)


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


def test_9_01_only_filing_needs_an_ex99_exhibit(monkeypatch) -> None:
    assert fetcher._match_item_code("9.01") == "9.01"
    assert fetcher._match_item_code("5.02,9.01") is None

    monkeypatch.setattr(fetcher, "_get", lambda url, **_: _Resp(_INDEX_HTML.format(ex99="")))
    assert fetcher._get_all_filing_doc_urls(
        "000008066115000034", "0000080661-15-000034", require_ex99=True
    ) == []

    monkeypatch.setattr(
        fetcher, "_get", lambda url, **_: _Resp(_INDEX_HTML.format(ex99=_EX99_ROW))
    )
    urls = fetcher._get_all_filing_doc_urls(
        "000008066115000034", "0000080661-15-000034", require_ex99=True
    )
    assert urls[0].endswith("exhibit99may2015earningsre.htm")


# ---------------------------------------------------------------------------
# F18: equity and debt anchoring
# ---------------------------------------------------------------------------


def _balance_sheet(rows: list[tuple[str, str]]) -> object:
    from bs4 import BeautifulSoup

    body = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows)
    return BeautifulSoup(f"<table>{body}</table>", "lxml").find("table")


def test_no_equity_line_uses_bvps_times_shares() -> None:
    """Dec-2004 release (0000950152-05-000341): no total-equity line.

    It shows BVPS $25.73, ROE 30.0 %, debt/capital 19.9 % and 200.4M shares;
    the FY2004 10-K gives equity of $5,155.4M.
    """
    html = (
        "<html><body><table>"
        "<tr><td>Total assets</td><td>$17,184.3</td></tr>"
        "<tr><td>Common shares outstanding</td><td>200.4</td></tr>"
        "<tr><td>Book value per share</td><td>$25.73</td></tr>"
        "<tr><td>Return on average shareholders’ equity</td><td>30.0%</td></tr>"
        "<tr><td>Debt to total capital ratio</td><td>19.9%</td></tr>"
        "<tr><td>Combined ratio</td><td>87.4</td></tr>"
        "</table></body></html>"
    )
    parsed = _parse_html_exhibit(html, "2005-01-19")
    assert parsed is not None
    assert parsed["shareholders_equity"] == pytest.approx(25.73 * 200.4, abs=0.1)
    assert parsed["shareholders_equity"] == pytest.approx(5155.4, rel=0.01)
    assert "shareholders_equity" in parsed["derived_fields"]
    assert parsed["debt"] is None
    assert parsed["debt_to_total_capital"] == pytest.approx(19.9)


def test_total_equity_line_is_used_not_roe_row() -> None:
    tbl = _balance_sheet([
        ("Return on average shareholders' equity", "16.3%"),
        ("Shareholders' equity:", ""),
        ("Total shareholders' equity", "$9,289.4"),
        ("Debt-to-total capital ratio", "25.4%"),
        ("Debt", "3,148.2"),
        ("Book value per common share", "$15.97"),
        ("Common shares outstanding", "581.6"),
    ])
    out = _extract_balance_sheet(tbl, "2017-09-30")
    assert out["shareholders_equity"] == pytest.approx(9289.4)
    assert out["debt"] == pytest.approx(3148.2)
    assert out["debt_to_total_capital"] == pytest.approx(25.4)


# ---------------------------------------------------------------------------
# F11 / F16: calendar YoY, leap years, PIF definition, Gainshare
# ---------------------------------------------------------------------------


def test_prior_year_key_uses_calendar_month() -> None:
    assert _prior_year_key("2025-02-28") == "2024-02-29"
    assert _prior_year_key("2024-02-29") == "2023-02-28"
    assert _prior_year_key("2021-02-28") == "2020-02-29"


def _pif_record(month_end: str, scale: float, property_pif: float = 3_000.0) -> dict:
    return {
        "month_end": month_end,
        "combined_ratio": 90.0,
        "pif_agency_auto": 10_000.0 * scale,
        "pif_direct_auto": 15_000.0 * scale,
        "pif_special_lines": 7_000.0 * scale,
        "pif_commercial_lines": 1_200.0 * scale,
        "pif_property": property_pif,
        "net_premiums_written": 6_000.0 * scale,
    }


def test_february_after_leap_year_has_yoy() -> None:
    records = [_pif_record("2024-02-29", 1.0), _pif_record("2025-02-28", 1.1)]
    _compute_derived_fields(records)
    assert records[1]["pif_growth_yoy"] == pytest.approx(0.1)
    assert records[1]["npw_growth_yoy"] == pytest.approx(0.1)


def test_pif_total_excludes_property_whatever_the_release_printed() -> None:
    # The release printed a companywide total that includes property.
    rec = _pif_record("2025-03-31", 1.0) | {"pif_total": 36_200.0}
    _compute_derived_fields([rec])
    assert rec["pif_total"] == pytest.approx(33_200.0)
    assert rec["pif_total_personal_lines"] == pytest.approx(32_000.0)


def test_pif_total_needs_every_component() -> None:
    rec = _pif_record("2025-03-31", 1.0) | {"pif_special_lines": None, "pif_total": 30_000.0}
    _compute_derived_fields([rec])
    assert rec["pif_total"] is None


def test_yoy_is_nan_when_base_month_is_missing() -> None:
    """F16: a missing month must give NaN a year later, not a 13-month change."""
    months = pd.date_range("2014-01-31", "2016-06-30", freq="ME")
    records = [
        _pif_record(m.strftime("%Y-%m-%d"), 1.0 + 0.01 * i)
        for i, m in enumerate(months)
        if m != pd.Timestamp("2015-05-31")
    ]
    _compute_derived_fields(records)
    by_month = {r["month_end"]: r for r in records}
    assert by_month["2016-05-31"]["npw_growth_yoy"] is None
    assert by_month["2016-05-31"]["pif_growth_yoy"] is None
    assert by_month["2016-04-30"]["npw_growth_yoy"] == pytest.approx(1.27 / 1.15 - 1)


def test_load_from_csv_missing_month_gives_nan_yoy(tmp_path: Path) -> None:
    months = [m for m in pd.date_range("2014-01-31", "2016-06-30", freq="ME")
              if m != pd.Timestamp("2015-05-31")]
    df = pd.DataFrame({
        "report_period": [m.strftime("%Y-%m") for m in months],
        "net_premiums_written": np.linspace(1_000, 1_300, len(months)),
        "unearned_premiums": np.linspace(5_000, 6_000, len(months)),
        "combined_ratio": 90.0,
    })
    csv_path = tmp_path / "cache.csv"
    df.to_csv(csv_path, index=False)
    conn = _fresh_db(tmp_path)
    load_from_csv(conn, str(csv_path))
    got = dict(conn.execute(
        "SELECT month_end, npw_growth_yoy FROM pgr_edgar_monthly"
    ).fetchall())
    uprem = dict(conn.execute(
        "SELECT month_end, unearned_premium_growth_yoy FROM pgr_edgar_monthly"
    ).fetchall())
    conn.close()
    assert got["2016-05-31"] is None
    assert uprem["2016-05-31"] is None
    assert got["2016-04-30"] is not None


def test_one_gainshare_formula_everywhere(tmp_path: Path, monkeypatch) -> None:
    """Every producer of gainshare_estimate / pif_growth_yoy gives the same values."""
    from src.ingestion import edgar_8k_fetcher as ingestion_fetcher
    from src.ingestion import pgr_monthly_loader

    months = pd.date_range("2023-01-31", periods=16, freq="ME")
    pif = [30_000.0] * 12 + [31_500.0, 33_000.0, 36_000.0, 45_000.0]
    cr = [90.0] * 12 + [86.0, 101.0, 70.0, np.nan]
    expected = [np.nan] * 12 + [
        0.5 * 1.0 + 0.5 * 0.5,  # growth 5 %
        0.5 * 0.0 + 0.5 * 1.0,  # CR above target, growth 10 %
        0.5 * 2.0 + 0.5 * 2.0,  # both capped at 2
        np.nan,                 # CR missing
    ]

    frame = pd.DataFrame({"combined_ratio": cr, "pif_total": pif}, index=months)
    shared = pgr_edgar_derived.gainshare_series(
        frame["combined_ratio"], pgr_edgar_derived.yoy_growth_series(frame["pif_total"])
    )
    legacy = ingestion_fetcher._compute_gainshare(frame)["gainshare_estimate"]

    records = [
        {"month_end": m.strftime("%Y-%m-%d"), "combined_ratio": None if np.isnan(c) else c,
         "pif_agency_auto": p, "pif_direct_auto": 0.0, "pif_special_lines": 0.0,
         "pif_commercial_lines": 0.0}
        for m, c, p in zip(months, cr, pif)
    ]
    script = [r["gainshare_estimate"] for r in _compute_derived_fields(records)]

    csv_path = tmp_path / "cache.csv"
    pd.DataFrame({
        "report_period": [m.strftime("%Y-%m") for m in months],
        "combined_ratio": cr, "pif_total": pif,
    }).to_csv(csv_path, index=False)
    monkeypatch.setattr(pgr_monthly_loader, "_EDGAR_CACHE_PATH", str(csv_path))
    monkeypatch.setattr(pgr_monthly_loader, "_PROCESSED_PATH", str(tmp_path / "out.parquet"))
    loader = pgr_monthly_loader.load(force_refresh=True, apply_filing_lag=False)[
        "gainshare_estimate"
    ]

    for i, exp in enumerate(expected):
        got = [shared.iloc[i], legacy.iloc[i], script[i], loader.iloc[i]]
        got = [np.nan if g is None else g for g in got]
        if np.isnan(exp):
            assert all(np.isnan(g) for g in got), (i, got)
        else:
            assert got == pytest.approx([exp] * 4), (i, got)


# ---------------------------------------------------------------------------
# F33: provenance
# ---------------------------------------------------------------------------


def _row(conn: sqlite3.Connection, month_end: str) -> dict:
    cur = conn.execute("SELECT * FROM pgr_edgar_monthly WHERE month_end = ?", (month_end,))
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone()))


def test_later_filing_does_not_overwrite_provenance(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    first = {
        "month_end": "2025-09-30", "filing_date": "2025-10-15",
        "accession_number": "000008066125000126", "combined_ratio": 100.4,
        "expense_ratio": 34.7,
    }
    later = {
        "month_end": "2025-09-30", "filing_date": "2025-11-19",
        "accession_number": "0000080661-25-000134", "combined_ratio": 88.7,
        "expense_ratio": 23.0, "net_income": 305.0,
    }
    db_client.upsert_pgr_edgar_monthly(conn, [first])
    db_client.upsert_pgr_edgar_monthly(conn, [later])
    row = _row(conn, "2025-09-30")
    conn.close()
    assert row["accession_number"] == "0000080661-25-000126"
    assert row["filing_date"] == "2025-10-15"
    assert row["combined_ratio"] == pytest.approx(100.4)
    assert row["net_income"] is None  # nothing from the later filing leaks in


def test_earlier_filing_replaces_the_whole_row(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    db_client.upsert_pgr_edgar_monthly(conn, [{
        "month_end": "2019-04-30", "filing_date": "2019-06-01",
        "accession_number": "0000080661-19-000099", "combined_ratio": 80.0,
        "net_income": 1.0,
    }])
    db_client.upsert_pgr_edgar_monthly(conn, [{
        "month_end": "2019-04-30", "filing_date": "2019-05-15",
        "accession_number": "0000080661-19-000027", "combined_ratio": 87.4,
    }])
    row = _row(conn, "2019-04-30")
    conn.close()
    assert row["accession_number"] == "0000080661-19-000027"
    assert row["combined_ratio"] == pytest.approx(87.4)
    assert row["net_income"] is None


def test_same_filing_reparse_updates_values_not_provenance(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    base = {"month_end": "2017-08-31", "filing_date": "2017-09-20",
            "accession_number": "0000080661-17-000060"}
    db_client.upsert_pgr_edgar_monthly(conn, [base | {"net_income": 16.8, "eps_basic": 0.03}])
    db_client.upsert_pgr_edgar_monthly(
        conn, [base | {"accession_number": "000008066117000060", "net_income": -16.8}]
    )
    row = _row(conn, "2017-08-31")
    conn.close()
    assert row["net_income"] == pytest.approx(-16.8)
    assert row["eps_basic"] == pytest.approx(0.03)
    assert row["accession_number"] == "0000080661-17-000060"


def test_raw_table_is_append_only_and_first_reported_view(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    rec = {"month_end": "2019-04-30", "filing_date": "2019-05-15",
           "accession_number": "0000080661-19-000027", "combined_ratio": 87.4,
           "document_url": "https://www.sec.gov/x.htm", "fetched_at": "2026-09-25T00:00:00+00:00"}
    assert db_client.record_pgr_edgar_raw(conn, [rec], "v1") == 1
    assert db_client.record_pgr_edgar_raw(conn, [rec], "v1") == 0  # same key: ignored
    db_client.record_pgr_edgar_raw(conn, [rec | {"combined_ratio": 87.5}], "v2")
    # A later filing for the same month does not change the first-reported value.
    db_client.record_pgr_edgar_raw(conn, [rec | {
        "accession_number": "0000080661-19-000040", "filing_date": "2019-06-20",
        "combined_ratio": 99.9,
    }], "v2")
    for table, column in (
        ("pgr_edgar_monthly_raw", "value_real"),
        ("pgr_edgar_filing_parses", "fetched_at"),
    ):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(f"UPDATE {table} SET {column} = NULL")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")
    flat = pd.read_sql_query(
        "SELECT * FROM pgr_edgar_monthly_raw_values WHERE field = 'combined_ratio'", conn
    )
    assert len(flat) == 3
    assert set(flat["fetched_at"]) == {"2026-09-25T00:00:00+00:00"}
    assert set(flat["source_url"]) == {"https://www.sec.gov/x.htm"}
    first = db_client.get_pgr_edgar_first_reported(conn)
    conn.close()
    row = first[(first["month_end"] == "2019-04-30") & (first["field"] == "combined_ratio")]
    assert len(row) == 1
    assert row["value_real"].iloc[0] == pytest.approx(87.5)  # latest parser, first filing
    assert row["accession_number"].iloc[0] == "0000080661-19-000027"
    assert row["parser_version"].iloc[0] == "v2"


def test_load_from_csv_is_idempotent_against_newer_rows(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    live = {
        "month_end": "2025-09-30", "filing_date": "2025-10-15",
        "accession_number": "0000080661-25-000126", "combined_ratio": 100.4,
        "investment_book_yield": 4.1,
    }
    db_client.upsert_pgr_edgar_monthly(conn, [live])
    csv_path = tmp_path / "cache.csv"
    pd.DataFrame({
        "report_period": ["2025-08", "2025-09"],
        "accession_number": ["0000080661-25-000079", "0000080661-25-000126"],
        "filing_date": ["2025-09-17", "2025-10-15"],
        "combined_ratio": [87.0, 88.7],
        "investment_book_yield": [0.041, 0.041],
    }).to_csv(csv_path, index=False)

    assert load_from_csv(conn, str(csv_path)) == 1  # only 2025-08 is new
    snapshot = conn.execute("SELECT * FROM pgr_edgar_monthly ORDER BY month_end").fetchall()
    assert load_from_csv(conn, str(csv_path)) == 0
    assert conn.execute(
        "SELECT * FROM pgr_edgar_monthly ORDER BY month_end"
    ).fetchall() == snapshot
    row = _row(conn, "2025-09-30")
    conn.close()
    assert row["combined_ratio"] == pytest.approx(100.4)
    assert row["investment_book_yield"] == pytest.approx(4.1)


def test_recompute_derived_fields_covers_whole_table(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    db_client.upsert_pgr_edgar_monthly(conn, [
        _pif_record("2024-02-29", 1.0) | {"pif_growth_yoy": 0.99, "gainshare_estimate": 1.9},
        _pif_record("2025-02-28", 1.1),
    ])
    recompute_derived_fields(conn)
    old, new = _row(conn, "2024-02-29"), _row(conn, "2025-02-28")
    conn.close()
    assert old["pif_growth_yoy"] is None and old["gainshare_estimate"] is None
    assert new["pif_growth_yoy"] == pytest.approx(0.1)
    assert new["gainshare_estimate"] == pytest.approx(0.5 * 0.6 + 0.5 * 1.0)
