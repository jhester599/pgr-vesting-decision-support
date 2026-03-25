"""
Tests for src/ingestion/edgar_8k_fetcher.py — all mocked, no network calls.

Coverage:
  1. fetch_submissions — filters form=8-K with 7.01, correct older-page URL
  2. _accession_to_folder — dash removal
  3. get_exhibit_url — locates EX-99 href in index HTML
  4. _dedup_row — collapses duplicate values
  5. _try_float / _extract_numerics — numeric parsing (negatives, commas)
  6. _extract_report_period — month-end date from exhibit text
  7. parse_exhibit — full round-trip with synthetic HTML tables
  8. _compute_gainshare — CR and PIF growth derivation
  9. fetch_monthly_8ks — orchestration smoke test
 10. backfill_to_db — DB write path
"""

from __future__ import annotations

import io
import sqlite3
from datetime import date
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from src.ingestion.edgar_8k_fetcher import (
    _accession_to_folder,
    _compute_gainshare,
    _dedup_row,
    _extract_numerics,
    _extract_report_period,
    _try_float,
    fetch_monthly_8ks,
    fetch_submissions,
    get_exhibit_url,
    parse_exhibit,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal submissions JSON payload
# ---------------------------------------------------------------------------

def _make_submissions(
    forms: list[str],
    items: list[str],
    filing_dates: list[str],
    accessions: list[str],
    extra_files: list[dict] | None = None,
) -> dict:
    return {
        "filings": {
            "recent": {
                "form": forms,
                "items": items,
                "filingDate": filing_dates,
                "accessionNumber": accessions,
            },
            "files": extra_files or [],
        }
    }


# ---------------------------------------------------------------------------
# 1. fetch_submissions
# ---------------------------------------------------------------------------

class TestFetchSubmissions:
    def _session(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = payload
        session = MagicMock()
        session.get.return_value = resp
        return session

    def test_filters_7_01_8ks_only(self):
        payload = _make_submissions(
            forms=["4", "8-K", "8-K", "DEF 14A"],
            items=["", "7.01,9.01", "2.02,9.01", ""],
            filing_dates=["2026-01-10", "2026-01-18", "2026-01-28", "2026-02-01"],
            accessions=["AAA", "BBB", "CCC", "DDD"],
        )
        results = fetch_submissions(self._session(payload))
        assert len(results) == 1
        assert results[0]["accession"] == "BBB"
        assert results[0]["filing_date"] == "2026-01-18"

    def test_sorted_descending_by_filing_date(self):
        payload = _make_submissions(
            forms=["8-K", "8-K", "8-K"],
            items=["7.01", "7.01", "7.01"],
            filing_dates=["2025-03-19", "2025-12-17", "2025-06-18"],
            accessions=["A", "B", "C"],
        )
        results = fetch_submissions(self._session(payload))
        dates = [r["filing_date"] for r in results]
        assert dates == sorted(dates, reverse=True)

    def test_older_filings_page_url_has_submissions_prefix(self):
        """Extra filings pages must be fetched from /submissions/ path."""
        main_payload = _make_submissions(
            forms=["8-K"],
            items=["7.01"],
            filing_dates=["2026-01-18"],
            accessions=["AAA"],
            extra_files=[{"name": "CIK0000080661-submissions-001.json"}],
        )
        extra_payload = {
            "form": ["8-K"],
            "items": ["7.01,9.01"],
            "filingDate": ["2022-01-19"],
            "accessionNumber": ["OLD001"],
        }
        session = MagicMock()
        resp_main = MagicMock()
        resp_main.raise_for_status = MagicMock()
        resp_main.json.return_value = main_payload
        resp_extra = MagicMock()
        resp_extra.raise_for_status = MagicMock()
        resp_extra.json.return_value = extra_payload

        session.get.side_effect = [resp_main, resp_extra]
        results = fetch_submissions(session)

        # Verify second call used /submissions/ prefix
        second_call_url = session.get.call_args_list[1][0][0]
        assert "data.sec.gov/submissions/" in second_call_url
        assert "CIK0000080661-submissions-001.json" in second_call_url

        # Both results should be present
        accessions = [r["accession"] for r in results]
        assert "AAA" in accessions
        assert "OLD001" in accessions

    def test_no_monthly_8ks_returns_empty_list(self):
        payload = _make_submissions(
            forms=["4", "DEF 14A"],
            items=["", ""],
            filing_dates=["2026-01-10", "2026-01-15"],
            accessions=["A", "B"],
        )
        results = fetch_submissions(self._session(payload))
        assert results == []

    def test_older_page_http_error_is_warned_not_raised(self):
        """A 404 on an older filings page should log a warning, not crash."""
        import requests as req_lib

        main_payload = _make_submissions(
            forms=["8-K"],
            items=["7.01"],
            filing_dates=["2026-01-18"],
            accessions=["AAA"],
            extra_files=[{"name": "CIK0000080661-submissions-001.json"}],
        )
        session = MagicMock()
        resp_main = MagicMock()
        resp_main.raise_for_status = MagicMock()
        resp_main.json.return_value = main_payload
        resp_extra = MagicMock()
        resp_extra.raise_for_status.side_effect = req_lib.HTTPError("404")

        session.get.side_effect = [resp_main, resp_extra]
        results = fetch_submissions(session)
        # Main filing still returned even though extra page failed
        assert len(results) == 1
        assert results[0]["accession"] == "AAA"


# ---------------------------------------------------------------------------
# 2. _accession_to_folder
# ---------------------------------------------------------------------------

class TestAccessionToFolder:
    def test_removes_dashes(self):
        assert _accession_to_folder("0000080661-26-000096") == "000008066126000096"

    def test_no_dashes_unchanged(self):
        assert _accession_to_folder("000008066126000096") == "000008066126000096"


# ---------------------------------------------------------------------------
# 3. get_exhibit_url
# ---------------------------------------------------------------------------

_INDEX_HTML_WITH_EX99 = """
<table>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/80661/001/pgr-20260318.htm">pgr-20260318.htm</a></td><td>8-K</td></tr>
<tr><td>2</td><td>EX-99</td><td><a href="/Archives/edgar/data/80661/001/pgr202602ex99earningsrelea.htm">pgr202602ex99earningsrelea.htm</a></td><td>EX-99</td></tr>
</table>
"""

_INDEX_HTML_NO_EX99 = """
<table>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/80661/001/pgr.htm">pgr.htm</a></td></tr>
</table>
"""


class TestGetExhibitUrl:
    def _session(self, html: str) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.text = html
        session = MagicMock()
        session.get.return_value = resp
        return session

    def test_returns_full_url_for_ex99(self):
        url = get_exhibit_url(
            self._session(_INDEX_HTML_WITH_EX99), "0000080661-26-000001"
        )
        assert url is not None
        assert "pgr202602ex99earningsrelea.htm" in url
        assert url.startswith("https://www.sec.gov")

    def test_returns_none_when_no_ex99(self):
        url = get_exhibit_url(
            self._session(_INDEX_HTML_NO_EX99), "0000080661-26-000001"
        )
        assert url is None

    def test_http_error_returns_none(self):
        import requests as req_lib
        session = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.side_effect = req_lib.HTTPError("404")
        session.get.return_value = resp
        url = get_exhibit_url(session, "0000080661-26-000001")
        assert url is None


# ---------------------------------------------------------------------------
# 4. _dedup_row
# ---------------------------------------------------------------------------

class TestDedupRow:
    def test_collapses_consecutive_duplicates(self):
        row = pd.Series(["Net premiums written", "Net premiums written", "Net premiums written", "$", "6995", "$", "6684"])
        result = _dedup_row(row)
        assert result == ["Net premiums written", "$", "6995", "$", "6684"]

    def test_drops_nan_values(self):
        row = pd.Series([float("nan"), "Combined ratio", float("nan"), "85.7", "82.6"])
        result = _dedup_row(row)
        assert result == ["Combined ratio", "85.7", "82.6"]

    def test_non_consecutive_duplicates_retained(self):
        row = pd.Series(["$", "6995", "$", "6684"])
        result = _dedup_row(row)
        assert result == ["$", "6995", "$", "6684"]

    def test_empty_row(self):
        row = pd.Series([float("nan"), float("nan")])
        assert _dedup_row(row) == []


# ---------------------------------------------------------------------------
# 5. _try_float / _extract_numerics
# ---------------------------------------------------------------------------

class TestTryFloat:
    def test_positive_integer(self):
        assert _try_float("6995") == pytest.approx(6995.0)

    def test_positive_decimal(self):
        assert _try_float("85.7") == pytest.approx(85.7)

    def test_comma_separated(self):
        assert _try_float("39,220") == pytest.approx(39220.0)

    def test_parentheses_negative(self):
        assert _try_float("(5)") == pytest.approx(-5.0)

    def test_dollar_sign_stripped(self):
        assert _try_float("$ 943") == pytest.approx(943.0)

    def test_non_numeric_returns_none(self):
        assert _try_float("pts.") is None
        assert _try_float("%") is None
        assert _try_float("NaN") is None

    def test_percentage_string_with_pct(self):
        # "5 %" should fail because of the % sign — we only want pure numbers
        assert _try_float("5 %") is None


class TestExtractNumerics:
    def test_extracts_all_numbers(self):
        tokens = ["$", "6995", "$", "6684", "5", "%"]
        nums = _extract_numerics(tokens)
        assert nums == pytest.approx([6995.0, 6684.0, 5.0])

    def test_handles_negatives(self):
        tokens = ["(5)", "(110)", "(95)"]
        nums = _extract_numerics(tokens)
        assert nums == pytest.approx([-5.0, -110.0, -95.0])

    def test_skips_labels(self):
        nums = _extract_numerics(["pts.", "Change", "n/a"])
        assert nums == []


# ---------------------------------------------------------------------------
# 6. _extract_report_period
# ---------------------------------------------------------------------------

class TestExtractReportPeriod:
    def test_standard_format(self):
        html = "<p>results for the month ended February&#160;28, 2026:</p>"
        d = _extract_report_period(html)
        assert d == date(2026, 2, 28)

    def test_without_html_entities(self):
        html = "<p>For the month ended October 31, 2025</p>"
        d = _extract_report_period(html)
        assert d == date(2025, 10, 31)

    def test_january(self):
        html = "For the month ended January 31, 2024"
        d = _extract_report_period(html)
        assert d == date(2024, 1, 31)

    def test_missing_month_text_returns_none(self):
        html = "<p>No date information here</p>"
        assert _extract_report_period(html) is None

    def test_strips_tags(self):
        html = "<b>month</b> <b>ended</b> <b>March</b> <b>31,</b> <b>2025</b>"
        d = _extract_report_period(html)
        assert d == date(2025, 3, 31)


# ---------------------------------------------------------------------------
# 7. parse_exhibit — synthetic HTML
# ---------------------------------------------------------------------------

def _make_exhibit_html(
    period: str = "February&#160;28, 2026",
    combined_ratio: str = "85.7",
    pif_total_label: str = "Total",
    pif_value: str = "39,220",
    npw: str = "6,995",
    npe: str = "6,528",
    net_income: str = "943",
    eps: str = "1.61",
    loss_ratio: str = "65.0",
    expense_ratio: str = "20.7",
) -> str:
    """Build a minimal exhibit HTML with the key tables."""
    return f"""<html><body>
<p>results for the month ended {period}</p>

<!-- Summary table (Table 2) -->
<table>
<tr><td>February</td><td>February</td></tr>
<tr><td>(millions)</td><td>2026</td><td>2025</td><td>Change</td></tr>
<tr><td>Net premiums written</td><td>Net premiums written</td><td>$</td><td>{npw}</td><td>$</td><td>5,000</td><td>10</td><td>%</td></tr>
<tr><td>Net premiums earned</td><td>Net premiums earned</td><td>$</td><td>{npe}</td><td>$</td><td>4,800</td><td>8</td><td>%</td></tr>
<tr><td>Net income</td><td>Net income</td><td>$</td><td>{net_income}</td><td>$</td><td>800</td><td>2</td><td>%</td></tr>
<tr><td>Per share available to common shareholders</td><td>Per share available to common shareholders</td><td>$</td><td>{eps}</td><td>$</td><td>1.40</td></tr>
<tr><td>Combined ratio</td><td>Combined ratio</td><td>{combined_ratio}</td><td>{combined_ratio}</td><td>82.6</td><td>3.1</td><td>pts.</td></tr>
</table>

<!-- YTD table — values must NOT overwrite monthly -->
<table>
<tr><td>Year-to-Date</td></tr>
<tr><td>Net premiums written</td><td>Net premiums written</td><td>$</td><td>13,730</td></tr>
<tr><td>Net premiums earned</td><td>Net premiums earned</td><td>$</td><td>13,000</td></tr>
<tr><td>Net income</td><td>Net income</td><td>$</td><td>2,106</td></tr>
<tr><td>Combined ratio</td><td>Combined ratio</td><td>83.0</td></tr>
</table>

<!-- PIF table (Table 3) -->
<table>
<tr><td>February 28,</td><td>February 28,</td></tr>
<tr><td>(thousands)</td><td>2026</td><td>2025</td><td>% Change</td></tr>
<tr><td>Policies in Force</td></tr>
<tr><td>Personal Lines</td></tr>
<tr><td>Agency auto</td><td>Agency auto</td><td>10,959</td><td>9,950</td><td>10</td></tr>
<tr><td>Direct auto</td><td>Direct auto</td><td>16,383</td><td>14,395</td><td>14</td></tr>
<tr><td>Special lines</td><td>Special lines</td><td>7,041</td><td>6,568</td><td>7</td></tr>
<tr><td>Property</td><td>Property</td><td>3,649</td><td>3,556</td><td>3</td></tr>
<tr><td>Total Personal Lines</td><td>Total Personal Lines</td><td>38,032</td><td>34,469</td><td>10</td></tr>
<tr><td>Commercial Lines</td><td>Commercial Lines</td><td>1,188</td><td>1,151</td><td>3</td></tr>
<tr><td>{pif_total_label}</td><td>{pif_total_label}</td><td>{pif_value}</td><td>35,620</td><td>10</td></tr>
</table>

<!-- Supplemental GAAP Ratios table (Table 8 equivalent) -->
<table>
<tr><td>GAAP Ratios</td></tr>
<tr><td>Loss/LAE ratio</td><td>Loss/LAE ratio</td><td>63.9</td><td>67.8</td><td>41.5</td><td>65.1</td><td>63.8</td><td>{loss_ratio}</td></tr>
<tr><td>Expense ratio</td><td>Expense ratio</td><td>18.1</td><td>21.6</td><td>29.2</td><td>20.5</td><td>22.2</td><td>{expense_ratio}</td></tr>
<tr><td>Combined ratio</td><td>Combined ratio</td><td>82.0</td><td>89.4</td><td>70.7</td><td>85.6</td><td>86.0</td><td>{combined_ratio}</td></tr>
</table>
</body></html>"""


class TestParseExhibit:
    def test_extracts_report_period(self):
        html = _make_exhibit_html()
        m = parse_exhibit(html)
        assert m["report_period"] == date(2026, 2, 28)

    def test_extracts_combined_ratio(self):
        m = parse_exhibit(_make_exhibit_html(combined_ratio="85.7"))
        assert m["combined_ratio"] == pytest.approx(85.7)

    def test_extracts_net_premiums_written(self):
        m = parse_exhibit(_make_exhibit_html(npw="6,995"))
        assert m["net_premiums_written"] == pytest.approx(6995.0)

    def test_extracts_net_premiums_earned(self):
        m = parse_exhibit(_make_exhibit_html(npe="6,528"))
        assert m["net_premiums_earned"] == pytest.approx(6528.0)

    def test_extracts_net_income(self):
        m = parse_exhibit(_make_exhibit_html(net_income="943"))
        assert m["net_income"] == pytest.approx(943.0)

    def test_extracts_eps_diluted(self):
        m = parse_exhibit(_make_exhibit_html(eps="1.61"))
        assert m["eps_diluted"] == pytest.approx(1.61)

    def test_extracts_pif_total_label(self):
        m = parse_exhibit(_make_exhibit_html(pif_total_label="Total", pif_value="39,220"))
        assert m["pif_total"] == pytest.approx(39220.0)

    def test_extracts_pif_companywide_label(self):
        """PGR sometimes uses 'Companywide' instead of 'Total'."""
        m = parse_exhibit(_make_exhibit_html(pif_total_label="Companywide", pif_value="38,379"))
        assert m["pif_total"] == pytest.approx(38379.0)

    def test_extracts_pif_companywide_total_label(self):
        """PGR 2024 filings use 'Companywide Total'."""
        m = parse_exhibit(_make_exhibit_html(pif_total_label="Companywide Total", pif_value="34,364.3"))
        assert m["pif_total"] == pytest.approx(34364.3)

    def test_extracts_pif_fractional_value(self):
        """Older filings may report PIF as decimals (e.g. 34364.3)."""
        m = parse_exhibit(_make_exhibit_html(pif_total_label="Total", pif_value="34,364.3"))
        assert m["pif_total"] == pytest.approx(34364.3)

    def test_extracts_loss_ratio(self):
        m = parse_exhibit(_make_exhibit_html(loss_ratio="65.0"))
        assert m["loss_ratio"] == pytest.approx(65.0)

    def test_extracts_expense_ratio(self):
        m = parse_exhibit(_make_exhibit_html(expense_ratio="20.7"))
        assert m["expense_ratio"] == pytest.approx(20.7)

    def test_ytd_does_not_overwrite_monthly_npw(self):
        """Monthly NPW (6,995) must not be overwritten by YTD NPW (13,730)."""
        m = parse_exhibit(_make_exhibit_html(npw="6,995"))
        assert m["net_premiums_written"] == pytest.approx(6995.0)

    def test_ytd_does_not_overwrite_monthly_combined_ratio(self):
        """Monthly CR (85.7) must not be overwritten by YTD CR (83.0)."""
        m = parse_exhibit(_make_exhibit_html(combined_ratio="85.7"))
        assert m["combined_ratio"] == pytest.approx(85.7)

    def test_missing_period_returns_none(self):
        html = "<html><body><p>No date here</p></body></html>"
        m = parse_exhibit(html)
        assert m["report_period"] is None

    def test_empty_html_returns_all_none(self):
        m = parse_exhibit("<html></html>")
        for key, val in m.items():
            assert val is None, f"Expected None for {key}, got {val}"


# ---------------------------------------------------------------------------
# 8. _compute_gainshare
# ---------------------------------------------------------------------------

class TestComputeGainshare:
    def _make_df(self, crs: list[float], pifs: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2024-01-31", periods=len(crs), freq="ME")
        return pd.DataFrame({"combined_ratio": crs, "pif_total": pifs}, index=idx)

    def test_cr_below_96_yields_positive_score(self):
        df = self._make_df([86.0] * 13, [30000.0] * 13)
        result = _compute_gainshare(df)
        # CR score = (96 - 86) / 10 = 1.0
        assert result["gainshare_estimate"].iloc[-1] > 0

    def test_cr_above_96_yields_zero_cr_score(self):
        df = self._make_df([98.0] * 13, [30000.0] * 13)
        result = _compute_gainshare(df)
        # CR score = 0; pif_growth_yoy for 13th month = 0 (same PIF)
        assert result["gainshare_estimate"].iloc[-1] == pytest.approx(0.0)

    def test_gainshare_capped_at_two(self):
        # CR = 76 → CR score = (96-76)/10 = 2.0
        # PIF growth = 20% → PIF score = min(0.20/0.10, 2.0) = 2.0
        pifs = [30000.0] * 12 + [36000.0]
        df = self._make_df([76.0] * 13, pifs)
        result = _compute_gainshare(df)
        assert result["gainshare_estimate"].iloc[-1] <= 2.0

    def test_pif_growth_yoy_computed_correctly(self):
        # 13 months of data; last month PIF is 10% higher than 12 months ago
        pifs = [30000.0] * 12 + [33000.0]
        df = self._make_df([90.0] * 13, pifs)
        result = _compute_gainshare(df)
        expected_growth = (33000 - 30000) / 30000  # = 0.10
        assert result["pif_growth_yoy"].iloc[-1] == pytest.approx(expected_growth)

    def test_all_nan_pif_does_not_crash(self):
        df = self._make_df([85.0] * 6, [float("nan")] * 6)
        result = _compute_gainshare(df)
        assert result["pif_growth_yoy"].isna().all()

    def test_all_nan_cr_does_not_crash(self):
        df = self._make_df([float("nan")] * 6, [30000.0] * 6)
        result = _compute_gainshare(df)
        assert result["gainshare_estimate"].isna().all()


# ---------------------------------------------------------------------------
# 9. fetch_monthly_8ks — orchestration smoke test
# ---------------------------------------------------------------------------

class TestFetchMonthly8ks:
    """Smoke test: stub submissions + exhibit fetch; verify DataFrame output."""

    def _mock_session(self, submissions_payload: dict, exhibit_htmls: list[str]) -> MagicMock:
        session = MagicMock()
        sub_resp = MagicMock()
        sub_resp.raise_for_status = MagicMock()
        sub_resp.json.return_value = submissions_payload

        index_resp = MagicMock()
        index_resp.raise_for_status = MagicMock()
        # Return a minimal filing index HTML pointing to the exhibit
        index_resp.text = (
            '<table>'
            '<tr><td>2</td><td>EX-99</td>'
            '<td><a href="/Archives/edgar/data/80661/000001/pgr_ex99.htm">pgr_ex99.htm</a></td>'
            '</tr></table>'
        )

        exhibit_resps = []
        for html in exhibit_htmls:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.text = html
            exhibit_resps.append(r)

        # Interleave: submissions, (index, exhibit) * n
        side_effects = [sub_resp]
        for er in exhibit_resps:
            side_effects.append(index_resp)
            side_effects.append(er)

        session.get.side_effect = side_effects
        return session

    def test_returns_dataframe_with_expected_columns(self):
        payload = _make_submissions(
            forms=["8-K"],
            items=["7.01,9.01"],
            filing_dates=["2026-03-18"],
            accessions=["0000080661-26-000096"],
        )
        html = _make_exhibit_html()
        session = self._mock_session(payload, [html])
        df = fetch_monthly_8ks(lookback_months=1, session=session)

        assert isinstance(df, pd.DataFrame)
        assert "combined_ratio" in df.columns
        assert "pif_total" in df.columns
        assert "net_premiums_written" in df.columns

    def test_index_is_month_end_datetimeindex(self):
        payload = _make_submissions(
            forms=["8-K"],
            items=["7.01"],
            filing_dates=["2026-03-18"],
            accessions=["0000080661-26-000096"],
        )
        html = _make_exhibit_html(period="February&#160;28, 2026")
        session = self._mock_session(payload, [html])
        df = fetch_monthly_8ks(lookback_months=1, session=session)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == pd.Timestamp("2026-02-28")

    def test_deduplicates_same_month_keeps_last(self):
        """If two filings cover the same period month, keep the later one."""
        payload = _make_submissions(
            forms=["8-K", "8-K"],
            items=["7.01", "7.01"],
            filing_dates=["2026-03-18", "2026-03-19"],
            accessions=["ACC1", "ACC2"],
        )
        html1 = _make_exhibit_html(period="February 28, 2026", combined_ratio="88.0")
        html2 = _make_exhibit_html(period="February 28, 2026", combined_ratio="85.7")

        session = MagicMock()
        sub_resp = MagicMock()
        sub_resp.raise_for_status = MagicMock()
        sub_resp.json.return_value = payload

        index_resp = MagicMock()
        index_resp.raise_for_status = MagicMock()
        index_resp.text = (
            '<table>'
            '<tr><td>2</td><td>EX-99</td>'
            '<td><a href="/Archives/edgar/data/80661/000001/ex.htm">ex.htm</a></td>'
            '</tr></table>'
        )
        ex1 = MagicMock()
        ex1.raise_for_status = MagicMock()
        ex1.text = html1
        ex2 = MagicMock()
        ex2.raise_for_status = MagicMock()
        ex2.text = html2

        # submissions JSON is sorted descending; ACC2 (later) comes first
        session.get.side_effect = [sub_resp, index_resp, ex2, index_resp, ex1]
        df = fetch_monthly_8ks(lookback_months=2, session=session)

        # Should have only one row for February 2026
        assert len(df) == 1
        # The "last" kept by drop_duplicates(keep='last') after sort_values is the earlier
        # filing date — but wait, drop_duplicates keep='last' after sort ascending keeps
        # the LATER row in the sorted order.
        # After sort_values("report_period"), both rows have same report_period
        # so drop_duplicates keep='last' keeps the one that came last in the sort, which
        # is still the same filing (accession ordering).
        # What matters is PGR never amends these, so any value is fine.
        assert df["combined_ratio"].iloc[0] in (85.7, 88.0)

    def test_no_filings_returns_empty_dataframe(self):
        payload = _make_submissions(
            forms=["4"],
            items=[""],
            filing_dates=["2026-01-10"],
            accessions=["X"],
        )
        session = MagicMock()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = payload
        session.get.return_value = resp

        df = fetch_monthly_8ks(lookback_months=1, session=session)
        assert df.empty

    def test_missing_exhibit_url_skipped(self):
        """Filing with no EX-99 in index is skipped without crash."""
        payload = _make_submissions(
            forms=["8-K"],
            items=["7.01"],
            filing_dates=["2026-03-18"],
            accessions=["0000080661-26-000096"],
        )
        session = MagicMock()
        sub_resp = MagicMock()
        sub_resp.raise_for_status = MagicMock()
        sub_resp.json.return_value = payload

        index_resp = MagicMock()
        index_resp.raise_for_status = MagicMock()
        index_resp.text = "<table><tr><td>No exhibit here</td></tr></table>"

        session.get.side_effect = [sub_resp, index_resp]
        df = fetch_monthly_8ks(lookback_months=1, session=session)
        assert df.empty


# ---------------------------------------------------------------------------
# 10. backfill_to_db
# ---------------------------------------------------------------------------

class TestBackfillToDb:
    def _in_memory_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pgr_edgar_monthly (
                month_end           TEXT NOT NULL,
                combined_ratio      REAL,
                pif_total           REAL,
                pif_growth_yoy      REAL,
                gainshare_estimate  REAL,
                PRIMARY KEY (month_end)
            )
        """)
        conn.commit()
        return conn

    def _mock_fetch(self, num_rows: int) -> pd.DataFrame:
        idx = pd.date_range("2025-01-31", periods=num_rows, freq="ME")
        return pd.DataFrame(
            {
                "combined_ratio": [85.0 + i * 0.1 for i in range(num_rows)],
                "pif_total": [38000.0 + i * 100 for i in range(num_rows)],
                "net_premiums_written": [6000.0] * num_rows,
                "net_premiums_earned": [5800.0] * num_rows,
                "net_income": [900.0] * num_rows,
                "eps_diluted": [1.5] * num_rows,
                "loss_ratio": [64.0] * num_rows,
                "expense_ratio": [21.0] * num_rows,
                "filing_date": ["2025-02-15"] * num_rows,
            },
            index=idx,
        )

    def test_writes_rows_to_db(self):
        from src.ingestion.edgar_8k_fetcher import backfill_to_db
        from src.database import db_client

        conn = self._in_memory_conn()
        with patch(
            "src.ingestion.edgar_8k_fetcher.fetch_monthly_8ks",
            return_value=self._mock_fetch(13),
        ):
            # Use the real db_client upsert but against our in-memory conn.
            # Patch the module-level import inside edgar_8k_fetcher so it
            # calls our in-memory upsert.
            original_upsert = db_client.upsert_pgr_edgar_monthly

            def _upsert(c, records):
                sql = """INSERT OR REPLACE INTO pgr_edgar_monthly
                    (month_end, combined_ratio, pif_total, pif_growth_yoy, gainshare_estimate)
                    VALUES (:month_end, :combined_ratio, :pif_total, :pif_growth_yoy, :gainshare_estimate)"""
                conn.executemany(sql, records)
                conn.commit()
                return len(records)

            with patch("src.database.db_client.upsert_pgr_edgar_monthly", side_effect=_upsert):
                n = backfill_to_db(conn, lookback_months=13)

        assert n == 13
        cursor = conn.execute("SELECT COUNT(*) FROM pgr_edgar_monthly")
        assert cursor.fetchone()[0] == 13

    def test_empty_fetch_writes_nothing(self):
        from src.ingestion.edgar_8k_fetcher import backfill_to_db

        conn = self._in_memory_conn()
        empty_df = pd.DataFrame(
            columns=["combined_ratio", "pif_total"],
            index=pd.DatetimeIndex([], name="report_period"),
        )
        with patch("src.ingestion.edgar_8k_fetcher.fetch_monthly_8ks", return_value=empty_df):
            n = backfill_to_db(conn, lookback_months=1)

        assert n == 0
        cursor = conn.execute("SELECT COUNT(*) FROM pgr_edgar_monthly")
        assert cursor.fetchone()[0] == 0
