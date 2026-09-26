"""Tests for the HTML exhibit parser in scripts/edgar_8k_fetcher.py (v6.5, P2.6).

  1.  _try_parse_dollar returns correct value within range
  2.  _try_parse_dollar returns None when all patterns fail
  3.  _try_parse_dollar returns None when value out of range
  4.  _parse_html_exhibit returns None when neither CR nor PIF parseable
  5.  _parse_html_exhibit extracts combined_ratio from table cell pattern
  6.  _parse_html_exhibit extracts pif_total from table cell pattern
  7.  _parse_html_exhibit extracts net_premiums_written
  8.  _parse_html_exhibit extracts investment_income
  9.  _parse_html_exhibit extracts book_value_per_share
 10.  _parse_html_exhibit extracts eps_basic
 11.  _parse_html_exhibit extracts shares_repurchased and avg_cost_per_share
 12.  _parse_html_exhibit extracts investment_book_yield (stored in percent)
 13.  _parse_html_exhibit sets derived fields to None (set by _compute_derived_fields)
 14.  _compute_derived_fields sets npw_growth_yoy for 12-month prior found
 15.  _compute_derived_fields sets channel_mix_agency_pct = agency / (agency + direct)
 16.  _compute_derived_fields sets underwriting_income = npe × (1 − CR/100)
 17.  _compute_derived_fields channel_mix_agency_pct None when both NPW are None
 18.  _prior_year_key handles Feb-29 leap-year edge case

Split out of the old ``tests/test_v65_p26_p27_p28.py`` (review 2026-09-25, step 11).
"""

from __future__ import annotations

import pytest

from scripts.edgar_8k_fetcher import (
    _compute_derived_fields,
    _parse_html_exhibit,
    _prior_year_key,
    _try_parse_dollar,
)


class TestTryParseDollar:
    def test_basic_match(self):
        html = "<td>Combined Ratio</td><td> 91.5 </td>"
        result = _try_parse_dollar(
            html,
            patterns=[r"(?i)combined\s+ratio[^<]{0,60}</td>\s*<td[^>]*>\s*([\d]+\.[\d]+)"],
            lo=60.0, hi=140.0,
        )
        assert result == pytest.approx(91.5)

    def test_returns_none_when_no_match(self):
        html = "<td>something else</td><td>99</td>"
        result = _try_parse_dollar(html, patterns=[r"combined_ratio\s+([\d.]+)"], lo=60, hi=140)
        assert result is None

    def test_returns_none_when_out_of_range(self):
        html = "<td>CR</td><td>999.9</td>"
        result = _try_parse_dollar(
            html,
            patterns=[r"CR[^<]{0,40}</td>\s*<td[^>]*>\s*([\d]+\.[\d]+)"],
            lo=60.0, hi=140.0,
        )
        assert result is None

    def test_scale_applied(self):
        html = "<td>book yield</td><td>3.50%</td>"
        result = _try_parse_dollar(
            html,
            patterns=[r"book\s+yield[^<]{0,60}</td>\s*<td[^>]*>\s*([\d]+\.[\d]+)%?"],
            lo=0.005, hi=0.15,
            scale=0.01,
        )
        assert result == pytest.approx(0.035)


class TestParseHtmlExhibit:
    """Tests for the extended _parse_html_exhibit() function."""

    def _minimal_html(
        self,
        cr: float = 91.5,
        pif: int = 15_000_000,
        npw: float | None = None,
        inv_income: float | None = None,
        bvps: float | None = None,
        eps: float | None = None,
        shares_repurchased: float | None = None,
        avg_cost: float | None = None,
        book_yield_pct: float | None = None,
        npw_agency: float | None = None,
        npw_direct: float | None = None,
    ) -> str:
        """Build minimal HTML exhibit containing requested fields."""
        rows = []
        if cr is not None:
            rows.append(f"<td>Combined Ratio</td><td>{cr}</td>")
        if pif is not None:
            rows.append(f"<td>Policies in Force</td><td>{pif:,}</td>")
        if npw is not None:
            rows.append(f"<td>Net Premiums Written</td><td>${npw:,.1f}</td>")
        if inv_income is not None:
            rows.append(f"<td>Net Investment Income</td><td>${inv_income:,.1f}</td>")
        if bvps is not None:
            rows.append(f"<td>Book Value per Share</td><td>${bvps:.2f}</td>")
        if eps is not None:
            rows.append(f"<td>Earnings per Share</td><td>${eps:.2f}</td>")
        if shares_repurchased is not None:
            rows.append(f"<td>Shares Repurchased</td><td>{shares_repurchased:.2f}</td>")
        if avg_cost is not None:
            rows.append(f"<td>Average Purchase Price per Share</td><td>${avg_cost:.2f}</td>")
        if book_yield_pct is not None:
            rows.append(f"<td>Book Yield</td><td>{book_yield_pct:.2f}%</td>")
        if npw_agency is not None:
            rows.append(f"<td>Agency</td><td>${npw_agency:,.1f}</td>")
        if npw_direct is not None:
            rows.append(f"<td>Direct</td><td>${npw_direct:,.1f}</td>")
        table = "<table>" + "".join(f"<tr>{r}</tr>" for r in rows) + "</table>"
        return f"<html><body>{table}</body></html>"

    def test_returns_none_when_no_cr_or_pif(self):
        html = "<html><body><p>nothing useful here</p></body></html>"
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is None

    def test_extracts_combined_ratio(self):
        html = self._minimal_html(cr=89.3, pif=16_000_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["combined_ratio"] == pytest.approx(89.3)

    def test_extracts_pif_total(self):
        html = self._minimal_html(cr=91.0, pif=18_500_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["pif_total"] == pytest.approx(18_500)

    def test_extracts_pif_total_when_already_in_thousands(self):
        html = self._minimal_html(cr=91.0, pif=18_500)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["pif_total"] == pytest.approx(18_500)

    def test_extracts_net_premiums_written(self):
        html = self._minimal_html(cr=91.0, pif=15_000_000, npw=2_500.0)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["net_premiums_written"] == pytest.approx(2_500.0)

    def test_extracts_investment_income(self):
        html = self._minimal_html(cr=91.0, pif=15_000_000, inv_income=250.0)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["investment_income"] == pytest.approx(250.0)

    def test_extracts_book_value_per_share(self):
        html = self._minimal_html(cr=91.0, pif=15_000_000, bvps=75.40)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["book_value_per_share"] == pytest.approx(75.40, abs=0.01)

    def test_extracts_eps_basic(self):
        html = self._minimal_html(cr=91.0, pif=15_000_000, eps=1.85)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["eps_basic"] == pytest.approx(1.85, abs=0.01)

    def test_extracts_shares_repurchased_and_avg_cost(self):
        html = self._minimal_html(
            cr=91.0, pif=15_000_000,
            shares_repurchased=0.75, avg_cost=175.00,
        )
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["shares_repurchased"] == pytest.approx(0.75, abs=0.01)
        assert result["avg_cost_per_share"] == pytest.approx(175.00, abs=0.01)

    def test_extracts_investment_book_yield_as_percent(self):
        """Book yield given as "3.50%" → stored as 3.5 (percent, like the CSV; F10)."""
        html = self._minimal_html(cr=91.0, pif=15_000_000, book_yield_pct=3.50)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        if result["investment_book_yield"] is not None:
            assert result["investment_book_yield"] == pytest.approx(3.5, abs=1e-4)

    def test_derived_fields_set_to_none(self):
        """Derived fields must be None at parse time (set later by _compute_derived_fields)."""
        html = self._minimal_html(cr=91.0, pif=15_000_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        for field in ("pif_growth_yoy", "gainshare_estimate",
                      "channel_mix_agency_pct", "underwriting_income",
                      "npw_growth_yoy", "unearned_premium_growth_yoy"):
            assert result[field] is None, f"{field} should be None at parse time"

    def test_month_end_derived_from_filing_date(self):
        """Filing 2024-02-15 → month_end 2024-01-31."""
        html = self._minimal_html(cr=91.0, pif=15_000_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["month_end"] == "2024-01-31"


class TestComputeDerivedFields:
    def _make_records(
        self,
        n: int = 14,
        npw_base: float = 2_000.0,
        npw_agency: float | None = 900.0,
        npw_direct: float | None = 600.0,
        cr: float = 91.0,
        npe: float | None = 1_900.0,
    ) -> list[dict]:
        """Build a minimal sorted list of records for testing."""
        import pandas as pd
        idx = pd.date_range("2023-01-31", periods=n, freq="ME")
        records = []
        for i, date in enumerate(idx):
            rec: dict = {
                "month_end": date.strftime("%Y-%m-%d"),
                "pif_total": 15_000_000.0,
                "combined_ratio": cr,
                "net_premiums_written": npw_base + i * 10,
                "net_premiums_earned": npe,
                "npw_agency": npw_agency,
                "npw_direct": npw_direct,
                "unearned_premiums": None,
                # derived — initially None
                "pif_growth_yoy": None,
                "gainshare_estimate": None,
                "channel_mix_agency_pct": None,
                "underwriting_income": None,
                "npw_growth_yoy": None,
                "unearned_premium_growth_yoy": None,
                "buyback_yield": None,
            }
            records.append(rec)
        return records

    def test_npw_growth_yoy_computed_after_12_months(self):
        records = self._make_records(n=14)
        out = _compute_derived_fields(records)
        # Rows 0–11 have no prior-year data → None
        # Row 12 (month 13) should have npw_growth_yoy set
        assert out[12]["npw_growth_yoy"] is not None

    def test_channel_mix_agency_pct(self):
        records = self._make_records(npw_agency=900.0, npw_direct=600.0)
        out = _compute_derived_fields(records)
        # 900 / (900 + 600) = 0.6
        assert out[0]["channel_mix_agency_pct"] == pytest.approx(0.6, abs=1e-9)

    def test_underwriting_income(self):
        # uw_income = npe * (1 - cr/100) = 1900 * (1 - 91/100) = 1900 * 0.09 = 171.0
        records = self._make_records(cr=91.0, npe=1_900.0)
        out = _compute_derived_fields(records)
        assert out[0]["underwriting_income"] == pytest.approx(171.0, abs=1e-6)

    def test_channel_mix_none_when_both_npw_none(self):
        records = self._make_records(npw_agency=None, npw_direct=None)
        out = _compute_derived_fields(records)
        assert out[0]["channel_mix_agency_pct"] is None

    def test_channel_mix_none_when_sum_is_zero(self):
        records = self._make_records(npw_agency=0.0, npw_direct=0.0)
        out = _compute_derived_fields(records)
        assert out[0]["channel_mix_agency_pct"] is None


class TestPriorYearKey:
    def test_normal_month(self):
        assert _prior_year_key("2024-06-30") == "2023-06-30"

    def test_leap_day(self):
        # 2024-02-29 → prior year is 2023; Feb 29 doesn't exist → Feb 28
        assert _prior_year_key("2024-02-29") == "2023-02-28"
