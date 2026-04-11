"""
v6.5 tests — P2.6 / P2.7 / P2.8.

P2.6 — edgar_8k_fetcher.py HTML parser extension:
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
 12.  _parse_html_exhibit extracts investment_book_yield (percent → decimal)
 13.  _parse_html_exhibit sets derived fields to None (set by _compute_derived_fields)
 14.  _compute_derived_fields sets npw_growth_yoy for 12-month prior found
 15.  _compute_derived_fields sets channel_mix_agency_pct = agency / (agency + direct)
 16.  _compute_derived_fields sets underwriting_income = npe × (1 − CR/100)
 17.  _compute_derived_fields channel_mix_agency_pct None when both NPW are None
 18.  _prior_year_key handles Feb-29 leap-year edge case

P2.7 — calibration reliability diagram:
 19.  _plot_calibration_curve returns None when fewer than 4 observations
 20.  _plot_calibration_curve returns None when method is "uncalibrated"
 21.  _plot_calibration_curve writes a PNG file when data is sufficient

P2.8 — email_sender module:
 22.  build_email_message extracts signal from body
 23.  build_email_message sets correct From / To / Subject headers
 24.  build_email_message defaults month_label to current UTC month
 25.  send_monthly_email dry_run returns subject without SMTP call
 26.  send_monthly_email raises FileNotFoundError when report missing
 27.  send_monthly_email raises ValueError when SMTP config missing
 28.  send_monthly_email calls SMTP_SSL for port 465 with correct args
 29.  send_monthly_email calls STARTTLS for port 587
"""

from __future__ import annotations

import os
import re
import smtplib
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ===========================================================================
# P2.6 — HTML parser extension (edgar_8k_fetcher)
# ===========================================================================

# Import functions directly from scripts package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from edgar_8k_fetcher import (
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

    def test_extracts_investment_book_yield_as_decimal(self):
        """Book yield given as "3.50%" → stored as 0.035."""
        html = self._minimal_html(cr=91.0, pif=15_000_000, book_yield_pct=3.50)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        if result["investment_book_yield"] is not None:
            assert result["investment_book_yield"] == pytest.approx(0.035, abs=1e-4)

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


# ===========================================================================
# P2.7 — Calibration reliability diagram
# ===========================================================================

class TestPlotCalibrationCurve:
    """Tests for _plot_calibration_curve in monthly_decision.py."""

    def _import(self):
        """Import _plot_calibration_curve lazily (avoids matplotlib import at collection time)."""
        import importlib
        import sys
        # Ensure scripts/ is on sys.path
        scripts_dir = str(Path(__file__).parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        # monthly_decision imports config and src modules; mock heavy dependencies
        with patch.dict("os.environ", {"AV_API_KEY": "test", "FMP_API_KEY": "test",
                                        "FRED_API_KEY": "test"}):
            mod = importlib.import_module("monthly_decision")
        return mod._plot_calibration_curve

    def test_returns_none_when_fewer_than_4_obs(self, tmp_path):
        plot_fn = self._import()
        from src.models.calibration import CalibrationResult
        cal = CalibrationResult(n_obs=3, method="platt", ece=0.05,
                                ece_ci_lower=0.02, ece_ci_upper=0.10)
        result = plot_fn(tmp_path, np.array([0.6, 0.7, 0.8]), np.array([1, 0, 1]), cal)
        assert result is None

    def test_returns_none_when_uncalibrated(self, tmp_path):
        plot_fn = self._import()
        from src.models.calibration import CalibrationResult
        cal = CalibrationResult(n_obs=50, method="uncalibrated", ece=0.05,
                                ece_ci_lower=0.02, ece_ci_upper=0.10)
        probs = np.random.default_rng(42).uniform(0.3, 0.8, 50)
        outcomes = (probs > 0.5).astype(int)
        result = plot_fn(tmp_path, probs, outcomes, cal)
        assert result is None

    def test_writes_png_when_sufficient_data(self, tmp_path):
        plot_fn = self._import()
        from src.models.calibration import CalibrationResult
        cal = CalibrationResult(n_obs=100, method="platt", ece=0.03,
                                ece_ci_lower=0.01, ece_ci_upper=0.07)
        rng = np.random.default_rng(0)
        probs = rng.uniform(0.2, 0.9, 100)
        outcomes = (probs > 0.5).astype(int)
        save_path = plot_fn(tmp_path, probs, outcomes, cal)
        assert save_path is not None
        assert Path(save_path).exists()
        assert Path(save_path).suffix == ".png"


# ===========================================================================
# P2.8 — email_sender module
# ===========================================================================

from src.reporting.email_sender import build_email_message, send_monthly_email


class TestBuildEmailMessage:
    _SAMPLE_BODY = textwrap.dedent("""\
        # PGR Monthly Decision — April 2026

        **As-of Date:** 2026-04-03  
        **Recommendation Layer:** v13.1 promoted simpler diversification-first recommendation layer + live-stack cross-check  

        ## Executive Summary

        - What changed since last month: Outlook weakened.
        - Current model view: PGR is projected to outperform the benchmark set by +6.0% over the next 6 months. Recommendation mode remains ACTIONABLE.
        - What to do at the next vest: Default 50% sale.

        ---

        ## Consensus Signal

        | Field | Value |
        |-------|-------|
        | Signal | **OUTPERFORM (HIGH CONFIDENCE)** |
        | Recommendation Mode | **ACTIONABLE** |
        | Recommended Sell % | **20%** |
        | Predicted 6M Relative Return | +6.00% |
        | P(Outperform, calibrated) | 66.0% |
        | Mean IC (across benchmarks) | 0.0800 |
        | Mean Hit Rate | 56.0% |
        | Aggregate OOS R^2 | -20.00% |

        ---

        ## Confidence Snapshot

        - 2/4 core gates pass. The signal may still be directionally interesting, but the quality gate remains too weak for a prediction-led vest action.

        | Check | Current | Threshold | Status | Meaning |
        |-------|---------|-----------|--------|---------|
        | Mean IC | 0.0800 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
        | Mean hit rate | 56.0% | >= 55.0% | **PASS** | Directional accuracy versus zero. |
        | Aggregate OOS R^2 | -20.00% | >= 2.00% | **FAIL** | Calibration / fit versus a naive benchmark. |
        | Representative CPCV | FAIL | not FAIL | **FAIL** | Stability across purged cross-validation paths. |

        ## Next Vest Decision

        | Field | Value |
        |-------|-------|
        | Recommendation mode | **ACTIONABLE** |
        | Next vest date | 2026-07-17 |
        | RSU type | performance |
        | Current PGR price | $198.84 |
        | Current in-scope shares | 8.00 |
        | Average cost basis used | $133.38 |
        | Suggested default vest action | Sell 20% of the vesting tranche |

        | Scenario | Timing | Tax Rate | Predicted Return | Probability | Use when |
        |----------|--------|----------|------------------|-------------|----------|
        | Sell at vest (STCG) | 2026-07-17 | 37% | +0.00% | 100.0% | Use the default diversification / tax-discipline rule or when the model edge is weak. |
        | Hold to LTCG date | 2027-07-18 | 20% | +8.36% | 67.4% | Use only when the edge is strong enough to justify waiting for lower long-term tax treatment. |

        ## Existing Holdings Guidance

        - LOSS: 2025-01-21 @ $240.00 (1.00 share(s)). Trim loss lots first when reducing concentration.
        - LTCG: 2024-01-01 @ $161.00 (1.00 share(s)). After losses, trim LTCG gain lots next.

        ## Redeploy Guidance

        - Broad US Equity: VOO. Broad US equity diversifies away from single-stock risk.
        - Fixed Income: BND, VMBS. Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.

        ## Suggested Redeploy Portfolio

        - Default posture: `95%` equities / `5%` bonds across the curated investable universe.

        | Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
        |------|------------|--------|--------------------|-----------------|-----------------|------------------------|
        | VOO | 40% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.14 | Preferred this month (-4.0%) | 60.0% |
        | VGT | 21% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.36 | Highest-conviction buy (-6.0%) | 63.0% |
        | SCHD | 14% | Value / dividend tilt | Closest current project proxy for a value sleeve. | 0.29 | Supportive (-2.0%) | 53.0% |
        | VXUS | 11% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.28 | Preferred this month (-3.0%) | 57.0% |
        | VWO | 9% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.30 | Keep near base (-1.0%) | 52.0% |
        | BND | 5% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.04 | Keep near base (+1.0%) | 47.0% |

        ## Simple-Baseline Cross-Check

        | Path | Candidate | Policy | Signal | Recommendation Mode | Sell % | Predicted 6M Return | Aggregate OOS R^2 |
        |------|-----------|--------|--------|---------------------|--------|---------------------|------------------|
        | Live production | `production_4_model_ensemble` | `current_production_mapping` | OUTPERFORM | **ACTIONABLE** | **20%** | +6.00% | -20.00% |
        | Simpler baseline | `baseline_historical_mean` | `neutral_band_3pct` | OUTPERFORM | **ACTIONABLE** | **20%** | +5.00% | -10.00% |

        ## Per-Benchmark Signals

        - Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
        - Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

        | Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
        |-----------|----------------|-------------|------------------|----------|----------|----|----------|--------|--------|------------|--------|
        | VTI | Forecast only | Total Stock Market | +7.0% | -29.1% | +43.1% | 0.0407 | 51.6% | 70.9% | 59.3% | HIGH | NEUTRAL |
        | VHT | Forecast only | Health Care | +6.0% | -28.4% | +40.4% | 0.1665 | 58.5% | 68.3% | 66.1% | MODERATE | OUTPERFORM |
    """)

    def _write_lots(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "position_lots.csv"
        csv_path.write_text(
            textwrap.dedent("""\
                vest_date,rsu_type,shares,cost_basis_per_share
                2024-01-01,time,1,161
                2025-01-21,time,1,240
                2026-01-20,time,1,201
            """),
            encoding="utf-8",
        )
        return csv_path

    def test_extracts_signal_in_subject(self):
        msg = build_email_message(self._SAMPLE_BODY, "a@b.com", "c@d.com", "April 2026")
        assert "OUTPERFORM (HIGH CONFIDENCE)" in msg["Subject"]
        assert "April 2026" in msg["Subject"]

    def test_from_to_headers(self):
        msg = build_email_message(self._SAMPLE_BODY, "from@x.com", "to@y.com", "April 2026")
        assert msg["From"] == "from@x.com"
        assert msg["To"] == "to@y.com"

    def test_body_attached_as_plain_text(self, tmp_path):
        lots_path = self._write_lots(tmp_path)
        msg = build_email_message(
            self._SAMPLE_BODY,
            "a@b.com",
            "c@d.com",
            "April 2026",
            lots_csv_path=lots_path,
        )
        payloads = msg.get_payload()
        body = payloads[0].get_payload(decode=True).decode()
        assert "PGR Monthly Decision Summary" in body
        assert "Recommendation mode: ACTIONABLE" in body
        assert "What's changed:" in body
        assert "Existing shares already held:" in body
        assert "Confidence checks:" in body
        assert "Consensus cross-check:" in body
        assert "Active layer: v13.1 promoted simpler diversification-first recommendation layer + live-stack cross-check" in body
        assert "If redeploying sold exposure:" in body
        assert "Suggested redeploy portfolio:" in body
        assert "Full report:" in body

    def test_html_body_attached_and_contains_structured_sections(self, tmp_path):
        lots_path = self._write_lots(tmp_path)
        msg = build_email_message(
            self._SAMPLE_BODY,
            "a@b.com",
            "c@d.com",
            "April 2026",
            lots_csv_path=lots_path,
        )
        payloads = msg.get_payload()
        html_body = payloads[1].get_payload(decode=True).decode()
        assert "<html>" in html_body
        assert "New vested shares" in html_body
        assert "Existing shares already held" in html_body
        assert "Confidence snapshot" in html_body
        assert "Consensus cross-check" in html_body
        assert "Active layer:" in html_body
        assert "If redeploying sold exposure" in html_body
        assert "Suggested redeploy portfolio" in html_body
        assert "Benchmark detail" in html_body
        assert "Loss lots first" in html_body

    def test_unknown_signal_when_no_match(self):
        msg = build_email_message("No signal line here", "a@b.com", "c@d.com", "April 2026")
        assert "UNKNOWN" in msg["Subject"]

    def test_default_month_label_is_current_utc(self):
        expected = datetime.now(timezone.utc).strftime("%B %Y")
        msg = build_email_message("| Signal | **NEUTRAL** |", "a@b.com", "c@d.com")
        assert expected in msg["Subject"]


class TestSendMonthlyEmail:
    _SAMPLE_BODY = "| Signal | **OUTPERFORM (HIGH CONFIDENCE)** |"

    def _write_report(self, tmp_path: Path, body: str) -> Path:
        report = tmp_path / "recommendation.md"
        report.write_text(body, encoding="utf-8")
        return report

    def test_dry_run_returns_subject_no_smtp(self, tmp_path):
        report = self._write_report(tmp_path, self._SAMPLE_BODY)
        subject = send_monthly_email(
            report_path=report,
            from_addr="a@b.com", to_addr="c@d.com",
            month_label="April 2026",
            dry_run=True,
        )
        assert "OUTPERFORM" in subject
        assert "April 2026" in subject

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            send_monthly_email(
                report_path=tmp_path / "nonexistent.md",
                from_addr="a@b.com", to_addr="c@d.com",
                dry_run=True,
            )

    def test_raises_value_error_when_config_missing(self, tmp_path):
        report = self._write_report(tmp_path, self._SAMPLE_BODY)
        with pytest.raises(ValueError, match="missing SMTP configuration"):
            send_monthly_email(
                report_path=report,
                month_label="April 2026",
                # No SMTP config provided; no env-vars set
                dry_run=False,
            )

    def test_smtp_ssl_called_for_port_465(self, tmp_path):
        report = self._write_report(tmp_path, self._SAMPLE_BODY)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP_SSL", return_value=mock_smtp) as smtp_cls:
            send_monthly_email(
                report_path=report,
                smtp_server="smtp.example.com",
                smtp_port=465,
                username="user",
                password="pass",
                from_addr="a@b.com",
                to_addr="c@d.com",
                month_label="April 2026",
            )
            smtp_cls.assert_called_once()
            # login and sendmail must have been called
            mock_smtp.login.assert_called_once_with("user", "pass")
            mock_smtp.sendmail.assert_called_once()
            # Verify To address appears in sendmail args
            call_args = mock_smtp.sendmail.call_args
            assert "c@d.com" in call_args[0][1]

    def test_starttls_called_for_port_587(self, tmp_path):
        report = self._write_report(tmp_path, self._SAMPLE_BODY)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp) as smtp_cls:
            send_monthly_email(
                report_path=report,
                smtp_server="smtp.example.com",
                smtp_port=587,
                username="user",
                password="pass",
                from_addr="a@b.com",
                to_addr="c@d.com",
                month_label="April 2026",
            )
            smtp_cls.assert_called_once()
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("user", "pass")
            mock_smtp.sendmail.assert_called_once()
