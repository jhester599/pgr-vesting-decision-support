"""Tests for src/reporting/email_sender.py (v6.5, P2.8).

 1.  build_email_message extracts signal from body
 2.  build_email_message sets correct From / To / Subject headers
 3.  build_email_message defaults month_label to current UTC month
 4.  send_monthly_email dry_run returns subject without SMTP call
 5.  send_monthly_email raises FileNotFoundError when report missing
 6.  send_monthly_email raises ValueError when SMTP config missing
 7.  send_monthly_email calls SMTP_SSL for port 465 with correct args
 8.  send_monthly_email calls STARTTLS for port 587

Split out of the old ``tests/test_v65_p26_p27_p28.py`` (review 2026-09-25, step 11).
"""

from __future__ import annotations

import smtplib
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
