"""
Tests for src/ingestion/edgar_client.py.

Validates:
  - _extract_flow_concept: quarter filtering, Q4 = FY - 9M, earliest-filed
    values, empty fallback
  - _extract_instant_concept: correct instant item handling
  - fetch_pgr_fundamentals_quarterly: field mapping, TTM / average-equity ROE,
    output schema matches pgr_fundamentals_quarterly DB columns

All tests use synthetic in-memory companyfacts dicts; no HTTP calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import config
from src.ingestion.edgar_client import (
    _extract_flow_concept,
    _extract_instant_concept,
    fetch_pgr_fundamentals_quarterly,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic companyfacts structures
# ---------------------------------------------------------------------------

def _make_facts(taxonomy: str, concept: str, unit: str, records: list) -> dict:
    """Build a minimal companyfacts dict for one concept."""
    return {
        "facts": {
            taxonomy: {
                concept: {
                    "units": {
                        unit: records
                    }
                }
            }
        }
    }


# Single 10-Q quarterly revenue record (3-month period).
_Q1_REVENUE = {
    "start": "2023-01-01",
    "end": "2023-03-31",
    "val": 15_000_000_000,
    "accn": "0000080661-23-000001",
    "filed": "2023-05-01",
    "form": "10-Q",
    "fp": "Q1",
}

# 10-K full-year revenue record.
_FY_REVENUE = {
    "start": "2022-01-01",
    "end": "2022-12-31",
    "val": 55_000_000_000,
    "accn": "0000080661-23-000002",
    "filed": "2023-02-15",
    "form": "10-K",
    "fp": "FY",
}

# YTD 6-month revenue record (Q2, cumulative) — should be excluded.
_Q2_YTD_REVENUE = {
    "start": "2023-01-01",
    "end": "2023-06-30",
    "val": 30_500_000_000,
    "accn": "0000080661-23-000003",
    "filed": "2023-08-01",
    "form": "10-Q",
    "fp": "Q2",
}

# Amended Q1 10-Q: same period, filed later — the original (first-reported)
# value is kept.
_Q1_REVENUE_AMENDED = {
    "start": "2023-01-01",
    "end": "2023-03-31",
    "val": 15_100_000_000,  # restated value
    "accn": "0000080661-23-000099",
    "filed": "2023-06-15",   # filed after original
    "form": "10-Q",
    "fp": "Q1",
}

# Equity (instant balance-sheet item — no "start" key).
_Q1_EQUITY = {
    "end": "2023-03-31",
    "val": 20_000_000_000,
    "accn": "0000080661-23-000001",
    "filed": "2023-05-01",
    "form": "10-Q",
    "fp": "Q1",
}

# EPS record.
_Q1_EPS = {
    "start": "2023-01-01",
    "end": "2023-03-31",
    "val": 3.25,
    "accn": "0000080661-23-000001",
    "filed": "2023-05-01",
    "form": "10-Q",
    "fp": "Q1",
}


# ---------------------------------------------------------------------------
# Tests: _extract_flow_concept
# ---------------------------------------------------------------------------

class TestExtractFlowConcept:
    def test_single_quarter_extracted(self):
        """A standard 10-Q Q1 record with ~90-day period is kept."""
        facts = _make_facts("us-gaap", "Revenues", "USD", [_Q1_REVENUE])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert not series.empty
        assert "2023-03-31" in series.index
        assert series["2023-03-31"] == pytest.approx(15_000_000_000)

    def test_ytd_period_excluded(self):
        """A 6-month YTD 10-Q record (180 days) must be excluded."""
        facts = _make_facts("us-gaap", "Revenues", "USD", [_Q2_YTD_REVENUE])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert series.empty

    def test_annual_10k_without_nine_months_is_dropped(self):
        """A full-year fact is never stored as a quarter (F09)."""
        facts = _make_facts("us-gaap", "Revenues", "USD", [_FY_REVENUE])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert series.empty

    def test_q4_is_full_year_less_nine_months(self):
        """Q4 = 10-K full year − 10-Q nine-month YTD with the same start (F09)."""
        nine_months = {
            "start": "2022-01-01", "end": "2022-09-30", "val": 40_000_000_000,
            "filed": "2022-11-01", "form": "10-Q", "accn": "q3",
        }
        facts = _make_facts("us-gaap", "Revenues", "USD", [_FY_REVENUE, nine_months])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert list(series.index) == ["2022-12-31"]
        assert series["2022-12-31"] == pytest.approx(15_000_000_000)

    def test_earliest_filed_value_kept(self):
        """When the same period appears twice, the first-filed value is kept."""
        facts = _make_facts(
            "us-gaap", "Revenues", "USD", [_Q1_REVENUE_AMENDED, _Q1_REVENUE]
        )
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert len(series) == 1
        assert series["2023-03-31"] == pytest.approx(15_000_000_000)

    def test_missing_concept_returns_empty(self):
        """If the concept is absent from the facts dict, return an empty Series."""
        facts = {"facts": {"us-gaap": {}}}
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert series.empty

    def test_non_10q_10k_forms_excluded(self):
        """Records with form='8-K' or form='DEF 14A' must be excluded."""
        rec = {**_Q1_REVENUE, "form": "8-K"}
        facts = _make_facts("us-gaap", "Revenues", "USD", [rec])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert series.empty

    def test_series_name_is_concept(self):
        facts = _make_facts("us-gaap", "Revenues", "USD", [_Q1_REVENUE])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert series.name == "Revenues"


# ---------------------------------------------------------------------------
# Tests: _extract_instant_concept
# ---------------------------------------------------------------------------

class TestExtractInstantConcept:
    def test_instant_item_extracted(self):
        """Balance-sheet instant items (no 'start') are extracted correctly."""
        facts = _make_facts("us-gaap", "StockholdersEquity", "USD", [_Q1_EQUITY])
        series = _extract_instant_concept(facts, "us-gaap", "StockholdersEquity")
        assert not series.empty
        assert "2023-03-31" in series.index
        assert series["2023-03-31"] == pytest.approx(20_000_000_000)

    def test_missing_concept_returns_empty(self):
        facts = {"facts": {"us-gaap": {}}}
        series = _extract_instant_concept(facts, "us-gaap", "StockholdersEquity")
        assert series.empty

    def test_non_quarterly_forms_excluded(self):
        rec = {**_Q1_EQUITY, "form": "8-K"}
        facts = _make_facts("us-gaap", "StockholdersEquity", "USD", [rec])
        series = _extract_instant_concept(facts, "us-gaap", "StockholdersEquity")
        assert series.empty


# ---------------------------------------------------------------------------
# Tests: fetch_pgr_fundamentals_quarterly (with mocked companyfacts)
# ---------------------------------------------------------------------------

def _make_full_facts() -> dict:
    """Build a multi-concept companyfacts dict for integration tests."""
    net_income_rec = {
        "start": "2023-01-01",
        "end": "2023-03-31",
        "val": 1_500_000_000,
        "accn": "0000080661-23-000001",
        "filed": "2023-05-01",
        "form": "10-Q",
    }
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [_Q1_REVENUE]}
                },
                "NetIncomeLoss": {
                    "units": {"USD": [net_income_rec]}
                },
                "EarningsPerShareBasic": {
                    "units": {"USD/shares": [_Q1_EPS]}
                },
                "StockholdersEquity": {
                    "units": {"USD": [_Q1_EQUITY]}
                },
            }
        }
    }


class TestFetchPGRFundamentalsQuarterly:
    def test_returns_list_of_dicts(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        records = fetch_pgr_fundamentals_quarterly()
        assert isinstance(records, list)
        assert len(records) == 1

    def test_required_db_keys_present(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        expected_keys = {
            "period_end", "roe", "eps", "revenue", "net_income",
            "filing_date", "source",
        }
        assert set(rec.keys()) == expected_keys

    def test_period_end_is_iso_string(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert rec["period_end"] == "2023-03-31"

    def test_revenue_and_net_income_correct(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert rec["revenue"] == pytest.approx(15_000_000_000)
        assert rec["net_income"] == pytest.approx(1_500_000_000)

    def test_eps_correct(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert rec["eps"] == pytest.approx(3.25)

    def test_roe_needs_four_quarters(self, monkeypatch):
        """ROE is TTM net income / average equity: one quarter gives None."""
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert rec["roe"] is None

    def test_filing_date_is_net_income_filing(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert rec["filing_date"] == "2023-05-01"

    def test_source_is_edgar(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert rec["source"] == "edgar"

    def test_empty_facts_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: {"facts": {"us-gaap": {}}},
        )
        records = fetch_pgr_fundamentals_quarterly()
        assert records == []

    def test_revenue_fallback_to_premiums(self, monkeypatch):
        """If Revenues is absent, fall back to PremiumsEarnedNet."""
        premiums_rec = {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "val": 14_000_000_000,
            "accn": "0000080661-23-000001",
            "filed": "2023-05-01",
            "form": "10-Q",
        }
        facts = {
            "facts": {
                "us-gaap": {
                    "PremiumsEarnedNet": {
                        "units": {"USD": [premiums_rec]}
                    }
                }
            }
        }
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: facts,
        )
        records = fetch_pgr_fundamentals_quarterly()
        assert len(records) == 1
        assert records[0]["revenue"] == pytest.approx(14_000_000_000)

    def test_equity_fallback_to_attributable(self, monkeypatch):
        """If StockholdersEquity absent, fall back to StockholdersEquityAttributableToParent."""
        facts = _annual_facts()
        us_gaap = facts["facts"]["us-gaap"]
        us_gaap["StockholdersEquityAttributableToParent"] = us_gaap.pop(
            "StockholdersEquity"
        )
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: facts,
        )
        rec = {r["period_end"]: r for r in fetch_pgr_fundamentals_quarterly()}
        assert rec["2018-12-31"]["roe"] == pytest.approx(2615.3e6 / _AVG_EQUITY_2018)

    def test_multiple_quarters_returned(self, monkeypatch):
        """Multiple quarters produce one record per period_end."""
        q2_revenue = {
            "start": "2023-04-01", "end": "2023-06-30",
            "val": 16_000_000_000, "filed": "2023-08-01", "form": "10-Q",
            "accn": "0000080661-23-000004",
        }
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {"USD": [_Q1_REVENUE, q2_revenue]}
                    }
                }
            }
        }
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: facts,
        )
        records = fetch_pgr_fundamentals_quarterly()
        period_ends = [r["period_end"] for r in records]
        assert "2023-03-31" in period_ends
        assert "2023-06-30" in period_ends
        assert len(records) == 2


def _annual_facts() -> dict:
    """PGR FY2018 facts as filed in XBRL (companyfacts), in USD.

    Net income: Q1 718.0M, Q2 704.2M, Q3 928.4M, 6M 1,422.2M, 9M 2,350.6M
    (10-Q 0000080661-18-000050), FY 2,615.3M (10-K 0000080661-19-000008), so
    Q4 = 264.7M.  Basic EPS: Q1 1.23, Q2 1.20, Q3 1.58, 9M 4.01, FY 4.45, so
    Q4 = 0.44.  Equity: 9,284.8M (2017-12-31), 10,323.2M, 11,000.8M,
    11,858.8M, 10,821.8M (2018-12-31).  PGR printed a trailing-12-month ROE of
    24.7 % for December 2018 (8-K 0000080661-19-000003).  The restated 2020
    comparative is illustrative: it checks that the first filing wins.
    """
    def dur(start, end, val, filed, form):
        return {"start": start, "end": end, "val": val, "filed": filed, "form": form}

    ni = [
        dur("2018-01-01", "2018-03-31", 718.0e6, "2018-05-02", "10-Q"),
        dur("2018-04-01", "2018-06-30", 704.2e6, "2018-07-31", "10-Q"),
        dur("2018-07-01", "2018-09-30", 928.4e6, "2018-10-31", "10-Q"),
        dur("2018-01-01", "2018-06-30", 1422.2e6, "2018-07-31", "10-Q"),
        dur("2018-01-01", "2018-09-30", 2350.6e6, "2018-10-31", "10-Q"),
        dur("2018-01-01", "2018-12-31", 2615.3e6, "2019-02-27", "10-K"),
        dur("2018-01-01", "2018-12-31", 2600.0e6, "2020-03-02", "10-K"),
    ]
    eps = [
        dur("2018-01-01", "2018-03-31", 1.23, "2018-05-02", "10-Q"),
        dur("2018-04-01", "2018-06-30", 1.20, "2018-07-31", "10-Q"),
        dur("2018-07-01", "2018-09-30", 1.58, "2018-10-31", "10-Q"),
        dur("2018-01-01", "2018-09-30", 4.01, "2018-10-31", "10-Q"),
        dur("2018-01-01", "2018-12-31", 4.45, "2019-02-27", "10-K"),
    ]
    equity = [
        {"end": "2017-12-31", "val": 9284.8e6, "filed": "2018-02-27", "form": "10-K"},
        {"end": "2018-03-31", "val": 10323.2e6, "filed": "2018-05-02", "form": "10-Q"},
        {"end": "2018-06-30", "val": 11000.8e6, "filed": "2018-07-31", "form": "10-Q"},
        {"end": "2018-09-30", "val": 11858.8e6, "filed": "2018-10-31", "form": "10-Q"},
        {"end": "2018-12-31", "val": 10821.8e6, "filed": "2019-02-27", "form": "10-K"},
    ]
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": ni}},
                "EarningsPerShareBasic": {"units": {"USD/shares": eps}},
                "StockholdersEquity": {"units": {"USD": equity}},
            }
        }
    }


_AVG_EQUITY_2018 = (9284.8 + 10323.2 + 11000.8 + 11858.8 + 10821.8) / 5 * 1e6


class TestAnnualFactsF09:
    """F09: 10-K annual + 10-Q YTD facts give discrete quarters and sane ROE."""

    def _records(self, monkeypatch) -> dict:
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _annual_facts(),
        )
        return {r["period_end"]: r for r in fetch_pgr_fundamentals_quarterly()}

    def test_q4_eps_is_full_year_less_nine_months(self, monkeypatch):
        rec = self._records(monkeypatch)
        assert rec["2018-12-31"]["eps"] == pytest.approx(4.45 - 4.01)
        assert rec["2018-12-31"]["net_income"] == pytest.approx((2615.3 - 2350.6) * 1e6)

    def test_q4_uses_first_filed_annual_value(self, monkeypatch):
        rec = self._records(monkeypatch)
        assert rec["2018-12-31"]["filing_date"] == "2019-02-27"

    def test_ytd_facts_never_stored_as_quarters(self, monkeypatch):
        rec = self._records(monkeypatch)
        assert sorted(rec) == ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31"]
        assert rec["2018-06-30"]["net_income"] == pytest.approx(704.2e6)

    def test_roe_is_ttm_over_average_equity(self, monkeypatch):
        rec = self._records(monkeypatch)
        roe = rec["2018-12-31"]["roe"]
        assert roe == pytest.approx(2615.3e6 / _AVG_EQUITY_2018)
        assert roe == pytest.approx(0.247, abs=0.005)  # PGR printed 24.7 %
        for r in rec.values():
            assert r["roe"] is None or -0.5 <= r["roe"] <= 0.6


class TestEdgarHeaders:
    def test_build_edgar_headers_uses_env_override(self, monkeypatch):
        monkeypatch.setenv("EDGAR_USER_AGENT", "Unit Test qa@example.com")
        headers = config.build_edgar_headers("data.sec.gov")
        assert headers["User-Agent"] == "Unit Test qa@example.com"
        assert headers["Host"] == "data.sec.gov"

    def test_build_edgar_headers_falls_back_to_generic_value(self, monkeypatch):
        monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
        headers = config.build_edgar_headers()
        assert headers["User-Agent"] == config.EDGAR_USER_AGENT_FALLBACK
        assert "Host" not in headers

    def test_fetch_companyfacts_uses_configured_edgar_user_agent(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("src.ingestion.edgar_client._cache_path", lambda: str(tmp_path / "companyfacts.json"))
        monkeypatch.setattr("src.ingestion.edgar_client._is_cache_valid", lambda *args, **kwargs: False)
        monkeypatch.setenv("EDGAR_USER_AGENT", "Header Test test@example.com")

        mock_response = MagicMock()
        mock_response.json.return_value = {"facts": {}}
        mock_response.raise_for_status = MagicMock()

        captured: dict[str, object] = {}

        def _mock_get(url, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            return mock_response

        mock_session = MagicMock()
        mock_session.get.side_effect = _mock_get
        monkeypatch.setattr("src.ingestion.edgar_client.build_retry_session", lambda: mock_session)

        from src.ingestion.edgar_client import fetch_companyfacts

        assert fetch_companyfacts(force_refresh=True) == {"facts": {}}
        assert captured["headers"]["User-Agent"] == "Header Test test@example.com"
