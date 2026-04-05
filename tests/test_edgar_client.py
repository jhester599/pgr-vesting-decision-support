"""
Tests for src/ingestion/edgar_client.py.

Validates:
  - _extract_flow_concept: correct quarter filtering, deduplication, empty fallback
  - _extract_instant_concept: correct instant item handling
  - fetch_pgr_fundamentals_quarterly: field mapping, ROE derivation, NULL fields,
    output schema matches pgr_fundamentals_quarterly DB columns

All tests use synthetic in-memory companyfacts dicts; no HTTP calls are made.
"""

from __future__ import annotations

import math
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

# Amended Q1 10-Q: same period, filed later — should replace the original.
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

    def test_annual_10k_included(self):
        """A 10-K annual record (~365 days) is kept."""
        facts = _make_facts("us-gaap", "Revenues", "USD", [_FY_REVENUE])
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert not series.empty
        assert "2022-12-31" in series.index
        assert series["2022-12-31"] == pytest.approx(55_000_000_000)

    def test_amendment_replaces_original(self):
        """When the same period_end appears twice, the later-filed value wins."""
        facts = _make_facts(
            "us-gaap", "Revenues", "USD", [_Q1_REVENUE, _Q1_REVENUE_AMENDED]
        )
        series = _extract_flow_concept(facts, "us-gaap", "Revenues")
        assert len(series) == 1
        assert series["2023-03-31"] == pytest.approx(15_100_000_000)

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
            "period_end", "pe_ratio", "pb_ratio", "roe",
            "eps", "revenue", "net_income", "source",
        }
        assert expected_keys.issubset(rec.keys())

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

    def test_roe_annualised(self, monkeypatch):
        """ROE = (quarterly net_income × 4) / equity."""
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        expected_roe = (1_500_000_000 * 4) / 20_000_000_000   # = 0.30
        assert rec["roe"] == pytest.approx(expected_roe)

    def test_pe_pb_are_nan(self, monkeypatch):
        """pe_ratio and pb_ratio must be NaN (not available from XBRL alone)."""
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: _make_full_facts(),
        )
        rec = fetch_pgr_fundamentals_quarterly()[0]
        assert math.isnan(rec["pe_ratio"])
        assert math.isnan(rec["pb_ratio"])

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
        net_income_rec = {
            "start": "2023-01-01", "end": "2023-03-31",
            "val": 1_000_000_000, "filed": "2023-05-01", "form": "10-Q",
            "accn": "x",
        }
        attr_equity = {
            "end": "2023-03-31",
            "val": 10_000_000_000,
            "filed": "2023-05-01",
            "form": "10-Q",
            "accn": "x",
        }
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {"units": {"USD": [_Q1_REVENUE]}},
                    "NetIncomeLoss": {"units": {"USD": [net_income_rec]}},
                    "StockholdersEquityAttributableToParent": {
                        "units": {"USD": [attr_equity]}
                    },
                }
            }
        }
        monkeypatch.setattr(
            "src.ingestion.edgar_client.fetch_companyfacts",
            lambda **_: facts,
        )
        records = fetch_pgr_fundamentals_quarterly()
        assert len(records) >= 1
        rec = records[0]
        # ROE derived from attributable equity fallback
        expected_roe = (1_000_000_000 * 4) / 10_000_000_000
        assert rec["roe"] == pytest.approx(expected_roe)

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
