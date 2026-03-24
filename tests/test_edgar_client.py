"""
Tests for src/ingestion/edgar_client.py

Covers:
  - _filter_quarterly deduplication logic (unit tests, no network)
  - fetch_pgr_quarterly_fundamentals output shape and types (mocked HTTP)
  - start_date filtering
  - graceful handling of missing unit keys
  - fetch_pgr_latest_quarter convenience wrapper
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.edgar_client import (
    _filter_quarterly,
    fetch_pgr_latest_quarter,
    fetch_pgr_quarterly_fundamentals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_row(
    end: str,
    val: float,
    form: str = "10-Q",
    filed: str = "2023-11-14",
) -> dict:
    return {"end": end, "val": val, "form": form, "filed": filed}


def _concept_payload(rows: list[dict], unit_key: str = "USD") -> dict:
    return {"units": {unit_key: rows}}


def _make_requests_get_side_effect(concept_payloads: dict[str, dict]):
    """Return a side_effect callable that dispatches by URL concept name."""

    def side_effect(url: str, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        for concept, payload in concept_payloads.items():
            if concept in url:
                mock_resp.json.return_value = payload
                return mock_resp
        mock_resp.json.return_value = {"units": {}}
        return mock_resp

    return side_effect


# ---------------------------------------------------------------------------
# Unit tests: _filter_quarterly
# ---------------------------------------------------------------------------

class TestFilterQuarterly:
    def test_keeps_10q_and_10k(self):
        rows = [
            _make_row("2023-09-30", 1.0, form="10-Q"),
            _make_row("2023-12-31", 2.0, form="10-K"),
        ]
        result = _filter_quarterly(rows)
        ends = {r["end"] for r in result}
        assert ends == {"2023-09-30", "2023-12-31"}

    def test_drops_other_form_types(self):
        rows = [
            _make_row("2023-09-30", 1.0, form="8-K"),
            _make_row("2023-06-30", 2.0, form="10-Q/A"),
        ]
        result = _filter_quarterly(rows)
        assert result == []

    def test_deduplicates_on_end_date_keeps_latest_filed(self):
        rows = [
            _make_row("2023-09-30", 1.0, form="10-Q", filed="2023-11-01"),
            _make_row("2023-09-30", 1.05, form="10-Q", filed="2023-12-15"),  # amendment
        ]
        result = _filter_quarterly(rows)
        assert len(result) == 1
        assert result[0]["val"] == pytest.approx(1.05)

    def test_drops_rows_without_end_date(self):
        rows = [{"val": 1.0, "form": "10-Q", "filed": "2023-11-01"}]
        result = _filter_quarterly(rows)
        assert result == []

    def test_empty_input(self):
        assert _filter_quarterly([]) == []

    def test_mixed_input_returns_only_valid(self):
        rows = [
            _make_row("2023-09-30", 1.0, form="10-Q"),
            {"val": 2.0, "form": "10-Q"},          # missing end
            _make_row("2023-06-30", 3.0, form="DEF 14A"),
        ]
        result = _filter_quarterly(rows)
        assert len(result) == 1
        assert result[0]["end"] == "2023-09-30"


# ---------------------------------------------------------------------------
# Integration-style tests with mocked HTTP
# ---------------------------------------------------------------------------

_EPS_ROWS = [
    _make_row("2022-09-30", 4.29, form="10-Q"),
    _make_row("2022-12-31", 5.10, form="10-K"),
    _make_row("2023-03-31", 3.88, form="10-Q"),
]
_REVENUE_ROWS = [
    _make_row("2022-09-30", 15_800_000_000.0, form="10-Q"),
    _make_row("2022-12-31", 17_200_000_000.0, form="10-K"),
    _make_row("2023-03-31", 16_100_000_000.0, form="10-Q"),
]
_NI_ROWS = [
    _make_row("2022-09-30", 750_000_000.0, form="10-Q"),
    _make_row("2022-12-31", 900_000_000.0, form="10-K"),
    _make_row("2023-03-31", 680_000_000.0, form="10-Q"),
]

_MOCK_PAYLOADS = {
    "EarningsPerShareDiluted": _concept_payload(_EPS_ROWS, unit_key="USD/shares"),
    "Revenues": _concept_payload(_REVENUE_ROWS),
    "NetIncomeLoss": _concept_payload(_NI_ROWS),
}


@patch("src.ingestion.edgar_client.time.sleep")
@patch("src.ingestion.edgar_client.requests.get")
class TestFetchPgrQuarterlyFundamentals:
    def test_returns_list_of_dicts(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_record_has_required_keys(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals()
        required = {"period_end", "eps", "revenue", "net_income",
                    "pe_ratio", "pb_ratio", "roe", "source"}
        for record in result:
            assert required <= record.keys(), f"Missing keys in {record}"

    def test_source_is_edgar_xbrl(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals()
        assert all(r["source"] == "edgar_xbrl" for r in result)

    def test_market_ratio_fields_are_none(self, mock_get, mock_sleep):
        """pe_ratio, pb_ratio, roe not available from XBRL — must be None."""
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals()
        for r in result:
            assert r["pe_ratio"] is None
            assert r["pb_ratio"] is None
            assert r["roe"] is None

    def test_sorted_ascending_by_period_end(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals()
        ends = [r["period_end"] for r in result]
        assert ends == sorted(ends)

    def test_start_date_filter_excludes_older_rows(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals(start_date="2023-01-01")
        assert all(r["period_end"] >= "2023-01-01" for r in result)

    def test_start_date_filter_keeps_newer_rows(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result_all = fetch_pgr_quarterly_fundamentals()
        result_filtered = fetch_pgr_quarterly_fundamentals(start_date="2023-01-01")
        assert len(result_filtered) <= len(result_all)

    def test_eps_value_correct(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_quarterly_fundamentals()
        q1_2023 = next(r for r in result if r["period_end"] == "2023-03-31")
        assert q1_2023["eps"] == pytest.approx(3.88)

    def test_empty_concept_data_returns_empty_list(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(
            {"EarningsPerShareDiluted": {"units": {}},
             "Revenues": {"units": {}},
             "NetIncomeLoss": {"units": {}}}
        )
        result = fetch_pgr_quarterly_fundamentals()
        assert result == []

    def test_http_error_propagates(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 429")
        mock_get.return_value = mock_resp
        with pytest.raises(Exception, match="HTTP 429"):
            fetch_pgr_quarterly_fundamentals()

    def test_inter_request_sleep_called(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        fetch_pgr_quarterly_fundamentals()
        assert mock_sleep.call_count == len(_MOCK_PAYLOADS)


# ---------------------------------------------------------------------------
# Tests for fetch_pgr_latest_quarter
# ---------------------------------------------------------------------------

@patch("src.ingestion.edgar_client.time.sleep")
@patch("src.ingestion.edgar_client.requests.get")
class TestFetchPgrLatestQuarter:
    def test_returns_most_recent_record(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_latest_quarter()
        assert result is not None
        assert result["period_end"] == "2023-03-31"

    def test_returns_none_when_no_data(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(
            {k: {"units": {}} for k in _MOCK_PAYLOADS}
        )
        result = fetch_pgr_latest_quarter()
        assert result is None

    def test_result_has_source_edgar_xbrl(self, mock_get, mock_sleep):
        mock_get.side_effect = _make_requests_get_side_effect(_MOCK_PAYLOADS)
        result = fetch_pgr_latest_quarter()
        assert result["source"] == "edgar_xbrl"
