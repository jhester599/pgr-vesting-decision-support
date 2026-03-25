"""
Tests for scripts/edgar_8k_fetcher.py.

Validates:
  - _collect_8k_filings: correct 8-K/item-7.01 filtering and cutoff handling
  - _parse_html_exhibit: combined_ratio and PIF extraction, month_end derivation
  - _compute_derived_fields: YoY PIF growth and gainshare estimate math
  - check_staleness: warning at >45 days, no warning when fresh, empty-table warning
  - upsert idempotency: INSERT OR REPLACE behaviour via db_client.upsert_pgr_edgar_monthly
  - load_from_csv: seeds pgr_edgar_monthly from pgr_edgar_cache.csv without HTTP calls

No HTTP calls are made; all network dependencies are stubbed with monkeypatch.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import date, timedelta

import pytest

# Resolve project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import tempfile

from scripts.edgar_8k_fetcher import (
    _collect_8k_filings,
    _compute_derived_fields,
    _parse_html_exhibit,
    check_staleness,
    load_from_csv,
)
from src.database import db_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_memory_conn() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the full schema applied."""
    conn = db_client.get_connection(":memory:")
    db_client.initialize_schema(conn)
    return conn


def _make_recent(
    forms: list[str],
    dates: list[str],
    items: list[str],
    accessions: list[str],
) -> dict:
    return {
        "form": forms,
        "filingDate": dates,
        "items": items,
        "accessionNumber": accessions,
    }


# ---------------------------------------------------------------------------
# _collect_8k_filings
# ---------------------------------------------------------------------------

class TestCollect8kFilings:
    def test_extracts_8k_item_701(self):
        recent = _make_recent(
            forms=["8-K", "10-Q", "8-K"],
            dates=["2024-02-15", "2024-02-20", "2024-03-15"],
            items=["7.01,9.01", "1.01", "7.01"],
            accessions=[
                "0000080661-24-000001",
                "0000080661-24-000002",
                "0000080661-24-000003",
            ],
        )
        out: list[dict] = []
        _collect_8k_filings(recent, "2024-01-01", out)
        assert len(out) == 2
        # Accession numbers stored without dashes
        acc_numbers = [r["accession_number"] for r in out]
        assert "000008066124000001" in acc_numbers
        assert "000008066124000003" in acc_numbers

    def test_excludes_10q_form(self):
        recent = _make_recent(
            forms=["10-Q"],
            dates=["2024-02-15"],
            items=["1.01"],
            accessions=["0000080661-24-000001"],
        )
        out: list[dict] = []
        _collect_8k_filings(recent, "2024-01-01", out)
        assert len(out) == 0

    def test_excludes_8k_without_item_701(self):
        recent = _make_recent(
            forms=["8-K"],
            dates=["2024-02-15"],
            items=["2.02"],
            accessions=["0000080661-24-000001"],
        )
        out: list[dict] = []
        _collect_8k_filings(recent, "2024-01-01", out)
        assert len(out) == 0

    def test_excludes_filings_before_cutoff(self):
        recent = _make_recent(
            forms=["8-K"],
            dates=["2019-06-15"],
            items=["7.01"],
            accessions=["0000080661-19-000001"],
        )
        out: list[dict] = []
        _collect_8k_filings(recent, "2024-01-01", out)
        assert len(out) == 0

    def test_returns_true_when_oldest_before_cutoff(self):
        """Signal caller to stop pagination when a pre-cutoff date is seen."""
        recent = _make_recent(
            forms=["8-K", "8-K"],
            dates=["2024-02-15", "2019-05-10"],
            items=["7.01", "7.01"],
            accessions=[
                "0000080661-24-000001",
                "0000080661-19-000001",
            ],
        )
        out: list[dict] = []
        stop = _collect_8k_filings(recent, "2024-01-01", out)
        assert stop is True

    def test_returns_false_when_all_after_cutoff(self):
        recent = _make_recent(
            forms=["8-K"],
            dates=["2024-02-15"],
            items=["7.01"],
            accessions=["0000080661-24-000001"],
        )
        out: list[dict] = []
        stop = _collect_8k_filings(recent, "2024-01-01", out)
        assert stop is False

    def test_multi_item_string_with_leading_space(self):
        """Items like " 7.01, 9.01 " should still match after strip."""
        recent = _make_recent(
            forms=["8-K"],
            dates=["2024-03-18"],
            items=[" 7.01, 9.01 "],
            accessions=["0000080661-24-000005"],
        )
        out: list[dict] = []
        _collect_8k_filings(recent, "2024-01-01", out)
        assert len(out) == 1

    def test_stores_dashed_accession(self):
        recent = _make_recent(
            forms=["8-K"],
            dates=["2024-02-15"],
            items=["7.01"],
            accessions=["0000080661-24-000001"],
        )
        out: list[dict] = []
        _collect_8k_filings(recent, "2024-01-01", out)
        assert out[0]["accession_dashed"] == "0000080661-24-000001"


# ---------------------------------------------------------------------------
# _parse_html_exhibit
# ---------------------------------------------------------------------------

class TestParseHtmlExhibit:
    def _make_table_html(self, cr: float, pif: int) -> str:
        return (
            f"<html><body><table>"
            f"<tr><td>Combined Ratio</td><td>{cr}</td></tr>"
            f"<tr><td>Policies in Force</td><td>{pif:,}</td></tr>"
            f"</table></body></html>"
        )

    def test_extracts_combined_ratio(self):
        html = self._make_table_html(cr=92.4, pif=10_000_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["combined_ratio"] == pytest.approx(92.4)

    def test_extracts_pif(self):
        html = self._make_table_html(cr=91.0, pif=10_500_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["pif_total"] == pytest.approx(10_500_000)

    def test_month_end_is_last_day_of_prior_month(self):
        """A filing on 2024-02-15 covers January 2024 → month_end = 2024-01-31."""
        html = self._make_table_html(cr=91.0, pif=9_000_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["month_end"] == "2024-01-31"

    def test_month_end_march_filing_covers_february(self):
        html = self._make_table_html(cr=90.5, pif=9_100_000)
        result = _parse_html_exhibit(html, "2024-03-20")
        assert result is not None
        assert result["month_end"] == "2024-02-29"   # 2024 is a leap year

    def test_returns_none_on_empty_html(self):
        result = _parse_html_exhibit("<html><body></body></html>", "2024-02-15")
        assert result is None

    def test_ignores_implausible_cr_too_high(self):
        """Value of 200 is not a valid combined ratio; must not be stored."""
        html = (
            "<html><body>"
            "Combined Ratio 200.0 "
            "Policies in Force 9,000,000"
            "</body></html>"
        )
        result = _parse_html_exhibit(html, "2024-02-15")
        if result is not None:
            assert result["combined_ratio"] is None

    def test_ignores_implausible_pif_too_small(self):
        """Value of 500 is not a plausible PIF count."""
        html = (
            "<html><body>"
            "Combined Ratio 91.5 "
            "Policies in Force 500"
            "</body></html>"
        )
        result = _parse_html_exhibit(html, "2024-02-15")
        if result is not None:
            assert result["pif_total"] is None

    def test_pif_growth_yoy_is_none_initially(self):
        """pif_growth_yoy and gainshare_estimate must be None from the parser."""
        html = self._make_table_html(cr=92.0, pif=10_000_000)
        result = _parse_html_exhibit(html, "2024-02-15")
        assert result is not None
        assert result["pif_growth_yoy"] is None
        assert result["gainshare_estimate"] is None


# ---------------------------------------------------------------------------
# _compute_derived_fields
# ---------------------------------------------------------------------------

class TestComputeDerivedFields:
    def test_pif_growth_yoy_computed(self):
        records = [
            {
                "month_end": "2023-01-31", "combined_ratio": 91.0,
                "pif_total": 10_000_000.0, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
            {
                "month_end": "2024-01-31", "combined_ratio": 90.0,
                "pif_total": 11_000_000.0, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        # (11M - 10M) / 10M = 10%
        assert result[1]["pif_growth_yoy"] == pytest.approx(0.10)

    def test_pif_growth_none_when_no_prior_year(self):
        records = [
            {
                "month_end": "2024-01-31", "combined_ratio": 90.0,
                "pif_total": 11_000_000.0, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        assert result[0]["pif_growth_yoy"] is None

    def test_gainshare_both_components(self):
        """CR=91, PIF growth=10% → gainshare=0.75."""
        records = [
            {
                "month_end": "2023-01-31", "combined_ratio": 91.0,
                "pif_total": 10_000_000.0, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
            {
                "month_end": "2024-01-31", "combined_ratio": 91.0,
                "pif_total": 11_000_000.0, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        # cr_score = (96-91)/10 = 0.5; pif_score = 0.10/0.10 = 1.0
        # gainshare = 0.5*0.5 + 0.5*1.0 = 0.75
        assert result[1]["gainshare_estimate"] == pytest.approx(0.75)

    def test_gainshare_cr_only(self):
        """When PIF data absent, gainshare uses cr_score alone."""
        records = [
            {
                "month_end": "2024-01-31", "combined_ratio": 86.0,
                "pif_total": None, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        # cr_score = (96-86)/10 = 1.0
        assert result[0]["gainshare_estimate"] == pytest.approx(1.0)

    def test_gainshare_none_when_no_data(self):
        records = [
            {
                "month_end": "2024-01-31", "combined_ratio": None,
                "pif_total": None, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        assert result[0]["gainshare_estimate"] is None

    def test_gainshare_clamped_at_2(self):
        """Extremely good CR (e.g. 50) should be clamped to max 2.0."""
        records = [
            {
                "month_end": "2024-01-31", "combined_ratio": 50.0,
                "pif_total": None, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        assert result[0]["gainshare_estimate"] == pytest.approx(2.0)

    def test_gainshare_clamped_at_0(self):
        """CR above 96 should yield cr_score = 0."""
        records = [
            {
                "month_end": "2024-01-31", "combined_ratio": 105.0,
                "pif_total": None, "pif_growth_yoy": None,
                "gainshare_estimate": None,
            },
        ]
        result = _compute_derived_fields(records)
        assert result[0]["gainshare_estimate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------

class TestCheckStaleness:
    def test_warns_when_stale(self, caplog):
        conn = _in_memory_conn()
        stale_date = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        db_client.upsert_pgr_edgar_monthly(conn, [{
            "month_end": stale_date,
            "combined_ratio": 91.0,
            "pif_total": None,
            "pif_growth_yoy": None,
            "gainshare_estimate": None,
        }])
        with caplog.at_level(logging.WARNING):
            check_staleness(conn)
        assert any(
            "WARNING" in r.message and "days old" in r.message
            for r in caplog.records
        )

    def test_no_warning_when_fresh(self, caplog):
        conn = _in_memory_conn()
        fresh_date = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
        db_client.upsert_pgr_edgar_monthly(conn, [{
            "month_end": fresh_date,
            "combined_ratio": 90.0,
            "pif_total": None,
            "pif_growth_yoy": None,
            "gainshare_estimate": None,
        }])
        with caplog.at_level(logging.WARNING):
            check_staleness(conn)
        stale_warnings = [
            r for r in caplog.records
            if "WARNING" in r.message and "days old" in r.message
        ]
        assert len(stale_warnings) == 0

    def test_warns_when_empty_table(self, caplog):
        conn = _in_memory_conn()
        with caplog.at_level(logging.WARNING):
            check_staleness(conn)
        assert any("empty" in r.message.lower() for r in caplog.records)

    def test_staleness_threshold_is_45_days(self, caplog):
        """Exactly 45 days old should NOT trigger the warning."""
        conn = _in_memory_conn()
        exactly_45 = (date.today() - timedelta(days=45)).strftime("%Y-%m-%d")
        db_client.upsert_pgr_edgar_monthly(conn, [{
            "month_end": exactly_45,
            "combined_ratio": 90.0,
            "pif_total": None,
            "pif_growth_yoy": None,
            "gainshare_estimate": None,
        }])
        with caplog.at_level(logging.WARNING):
            check_staleness(conn)
        stale_warnings = [
            r for r in caplog.records
            if "WARNING" in r.message and "days old" in r.message
        ]
        assert len(stale_warnings) == 0

    def test_staleness_threshold_46_days(self, caplog):
        """46 days old SHOULD trigger the warning."""
        conn = _in_memory_conn()
        d46 = (date.today() - timedelta(days=46)).strftime("%Y-%m-%d")
        db_client.upsert_pgr_edgar_monthly(conn, [{
            "month_end": d46,
            "combined_ratio": 91.5,
            "pif_total": None,
            "pif_growth_yoy": None,
            "gainshare_estimate": None,
        }])
        with caplog.at_level(logging.WARNING):
            check_staleness(conn)
        assert any(
            "WARNING" in r.message and "days old" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Idempotent upsert — Task 1 core requirement
# ---------------------------------------------------------------------------

class TestIdempotentUpsert:
    """Verify INSERT OR REPLACE behaviour via db_client.upsert_pgr_edgar_monthly."""

    def _sample_record(
        self,
        month_end: str = "2024-01-31",
        cr: float = 91.5,
        pif: float = 10_000_000.0,
    ) -> dict:
        return {
            "month_end": month_end,
            "combined_ratio": cr,
            "pif_total": pif,
            "pif_growth_yoy": None,
            "gainshare_estimate": None,
        }

    def test_first_upsert_inserts_row(self):
        conn = _in_memory_conn()
        n = db_client.upsert_pgr_edgar_monthly(conn, [self._sample_record()])
        assert n == 1
        count = conn.execute(
            "SELECT COUNT(*) FROM pgr_edgar_monthly"
        ).fetchone()[0]
        assert count == 1

    def test_second_upsert_same_data_does_not_duplicate(self):
        conn = _in_memory_conn()
        rec = self._sample_record()
        db_client.upsert_pgr_edgar_monthly(conn, [rec])
        db_client.upsert_pgr_edgar_monthly(conn, [rec])
        count = conn.execute(
            "SELECT COUNT(*) FROM pgr_edgar_monthly"
        ).fetchone()[0]
        assert count == 1

    def test_second_upsert_updated_data_overwrites_not_duplicates(self):
        """Second upsert with a new CR value must overwrite, not add a new row."""
        conn = _in_memory_conn()
        db_client.upsert_pgr_edgar_monthly(conn, [self._sample_record(cr=91.5)])
        db_client.upsert_pgr_edgar_monthly(conn, [self._sample_record(cr=90.1)])

        row = conn.execute(
            "SELECT combined_ratio FROM pgr_edgar_monthly WHERE month_end = '2024-01-31'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(90.1)

        count = conn.execute(
            "SELECT COUNT(*) FROM pgr_edgar_monthly"
        ).fetchone()[0]
        assert count == 1

    def test_multiple_months_upserted_correctly(self):
        conn = _in_memory_conn()
        records = [
            self._sample_record("2024-01-31", cr=91.5, pif=10_000_000.0),
            self._sample_record("2024-02-29", cr=90.8, pif=10_100_000.0),
            self._sample_record("2024-03-31", cr=89.3, pif=10_200_000.0),
        ]
        db_client.upsert_pgr_edgar_monthly(conn, records)
        count = conn.execute(
            "SELECT COUNT(*) FROM pgr_edgar_monthly"
        ).fetchone()[0]
        assert count == 3

    def test_third_upsert_still_idempotent(self):
        conn = _in_memory_conn()
        rec = self._sample_record()
        for _ in range(3):
            db_client.upsert_pgr_edgar_monthly(conn, [rec])
        count = conn.execute(
            "SELECT COUNT(*) FROM pgr_edgar_monthly"
        ).fetchone()[0]
        assert count == 1

    def test_empty_records_list_writes_zero_rows(self):
        conn = _in_memory_conn()
        n = db_client.upsert_pgr_edgar_monthly(conn, [])
        assert n == 0
        count = conn.execute(
            "SELECT COUNT(*) FROM pgr_edgar_monthly"
        ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# load_from_csv
# ---------------------------------------------------------------------------

class TestLoadFromCsv:
    """Verify that load_from_csv correctly seeds pgr_edgar_monthly from CSV."""

    def _write_csv(self, rows: list[dict], extra_cols: bool = False) -> str:
        """Write a minimal pgr_edgar_cache.csv to a temp file and return path."""
        import csv
        cols = ["report_period", "combined_ratio", "pif_total"]
        if extra_cols:
            cols += ["net_premiums_written", "eps_diluted"]
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in cols})
        return path

    def test_basic_load(self):
        path = self._write_csv([
            {"report_period": "2024-01", "combined_ratio": 91.5, "pif_total": 10_000_000},
            {"report_period": "2024-02", "combined_ratio": 90.8, "pif_total": 10_100_000},
        ])
        conn = _in_memory_conn()
        n = load_from_csv(conn, path)
        assert n == 2
        count = conn.execute("SELECT COUNT(*) FROM pgr_edgar_monthly").fetchone()[0]
        assert count == 2
        os.unlink(path)

    def test_month_end_is_last_day_of_month(self):
        path = self._write_csv([
            {"report_period": "2024-02", "combined_ratio": 90.0, "pif_total": 9_000_000},
        ])
        conn = _in_memory_conn()
        load_from_csv(conn, path)
        row = conn.execute("SELECT month_end FROM pgr_edgar_monthly").fetchone()
        assert row[0] == "2024-02-29"  # 2024 is a leap year
        os.unlink(path)

    def test_january_month_end(self):
        path = self._write_csv([
            {"report_period": "2024-01", "combined_ratio": 88.0, "pif_total": 9_500_000},
        ])
        conn = _in_memory_conn()
        load_from_csv(conn, path)
        row = conn.execute("SELECT month_end FROM pgr_edgar_monthly").fetchone()
        assert row[0] == "2024-01-31"
        os.unlink(path)

    def test_idempotent_when_called_twice(self):
        path = self._write_csv([
            {"report_period": "2024-01", "combined_ratio": 91.5, "pif_total": 10_000_000},
        ])
        conn = _in_memory_conn()
        load_from_csv(conn, path)
        load_from_csv(conn, path)  # second call — should not duplicate
        count = conn.execute("SELECT COUNT(*) FROM pgr_edgar_monthly").fetchone()[0]
        assert count == 1
        os.unlink(path)

    def test_dry_run_writes_nothing(self):
        path = self._write_csv([
            {"report_period": "2024-01", "combined_ratio": 91.5, "pif_total": 10_000_000},
        ])
        conn = _in_memory_conn()
        n = load_from_csv(conn, path, dry_run=True)
        assert n == 0
        count = conn.execute("SELECT COUNT(*) FROM pgr_edgar_monthly").fetchone()[0]
        assert count == 0
        os.unlink(path)

    def test_file_not_found_raises(self):
        conn = _in_memory_conn()
        with pytest.raises(FileNotFoundError):
            load_from_csv(conn, "/nonexistent/path/pgr_edgar_cache.csv")

    def test_extra_columns_ignored(self):
        """CSV columns beyond the expected ones should not cause errors."""
        path = self._write_csv(
            [{"report_period": "2024-03", "combined_ratio": 89.0, "pif_total": 10_200_000}],
            extra_cols=True,
        )
        conn = _in_memory_conn()
        n = load_from_csv(conn, path)
        assert n == 1
        os.unlink(path)

    def test_pif_growth_yoy_computed_from_csv(self):
        """Two rows exactly 12 months apart should yield pif_growth_yoy."""
        path = self._write_csv([
            {"report_period": "2023-01", "combined_ratio": 91.0, "pif_total": 10_000_000},
            {"report_period": "2024-01", "combined_ratio": 90.0, "pif_total": 11_000_000},
        ])
        conn = _in_memory_conn()
        load_from_csv(conn, path)
        row = conn.execute(
            "SELECT pif_growth_yoy FROM pgr_edgar_monthly WHERE month_end = '2024-01-31'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.10)
        os.unlink(path)

    def test_loads_actual_repo_csv_when_present(self):
        """Integration test: load the actual committed CSV if present."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(repo_root, "data", "processed", "pgr_edgar_cache.csv")
        if not os.path.exists(csv_path):
            pytest.skip("pgr_edgar_cache.csv not present in repo")
        conn = _in_memory_conn()
        n = load_from_csv(conn, csv_path)
        assert n > 100, f"Expected 100+ rows from the committed CSV, got {n}"
        # Verify the earliest and latest dates are plausible
        row = conn.execute(
            "SELECT MIN(month_end), MAX(month_end) FROM pgr_edgar_monthly"
        ).fetchone()
        assert row[0] <= "2010-12-31", f"Earliest date {row[0]} is later than expected"
        assert row[1] >= "2024-01-31", f"Latest date {row[1]} is earlier than expected"
