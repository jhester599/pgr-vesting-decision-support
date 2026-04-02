"""
v6.2 tests — pgr_edgar_monthly schema expansion + CSV loader.

Coverage:
  1. Schema migration: initialize_schema adds all v6.2 columns to an existing DB
  2. upsert_pgr_edgar_monthly: new columns round-trip through DB correctly
  3. upsert backward-compat: old callers passing only core fields still work
  4. get_pgr_edgar_monthly: returns all v6.2 columns
  5. load_from_csv: maps direct CSV fields into DB columns
  6. load_from_csv: derived fields are computed correctly
  7. load_from_csv: pif_growth_yoy and gainshare_estimate computed via existing helper
  8. load_from_csv: NaN rows produce NULL in DB (not a crash)
  9. load_from_csv: dry_run skips DB write
 10. load_from_csv: FileNotFoundError on missing CSV
 11. load_from_csv: ValueError on CSV missing report_period column
 12. channel_mix_agency_pct: zero denominator produces NULL (not division error)
"""

from __future__ import annotations

import io
import os
import sqlite3
import tempfile
from typing import Any

import pandas as pd
import pytest

import config
from src.database import db_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_memory_conn() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the v6.2 schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    db_client.initialize_schema(conn)
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _minimal_csv(rows: list[dict[str, Any]]) -> str:
    """Build a minimal pgr_edgar_cache.csv string from a list of row dicts."""
    # Always include report_period; fill other columns with empty string if absent
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    cols = ["report_period"] + sorted(all_keys - {"report_period"})
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Schema migration: initialize_schema adds all v6.2 columns to existing DB
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    _V62_COLUMNS = {
        # Foundational P&L
        "net_premiums_written", "net_premiums_earned", "net_income",
        "eps_diluted", "loss_lae_ratio", "expense_ratio",
        # Segment channels
        "npw_agency", "npw_direct", "npw_commercial", "npw_property",
        "npe_agency", "npe_direct", "npe_commercial", "npe_property",
        "pif_agency_auto", "pif_direct_auto", "pif_commercial_lines",
        "pif_total_personal_lines",
        # Company-level
        "investment_income", "total_revenues", "total_expenses",
        "income_before_income_taxes", "roe_net_income_ttm",
        "shareholders_equity", "total_assets",
        "unearned_premiums", "shares_repurchased", "avg_cost_per_share",
        # Investment portfolio
        "fte_return_total_portfolio", "investment_book_yield",
        "net_unrealized_gains_fixed", "fixed_income_duration",
        # Derived
        "channel_mix_agency_pct", "npw_growth_yoy", "underwriting_income",
        "unearned_premium_growth_yoy", "buyback_yield",
    }

    def test_new_columns_present_after_initialize_schema(self):
        conn = _in_memory_conn()
        cols = _table_columns(conn, "pgr_edgar_monthly")
        for col in self._V62_COLUMNS:
            assert col in cols, f"Column '{col}' missing from pgr_edgar_monthly"

    def test_pre_existing_columns_still_present(self):
        conn = _in_memory_conn()
        cols = _table_columns(conn, "pgr_edgar_monthly")
        for col in ("month_end", "combined_ratio", "pif_total",
                    "pif_growth_yoy", "gainshare_estimate",
                    "book_value_per_share", "eps_basic"):
            assert col in cols

    def test_initialize_schema_idempotent(self):
        """Running initialize_schema twice must not raise."""
        conn = _in_memory_conn()
        db_client.initialize_schema(conn)  # second call
        cols = _table_columns(conn, "pgr_edgar_monthly")
        assert "npw_agency" in cols

    def test_migration_on_pre_v62_db(self):
        """A DB with only the original 5 columns gets all v6.2 columns added."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE pgr_edgar_monthly (
                month_end           TEXT NOT NULL,
                combined_ratio      REAL,
                pif_total           REAL,
                pif_growth_yoy      REAL,
                gainshare_estimate  REAL,
                PRIMARY KEY (month_end)
            )
        """)
        conn.commit()
        db_client.initialize_schema(conn)
        cols = _table_columns(conn, "pgr_edgar_monthly")
        assert "npw_agency" in cols
        assert "underwriting_income" in cols
        # Original columns preserved
        assert "combined_ratio" in cols


# ---------------------------------------------------------------------------
# 2. upsert_pgr_edgar_monthly: new columns round-trip correctly
# ---------------------------------------------------------------------------

class TestUpsertPgrEdgarMonthly:
    def test_full_v62_record_round_trips(self):
        conn = _in_memory_conn()
        record = {
            "month_end":                   "2026-02-28",
            "combined_ratio":              85.7,
            "pif_total":                   39220.0,
            "pif_growth_yoy":              0.10,
            "gainshare_estimate":          1.5,
            "book_value_per_share":        76.61,
            "eps_basic":                   1.61,
            "net_premiums_written":        6995.0,
            "net_premiums_earned":         6528.0,
            "net_income":                  943.0,
            "eps_diluted":                 1.59,
            "loss_lae_ratio":              65.0,
            "expense_ratio":               20.7,
            "npw_agency":                  3100.0,
            "npw_direct":                  3895.0,
            "npw_commercial":              350.0,
            "npw_property":                650.0,
            "npe_agency":                  2900.0,
            "npe_direct":                  3628.0,
            "npe_commercial":              310.0,
            "npe_property":                600.0,
            "pif_agency_auto":             10959.0,
            "pif_direct_auto":             16383.0,
            "pif_commercial_lines":        1188.0,
            "pif_total_personal_lines":    38032.0,
            "investment_income":           240.0,
            "total_revenues":              7236.0,
            "total_expenses":              6100.0,
            "income_before_income_taxes":  1136.0,
            "roe_net_income_ttm":          0.22,
            "shareholders_equity":         14000.0,
            "total_assets":               85000.0,
            "unearned_premiums":           12000.0,
            "shares_repurchased":          500.0,
            "avg_cost_per_share":          220.0,
            "fte_return_total_portfolio":  0.0425,
            "investment_book_yield":       0.038,
            "net_unrealized_gains_fixed":  -800.0,
            "fixed_income_duration":       2.8,
            "channel_mix_agency_pct":      0.443,
            "npw_growth_yoy":              0.12,
            "underwriting_income":         930.0,
            "unearned_premium_growth_yoy": 0.09,
            "buyback_yield":               None,
        }
        db_client.upsert_pgr_edgar_monthly(conn, [record])
        df = db_client.get_pgr_edgar_monthly(conn)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["combined_ratio"] == pytest.approx(85.7)
        assert row["npw_agency"] == pytest.approx(3100.0)
        assert row["investment_income"] == pytest.approx(240.0)
        assert row["channel_mix_agency_pct"] == pytest.approx(0.443)
        assert row["roe_net_income_ttm"] == pytest.approx(0.22)
        assert pd.isna(row["buyback_yield"])

    def test_upsert_replace_semantics(self):
        """Second upsert with same month_end overwrites the row."""
        conn = _in_memory_conn()
        db_client.upsert_pgr_edgar_monthly(conn, [
            {"month_end": "2026-01-31", "combined_ratio": 88.0, "npw_agency": 3000.0}
        ])
        db_client.upsert_pgr_edgar_monthly(conn, [
            {"month_end": "2026-01-31", "combined_ratio": 85.0, "npw_agency": 3100.0}
        ])
        df = db_client.get_pgr_edgar_monthly(conn)
        assert len(df) == 1
        assert df.iloc[0]["combined_ratio"] == pytest.approx(85.0)
        assert df.iloc[0]["npw_agency"] == pytest.approx(3100.0)


# ---------------------------------------------------------------------------
# 3. Backward compatibility: old callers passing only core fields still work
# ---------------------------------------------------------------------------

class TestUpsertBackwardCompat:
    def test_core_only_record_does_not_raise(self):
        """Old-style record (no v6.2 fields) inserts without error."""
        conn = _in_memory_conn()
        db_client.upsert_pgr_edgar_monthly(conn, [
            {
                "month_end":          "2026-01-31",
                "combined_ratio":     88.0,
                "pif_total":          38000.0,
                "pif_growth_yoy":     0.08,
                "gainshare_estimate": 1.2,
            }
        ])
        df = db_client.get_pgr_edgar_monthly(conn)
        assert len(df) == 1
        assert df.iloc[0]["combined_ratio"] == pytest.approx(88.0)
        assert pd.isna(df.iloc[0]["npw_agency"])

    def test_empty_records_returns_zero(self):
        conn = _in_memory_conn()
        n = db_client.upsert_pgr_edgar_monthly(conn, [])
        assert n == 0


# ---------------------------------------------------------------------------
# 4. get_pgr_edgar_monthly: returns all v6.2 columns
# ---------------------------------------------------------------------------

class TestGetPgrEdgarMonthly:
    def test_returns_all_v62_columns(self):
        conn = _in_memory_conn()
        db_client.upsert_pgr_edgar_monthly(conn, [
            {"month_end": "2026-01-31", "combined_ratio": 85.0}
        ])
        df = db_client.get_pgr_edgar_monthly(conn)
        expected = {
            "combined_ratio", "npw_agency", "npw_direct",
            "investment_income", "channel_mix_agency_pct",
            "underwriting_income", "roe_net_income_ttm",
        }
        for col in expected:
            assert col in df.columns, f"Column '{col}' missing from get_pgr_edgar_monthly()"

    def test_empty_table_returns_empty_dataframe(self):
        conn = _in_memory_conn()
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.empty


# ---------------------------------------------------------------------------
# 5. load_from_csv: direct field mapping
# ---------------------------------------------------------------------------

class TestLoadFromCsvDirectMapping:
    def _run_load(self, csv_content: str, dry_run: bool = False) -> tuple[int, Any]:
        from scripts.edgar_8k_fetcher import load_from_csv
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            path = f.name
        try:
            n = load_from_csv(conn, path, dry_run=dry_run)
        finally:
            os.unlink(path)
        return n, conn

    def test_maps_core_fields(self):
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            "net_premiums_written": "6995",
            "net_premiums_earned": "6528",
        }])
        n, conn = self._run_load(csv)
        assert n == 1
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.iloc[0]["combined_ratio"] == pytest.approx(85.7)
        assert df.iloc[0]["net_premiums_written"] == pytest.approx(6995.0)
        assert df.iloc[0]["net_premiums_earned"] == pytest.approx(6528.0)

    def test_maps_segment_fields(self):
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            "npw_agency": "3100",
            "npw_direct": "3895",
            "npw_commercial": "350",
            "npw_property": "650",
            "pif_agency_auto": "10959",
            "pif_direct_auto": "16383",
        }])
        n, conn = self._run_load(csv)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.iloc[0]["npw_agency"] == pytest.approx(3100.0)
        assert df.iloc[0]["npw_direct"] == pytest.approx(3895.0)
        assert df.iloc[0]["pif_agency_auto"] == pytest.approx(10959.0)

    def test_maps_roe_net_income_trailing_12m_to_roe_net_income_ttm(self):
        """CSV column 'roe_net_income_trailing_12m' → DB 'roe_net_income_ttm'."""
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            "roe_net_income_trailing_12m": "0.22",
        }])
        n, conn = self._run_load(csv)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.iloc[0]["roe_net_income_ttm"] == pytest.approx(0.22)

    def test_maps_investment_portfolio_fields(self):
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            "investment_book_yield": "0.038",
            "fixed_income_duration": "2.8",
            "net_unrealized_gains_fixed": "-800",
        }])
        n, conn = self._run_load(csv)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.iloc[0]["investment_book_yield"] == pytest.approx(0.038)
        assert df.iloc[0]["fixed_income_duration"] == pytest.approx(2.8)
        assert df.iloc[0]["net_unrealized_gains_fixed"] == pytest.approx(-800.0)

    def test_month_end_conversion_to_last_day(self):
        """report_period '2026-02' → month_end '2026-02-28'."""
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
        }])
        n, conn = self._run_load(csv)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.index[0] == pd.Timestamp("2026-02-28")

    def test_multiple_rows_all_upserted(self):
        rows = [
            {"report_period": f"2024-{m:02d}", "combined_ratio": f"{85 + m * 0.1:.1f}", "pif_total": "38000"}
            for m in range(1, 13)
        ]
        csv = _minimal_csv(rows)
        n, conn = self._run_load(csv)
        assert n == 12
        df = db_client.get_pgr_edgar_monthly(conn)
        assert len(df) == 12


# ---------------------------------------------------------------------------
# 6. load_from_csv: derived fields
# ---------------------------------------------------------------------------

class TestLoadFromCsvDerivedFields:
    def _load_with_data(
        self,
        npw_agency: list[float],
        npw_direct: list[float],
        net_premiums_earned: list[float],
        combined_ratio: list[float],
        net_premiums_written: list[float],
        unearned_premiums: list[float],
        start_period: str = "2024-01",
    ) -> pd.DataFrame:
        from scripts.edgar_8k_fetcher import load_from_csv
        n = len(npw_agency)
        rows = []
        for i in range(n):
            period_dt = pd.Period(start_period, freq="M") + i
            rows.append({
                "report_period": str(period_dt),
                "combined_ratio": str(combined_ratio[i]),
                "pif_total": "38000",
                "npw_agency": str(npw_agency[i]),
                "npw_direct": str(npw_direct[i]),
                "net_premiums_written": str(net_premiums_written[i]),
                "net_premiums_earned": str(net_premiums_earned[i]),
                "unearned_premiums": str(unearned_premiums[i]),
            })
        csv = _minimal_csv(rows)
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            load_from_csv(conn, path)
        finally:
            os.unlink(path)
        return db_client.get_pgr_edgar_monthly(conn)

    def test_channel_mix_agency_pct_correct(self):
        df = self._load_with_data(
            npw_agency=[3000.0],
            npw_direct=[7000.0],
            net_premiums_earned=[6000.0],
            combined_ratio=[85.0],
            net_premiums_written=[10000.0],
            unearned_premiums=[12000.0],
        )
        # 3000 / (3000 + 7000) = 0.30
        assert df.iloc[0]["channel_mix_agency_pct"] == pytest.approx(0.30)

    def test_underwriting_income_correct(self):
        # underwriting_income = 6000 * (1 - 85/100) = 6000 * 0.15 = 900
        df = self._load_with_data(
            npw_agency=[3000.0],
            npw_direct=[7000.0],
            net_premiums_earned=[6000.0],
            combined_ratio=[85.0],
            net_premiums_written=[10000.0],
            unearned_premiums=[12000.0],
        )
        assert df.iloc[0]["underwriting_income"] == pytest.approx(900.0)

    def test_npw_growth_yoy_correct_after_12_months(self):
        # 12 months at 10000, then month 13 at 11000 → growth = 10%
        npw = [10000.0] * 12 + [11000.0]
        df = self._load_with_data(
            npw_agency=[3000.0] * 13,
            npw_direct=[7000.0] * 13,
            net_premiums_earned=[6000.0] * 13,
            combined_ratio=[85.0] * 13,
            net_premiums_written=npw,
            unearned_premiums=[12000.0] * 13,
        )
        assert df.iloc[-1]["npw_growth_yoy"] == pytest.approx(0.10)

    def test_unearned_premium_growth_yoy_correct(self):
        uep = [12000.0] * 12 + [13200.0]  # 10% growth at month 13
        df = self._load_with_data(
            npw_agency=[3000.0] * 13,
            npw_direct=[7000.0] * 13,
            net_premiums_earned=[6000.0] * 13,
            combined_ratio=[85.0] * 13,
            net_premiums_written=[10000.0] * 13,
            unearned_premiums=uep,
        )
        assert df.iloc[-1]["unearned_premium_growth_yoy"] == pytest.approx(0.10)

    def test_buyback_yield_is_null(self):
        """buyback_yield requires market_cap — must be NULL for CSV path."""
        df = self._load_with_data(
            npw_agency=[3000.0],
            npw_direct=[7000.0],
            net_premiums_earned=[6000.0],
            combined_ratio=[85.0],
            net_premiums_written=[10000.0],
            unearned_premiums=[12000.0],
        )
        assert pd.isna(df.iloc[0]["buyback_yield"])


# ---------------------------------------------------------------------------
# 7. pif_growth_yoy and gainshare_estimate via existing helper
# ---------------------------------------------------------------------------

class TestLoadFromCsvGainshare:
    def test_pif_growth_yoy_computed_after_12_months(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        rows = [
            {"report_period": str(pd.Period("2024-01", freq="M") + i),
             "combined_ratio": "85.0", "pif_total": "30000"}
            for i in range(12)
        ]
        rows.append({
            "report_period": "2025-01",
            "combined_ratio": "85.0",
            "pif_total": "33000",  # 10% growth
        })
        csv = _minimal_csv(rows)
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            load_from_csv(conn, path)
        finally:
            os.unlink(path)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.iloc[-1]["pif_growth_yoy"] == pytest.approx(0.10)

    def test_gainshare_estimate_computed(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        # CR = 86 → cr_score = (96-86)/10 = 1.0; PIF flat → pif_score = 0 → gainshare = 0.5
        rows = [
            {"report_period": str(pd.Period("2024-01", freq="M") + i),
             "combined_ratio": "86.0", "pif_total": "30000"}
            for i in range(13)
        ]
        csv = _minimal_csv(rows)
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            load_from_csv(conn, path)
        finally:
            os.unlink(path)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.iloc[-1]["gainshare_estimate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 8. NaN rows produce NULL in DB
# ---------------------------------------------------------------------------

class TestLoadFromCsvNullHandling:
    def test_missing_optional_fields_produce_null(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            # npw_agency, npw_direct, etc. intentionally absent
        }])
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            load_from_csv(conn, path)
        finally:
            os.unlink(path)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert pd.isna(df.iloc[0]["npw_agency"])
        assert pd.isna(df.iloc[0]["channel_mix_agency_pct"])

    def test_empty_string_in_numeric_field_becomes_null(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            "npw_agency": "",  # empty → NaN → NULL
        }])
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            load_from_csv(conn, path)
        finally:
            os.unlink(path)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert pd.isna(df.iloc[0]["npw_agency"])


# ---------------------------------------------------------------------------
# 9. dry_run skips DB write
# ---------------------------------------------------------------------------

class TestLoadFromCsvDryRun:
    def test_dry_run_returns_zero_and_writes_nothing(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
        }])
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            n = load_from_csv(conn, path, dry_run=True)
        finally:
            os.unlink(path)
        assert n == 0
        df = db_client.get_pgr_edgar_monthly(conn)
        assert df.empty


# ---------------------------------------------------------------------------
# 10. FileNotFoundError on missing CSV
# ---------------------------------------------------------------------------

class TestLoadFromCsvErrors:
    def test_missing_file_raises_file_not_found(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        conn = _in_memory_conn()
        with pytest.raises(FileNotFoundError):
            load_from_csv(conn, "/nonexistent/path/pgr_edgar_cache.csv")

    def test_missing_report_period_column_raises_value_error(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        csv = "combined_ratio,pif_total\n85.7,39220\n"
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            with pytest.raises(ValueError, match="report_period"):
                load_from_csv(conn, path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 11. channel_mix_agency_pct: zero denominator → NULL
# ---------------------------------------------------------------------------

class TestChannelMixEdgeCases:
    def test_zero_denominator_produces_null(self):
        from scripts.edgar_8k_fetcher import load_from_csv
        csv = _minimal_csv([{
            "report_period": "2026-02",
            "combined_ratio": "85.7",
            "pif_total": "39220",
            "npw_agency": "0",
            "npw_direct": "0",  # denominator = 0
        }])
        conn = _in_memory_conn()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            load_from_csv(conn, path)
        finally:
            os.unlink(path)
        df = db_client.get_pgr_edgar_monthly(conn)
        assert pd.isna(df.iloc[0]["channel_mix_agency_pct"])
