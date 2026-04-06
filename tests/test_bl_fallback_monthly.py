"""Tests for v34.0 — BL optimizer status section in monthly report (Tier 1.4)."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import scripts.monthly_decision as md
from src.portfolio.black_litterman import BLDiagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_conn(daily_prices: pd.DataFrame | None = None) -> sqlite3.Connection:
    """Build an in-memory SQLite DB with a minimal daily_prices table."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE daily_prices (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            proxy_fill INTEGER DEFAULT 0
        )
        """
    )
    if daily_prices is not None and not daily_prices.empty:
        daily_prices.to_sql("daily_prices", conn, if_exists="append", index=False)
    conn.commit()
    return conn


def _make_daily_prices(ticker: str, n_months: int = 24) -> pd.DataFrame:
    """Generate synthetic daily close prices for `n_months` months."""
    dates = pd.date_range("2023-01-01", periods=n_months * 21, freq="B")
    prices = 100 * np.cumprod(1 + np.random.default_rng(42).normal(0.001, 0.01, len(dates)))
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": 1_000_000,
        "source": "test",
        "proxy_fill": 0,
    })


def _minimal_md_kwargs(tmp_path: Path, **overrides) -> dict:
    """Return the minimum keyword arguments for _write_recommendation_md."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    conn.execute.return_value.fetchone.return_value = None
    signals = pd.DataFrame(
        {"signal": ["OUTPERFORM"], "predicted_relative_return": [0.04]},
        index=["VTI"],
    )
    base = dict(
        out_dir=tmp_path,
        as_of=date(2026, 4, 1),
        run_date=date(2026, 4, 6),
        conn=conn,
        signals=signals,
        consensus="OUTPERFORM",
        mean_predicted=0.04,
        mean_ic=0.10,
        mean_hr=0.60,
        sell_pct=0.0,
        dry_run=True,
        freshness_report={"warnings": []},
        recommendation_mode={
            "mode": "monitoring-only",
            "label": "Monitoring Only",
            "sell_pct": 0.5,
            "summary": "Test stub.",
        },
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: _load_etf_monthly_returns
# ---------------------------------------------------------------------------

def test_load_etf_monthly_returns_returns_expected_shape():
    """Helper should return a DataFrame with one column per ticker."""
    tickers = ["VTI", "VOO"]
    rows = pd.concat([_make_daily_prices(t, n_months=24) for t in tickers])
    conn = _stub_conn(rows)
    result = md._load_etf_monthly_returns(conn, tickers, date(2024, 12, 31))
    assert not result.empty, "Expected non-empty return matrix"
    assert set(result.columns) == set(tickers)
    # Monthly pct_change: ~24 months of data → ~23 valid return rows (one lost to pct_change)
    assert len(result) >= 20


def test_load_etf_monthly_returns_empty_when_no_data():
    """Helper returns empty DataFrame when no prices exist in the DB."""
    conn = _stub_conn()
    result = md._load_etf_monthly_returns(conn, ["VTI", "VOO"], date(2024, 12, 31))
    assert result.empty


# ---------------------------------------------------------------------------
# Tests: Portfolio Optimizer Status section in recommendation.md
# ---------------------------------------------------------------------------

def test_bl_section_converged_appears(tmp_path: Path):
    """When bl_diagnostics.fallback_used=False, section shows ✅ Converged."""
    bl = BLDiagnostics(
        fallback_used=False,
        fallback_reason=None,
        n_active_tickers=15,
        n_view_tickers=10,
    )
    md._write_recommendation_md(**_minimal_md_kwargs(tmp_path, bl_diagnostics=bl))
    content = (tmp_path / "recommendation.md").read_text(encoding="utf-8")
    assert "## Portfolio Optimizer Status" in content
    assert "✅ Converged" in content
    assert "15" in content   # n_active_tickers


def test_bl_section_fallback_shows_warning(tmp_path: Path):
    """When bl_diagnostics.fallback_used=True, section shows ⚠️ and fallback reason."""
    bl = BLDiagnostics(
        fallback_used=True,
        fallback_reason="optimization_failure",
        n_active_tickers=15,
        n_view_tickers=0,
    )
    md._write_recommendation_md(**_minimal_md_kwargs(tmp_path, bl_diagnostics=bl))
    content = (tmp_path / "recommendation.md").read_text(encoding="utf-8")
    assert "## Portfolio Optimizer Status" in content
    assert "⚠️" in content
    assert "optimization_failure" in content


def test_bl_section_not_run_when_none(tmp_path: Path):
    """When bl_diagnostics=None, section shows 'not run' message."""
    md._write_recommendation_md(**_minimal_md_kwargs(tmp_path, bl_diagnostics=None))
    content = (tmp_path / "recommendation.md").read_text(encoding="utf-8")
    assert "## Portfolio Optimizer Status" in content
    assert "not run" in content


def test_bl_section_appears_when_diagnostics_omitted(tmp_path: Path):
    """Default (no bl_diagnostics kwarg) should produce 'not run' section."""
    # Calls without bl_diagnostics — verifies backward-compat default
    kwargs = _minimal_md_kwargs(tmp_path)
    # Explicitly do NOT pass bl_diagnostics
    md._write_recommendation_md(**kwargs)
    content = (tmp_path / "recommendation.md").read_text(encoding="utf-8")
    assert "## Portfolio Optimizer Status" in content
