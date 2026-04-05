from __future__ import annotations

from datetime import date

from src.database import db_client
from src.reporting.decision_rendering import build_data_freshness_lines


def _make_conn():
    conn = db_client.get_connection(":memory:")
    db_client.initialize_schema(conn)
    return conn


def test_check_data_freshness_reports_all_feeds_ok() -> None:
    conn = _make_conn()
    db_client.upsert_prices(
        conn,
        [{"ticker": "PGR", "date": "2026-04-04", "close": 250.0}],
    )
    db_client.upsert_fred_macro(
        conn,
        [{"series_id": "T10Y2Y", "month_end": "2026-02-28", "value": 0.5}],
    )
    db_client.upsert_pgr_edgar_monthly(
        conn,
        [{"month_end": "2026-03-31", "combined_ratio": 90.0}],
    )

    report = db_client.check_data_freshness(conn, date(2026, 4, 5))

    assert report["overall_status"] == "OK"
    assert report["warnings"] == []
    statuses = {row["feed"]: row["status"] for row in report["checks"]}
    assert statuses["Daily prices"] == "OK"
    assert statuses["FRED macro"] == "OK"
    assert statuses["PGR monthly EDGAR"] == "OK"


def test_check_data_freshness_flags_stale_and_missing_feeds() -> None:
    conn = _make_conn()
    db_client.upsert_prices(
        conn,
        [{"ticker": "PGR", "date": "2026-03-01", "close": 250.0}],
    )

    report = db_client.check_data_freshness(conn, date(2026, 4, 5))

    assert report["overall_status"] == "WARNING"
    statuses = {row["feed"]: row["status"] for row in report["checks"]}
    assert statuses["Daily prices"] == "STALE"
    assert statuses["FRED macro"] == "MISSING"
    assert statuses["PGR monthly EDGAR"] == "MISSING"
    assert any("Daily prices is stale" in warning for warning in report["warnings"])


def test_build_data_freshness_lines_includes_warning_block() -> None:
    report = {
        "overall_status": "WARNING",
        "checks": [
            {
                "feed": "Daily prices",
                "latest_date": "2026-03-01",
                "age_days": 35,
                "max_age_days": 10,
                "status": "STALE",
            },
            {
                "feed": "FRED macro",
                "latest_date": None,
                "age_days": None,
                "max_age_days": 45,
                "status": "MISSING",
            },
        ],
        "warnings": [
            "Daily prices is stale: latest 2026-03-01 (35 days old, limit 10).",
            "FRED macro data is missing from fred_macro_monthly.",
        ],
    }

    lines = build_data_freshness_lines(report)
    text = "\n".join(lines)

    assert "## Data Freshness" in text
    assert "Some upstream data is stale or missing" in text
    assert "**STALE**" in text
    assert "**MISSING**" in text
    assert "Warnings:" in text
