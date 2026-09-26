from __future__ import annotations

from pathlib import Path

from src.database import db_client


def _write_csv(path: Path, n_rows: int) -> None:
    lines = ["report_period,value"]
    for i in range(n_rows):
        month = (i % 12) + 1
        year = 2004 + (i // 12)
        lines.append(f"{year:04d}-{month:02d},{i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_db_health_warns_when_csv_backfill_not_loaded(tmp_path):
    db_path = tmp_path / "test.db"
    csv_path = tmp_path / "pgr_edgar_cache.csv"
    _write_csv(csv_path, n_rows=5)

    conn = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn)
    conn.execute(
        """
        INSERT INTO pgr_edgar_monthly (month_end, combined_ratio, pif_total)
        VALUES ('2026-02-28', 90.0, 1000000.0)
        """
    )
    conn.commit()

    report = db_client.get_db_health_report(conn, csv_path=str(csv_path))
    conn.close()

    assert report["row_count"] == 1
    assert report["expected_csv_rows"] == 5
    assert any("committed CSV contains 5" in msg for msg in report["warnings"])


def test_db_health_is_clean_when_backfill_matches_csv(tmp_path):
    db_path = tmp_path / "test.db"
    csv_path = tmp_path / "pgr_edgar_cache.csv"
    _write_csv(csv_path, n_rows=2)

    conn = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn)
    conn.execute(
        """
        INSERT INTO pgr_edgar_monthly (
            month_end, combined_ratio, pif_total, book_value_per_share, eps_basic,
            net_premiums_written, net_premiums_earned, underwriting_income,
            investment_book_yield, channel_mix_agency_pct, buyback_yield
        ) VALUES
            ('2004-08-31', 89.2, 8577000.0, 76.61, 0.46, 1085.7, 1007.4, 108.8, 1.3, 0.65, NULL),
            ('2004-09-30', 88.1, 9049000.0, 28.25, 0.56, 1002.4, 1013.2, 120.6, 0.4, 0.67, NULL)
        """
    )
    conn.commit()

    report = db_client.get_db_health_report(conn, csv_path=str(csv_path))
    conn.close()

    assert report["row_count"] == 2
    assert report["min_month_end"] == "2004-08-31"
    assert report["warnings"] == []


def test_warn_if_db_behind_prints_contextual_warning(tmp_path, capsys):
    db_path = tmp_path / "test.db"
    csv_path = tmp_path / "pgr_edgar_cache.csv"
    _write_csv(csv_path, n_rows=3)

    conn = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn)
    warnings = db_client.warn_if_db_behind(conn, context="unit-test", csv_path=str(csv_path))
    conn.close()

    captured = capsys.readouterr()
    assert warnings
    assert "[db-health] WARNING (unit-test):" in captured.out
