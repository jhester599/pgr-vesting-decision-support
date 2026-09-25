"""DB-integrity guards for the committed ``fred_macro_monthly`` table.

Review 2026-09-25 findings F06 and F27: one row per (series, calendar
month), each labelled with the month's last business day, and the unique
(series, month) index from migration 005 in place.

The committed DB is opened read-only; these tests never write to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database import db_client

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "pgr_financials.db"

pytestmark = pytest.mark.skipif(
    not _DB_PATH.exists(), reason="committed DB not present"
)


@pytest.fixture(scope="module")
def ro_conn():
    conn = db_client.get_connection(str(_DB_PATH), read_only=True)
    yield conn
    conn.close()


def test_no_duplicate_series_month_rows(ro_conn) -> None:
    duplicates = ro_conn.execute(
        """
        SELECT series_id, substr(month_end, 1, 7) AS month, COUNT(*) AS n
        FROM fred_macro_monthly GROUP BY 1, 2 HAVING n > 1
        """
    ).fetchall()
    assert [tuple(r) for r in duplicates] == []


def test_every_label_is_a_business_month_end(ro_conn) -> None:
    labels = [r[0] for r in ro_conn.execute("SELECT DISTINCT month_end FROM fred_macro_monthly")]
    bad = [label for label in labels if db_client.fred_month_label(label) != label]
    assert bad == []


def test_unique_series_month_index_present(ro_conn) -> None:
    names = {
        r[0] for r in ro_conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'fred_macro_monthly'"
        )
    }
    assert "idx_fred_macro_monthly_series_month" in names
