"""2006-05 split-month buybacks from the Q2 2006 10-Q (review 2026-09-25, step 4c).

Offline: the HTML fixture reproduces the cells of the 10-Q's Part II Item 2
table (0000950152-06-006431), including its "$" cells and footnote markers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import edgar_8k_fetcher as fetcher
from scripts import repair_split_month_buybacks as repair
from src.database import db_client
from src.ingestion import edgar_10q_repurchases as q10

TEN_Q_TABLE = """
<table>
<tr><td colspan="6">ISSUER PURCHASES OF EQUITY SECURITIES</td></tr>
<tr><td></td><td>Total Number of</td><td></td><td>Maximum Number of</td></tr>
<tr><td>2006</td><td>Shares</td><td>Average Price Paid</td><td>Announced Plans or</td></tr>
<tr><td>Calendar Month</td><td>Purchased <sup>1</sup></td><td>per Share <sup>1</sup></td>
    <td>Programs</td><td>Programs</td></tr>
<tr><td>April</td><td>668,704</td><td>$</td><td>107.17</td><td>12,862,416</td><td>62,137,584</td></tr>
<tr><td>May pre-split</td><td>331,496</td><td>107.94</td><td>13,193,912</td><td>61,806,088</td></tr>
<tr><td>May post-split</td><td>1,932,200</td><td>27.16</td><td>15,126,112</td><td>65,292,152</td></tr>
<tr><td>June</td><td>4,137,204</td><td>26.01</td><td>19,263,316</td><td>61,154,948</td></tr>
<tr><td>Total</td><td>7,069,604</td><td>$</td><td>37.84</td><td><sup>2</sup></td></tr>
</table>
"""
MDA = (
    "<p>During the second quarter 2006, we repurchased 7.1 million Common Shares, "
    "at a total cost of $267.5 million.</p>"
)
EIGHT_K = (
    "<p><sup>3</sup> Includes .3 million Common Shares repurchased prior to our 4-for-1 "
    "stock split at an average cost of $107.94 per share and 2.0 million Common Shares "
    "repurchased after the stock split at an average cost of $27.16; we did not split "
    "treasury shares.</p>"
)
MAY = "2006-05-31"
EIGHT_K_ACCESSION = "0000950152-06-005098"


def _purchases() -> q10.SplitMonthPurchases:
    return q10.split_month_purchases(q10.parse_issuer_purchases(TEN_Q_TABLE), "May", 4.0)


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = db_client.get_connection(str(tmp_path / "t.db"))
    db_client.initialize_schema(conn)
    rows = [
        {"month_end": "2006-04-30", "filing_date": "2006-05-17",
         "accession_number": "0000950152-06-004553",
         "shares_repurchased": 0.7, "avg_cost_per_share": 107.17},
        {"month_end": MAY, "filing_date": "2006-06-14",
         "accession_number": EIGHT_K_ACCESSION,
         "shares_repurchased": 2.3, "book_value_per_share": 8.24,
         "document_url": "https://www.sec.gov/8k.htm",
         "fetched_at": "2026-09-25T00:00:00+00:00"},
        {"month_end": "2006-06-30", "filing_date": "2006-07-13",
         "accession_number": "0000950152-06-005810",
         "shares_repurchased": 4.1, "avg_cost_per_share": 26.01},
    ]
    db_client.record_pgr_edgar_raw(conn, rows, "8k-test")
    db_client.upsert_pgr_edgar_monthly(conn, rows)
    return conn


def _record(conn: sqlite3.Connection) -> int:
    return db_client.record_pgr_edgar_supplement(
        conn,
        accession_number="000095015206006431",
        parser_version=q10.PARSER_VERSION,
        month_end=MAY,
        filing_date="2006-08-03",
        source_url="https://www.sec.gov/10q.htm",
        fetched_at="2026-09-25T00:00:00+00:00",
        values=q10.raw_values(_purchases()),
    )


def test_parse_issuer_purchases_reads_every_month_row() -> None:
    rows = q10.parse_issuer_purchases(TEN_Q_TABLE)
    assert [(r.label, r.shares, r.average_price) for r in rows] == [
        ("April", 668_704, 107.17),
        ("May pre-split", 331_496, 107.94),
        ("May post-split", 1_932_200, 27.16),
        ("June", 4_137_204, 26.01),
    ]


def test_parse_issuer_purchases_requires_the_table() -> None:
    with pytest.raises(ValueError, match="Issuer Purchases"):
        q10.parse_issuer_purchases("<table><tr><td>April</td><td>1</td><td>2</td></tr></table>")


def test_split_month_is_combined_onto_the_post_split_basis() -> None:
    p = _purchases()
    assert p.shares_post_split_basis_m == pytest.approx(0.331496 * 4 + 1.9322)  # 3.258184
    assert p.dollars_m == pytest.approx(35.78167824 + 52.478552)  # $88.26M
    assert p.avg_cost_post_split_basis == pytest.approx(88.26023024 / 3.258184)  # $27.09
    # The printed 2.3M is pre + post shares, on neither basis.
    assert round(p.shares_as_printed_m, 1) == 2.3
    # Dollars do not depend on the basis the shares are counted on.
    assert p.shares_post_split_basis_m * p.avg_cost_post_split_basis == pytest.approx(p.dollars_m)
    # The post-split average lies between the two legs' post-split prices.
    assert 107.94 / 4 < p.avg_cost_post_split_basis < 27.16


def test_split_month_rows_must_be_unique() -> None:
    rows = q10.parse_issuer_purchases(TEN_Q_TABLE)
    with pytest.raises(ValueError, match="June pre-split"):
        q10.split_month_purchases(rows, "June", 4.0)
    with pytest.raises(ValueError, match="found 2"):
        q10.split_month_purchases(rows + rows, "May", 4.0)


def test_raw_values_mark_legs_parsed_and_month_derived() -> None:
    methods = {field: method for field, _, method in q10.raw_values(_purchases())}
    assert methods == {
        "shares_repurchased_pre_split": "parsed",
        "avg_cost_per_share_pre_split": "parsed",
        "shares_repurchased_post_split": "parsed",
        "avg_cost_per_share_post_split": "parsed",
        "shares_repurchased": "derived",
        "avg_cost_per_share": "derived",
    }
    assert q10.PARSER_VERSION in db_client.PGR_EDGAR_SUPPLEMENT_PARSER_VERSIONS


def test_supplement_is_append_only_and_overrides_only_its_columns(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    assert _record(conn) == 6
    assert _record(conn) == 0  # same (accession, parser): ignored
    assert db_client.apply_pgr_edgar_supplements(conn) == 2
    assert db_client.apply_pgr_edgar_supplements(conn) == 0
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM pgr_edgar_monthly_raw")
    row = conn.execute("SELECT * FROM pgr_edgar_monthly WHERE month_end = ?", (MAY,)).fetchone()
    assert row["shares_repurchased"] == pytest.approx(3.258184)
    assert row["avg_cost_per_share"] == pytest.approx(27.08878, abs=1e-5)
    assert row["book_value_per_share"] == pytest.approx(8.24)
    assert row["accession_number"] == EIGHT_K_ACCESSION
    supplements = db_client.get_pgr_edgar_supplements(conn)
    # Per-leg fields are provenance only; they are not monthly columns.
    assert set(supplements["field"]) == {"shares_repurchased", "avg_cost_per_share"}
    assert set(supplements["accession_number"]) == {"0000950152-06-006431"}
    first = db_client.get_pgr_edgar_first_reported(conn).set_index(["month_end", "field"])
    conn.close()
    # What was known first is still the 8-K's 2.3M.
    assert first.loc[(MAY, "shares_repurchased"), "value_real"] == pytest.approx(2.3)
    assert first.loc[(MAY, "avg_cost_per_share"), "accession_number"] == "0000950152-06-006431"


def test_supplement_survives_a_rebuild_from_the_8k(tmp_path: Path) -> None:
    """repair_edgar_history rebuilds the row from the 8-K, then re-applies it."""
    conn = _db(tmp_path)
    _record(conn)
    conn.execute("DELETE FROM pgr_edgar_monthly")
    db_client.upsert_pgr_edgar_monthly(conn, [{
        "month_end": MAY, "filing_date": "2006-06-14",
        "accession_number": EIGHT_K_ACCESSION, "shares_repurchased": 2.3,
    }])
    db_client.apply_pgr_edgar_supplements(conn)
    fetcher.recompute_derived_fields(conn)
    row = conn.execute("SELECT * FROM pgr_edgar_monthly WHERE month_end = ?", (MAY,)).fetchone()
    conn.close()
    assert row["shares_repurchased"] == pytest.approx(3.258184)
    assert row["avg_cost_per_share"] == pytest.approx(27.08878, abs=1e-5)


def test_only_listed_parser_versions_are_supplements(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    with pytest.raises(ValueError, match="not a supplement"):
        db_client.record_pgr_edgar_supplement(
            conn, accession_number="0000950152-06-006431", parser_version="8k-test",
            month_end=MAY, filing_date="2006-08-03", source_url="x", fetched_at=None,
            values=[("avg_cost_per_share", 1.0, "parsed")],
        )
    conn.close()


def test_cross_check_passes_on_the_filings_and_catches_a_mismatch(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    rows = q10.parse_issuer_purchases(TEN_Q_TABLE)
    ten_q = repair._plain_text(TEN_Q_TABLE + MDA)
    eight_k = repair._plain_text(EIGHT_K)
    assert repair.cross_check(conn, rows, _purchases(), ten_q, eight_k) == []

    wrong_price = repair._plain_text(EIGHT_K.replace("$27.16", "$27.61"))
    problems = repair.cross_check(conn, rows, _purchases(), ten_q, wrong_price)
    assert any("post-split price" in p for p in problems)

    wrong_total = repair._plain_text(TEN_Q_TABLE + MDA.replace("267.5", "260.0"))
    problems = repair.cross_check(conn, rows, _purchases(), wrong_total, eight_k)
    conn.close()
    assert any("Q2 cost" in p for p in problems)
