"""Review 2026-09-25, step 6 (WP10, F23): EDGAR rows placed by filing date.

The fixed 2-month lag put every monthly 8-K one month later than it was
public, and placed several 10-Ks before their filing date. Each row now
enters on the first decision date (business month-end) on or after its
``filing_date``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.database import db_client
from src.processing import feature_engineering
from src.processing.feature_engineering import build_feature_matrix_from_db

DB_PATH = Path(config.DB_PATH)


@pytest.mark.parametrize(
    ("period", "filed", "expected"),
    [
        ("2024-01-31", "2024-02-14", "2024-02-29"),  # review example; the lag gave 2024-03-29
        ("2024-12-31", "2025-03-03", "2025-03-31"),  # 10-K after the 2-month mark
        ("2025-08-31", "2025-09-30", "2025-09-30"),  # filed on a business month-end
        ("2025-05-31", "2025-06-30", "2025-06-30"),
        ("2026-08-31", "2026-09-18", "2026-09-30"),
    ],
)
def test_row_is_placed_on_the_first_business_month_end_on_or_after_filing(
    period: str, filed: str, expected: str
) -> None:
    from src.processing.feature_engineering import edgar_availability_dates

    placed = edgar_availability_dates(pd.DatetimeIndex([period]), pd.Series([filed]))
    assert placed[0] == pd.Timestamp(expected)


def test_missing_filing_date_falls_back_to_the_fixed_lag() -> None:
    from src.processing.feature_engineering import edgar_availability_dates

    placed = edgar_availability_dates(pd.DatetimeIndex(["2025-06-30"]), pd.Series([None]))
    assert placed[0] == pd.Timestamp("2025-08-29")  # BME two months later


def test_same_month_filings_keep_the_later_period() -> None:
    from src.processing.feature_engineering import place_edgar_rows_by_filing_date

    values = pd.Series(
        [1.0, 2.0], index=pd.DatetimeIndex(["2025-01-31", "2025-02-28"])
    )
    placed = place_edgar_rows_by_filing_date(values, pd.Series(["2025-03-05", "2025-03-20"]))
    assert list(placed.index) == [pd.Timestamp("2025-03-31")]
    assert placed.iloc[0] == 2.0


def _synthetic_db() -> tuple[sqlite3.Connection, pd.DataFrame]:
    conn = db_client.get_connection(":memory:")
    db_client.initialize_schema(conn)
    fridays = pd.date_range("2014-01-03", "2020-12-25", freq="W-FRI")
    db_client.upsert_prices(
        conn,
        [
            {"ticker": "PGR", "date": d.strftime("%Y-%m-%d"), "close": 50.0 + 0.05 * i}
            for i, d in enumerate(fridays)
        ],
    )
    rng = np.random.default_rng(7)
    months = pd.date_range("2015-01-31", "2020-06-30", freq="ME")
    rows = []
    for i, month_end in enumerate(months):
        # 9-22 days after month end, on a business day (EDGAR does not accept
        # weekend filings); the committed table's range is 9-29 days.
        filed = pd.offsets.BDay().rollforward(month_end + pd.Timedelta(days=int(rng.integers(9, 21))))
        rows.append(
            {
                "month_end": month_end.strftime("%Y-%m-%d"),
                "filing_date": filed.strftime("%Y-%m-%d"),
                "pif_growth_yoy": 0.001 * (i + 1),  # unique per month
                "combined_ratio": 90.0 + 0.01 * i,
            }
        )
    db_client.upsert_pgr_edgar_monthly(conn, rows)
    return conn, pd.DataFrame(rows)


def test_first_feature_row_with_an_edgar_row_is_on_or_after_filing_and_within_a_month(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The review's test. Under the fixed lag the first row came 32-52 days
    after filing (e.g. filed 2024-02-14, first used 2024-03-29)."""
    monkeypatch.setattr(feature_engineering, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    conn, rows = _synthetic_db()
    features = build_feature_matrix_from_db(conn, force_refresh=True)
    assert "pif_growth_yoy" in features.columns
    series = features["pif_growth_yoy"]
    checked = 0
    for row in rows.itertuples():
        hits = series.index[np.isclose(series.to_numpy(dtype=float), row.pif_growth_yoy)]
        if len(hits) == 0:
            continue
        first = hits[0]
        filed = pd.Timestamp(row.filing_date)
        assert first >= filed, (row.month_end, first, filed)
        assert (first - filed).days <= 31, (row.month_end, first, filed)
        checked += 1
    assert checked == len(rows)


@pytest.mark.skipif(not DB_PATH.exists(), reason="committed DB not available")
def test_every_committed_edgar_row_is_placed_on_or_after_filing_within_a_month() -> None:
    from src.processing.feature_engineering import edgar_availability_dates

    conn = db_client.get_connection(str(DB_PATH), read_only=True)
    try:
        monthly = db_client.get_pgr_edgar_monthly(conn)
        quarterly = db_client.get_pgr_fundamentals(conn)
    finally:
        conn.close()
    for frame in (monthly, quarterly):
        filed = pd.to_datetime(frame["filing_date"])
        placed = edgar_availability_dates(frame.index, frame["filing_date"])
        gap = (placed - pd.DatetimeIndex(filed)).days
        assert (gap >= 0).all()
        assert (gap <= 31).all()
