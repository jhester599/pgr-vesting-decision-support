"""Tests for the canonical split registry and AV split detection (review F03/F05).

Before the 2026-09-25 review fixes, splits lived in three hand-maintained lists
(``scripts/weekly_fetch.py``, ``scripts/apply_split_history.py`` and
``config/features.py``), none of which contained VOO's 2013 1-for-2 reverse
split or VGT's 2026 8-for-1 split.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

import config
from src.database import db_client


def _split(ticker: str, date: str) -> dict:
    matches = [
        s for s in config.KNOWN_SPLITS
        if s["ticker"] == ticker and s["split_date"] == date
    ]
    assert len(matches) == 1, f"{ticker} {date} missing from config.KNOWN_SPLITS"
    return matches[0]


class TestCanonicalSplitSource:
    def test_voo_reverse_split_present(self) -> None:
        split = _split("VOO", "2013-10-24")
        assert split["split_ratio"] == pytest.approx(0.5)
        assert split["numerator"] == 1.0
        assert split["denominator"] == 2.0

    def test_vgt_split_present(self) -> None:
        split = _split("VGT", "2026-04-21")
        assert split["split_ratio"] == pytest.approx(8.0)

    def test_every_split_has_consistent_ratio_and_evidence(self) -> None:
        keys = set()
        for split in config.KNOWN_SPLITS:
            assert split["split_ratio"] == pytest.approx(
                split["numerator"] / split["denominator"]
            )
            assert split["evidence"], f"{split['ticker']} {split['split_date']} lacks evidence"
            pd.Timestamp(split["split_date"])  # parses
            keys.add((split["ticker"], split["split_date"]))
        assert len(keys) == len(config.KNOWN_SPLITS)

    def test_pgr_known_splits_derived_from_canonical_list(self) -> None:
        pgr = [
            {"date": s["split_date"], "ratio": s["split_ratio"]}
            for s in config.KNOWN_SPLITS
            if s["ticker"] == "PGR"
        ]
        assert config.PGR_KNOWN_SPLITS == pgr
        assert len(pgr) == 3

    def test_scripts_have_no_private_split_lists(self) -> None:
        weekly = importlib.import_module("scripts.weekly_fetch")
        apply_mod = importlib.import_module("scripts.apply_split_history")
        assert not hasattr(weekly, "_KNOWN_SPLITS")
        assert not hasattr(apply_mod, "KNOWN_SPLITS")

    def test_weekly_seed_writes_canonical_list(self, tmp_path) -> None:
        weekly = importlib.import_module("scripts.weekly_fetch")
        conn = db_client.get_connection(str(tmp_path / "t.db"))
        db_client.initialize_schema(conn)
        weekly._seed_known_splits(conn)
        n = conn.execute("SELECT COUNT(*) FROM split_history").fetchone()[0]
        assert n == len(config.KNOWN_SPLITS)
        voo = db_client.get_splits(conn, "VOO")
        assert voo["split_ratio"].tolist() == [0.5]
        conn.close()


# ---------------------------------------------------------------------------
# Split detection from Alpha Vantage adjusted series
# ---------------------------------------------------------------------------

def _weekly_adjusted_payload(
    closes: list[float],
    adj_factors: list[float],
    start: str = "2026-03-27",
) -> dict:
    dates = pd.date_range(start=start, periods=len(closes), freq="W-FRI")
    series = {}
    for dt, close, factor in zip(dates, closes, adj_factors):
        series[dt.strftime("%Y-%m-%d")] = {
            "1. open": str(close),
            "2. high": str(close),
            "3. low": str(close),
            "4. close": str(close),
            "5. adjusted close": str(close * factor),
            "6. volume": "1000",
            "7. dividend amount": "0.0000",
        }
    return {"Meta Data": {}, "Weekly Adjusted Time Series": series}


class TestSplitDetection:
    def test_forward_split_detected_from_weekly_adjusted(self) -> None:
        from src.ingestion.split_detector import detect_splits_from_adjusted_series

        # 8-for-1 between the 3rd and 4th bar: pre-split closes ~800 are
        # adjusted down by 8 (factor 1/8), post-split factor 1.
        closes = [790.0, 800.0, 805.0, 104.0, 105.0]
        factors = [0.125, 0.125, 0.125, 1.0, 1.0]
        payload = _weekly_adjusted_payload(closes, factors)
        detected = detect_splits_from_adjusted_series(payload, "VGT")
        assert len(detected) == 1
        assert detected[0]["ticker"] == "VGT"
        assert detected[0]["split_ratio"] == pytest.approx(8.0)
        assert detected[0]["bar_date"] == "2026-04-17"

    def test_reverse_split_detected(self) -> None:
        from src.ingestion.split_detector import detect_splits_from_adjusted_series

        closes = [79.0, 79.9, 161.2, 161.3]
        factors = [2.0, 2.0, 1.0, 1.0]
        detected = detect_splits_from_adjusted_series(
            _weekly_adjusted_payload(closes, factors, start="2013-10-11"), "VOO"
        )
        assert [d["split_ratio"] for d in detected] == [pytest.approx(0.5)]
        assert detected[0]["numerator"] == 1.0
        assert detected[0]["denominator"] == 2.0

    def test_dividend_adjustments_are_not_splits(self) -> None:
        from src.ingestion.split_detector import detect_splits_from_adjusted_series

        closes = [100.0, 100.0, 100.0, 100.0]
        factors = [0.97, 0.98, 0.99, 1.0]  # 1 % dividend adjustments
        detected = detect_splits_from_adjusted_series(
            _weekly_adjusted_payload(closes, factors), "BND"
        )
        assert detected == []

    def test_daily_adjusted_split_coefficient_used_directly(self) -> None:
        from src.ingestion.split_detector import detect_splits_from_adjusted_series

        payload = {
            "Time Series (Daily)": {
                "2026-04-20": {"4. close": "805.0", "5. adjusted close": "100.6",
                               "8. split coefficient": "1.0"},
                "2026-04-21": {"4. close": "101.0", "5. adjusted close": "101.0",
                               "8. split coefficient": "8.0"},
            }
        }
        detected = detect_splits_from_adjusted_series(payload, "VGT")
        assert len(detected) == 1
        assert detected[0]["bar_date"] == "2026-04-21"
        assert detected[0]["split_ratio"] == pytest.approx(8.0)

    def test_reconcile_flags_unknown_and_accepts_known(self) -> None:
        from src.ingestion.split_detector import reconcile_detected_splits

        detected = [
            {"ticker": "VGT", "bar_date": "2026-04-24", "split_ratio": 8.0},
            {"ticker": "VOO", "bar_date": "2013-10-25", "split_ratio": 0.5},
            {"ticker": "XYZ", "bar_date": "2020-01-10", "split_ratio": 4.0},
        ]
        known = [
            {"ticker": "VGT", "split_date": "2026-04-21", "split_ratio": 8.0},
            {"ticker": "VOO", "split_date": "2013-10-24", "split_ratio": 2.0},
        ]
        issues = reconcile_detected_splits(detected, known)
        kinds = {(i["ticker"], i["issue"]) for i in issues}
        assert kinds == {("VOO", "ratio_mismatch"), ("XYZ", "unknown_split")}


class TestDetectSplitsScript:
    def test_reports_unknown_split_and_respects_budget(
        self, tmp_path, monkeypatch
    ) -> None:
        from unittest.mock import MagicMock

        import scripts.detect_splits as ds

        db = str(tmp_path / "t.db")
        conn = db_client.get_connection(db)
        db_client.initialize_schema(conn)
        conn.close()
        monkeypatch.setattr(ds, "_CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(ds.time, "sleep", lambda s: None)
        monkeypatch.setattr(config, "AV_API_KEY", "test-key")
        monkeypatch.setattr(config, "KNOWN_SPLITS", [])
        payload = _weekly_adjusted_payload(
            [790.0, 800.0, 805.0, 104.0], [0.125, 0.125, 0.125, 1.0]
        )
        session = MagicMock()
        session.get.return_value = MagicMock(json=lambda: payload)
        monkeypatch.setattr(ds, "build_retry_session", lambda: session)

        status = ds.main(["--tickers", "VGT", "VOO", "--max-calls", "1", "--db", db])

        assert status == 1  # VGT's split is not in the (emptied) registry
        assert session.get.call_count == 1  # budget of one call
        conn = db_client.get_connection(db)
        assert db_client.get_api_request_count(
            conn, "av", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        ) == 1
        conn.close()
        # Second run is served from the day's cache: no new call.
        ds.main(["--tickers", "VGT", "--db", db])
        assert session.get.call_count == 1
