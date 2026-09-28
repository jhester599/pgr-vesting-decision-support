"""Independent expected outputs for the frozen matched endpoint controls."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import pytest

import config
from pgr_vds.research_lib import controls


def endpoint_database() -> sqlite3.Connection:
    """Build a synthetic source with cash and report amounts known by hand."""
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE daily_prices (ticker TEXT,date TEXT,close REAL);"
        "CREATE TABLE daily_dividends (ticker TEXT,ex_date TEXT,amount REAL);"
        "CREATE TABLE split_history "
        "(ticker TEXT,split_date TEXT,split_ratio REAL);"
        "CREATE TABLE pgr_edgar_monthly "
        "(month_end TEXT,filing_date TEXT,book_value_per_share REAL);"
    )
    dates = pd.date_range("2018-01-31", "2022-08-31", freq="BME")
    connection.executemany(
        "INSERT INTO daily_prices VALUES ('PGR',?,100)",
        [(str(date.date()),) for date in dates],
    )
    payments = [
        (f"{year}-{month:02d}-02", amount)
        for year in range(2018, 2023)
        for month in (1, 4, 7, 10)
        for amount in [
            {2019: 0.5, 2020: 0.7, 2022: 0.9}.get(year, 0.1)
            if month == 1
            else 0.1
        ]
    ]
    payments += [("2018-12-15", 0.2), ("2019-12-15", 0.2)]
    connection.executemany(
        "INSERT INTO daily_dividends VALUES ('PGR',?,?)",
        payments,
    )
    connection.executemany(
        "INSERT INTO pgr_edgar_monthly VALUES (?,?,?)",
        [
            (f"{year}-10-31", f"{year}-11-15", value)
            for year, value in [
                (2018, 100.0),
                (2019, 120.0),
                (2020, 144.0),
                (2021, 172.8),
                (2022, 207.36),
            ]
        ],
    )
    return connection


def control_row(frame: pd.DataFrame, endpoint: str, date: str) -> pd.Series:
    """Select one independently named monthly endpoint fixture."""
    selected = frame.loc[
        (frame["endpoint"] == endpoint) & (frame["date"] == date)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_endpoint_cash_bvps_and_positive_annual_expected_outputs() -> None:
    connection = endpoint_database()
    features = pd.DataFrame(
        index=pd.date_range(
            "2018-11-30",
            "2021-11-30",
            freq="BME",
        )
    )
    frame, definitions = controls.endpoint_controls(
        connection,
        features,
        pd.Timestamp("2022-09-29"),
    )
    cash = control_row(frame, "cash_dividend_12m", "2019-11-29")
    assert cash["y_true"] == pytest.approx(1.2, abs=1e-14)
    assert cash["y_hat"] == pytest.approx(1.0, abs=1e-14)
    assert cash["residual"] == pytest.approx(0.2, abs=1e-14)
    growth = control_row(frame, "bvps_growth_12m", "2018-11-30")
    assert growth["y_true"] == pytest.approx(0.2, abs=1e-14)
    assert growth["available"] == pd.Timestamp("2019-11-15")
    first = control_row(frame, "annual_excess_to_bvps", "2018-11-30")
    second = control_row(frame, "annual_excess_to_bvps", "2019-11-29")
    assert first["y_true"] == pytest.approx(0.006, abs=1e-14)
    assert first["available"] == pd.Timestamp("2019-02-28")
    assert first["warmup"]
    assert second["y_true"] == pytest.approx(0.8 / 120.0, abs=1e-14)
    assert second["y_hat"] == pytest.approx(0.006, abs=1e-14)
    assert second["target_end"] == pd.Timestamp("2020-02-29")
    annual = frame.loc[frame["endpoint"] == "annual_excess_to_bvps"]
    assert annual["event_id"].nunique() == len(annual) == 3
    assert not (annual["date"] == pd.Timestamp("2020-11-30")).any()
    support = definitions["annual_excess_to_bvps"]["support"]
    assert support["n_unique_annual_events"] == 3
    assert support["status"] == "unscorable"
    connection.close()


def test_drip_control_is_signed_absolute_asset_return() -> None:
    connection = endpoint_database()
    connection.execute("DELETE FROM daily_dividends")
    connection.execute(
        "UPDATE daily_prices SET close=90 WHERE date>='2019-05-31'"
    )
    features = pd.DataFrame(index=pd.to_datetime(["2018-11-30"]))
    frame, _ = controls.endpoint_controls(
        connection,
        features,
        pd.Timestamp("2020-01-31"),
    )
    row = control_row(frame, "pgr_drip_return_6m", "2018-11-30")
    assert row["y_true"] == pytest.approx(-0.1, abs=1e-14)
    connection.close()


def test_no_quarantine_cash_or_unfiled_bvps_outcome_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = endpoint_database()
    boundary = pd.Timestamp("2021-09-30")
    observed_queries: list[str] = []
    connection.set_trace_callback(observed_queries.append)
    features = pd.DataFrame(
        index=pd.date_range(
            "2018-11-30",
            "2021-08-31",
            freq="BME",
        )
    )
    original = controls.cash_window

    def bounded_sum(
        values: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        assert end < boundary
        return original(values, start, end)

    monkeypatch.setattr(controls, "cash_window", bounded_sum)
    frame, _ = controls.endpoint_controls(connection, features, boundary)
    assert (frame["target_end"] < boundary).all()
    assert (frame["available"] < boundary).all()
    queries = [query for query in observed_queries if "SELECT" in query]
    assert queries
    assert all("2021-09-30" in query and " < " in query for query in queries)
    connection.close()


def test_cash_and_bvps_keep_one_origin_share_across_split() -> None:
    connection = endpoint_database()
    connection.execute("DELETE FROM daily_dividends")
    connection.execute("DELETE FROM pgr_edgar_monthly")
    connection.execute(
        "INSERT INTO split_history VALUES ('PGR','2019-06-01',2)"
    )
    connection.executemany(
        "INSERT INTO daily_dividends VALUES ('PGR',?,?)",
        [("2018-12-01", 1.0), ("2019-07-01", 0.5)],
    )
    connection.executemany(
        "INSERT INTO pgr_edgar_monthly VALUES (?,?,?)",
        [
            ("2018-10-31", "2018-11-15", 100.0),
            ("2019-10-31", "2019-11-15", 60.0),
        ],
    )
    features = pd.DataFrame(index=pd.to_datetime(["2018-11-30"]))
    frame, _ = controls.endpoint_controls(
        connection,
        features,
        pd.Timestamp("2020-01-31"),
    )
    assert (
        control_row(frame, "cash_dividend_12m", "2018-11-30")["y_true"] == 2.0
    )
    assert control_row(frame, "bvps_growth_12m", "2018-11-30")[
        "y_true"
    ] == pytest.approx(0.2, abs=1e-14)
    connection.close()


def classifier_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced constant-feature logistic has an independent .5 probability."""
    dates = pd.date_range("2000-01-31", periods=108, freq="BME")
    features = pd.DataFrame(
        0.0,
        index=dates,
        columns=config.MODEL_FEATURE_OVERRIDES["ridge"],
    )
    records: list[dict[str, Any]] = []
    for position, date in enumerate(dates):
        for benchmark in config.INVESTABLE_CLASSIFIER_BASE_WEIGHTS:
            records.append(
                {
                    "date": date,
                    "benchmark": benchmark,
                    "horizon": 6,
                    "available": date + pd.offsets.BMonthEnd(6),
                    "y_true": 0.1 if position % 2 == 0 else -0.1,
                }
            )
    return features, pd.DataFrame(records)


def test_path_b_fixed_probability_label_and_strict_fold_support() -> None:
    features, targets = classifier_inputs()
    frame, ledger = controls.path_b_stream(features, targets)
    assert len(frame) == 36
    assert frame.iloc[0]["date"] == features.index[72]
    np.testing.assert_allclose(frame["raw_probability"], 0.5, atol=1e-12)
    assert frame.iloc[0]["y_true"] == 0
    assert frame.iloc[1]["y_true"] == 1
    assert frame.iloc[0]["base_prediction"] == 0
    assert frame.iloc[0]["n_mature_labels"] == 67
    assert frame.iloc[:29]["calibration_warmup"].all()
    assert frame["probability"].notna().sum() == 7
    assert "lower" not in frame and "upper" not in frame
    inner = [entry for entry in ledger if entry["kind"] == "inner"]
    assert len(inner) == 18
    assert min(entry["n_train"] for entry in inner) == 30
    assert all(entry["gap"] == 12 for entry in inner)


def test_missing_benchmark_and_single_class_are_unscorable() -> None:
    features, targets = classifier_inputs()
    missing = targets.loc[targets["benchmark"] != "VGT"]
    frame, ledger = controls.path_b_stream(features, missing)
    assert frame.empty
    assert ledger and all(entry["status"] == "unscorable" for entry in ledger)
    targets["y_true"] = -0.1
    frame, ledger = controls.path_b_stream(features, targets)
    assert frame.empty
    assert any("class" in entry.get("reason", "").lower() for entry in ledger)


def test_late_availability_does_not_become_earlier_training() -> None:
    features, targets = classifier_inputs()
    targets.loc[targets["date"] < features.index[60], "available"] = (
        pd.Timestamp("2099-01-31")
    )
    frame, ledger = controls.path_b_stream(features, targets)
    assert frame.empty
    assert any(entry["n_train"] < 24 for entry in ledger)


def test_path_b_rejects_availability_before_target_end() -> None:
    features, targets = classifier_inputs()
    targets["available"] = targets["date"]
    with pytest.raises(ValueError, match="target end"):
        controls.path_b_stream(features, targets)


def test_cash_naive_is_mature_target_mean_not_past_cash_amount() -> None:
    connection = endpoint_database()
    connection.execute("DELETE FROM daily_prices")
    connection.execute("DELETE FROM daily_dividends")
    connection.executemany(
        "INSERT INTO daily_prices VALUES ('PGR',?,100)",
        [
            (str(date.date()),)
            for date in pd.date_range("2012-06-29", "2015-06-30", freq="BME")
        ],
    )
    connection.executemany(
        "INSERT INTO daily_dividends VALUES ('PGR',?,?)",
        [("2013-06-15", 5), ("2014-06-15", 7), ("2015-06-15", 9)],
    )
    frame, _ = controls.endpoint_controls(
        connection,
        pd.DataFrame(index=pd.to_datetime(["2014-06-30"])),
        pd.Timestamp("2015-07-01"),
    )
    cash = control_row(frame, "cash_dividend_12m", "2014-06-30")
    assert cash["y_true"] == 9
    assert cash["y_hat"] == 7
    assert cash["n_mature_labels"] == 13
    assert cash["naive"] == pytest.approx(67 / 13)
    connection.close()


def classifier_summary_inputs() -> pd.DataFrame:
    """Perfect direction with .25 probability error has known proper scores."""
    dates = pd.date_range("2000-01-31", periods=12, freq="BME")
    return pd.DataFrame(
        {
            "date": dates,
            "benchmark": "PATH_B",
            "y_true": np.tile([0, 1], 6),
            "raw_probability": np.tile([0.25, 0.75], 6),
            "probability": [0.99] * 6 + [0.25, 0.75] * 3,
            "calibration_warmup": [True] * 6 + [False] * 6,
            "base_prediction": 0,
            "base_positive_rate": 0.25,
            "n_mature_labels": 4,
            "available": dates + pd.offsets.BMonthEnd(6),
        }
    )


def test_classifier_summary_known_scores_skill_and_warmup() -> None:
    result = controls.classifier_summary(classifier_summary_inputs())
    raw = result["raw"]
    calibrated = result["calibrated"]
    for summary in (raw, calibrated):
        assert summary["brier"] == pytest.approx(0.0625)
        assert summary["log_loss"] == pytest.approx(-np.log(0.75))
        assert summary["ece"] == pytest.approx(0.25)
        assert summary["hit_rate"] == 1.0
        assert summary["base_hit_rate"] == 0.5
        assert summary["directional_skill"] == 0.5
        assert "oos_r2" not in summary
        assert "primary_p" not in summary
        assert "coverage_80" not in summary
    assert raw["n_probability"] == raw["n_dates"] == 12
    assert raw["directional_skill_p"] == pytest.approx(1 / 2001)
    assert raw["bootstrap_replicates"] == 2000
    assert raw["block_length"] == 6
    assert raw["seed"] == 20260926
    assert calibrated["n_probability"] == 6
    assert not calibrated["inference_supported"]
    assert np.isnan(calibrated["directional_skill_p"])
    assert result["support"]["n_calibration_warmup"] == 6
    assert result["support"]["n_base_warmup"] == 0
    assert result["endpoint"] == "path_b_same_label_classifier"


def test_classifier_summary_probability_and_majority_ties() -> None:
    frame = classifier_summary_inputs()
    frame["y_true"] = 0
    frame["raw_probability"] = 0.5
    frame["base_positive_rate"] = 0.5
    frame["base_prediction"] = 1
    result = controls.classifier_summary(frame)
    assert result["raw"]["hit_rate"] == 1.0
    assert result["raw"]["base_hit_rate"] == 0.0
    assert result["raw"]["directional_skill"] == 1.0
    frame["base_prediction"] = 0
    with pytest.raises(ValueError, match="majority"):
        controls.classifier_summary(frame)
    frame["base_prediction"] = 1
    frame.loc[0, "raw_probability"] = 1.01
    with pytest.raises(ValueError, match="probability"):
        controls.classifier_summary(frame)


def test_endpoint_summary_separate_scores_and_no_annual_inference() -> None:
    dates = pd.date_range("2000-01-31", periods=3, freq="BME")
    cash = pd.DataFrame(
        {
            "date": dates,
            "endpoint": "cash_dividend_12m",
            "y_true": [2.0, 4.0, 6.0],
            "y_hat": [1.0, 5.0, 5.0],
            "naive": [0.0, 2.0, 4.0],
            "warmup": False,
            "event_id": ["cash:1", "cash:2", "cash:3"],
        }
    )
    annual = cash.copy()
    annual["endpoint"] = "annual_excess_to_bvps"
    annual["event_id"] = ["annual:1", "annual:2", "annual:3"]
    bvps = cash.copy()
    bvps["endpoint"] = "bvps_growth_12m"
    bvps["event_id"] = "one-repeated-report"
    definitions = {
        "cash_dividend_12m": {"unit": "USD/share", "horizon": 12},
        "annual_excess_to_bvps": {"unit": "fraction", "horizon": 12},
        "bvps_growth_12m": {"unit": "fraction", "horizon": 12},
    }
    result = controls.endpoint_summary(
        pd.concat([cash, annual, bvps], ignore_index=True), definitions
    )
    assert result["cash_dividend_12m"]["oos_r2"] == pytest.approx(0.75)
    assert result["cash_dividend_12m"]["mae"] == 1.0
    assert result["cash_dividend_12m"]["rmse"] == 1.0
    assert result["cash_dividend_12m"]["n_scored"] == 3
    assert result["bvps_growth_12m"]["n_unique_events"] == 1
    assert result["bvps_growth_12m"]["inference_supported"] is False
    assert result["annual_excess_to_bvps"]["status"] == "unscorable"
    assert result["annual_excess_to_bvps"]["oos_r2"] is None
    assert result["annual_excess_to_bvps"]["mae"] is None
    assert result["annual_excess_to_bvps"]["n_unique_events"] == 3
    assert all("primary_p" not in summary for summary in result.values())
