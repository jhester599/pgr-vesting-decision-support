"""Independent expected outputs for research-only causal adapters."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from pgr_vds.research_lib.adapters import (
    AsOfConnection,
    bounded_features,
    drip_return,
    ensemble_stream,
    nested_ridge,
    prequential_intervals,
    regular_dividend_baseline,
    cash_window,
    excess_ratio,
    bvps_growth,
)


def test_split_fractional_drip_and_start_exclusion() -> None:
    prices = pd.Series(
        [100.0, 50.0, 55.0],
        index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"]),
    )
    dividends = pd.Series(
        [90.0, 1.0], index=pd.to_datetime(["2020-01-31", "2020-02-28"])
    )
    splits = pd.Series([2.0], index=pd.to_datetime(["2020-02-28"]))
    shares, result = drip_return(
        prices, dividends, splits, prices.index[0], prices.index[-1]
    )
    assert shares == pytest.approx(2.04, abs=1e-14)
    assert result == pytest.approx(0.122, abs=1e-14)


def test_off_bar_dividend_uses_last_observed_raw_close() -> None:
    prices = pd.Series(
        [100.0, 110.0], index=pd.to_datetime(["2020-01-31", "2020-02-28"])
    )
    div = pd.Series([1.0], index=pd.to_datetime(["2020-02-03"]))
    shares, result = drip_return(
        prices, div, pd.Series(dtype=float), prices.index[0], prices.index[-1]
    )
    assert shares == pytest.approx(1.01)
    assert result == pytest.approx(0.111)


def test_sql_adapter_excludes_unfiled_and_future_rows(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.db"
    with sqlite3.connect(path) as writer:
        writer.execute(
            "CREATE TABLE pgr_edgar_monthly "
            "(month_end TEXT, filing_date TEXT, value REAL)"
        )
        writer.executemany(
            "INSERT INTO pgr_edgar_monthly VALUES (?,?,?)",
            [
                ("2020-01-31", "2020-02-14", 1.0),
                ("2020-02-28", "2020-03-13", 999.0),
            ],
        )
    conn = sqlite3.connect(
        path.as_uri() + "?mode=ro&immutable=1",
        uri=True,
        factory=AsOfConnection,
    )
    conn.as_of = pd.Timestamp("2020-02-28")
    frame = pd.read_sql_query("SELECT * FROM pgr_edgar_monthly", conn)
    assert frame["value"].tolist() == [1.0]
    assert conn.execute("SELECT value FROM pgr_edgar_monthly").fetchall() == [
        (1.0,)
    ]
    conn.close()


def test_regular_cash_baseline_excludes_future_and_specials() -> None:
    cash = pd.Series(
        [0.1, 0.1, 2.0, 0.1, 999.0],
        index=pd.to_datetime(
            [
                "2019-01-01",
                "2019-04-01",
                "2020-01-01",
                "2020-04-01",
                "2020-12-01",
            ]
        ),
    )
    assert regular_dividend_baseline(cash, pd.Timestamp("2020-11-30")) == 0.1


def test_ensemble_uses_only_mature_past_and_warmup_is_unscored() -> None:
    dates = pd.date_range("2000-01-31", periods=10, freq="BME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "available": dates + pd.offsets.BMonthEnd(6),
            "benchmark": "A",
            "ridge": 1.0,
            "gbt": 3.0,
            "y_true": 1.0,
        }
    )
    history = frame.rename(columns={"available": "label_available"})
    first = ensemble_stream(frame, history, min_shrinkage=36)
    mutated = frame.copy()
    mutated.loc[6:, "y_true"] = 999.0
    second = ensemble_stream(mutated, history, min_shrinkage=36)
    np.testing.assert_array_equal(first["y_hat"], second["y_hat"])
    assert first.iloc[0]["y_hat"] == 2.0
    assert first.iloc[0]["naive"] != first.iloc[0]["naive"]
    assert first.iloc[6]["naive"] == 1.0
    intervals = prequential_intervals(first, min_support=2)
    assert intervals.iloc[:7]["interval_warmup"].all()
    assert intervals.iloc[7]["lower"] == pytest.approx(0.4)
    assert intervals.iloc[7]["upper"] == pytest.approx(2.4)


def test_nested_ridge_constant_expected_intercept_and_inner_support() -> None:
    dates = pd.date_range("2000-01-31", periods=60, freq="BME")
    x = pd.DataFrame({"constant": np.zeros(60)}, index=dates)
    y = pd.Series(np.repeat(2.0, 60), index=dates)
    pred, alpha, ledger = nested_ridge(x, y, x.iloc[[-1]], 6)
    assert pred.tolist() == pytest.approx([2.0], abs=1e-12)
    assert alpha == pytest.approx(1e-4)
    assert [row["n_train"] for row in ledger] == [30, 36, 42]


def test_nested_future_validation_features_do_not_change_first_transform() -> (
    None
):
    dates = pd.date_range("2000-01-31", periods=60, freq="BME")
    x = pd.DataFrame({"x": np.arange(60, dtype=float)}, index=dates)
    y = pd.Series(np.zeros(60), index=dates)
    _, _, before = nested_ridge(x, y, x.iloc[[-1]], 6)
    x.iloc[42:] = 99999.0
    _, _, after = nested_ridge(x, y, x.iloc[[-1]], 6)
    assert before[0]["scale_mean"] == after[0]["scale_mean"] == [14.5]


def test_fixed_endpoint_units_and_availability() -> None:
    cash = pd.Series(
        [10.0, 0.1, 2.0, 0.1],
        index=pd.to_datetime(
            ["2020-11-30", "2021-01-04", "2021-03-01", "2021-04-01"]
        ),
    )
    assert cash_window(
        cash, pd.Timestamp("2020-11-30"), pd.Timestamp("2021-03-31")
    ) == pytest.approx(2.1)
    assert excess_ratio(2.1, 0.1, 20.0) == pytest.approx(0.1)
    assert bvps_growth(20.0, 22.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        bvps_growth(0.0, 22.0)


def test_source_ledger_has_independent_filing_and_macro_placement() -> None:
    """March 3 filing and lagged January NFCI both first enter March 31."""
    from pgr_vds.research_lib import adapters

    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE pgr_edgar_monthly "
        "(month_end TEXT,filing_date TEXT,secret_value REAL);"
        "CREATE TABLE pgr_fundamentals_quarterly "
        "(period_end TEXT,filing_date TEXT,secret_value REAL);"
        "CREATE TABLE fred_macro_monthly "
        "(series_id TEXT,month_end TEXT,value REAL);"
    )
    connection.executemany(
        "INSERT INTO pgr_edgar_monthly VALUES (?,?,?)",
        [("2024-12-31", "2025-03-03", 1.0),
         ("2025-04-30", "2025-05-15", 99999.0)],
    )
    connection.execute(
        "INSERT INTO pgr_fundamentals_quarterly VALUES (?,?,?)",
        ("2024-12-31", "2025-03-03", 2.0),
    )
    connection.executemany(
        "INSERT INTO fred_macro_monthly VALUES (?,?,?)",
        [("NFCI", "2025-01-31", 3.0),
         ("GS10", "2025-02-28", 4.0),
         ("NFCI", "2025-03-31", 99999.0),
         ("UNUSED", "2025-01-31", 99999.0)],
    )
    queries: list[str] = []
    connection.set_trace_callback(queries.append)
    ledger = adapters.feature_source_ledger(
        connection, pd.Timestamp("2025-05-01")
    )
    assert len(ledger) == 4
    assert ledger["usable_from"].tolist() == [pd.Timestamp("2025-03-31")] * 4
    assert set(ledger["source"]) == {
        "pgr_edgar_monthly", "pgr_fundamentals_quarterly",
        "fred_macro_monthly",
    }
    macro = ledger.loc[ledger["source"] == "fred_macro_monthly"]
    assert set(macro["series_id"]) == {"NFCI", "GS10"}
    assert dict(zip(macro["series_id"], macro["lag_months"])) == {
        "NFCI": 2, "GS10": 1,
    }
    assert macro["historical_vintage_assumption"].str.contains(
        "current vintage", case=False
    ).all()
    assert "value" not in ledger and "secret_value" not in ledger
    assert all("2025-05-01" in query for query in queries)
    connection.close()


def test_source_ledger_checks_first_origin_and_unknown_filings() -> None:
    from pgr_vds.research_lib import adapters

    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE pgr_edgar_monthly "
        "(month_end TEXT,filing_date TEXT);"
        "CREATE TABLE pgr_fundamentals_quarterly "
        "(period_end TEXT,filing_date TEXT);"
        "CREATE TABLE fred_macro_monthly "
        "(series_id TEXT,month_end TEXT);"
        "INSERT INTO pgr_edgar_monthly VALUES "
        "('2024-12-31','2025-03-03');"
    )
    origins = pd.to_datetime(["2025-02-28", "2025-03-31"])
    ledger = adapters.feature_source_ledger(
        connection, pd.Timestamp("2025-05-01"), origins
    )
    assert ledger["first_origin_using_source"].tolist() == [
        pd.Timestamp("2025-03-31")
    ]
    assert ledger["source_available_by_first_origin"].tolist() == [True]
    connection.execute(
        "INSERT INTO pgr_edgar_monthly VALUES ('2025-01-31',NULL)"
    )
    with pytest.raises(ValueError, match="filing"):
        adapters.feature_source_ledger(
            connection, pd.Timestamp("2025-05-01"), origins
        )
    connection.close()


def test_aci_retains_production_finite_sample_adjustment() -> None:
    dates = pd.date_range("2000-01-31", periods=26, freq="BME")
    panel = pd.DataFrame(
        {
            "date": dates,
            "available": dates + pd.offsets.BMonthEnd(6),
            "benchmark": "A",
            "y_hat": 0.0,
            "z": 0.0,
            "y_true": np.arange(1, 27, dtype=float),
        }
    )
    result = prequential_intervals(panel)
    # Nineteen successive misses push alpha to .01. ceil(.99*21)/20
    # is capped at1, so the final radius is the mature maximum20.
    assert result.iloc[-1]["lower"] == -20.0
    assert result.iloc[-1]["upper"] == 20.0


def test_active_platt_coin_and_future_outcomes_are_causal() -> None:
    dates = pd.date_range("2000-01-31", periods=32, freq="BME")
    panel = pd.DataFrame(
        {
            "date": dates,
            "available": dates + pd.offsets.BMonthEnd(6),
            "benchmark": "A",
            "y_hat": 0.0,
            "z": 0.0,
            "y_true": np.tile([-1.0, 1.0], 16),
        }
    )
    result = prequential_intervals(panel)
    assert result.iloc[:25]["calibration_warmup"].all()
    assert result.iloc[25]["probability"] == pytest.approx(0.5, abs=1e-12)
    panel.loc[26:, "y_true"] = 999.0
    changed = prequential_intervals(panel)
    np.testing.assert_array_equal(
        result.iloc[:32]["probability"], changed.iloc[:32]["probability"]
    )


def test_delayed_inner_labels_fail_closed() -> None:
    dates = pd.date_range("2000-01-31", periods=60, freq="BME")
    x = pd.DataFrame({"x": np.zeros(60)}, index=dates)
    y = pd.Series(np.ones(60), index=dates)
    available = pd.Series(pd.Timestamp("2099-01-01"), index=dates)
    with pytest.raises(ValueError, match="Unsupported"):
        nested_ridge(x, y, x.iloc[[-1]], 6, available=available)


def test_empty_inner_history_is_never_an_implicit_first_penalty() -> None:
    dates = pd.date_range("2000-01-31", periods=20, freq="BME")
    x = pd.DataFrame({"x": np.zeros(20)}, index=dates)
    with pytest.raises(ValueError, match="Unsupported"):
        nested_ridge(x, pd.Series(np.ones(20), index=dates), x.iloc[[-1]], 6)


def test_research_feature_catalog_retains_sparse_and_empty_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Future source counts cannot choose an earlier feature catalog."""
    import config
    from src.processing import feature_engineering as features

    database = tmp_path / "empty.db"
    with sqlite3.connect(database):
        pass
    weeks = pd.date_range("2019-01-04", "2023-12-29", freq="W-FRI")
    prices = pd.DataFrame({"close": 100.0}, index=weeks)
    months = pd.date_range("2019-01-31", "2023-12-29", freq="BME")
    monthly = pd.DataFrame(
        {
            "pif_growth_yoy": np.where(months.year <= 2020, np.nan, 0.1),
            "investment_book_yield": np.nan,
        },
        index=months,
    )
    dividends = pd.DataFrame(
        {"dividend": pd.Series(dtype=float)}, index=pd.DatetimeIndex([])
    )
    splits = pd.DataFrame(
        {"split_ratio": pd.Series(dtype=float)}, index=pd.DatetimeIndex([])
    )

    def synthetic_sources(connection: AsOfConnection) -> pd.DataFrame:
        cutoff = connection.as_of
        return features.build_feature_matrix(
            prices.loc[:cutoff], dividends, splits,
            pgr_monthly=monthly.loc[:cutoff], force_refresh=True,
        )

    monkeypatch.setattr(
        features, "build_feature_matrix_from_db", synthetic_sources
    )
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
    original_threshold = config.WFO_MIN_GAINSHARE_OBS
    early = bounded_features(
        database, pd.Timestamp("2020-12-31"), tmp_path / "early"
    )
    later = bounded_features(
        database, pd.Timestamp("2023-12-29"), tmp_path / "later"
    )
    fixed = set(config.MODEL_FEATURE_OVERRIDES["ridge"]) | set(
        config.MODEL_FEATURE_OVERRIDES["gbt"]
    )
    assert fixed <= set(early.columns) & set(later.columns)
    assert early["pif_growth_yoy"].isna().all()
    assert early["investment_book_yield"].isna().all()
    pd.testing.assert_frame_equal(
        early.loc[:, sorted(fixed)],
        later.loc[early.index, sorted(fixed)],
    )
    assert config.WFO_MIN_GAINSHARE_OBS == original_threshold
