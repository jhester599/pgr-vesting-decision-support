"""Review 2026-09-25, step 9 (WP12, F28): tests for surviving mutations.

F28's mutation study changed one production formula at a time and ran the
related tests. After steps 1-7, 10 of its 18 mutations still survived. Each
test below names the mutation it kills (M-numbers as in the step 9 report,
``docs/reviews/2026-09-25_step9_test_hardening.md``). Expected values are
worked out by hand from the fixture, not recomputed with the production
formula.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.database import db_client
from src.models.conformal import aci_adjusted_interval, split_conformal_interval
from src.models.consensus_shadow import build_quality_weights
from src.models.wfo_engine import predict_current, run_wfo
from src.processing.feature_engineering import (
    build_feature_matrix,
    build_feature_matrix_from_db,
)
from src.reporting.decision_rendering import sell_pct_from_consensus


# ---------------------------------------------------------------------------
# Feature engineering (M02, M03)
# ---------------------------------------------------------------------------

def _weekly_prices(start: str, end: str) -> pd.DataFrame:
    fridays = pd.date_range(start, end, freq="W-FRI")
    close = 50.0 + 0.05 * np.arange(len(fridays))
    return pd.DataFrame({"close": close}, index=fridays)


def _empty_splits() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"],
        index=pd.DatetimeIndex([], name="split_date"),
    )


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(columns=["dividend", "source"], index=pd.DatetimeIndex([], name="ex_date"))


def _db_with_quarterly_roe() -> tuple[sqlite3.Connection, list[dict]]:
    conn = db_client.get_connection(":memory:")
    db_client.initialize_schema(conn)
    prices = _weekly_prices("2014-01-03", "2020-12-25")
    db_client.upsert_prices(
        conn,
        [
            {"ticker": "PGR", "date": d.strftime("%Y-%m-%d"), "close": float(c)}
            for d, c in prices["close"].items()
        ],
    )
    quarters = pd.date_range("2015-03-31", "2020-06-30", freq="QE")
    rows = []
    for i, q_end in enumerate(quarters):
        # 10-Qs are filed 30-45 days after the quarter; use 40 days, rolled
        # to a business day.
        filed = pd.offsets.BDay().rollforward(q_end + pd.Timedelta(days=40))
        rows.append(
            {
                "period_end": q_end.strftime("%Y-%m-%d"),
                "roe": 0.10 + 0.001 * i,  # unique per quarter
                "filing_date": filed.strftime("%Y-%m-%d"),
                "source": "test",
            }
        )
    db_client.upsert_pgr_fundamentals(conn, rows)
    return conn, rows


def test_m02_quarterly_roe_enters_on_or_after_its_filing_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M02: removing the filing-date placement of quarterly ROE
    (``build_feature_matrix_from_db``) puts each value on its quarter end,
    30-45 days before it was public."""
    from src.processing import feature_engineering

    monkeypatch.setattr(feature_engineering, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    conn, rows = _db_with_quarterly_roe()
    features = build_feature_matrix_from_db(conn, force_refresh=True)
    assert "roe" in features.columns
    roe = features["roe"]
    checked = 0
    for row in rows:
        hits = roe.index[np.isclose(roe.to_numpy(dtype=float), row["roe"])]
        if len(hits) == 0:
            continue
        first = hits[0]
        filed = pd.Timestamp(row["filing_date"])
        assert first >= filed, (row["period_end"], first, filed)
        assert (first - filed).days <= 31
        checked += 1
    assert checked >= len(rows) - 1  # the last filing may fall after the prices end


def test_m03_combined_ratio_ttm_is_a_twelve_month_mean() -> None:
    """M03: ``combined_ratio_ttm`` over 3 months instead of 12.

    Combined ratio is 90 for 12 months, then 102. Three months after the
    step, the trailing-12 mean is (9 x 90 + 3 x 102) / 12 = 93.0 (a 3-month
    window gives 102.0); twelve months after, it is 102.0.
    """
    months = pd.date_range("2016-01-31", periods=72, freq="BME")
    cr = np.where(np.arange(72) < 12, 90.0, 102.0)
    pgr_monthly = pd.DataFrame({"combined_ratio": cr}, index=months)
    features = build_feature_matrix(
        _weekly_prices("2015-06-05", "2022-06-24"),
        _empty_dividends(),
        _empty_splits(),
        pgr_monthly=pgr_monthly,
        force_refresh=True,
    )
    ttm = features["combined_ratio_ttm"]
    assert ttm.loc[months[11]] == pytest.approx(90.0)
    assert ttm.loc[months[14]] == pytest.approx(93.0)
    assert ttm.loc[months[20]] == pytest.approx((3 * 90.0 + 9 * 102.0) / 12)
    assert ttm.loc[months[23]] == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# WFO (M07, M08)
# ---------------------------------------------------------------------------

def _linear_panel(n_rows: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n_rows, freq="BME")
    X = pd.DataFrame(rng.normal(size=(n_rows, 2)), index=idx, columns=["a", "b"])
    y = pd.Series(X["a"].to_numpy() + rng.normal(0, 0.3, n_rows), index=idx, name="y")
    return X, y


def test_m07_training_window_is_capped_at_sixty_months() -> None:
    """M07: ``TimeSeriesSplit(max_train_size=None)`` trains every later fold
    on all earlier history instead of the rolling 60-month window."""
    X, y = _linear_panel(160)
    result = run_wfo(X, y, model_type="ridge", target_horizon_months=6, feature_columns=["a", "b"])
    assert len(result.folds) >= 3
    assert [f.n_train for f in result.folds] == [config.WFO_TRAIN_WINDOW_MONTHS] * len(result.folds)
    last = result.folds[-1]
    assert X.index.get_loc(last.train_start) == X.index.get_loc(last.train_end) - 59


def test_m08_live_refit_trains_only_on_the_recent_window() -> None:
    """M08: ``predict_current`` refitting on the full history. Rows older
    than ``train_window_months`` are replaced by a different relation; the
    live prediction must not move."""
    X, y = _linear_panel(120, seed=3)
    wfo = run_wfo(X, y, model_type="ridge", target_horizon_months=6, feature_columns=["a", "b"])
    X_current = pd.DataFrame({"a": [1.5], "b": [0.0]}, index=[X.index[-1]])
    base = predict_current(X, y, X_current, wfo, model_type="ridge", train_window_months=60)
    y_old = y.copy()
    y_old.iloc[:60] = -3.0 * X["a"].iloc[:60]
    moved = predict_current(X, y_old, X_current, wfo, model_type="ridge", train_window_months=60)
    assert moved["predicted_return"] == pytest.approx(base["predicted_return"], rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# Quality-weighted consensus (M10, M11, M12)
# ---------------------------------------------------------------------------

_BENCHMARKS = pd.Index(["VOO", "BND", "GLD"], name="benchmark")


def _quality(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"benchmark": list(_BENCHMARKS), "nw_ic": scores})


def test_m10_lambda_weights_the_quality_share_not_the_equal_share() -> None:
    """M10: swapping the lambda mix. With IC scores 0.30/0.10/0.00 and
    lambda 0.25: w = 0.75 x 1/3 + 0.25 x (0.75, 0.25, 0) = (0.4375, 0.3125, 0.25)."""
    weights = build_quality_weights(_BENCHMARKS, _quality([0.30, 0.10, 0.00]), lambda_mix=0.25)
    assert weights.to_dict() == pytest.approx({"VOO": 0.4375, "BND": 0.3125, "GLD": 0.25})


def test_m11_negative_ic_is_clipped_to_zero_quality() -> None:
    """M11: without the clip a negative IC takes weight away from equal
    weight. IC 0.30/-0.20/0.10 with lambda 0.25 gives normalised quality
    (0.75, 0, 0.25) and w = (0.4375, 0.25, 0.3125); unclipped it would be
    (0.625, 0.0, 0.375)."""
    weights = build_quality_weights(_BENCHMARKS, _quality([0.30, -0.20, 0.10]), lambda_mix=0.25)
    assert weights.to_dict() == pytest.approx({"VOO": 0.4375, "BND": 0.25, "GLD": 0.3125})
    assert (weights >= (1 - 0.25) / 3 - 1e-12).all()


@pytest.mark.parametrize(("given", "clipped"), [(1.5, 1.0), (-0.5, 0.0)])
def test_m12_lambda_outside_unit_interval_is_clipped(given: float, clipped: float) -> None:
    """M12: without the clip lambda 1.5 gives negative weights and -0.5
    over-weights the low-IC benchmark."""
    quality = _quality([0.30, 0.10, 0.00])
    got = build_quality_weights(_BENCHMARKS, quality, lambda_mix=given)
    expected = build_quality_weights(_BENCHMARKS, quality, lambda_mix=clipped)
    pd.testing.assert_series_equal(got, expected)
    assert (got >= 0).all()


# ---------------------------------------------------------------------------
# ACTIONABLE sell mapping (M13)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("consensus", "mean_predicted", "mean_ic", "expected"),
    [
        ("UNDERPERFORM", -0.08, 0.10, 1.00),
        ("UNDERPERFORM", -0.01, 0.05, 1.00),
        ("OUTPERFORM", 0.20, 0.10, 0.25),
        ("OUTPERFORM", 0.15, 0.10, 0.50),
        ("OUTPERFORM", 0.03, 0.10, 0.50),
        ("NEUTRAL", 0.00, 0.10, 0.50),
        ("UNDERPERFORM", -0.08, 0.04, 0.50),
    ],
)
def test_m13_sell_mapping_table(consensus: str, mean_predicted: float, mean_ic: float, expected: float) -> None:
    """M13: UNDERPERFORM with IC >= 0.05 sells everything, not 75 %.
    The rest of the table is the step 6 mapping (F20)."""
    assert sell_pct_from_consensus(consensus, mean_predicted, mean_ic) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Conformal intervals (M15, M16)
# ---------------------------------------------------------------------------

def test_m15_split_conformal_uses_the_finite_sample_quantile() -> None:
    """M15: dropping the ceil((1 - a)(n + 1)) / n correction.

    Residuals 1..10 at 80 % coverage: level = ceil(0.8 x 11) / 10 = 0.9, so
    q = 1 + 0.9 x 9 = 9.1 (linear interpolation); the uncorrected 0.8
    quantile would be 8.2.
    """
    result = split_conformal_interval(0.0, np.arange(1.0, 11.0), coverage=0.80)
    assert result.upper == pytest.approx(9.1)
    assert result.lower == pytest.approx(-9.1)
    # Small samples cap the level at 1: n = 3 at 80 % gives ceil(3.2) / 3 > 1.
    capped = split_conformal_interval(0.0, np.array([1.0, 2.0, 3.0]), coverage=0.80)
    assert capped.upper == pytest.approx(3.0)


def test_m16_aci_narrows_when_covered_and_widens_after_misses() -> None:
    """M16: flipping the ACI update sign.

    Constant residuals are always covered (|e| <= q), so alpha rises by
    gamma x alpha_nominal = 0.01 per step: after 9 steps alpha = 0.29 and the
    effective coverage is 0.71. Residuals that grow every step are never
    covered, so alpha falls by 0.04 per step to the 0.01 floor (coverage 0.99).
    """
    covered = aci_adjusted_interval(0.0, np.ones(10), nominal_coverage=0.80, gamma=0.05)
    assert covered.coverage_level == pytest.approx(0.71)
    missed = aci_adjusted_interval(0.0, np.arange(1.0, 11.0), nominal_coverage=0.80, gamma=0.05)
    assert missed.coverage_level == pytest.approx(0.99)
