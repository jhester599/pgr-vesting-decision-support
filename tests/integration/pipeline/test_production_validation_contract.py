"""R3 (pre-v200 remediation): chronological validation and readiness contract.

Session R3 of ``docs/reviews/PRE_V200_FIX_PROMPTS_codex.md`` retires the
representative CPCV (a combinatorial K-fold, review F02) from the monthly
decision and replaces its completeness gate with two explicit gates:

- ``wfo_completed``: every required model/benchmark pair has non-empty
  walk-forward folds with finite OOS predictions, a chronological split audit
  that holds, outcomes realised by the as-of date and a finite live forecast;
- ``data_ready``: every required live feature is finite before imputation and
  every required feed (prices, FRED, EDGAR, dividends) is fresh at the
  decision's as-of date.

Unknown or false values fail closed. Each negative fixture changes one field
of the healthy fixture and must give DEFER-TO-TAX-DEFAULT at 50 %, with the
reason named in every output surface.

Expected values are set by hand from the fixtures, never recomputed with the
function under test. Some tests call the gate functions through
``_legacy_kwargs`` so that the same test can run against the pre-R3 code
(whose gate functions still take ``representative_cpcv``) for the recorded
red/green counterfactual.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import config
from pgr_vds.decision import (
    artifacts,
    diagnostic_report,
    health,
    pipeline,
    portfolio,
    refresh,
    signal_generation,
    tax_lots,
)
from src.database import db_client
from src.models import wfo_engine
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.models.wfo_engine import FoldResult, WFOResult
from src.reporting import decision_rendering

# ---------------------------------------------------------------------------
# The hand-controlled healthy gate fixture (R3 prompt)
# ---------------------------------------------------------------------------

HEALTHY: dict[str, Any] = {
    "oos_r2": 0.03,
    "pt_p_value": 0.01,
    "agg_hit": 0.70,
    "constant_rule_hit_rate": 0.50,
    "wfo_completed": True,
    "data_ready": True,
    "missing_live_features": [],
    "stale_required_feeds": [],
}
MEAN_IC = 0.09
EXPECTED_GATES = ["oos_r2", "mean_ic", "directional_skill", "wfo_completed", "data_ready"]

# One field changed per negative fixture: (id, field, value, failing gate,
# text that must appear in every output surface).
NEGATIVE_FIXTURES: list[tuple[str, str, Any, str, str]] = [
    ("oos_r2_below_zero", "oos_r2", -0.01, "oos_r2", "oos_r2"),
    ("directional_skill_fails", "pt_p_value", 0.50, "directional_skill", "directional_skill"),
    ("wfo_incomplete", "wfo_completed", False, "wfo_completed", "wfo_completed"),
    ("data_not_ready", "data_ready", False, "data_ready", "data_ready"),
    (
        "missing_live_feature",
        "missing_live_features",
        ["rate_adequacy_gap_yoy"],
        "data_ready",
        "rate_adequacy_gap_yoy",
    ),
    (
        "stale_required_dividend",
        "stale_required_feeds",
        ["Dividends VOO"],
        "data_ready",
        "Dividends VOO",
    ),
]


class _CompletedLegacyCPCV:
    """A CPCV result that passes the pre-R3 completeness gate (verdict GOOD).

    Only used for the counterfactual run against the old code, so that a
    negative fixture is not rejected there by the CPCV gate instead of the
    field it changes.
    """

    stability_verdict = "GOOD"
    n_positive_paths = 7
    n_paths = 7
    path_ics = [0.1] * 7
    mean_ic = 0.1


def _legacy_kwargs(func: Callable[..., Any], legacy_cpcv: Any) -> dict[str, Any]:
    """``representative_cpcv`` for the pre-R3 signature; nothing afterwards."""
    if "representative_cpcv" in inspect.signature(func).parameters:
        return {"representative_cpcv": legacy_cpcv}
    return {}


def _gates(health_dict: dict[str, Any] | None, mean_ic: float = MEAN_IC, legacy_cpcv: Any = None):
    func = decision_rendering.evaluate_quality_gates
    return func(mean_ic, health_dict, **_legacy_kwargs(func, legacy_cpcv))


def _mode(
    health_dict: dict[str, Any] | None,
    mean_ic: float = MEAN_IC,
    legacy_cpcv: Any = None,
    consensus: str = "UNDERPERFORM",
) -> dict[str, Any]:
    func = decision_rendering.determine_recommendation_mode
    return func(
        consensus,
        -0.04,
        mean_ic,
        0.70,
        health_dict,
        **_legacy_kwargs(func, legacy_cpcv),
    )


def _with(field: str, value: Any) -> dict[str, Any]:
    changed = dict(HEALTHY)
    changed[field] = value
    return changed


def _assert_deferred(mode: dict[str, Any], gate: str) -> None:
    assert mode["mode"] == "defer-to-tax-default"
    assert mode["label"] == "DEFER-TO-TAX-DEFAULT"
    assert mode["sell_pct"] == 0.50
    assert gate in mode.get("failed_gates", []), mode
    assert gate in str(mode["summary"]), mode["summary"]


# ---------------------------------------------------------------------------
# Gate-level contract
# ---------------------------------------------------------------------------


def test_healthy_wfo_is_independent_of_retired_cpcv() -> None:
    """The healthy fixture passes every gate with no CPCV result at all.

    Old code: the missing CPCV makes its ``cpcv_completed`` gate FAIL, so the
    healthy run defers. New code: only the five chronological, readiness and
    quality gates exist, all PASS, and the run is ACTIONABLE (an
    UNDERPERFORM consensus with IC 0.09 sells 100 %, unchanged mapping).
    """
    gates = _gates(dict(HEALTHY), legacy_cpcv=None)
    assert [gate.name for gate in gates] == EXPECTED_GATES
    assert [gate.status for gate in gates] == ["PASS"] * 5

    mode = _mode(dict(HEALTHY), legacy_cpcv=None)
    assert mode["mode"] == "actionable"
    assert mode["label"] == "ACTIONABLE"
    assert mode["sell_pct"] == 1.00
    assert mode.get("failed_gates") == []


@pytest.mark.parametrize(
    "value",
    [False, None, "true", 1, "missing"],
    ids=["false", "none", "truthy-string", "truthy-int", "missing-key"],
)
def test_incomplete_wfo_blocks_actionable(value: Any) -> None:
    """``wfo_completed`` must be exactly True; anything else defers."""
    fixture = _with("wfo_completed", value)
    if value == "missing":
        del fixture["wfo_completed"]
    statuses = {gate.name: gate.status for gate in _gates(fixture, legacy_cpcv=_CompletedLegacyCPCV())}
    assert statuses.get("wfo_completed") == "FAIL"
    _assert_deferred(_mode(fixture, legacy_cpcv=_CompletedLegacyCPCV()), "wfo_completed")


def test_missing_live_feature_blocks_actionable() -> None:
    """A required live feature that is NaN or infinite before imputation defers.

    Gate level: one named missing feature fails ``data_ready`` even if the
    flag says ready. Input level: the NaN and the +inf feature are both
    reported, in model-feature order, and the inputs are not ready.
    """
    fixture = _with("missing_live_features", ["rate_adequacy_gap_yoy"])
    mode = _mode(fixture, legacy_cpcv=_CompletedLegacyCPCV())
    _assert_deferred(mode, "data_ready")
    assert "rate_adequacy_gap_yoy" in str(mode["summary"])

    ridge = list(config.MODEL_FEATURE_OVERRIDES["ridge"])
    gbt = list(config.MODEL_FEATURE_OVERRIDES["gbt"])
    columns = list(dict.fromkeys(ridge + gbt))
    row = pd.DataFrame([[0.1] * len(columns)], columns=columns, index=[pd.Timestamp("2026-08-31")])
    row.loc[:, "vix"] = np.nan  # ridge feature, 8th in the ridge list
    row.loc[:, "rate_adequacy_gap_yoy"] = np.inf  # gbt-only feature
    readiness = health.assess_data_readiness(
        live_row=row,
        feed_readiness={"stale_required_feeds": [], "checks": []},
        as_of=date(2026, 9, 18),
        run_date=date(2026, 9, 18),
    )
    assert readiness["missing_live_features"] == ["vix", "rate_adequacy_gap_yoy"]
    assert readiness["data_ready"] is False


def test_stale_required_dividend_blocks_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A required benchmark whose dividends lag its prices is not ready.

    Fixture (as of 2026-03-20): VOO ex-dates 2025-03-27, 06-27, 09-26 give
    gaps of 92 and 91 days, median 91.5; due by 2026-03-20 - 137 days =
    2025-11-03, so the 2025-09-26 dividend is stale. GLD is an audited
    non-payer (no history is OK); DBC has no dividend rows and is not
    audited, so it is not ready either: a failed or empty request is never
    turned into "non-payer".
    """
    _configure_small_universe(monkeypatch, ["VOO", "GLD", "DBC"])
    conn = _synthetic_feed_db(tmp_path, voo_dividends=["2025-03-27", "2025-06-27", "2025-09-26"])
    feeds = db_client.check_required_feed_readiness(conn, date(2026, 3, 20))
    conn.close()
    assert "Dividends VOO" in feeds["stale_required_feeds"]
    assert "Dividends DBC" in feeds["stale_required_feeds"]
    assert "Dividends GLD" not in feeds["stale_required_feeds"]
    assert "Dividends PGR" not in feeds["stale_required_feeds"]

    fixture = _with("stale_required_feeds", ["Dividends VOO"])
    mode = _mode(fixture, legacy_cpcv=_CompletedLegacyCPCV())
    _assert_deferred(mode, "data_ready")
    assert "Dividends VOO" in str(mode["summary"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_ready", None),
        ("data_ready", "true"),
        ("data_ready", 1),
        ("data_ready", "missing"),
        ("missing_live_features", None),
        ("missing_live_features", "missing"),
        ("stale_required_feeds", None),
        ("stale_required_feeds", "missing"),
    ],
    ids=[
        "ready-none",
        "ready-truthy-string",
        "ready-truthy-int",
        "ready-missing-key",
        "features-unknown",
        "features-missing-key",
        "feeds-unknown",
        "feeds-missing-key",
    ],
)
def test_unknown_data_readiness_blocks_actionable(field: str, value: Any) -> None:
    """Unknown readiness is not ready."""
    fixture = _with(field, value)
    if value == "missing":
        del fixture[field]
    statuses = {gate.name: gate.status for gate in _gates(fixture, legacy_cpcv=_CompletedLegacyCPCV())}
    assert statuses.get("data_ready") == "FAIL"
    _assert_deferred(_mode(fixture, legacy_cpcv=_CompletedLegacyCPCV()), "data_ready")


def test_unknown_feed_readiness_is_not_ready() -> None:
    """No feed report (e.g. the check itself failed) means not ready."""
    columns = list(config.MODEL_FEATURE_OVERRIDES["ridge"])
    row = pd.DataFrame([[0.1] * len(columns)], columns=columns, index=[pd.Timestamp("2026-08-31")])
    readiness = health.assess_data_readiness(
        live_row=row, feed_readiness=None, as_of=date(2026, 9, 18), run_date=date(2026, 9, 18)
    )
    assert readiness["data_ready"] is False
    assert readiness["stale_required_feeds"]


@pytest.mark.parametrize(
    ("field", "value", "mean_ic", "gate"),
    [
        ("oos_r2", float("nan"), MEAN_IC, "oos_r2"),
        ("oos_r2", None, MEAN_IC, "oos_r2"),
        ("oos_r2", "missing", MEAN_IC, "oos_r2"),
        ("pt_p_value", float("nan"), MEAN_IC, "directional_skill"),
        ("pt_p_value", "missing", MEAN_IC, "directional_skill"),
        (None, None, float("nan"), "mean_ic"),
        (None, None, float("inf"), "mean_ic"),
    ],
    ids=[
        "r2-nan",
        "r2-none",
        "r2-missing",
        "pt-nan",
        "pt-missing",
        "ic-nan",
        "ic-inf",
    ],
)
def test_missing_or_nonfinite_quality_metric_blocks_actionable(
    field: str | None, value: Any, mean_ic: float, gate: str
) -> None:
    """Already green before R3 (kept): a missing or non-finite metric fails."""
    fixture = dict(HEALTHY)
    if field is not None:
        if value == "missing":
            del fixture[field]
        else:
            fixture[field] = value
    mode = _mode(fixture, mean_ic=mean_ic, legacy_cpcv=_CompletedLegacyCPCV())
    assert mode["mode"] == "defer-to-tax-default"
    assert mode["sell_pct"] == 0.50
    statuses = {g.name: g.status for g in _gates(fixture, mean_ic=mean_ic, legacy_cpcv=_CompletedLegacyCPCV())}
    assert statuses[gate] == "FAIL"


def test_no_health_at_all_blocks_actionable() -> None:
    """``aggregate_health=None`` (too few OOS rows) fails every gate."""
    mode = _mode(None, legacy_cpcv=_CompletedLegacyCPCV())
    assert mode["mode"] == "defer-to-tax-default"
    assert mode["sell_pct"] == 0.50


# ---------------------------------------------------------------------------
# wfo_completed: derived from the required pairs and a split audit
# ---------------------------------------------------------------------------


def _fold_records(
    model_type: str,
    benchmark: str,
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    gap_months: int = 9,
) -> WFOResult:
    """Six-row test folds; the training window ends ``gap_months`` before each."""
    folds: list[FoldResult] = []
    for fold_idx, start in enumerate(range(0, len(dates), 6)):
        stop = min(start + 6, len(dates))
        fold_dates = dates[start:stop]
        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=fold_dates[0] - pd.DateOffset(months=gap_months + 59),
            train_end=(fold_dates[0] - pd.DateOffset(months=gap_months)) + pd.offsets.MonthEnd(0),
            test_start=fold_dates[0],
            test_end=fold_dates[-1],
            y_true=np.asarray(y_true[start:stop], dtype=float),
            y_hat=np.asarray(y_hat[start:stop], dtype=float),
            optimal_alpha=0.0,
            feature_importances={"f": 1.0},
            n_train=60,
            n_test=len(fold_dates),
        )
        fold._test_dates = list(fold_dates)
        folds.append(fold)
    return WFOResult(folds=folds, benchmark=benchmark, target_horizon=6, model_type=model_type)


def _complete_ensembles(
    benchmarks: tuple[str, ...] = ("VOO", "BND"),
    models: tuple[str, ...] = ("ridge", "gbt"),
) -> dict[str, EnsembleWFOResult]:
    """24 OOS months, 2020-01 .. 2021-12, for each benchmark and model."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(5)
    out: dict[str, EnsembleWFOResult] = {}
    for bench in benchmarks:
        y_true = rng.normal(0.0, 0.1, len(dates))
        model_results = {m: _fold_records(m, bench, dates, y_true, y_true * 0.5) for m in models}
        out[bench] = EnsembleWFOResult(
            benchmark=bench,
            target_horizon=6,
            mean_ic=0.1,
            mean_hit_rate=0.6,
            mean_mae=0.05,
            model_results=model_results,
        )
    return out


def _live_signals(benchmarks: tuple[str, ...] = ("VOO", "BND")) -> pd.DataFrame:
    return pd.DataFrame(
        {"predicted_relative_return": [0.01] * len(benchmarks)},
        index=pd.Index(list(benchmarks), name="benchmark"),
    )


# The last OOS row is 2021-12-31; its 6-month window ends 2022-06-30, so
# every outcome is realised by 2022-07-20.
WFO_AS_OF = date(2022, 7, 20)


def _assess(ensembles: dict, signals: pd.DataFrame, as_of: date = WFO_AS_OF) -> dict[str, Any]:
    return health.assess_wfo_completion(
        ensembles,
        signals,
        as_of=as_of,
        target_horizon_months=6,
        required_benchmarks=["VOO", "BND"],
        required_models=["ridge", "gbt"],
    )


def test_wfo_completed_when_every_required_pair_passes_the_audit() -> None:
    result = _assess(_complete_ensembles(), _live_signals())
    assert result["wfo_completed"] is True
    assert result["wfo_failed_pairs"] == []
    assert result["wfo_required_pairs"] == ["VOO/ridge", "VOO/gbt", "BND/ridge", "BND/gbt"]
    assert result["wfo_optional_excluded"] == []


def _drop_benchmark(ens: dict) -> None:
    del ens["BND"]


def _drop_model(ens: dict) -> None:
    del ens["BND"].model_results["gbt"]


def _no_folds(ens: dict) -> None:
    ens["BND"].model_results["gbt"].folds = []


def _empty_fold(ens: dict) -> None:
    fold = ens["BND"].model_results["gbt"].folds[1]
    fold.y_true = np.array([], dtype=float)
    fold.y_hat = np.array([], dtype=float)
    fold.n_test = 0
    fold._test_dates = []


def _nan_prediction(ens: dict) -> None:
    ens["BND"].model_results["gbt"].folds[2].y_hat[0] = np.nan


def _short_gap(ens: dict) -> None:
    # Training ends 3 months before the test fold: its last 6-month label
    # window is not realised when the test forecast is made.
    fold = ens["BND"].model_results["gbt"].folds[1]
    fold.train_end = (fold.test_start - pd.DateOffset(months=3)) + pd.offsets.MonthEnd(0)


def _overlapping_folds(ens: dict) -> None:
    folds = ens["BND"].model_results["gbt"].folds
    folds[1].test_start = folds[0].test_start
    folds[1]._test_dates = list(folds[0]._test_dates)


PAIR_DEFECTS: list[tuple[str, Callable[[dict], None], str]] = [
    ("missing-benchmark", _drop_benchmark, "BND/ridge"),
    ("one-model-only", _drop_model, "BND/gbt"),
    ("no-folds", _no_folds, "BND/gbt"),
    ("empty-fold", _empty_fold, "BND/gbt"),
    ("nan-oos-prediction", _nan_prediction, "BND/gbt"),
    ("training-label-overlaps-test", _short_gap, "BND/gbt"),
    ("test-folds-not-chronological", _overlapping_folds, "BND/gbt"),
]


@pytest.mark.parametrize(
    ("defect", "pair"),
    [(defect, pair) for _, defect, pair in PAIR_DEFECTS],
    ids=[name for name, _, _ in PAIR_DEFECTS],
)
def test_wfo_completed_requires_every_required_pair(defect: Callable[[dict], None], pair: str) -> None:
    """One broken pair is enough: one successful model is not completion."""
    ensembles = _complete_ensembles()
    defect(ensembles)
    result = _assess(ensembles, _live_signals())
    assert result["wfo_completed"] is False
    assert pair in {f"{p['benchmark']}/{p['model']}" for p in result["wfo_failed_pairs"]}


def test_wfo_outcome_not_realised_by_as_of_is_incomplete() -> None:
    """2021-12-31's window ends 2022-06-30: not realised on 2022-06-20."""
    result = _assess(_complete_ensembles(), _live_signals(), as_of=date(2022, 6, 20))
    assert result["wfo_completed"] is False
    reasons = " ".join(p["reason"] for p in result["wfo_failed_pairs"])
    assert "realised" in reasons


def test_wfo_without_a_finite_live_forecast_is_incomplete() -> None:
    signals = _live_signals()
    signals.loc["BND", "predicted_relative_return"] = np.nan
    result = _assess(_complete_ensembles(), signals)
    assert result["wfo_completed"] is False
    assert "BND/ensemble" in {f"{p['benchmark']}/{p['model']}" for p in result["wfo_failed_pairs"]}


def test_wfo_protocol_is_unchanged_and_trains_only_on_realised_labels() -> None:
    """Audit of the production split (R3): TimeSeriesSplit, 60-row window,
    6-row test folds, gap = 6 + 2 = 8 rows, fold-local imputation/scaling.

    On contiguous monthly rows every fold's test start is 9 months after its
    training end, so the last training label (6-month window) ends 3 months
    before the first test forecast. Perturbing targets after fold k's test
    end cannot change fold k's predictions.
    """
    assert config.WFO_TRAIN_WINDOW_MONTHS == 60
    assert config.WFO_TEST_WINDOW_MONTHS == 6
    assert config.WFO_PURGE_BUFFER_6M == 2

    n = 110
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["a", "b", "c"])
    y = pd.Series(X["a"].to_numpy() * 0.3 + rng.normal(0, 0.1, n), index=idx, name="y")
    base = wfo_engine.run_wfo(X, y, model_type="ridge", target_horizon_months=6, feature_columns=["a", "b", "c"])
    # (110 - 60 - 8) // 6 = 7 folds of 6 rows.
    assert len(base.folds) == 7
    for fold in base.folds:
        assert fold.n_train == 60
        assert fold.n_test == 6
        months = (fold.test_start.year - fold.train_end.year) * 12 + fold.test_start.month - fold.train_end.month
        assert months == 9
    k = 2
    cut = base.folds[k].test_end
    y_future = y.copy()
    y_future[y_future.index > cut] += 5.0
    bumped = wfo_engine.run_wfo(
        X, y_future, model_type="ridge", target_horizon_months=6, feature_columns=["a", "b", "c"]
    )
    for i in range(k + 1):
        np.testing.assert_allclose(bumped.folds[i].y_hat, base.folds[i].y_hat, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# data_ready at the decision's as-of date (synthetic DB)
# ---------------------------------------------------------------------------


def _configure_small_universe(monkeypatch: pytest.MonkeyPatch, universe: list[str]) -> None:
    """Required feeds: PGR and ``universe`` prices/dividends, FRED NFCI, EDGAR."""
    monkeypatch.setattr(config, "PRIMARY_FORECAST_UNIVERSE", list(universe))
    monkeypatch.setattr(config, "ENSEMBLE_MODELS", ["ridge"])
    monkeypatch.setattr(config, "MODEL_FEATURE_OVERRIDES", {"ridge": ["nfci"]})


def _synthetic_feed_db(
    tmp_path: Path,
    voo_dividends: list[str],
    price_end: str = "2026-03-20",
    later_voo_dividends: tuple[str, ...] = (),
    edgar: tuple[tuple[str, str], ...] = (
        ("2025-11-30", "2025-12-17"),
        ("2025-12-31", "2026-01-28"),
        ("2026-01-31", "2026-02-18"),
    ),
) -> sqlite3.Connection:
    """Weekly prices to ``price_end``; PGR quarterly dividends; NFCI to 2025-12."""
    path = tmp_path / "feeds.db"
    conn = db_client.get_connection(str(path))
    db_client.initialize_schema(conn)
    fridays = pd.date_range("2024-01-05", price_end, freq="W-FRI")
    for ticker in ("PGR", "VOO", "GLD", "DBC"):
        conn.executemany(
            "INSERT INTO daily_prices (ticker, date, close, proxy_fill) VALUES (?, ?, ?, 0)",
            [(ticker, d.date().isoformat(), 100.0) for d in fridays],
        )
    pgr_divs = ["2025-01-02", "2025-04-03", "2025-07-03", "2025-10-02", "2026-01-02"]
    conn.executemany(
        "INSERT INTO daily_dividends (ticker, ex_date, amount, source) VALUES ('PGR', ?, 0.1, 'test')",
        [(d,) for d in pgr_divs],
    )
    conn.executemany(
        "INSERT INTO daily_dividends (ticker, ex_date, amount, source) VALUES ('VOO', ?, 1.0, 'test')",
        [(d,) for d in [*voo_dividends, *later_voo_dividends]],
    )
    months = pd.date_range("2023-01-31", "2025-12-31", freq="BME")
    conn.executemany(
        "INSERT INTO fred_macro_monthly (series_id, month_end, value) VALUES ('NFCI', ?, -0.5)",
        [(m.date().isoformat(),) for m in months],
    )
    conn.executemany(
        "INSERT INTO pgr_edgar_monthly (month_end, filing_date) VALUES (?, ?)",
        list(edgar),
    )
    conn.commit()
    return conn


def test_backdated_readiness_uses_decision_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feeds that were fresh on the as-of date are ready for that decision.

    As of 2026-03-20: prices end that day (age 0 <= 10); PGR's last ex-date
    2026-01-02 >= due 2025-11-04; VOO's 2025-12-22 >= due 2025-11-04 (gaps 92,
    91, 87, median 91); the decision row is February, NFCI (lag 2) needs
    2025-12 and has it; the February 8-K is not due until 2026-03-25, so
    January (filed 2026-02-18) is enough. Judged at the 2026-09-27 run date
    instead, every price would be 191 days old.
    """
    _configure_small_universe(monkeypatch, ["VOO", "GLD"])
    conn = _synthetic_feed_db(
        tmp_path, voo_dividends=["2025-03-27", "2025-06-27", "2025-09-26", "2025-12-22"]
    )
    feeds = db_client.check_required_feed_readiness(conn, date(2026, 3, 20))
    assert feeds["reference_date"] == "2026-03-20"
    assert feeds["stale_required_feeds"] == []

    # The pipeline evaluates freshness and readiness at the as-of date, not
    # at the run date.
    seen: dict[str, list[date]] = {"freshness": [], "readiness": []}
    real_freshness = db_client.check_data_freshness

    def _spy_freshness(conn_arg, reference_date, *args, **kwargs):
        seen["freshness"].append(reference_date)
        return real_freshness(conn_arg, reference_date, *args, **kwargs)

    def _spy_readiness(conn_arg, as_of, *args, **kwargs):
        seen["readiness"].append(as_of)
        return {"reference_date": as_of.isoformat(), "stale_required_feeds": [], "checks": [], "warnings": [],
                "overall_status": "OK"}

    monkeypatch.setattr(db_client, "check_data_freshness", _spy_freshness)
    monkeypatch.setattr(db_client, "check_required_feed_readiness", _spy_readiness, raising=False)
    out = _run_stubbed_pipeline(monkeypatch, tmp_path / "run", dict(HEALTHY), as_of="2026-03-20",
                                stub_readiness=False)
    assert seen["readiness"] == [date(2026, 3, 20)]
    assert all(ref == date(2026, 3, 20) for ref in seen["freshness"]), seen
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision_gates"]["readiness_basis"] == "backdated_reconstruction"
    conn.close()


def test_later_data_cannot_make_backdated_inputs_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows dated or filed after the as-of date do not count for it.

    As of 2026-03-20 VOO's last ex-date is 2025-09-26, due 2025-11-03 (gaps
    92 and 91, median 91.5): stale, although dividends of 2026-03-26 and
    2026-06-26 were stored later. The January 8-K was filed 2026-04-01, after
    the as-of date, and the February one 2026-04-10; the latest filed by
    2026-03-20 is December, but January is due: stale. Without the as-of
    bound the later rows would make both look fresh.
    """
    _configure_small_universe(monkeypatch, ["VOO", "GLD"])
    conn = _synthetic_feed_db(
        tmp_path,
        voo_dividends=["2025-03-27", "2025-06-27", "2025-09-26"],
        price_end="2026-09-25",
        later_voo_dividends=("2026-03-26", "2026-06-26"),
        edgar=(
            ("2025-11-30", "2025-12-17"),
            ("2025-12-31", "2026-01-28"),
            ("2026-01-31", "2026-04-01"),
            ("2026-02-28", "2026-04-10"),
        ),
    )
    feeds = db_client.check_required_feed_readiness(conn, date(2026, 3, 20))
    conn.close()
    assert "Dividends VOO" in feeds["stale_required_feeds"]
    assert "PGR monthly EDGAR" in feeds["stale_required_feeds"]
    assert not any(feed.startswith("Prices") for feed in feeds["stale_required_feeds"])


def test_backdated_decision_row_must_be_the_as_of_month() -> None:
    """The live row for 2026-09-18 is August's (September's BME is later)."""
    columns = list(config.MODEL_FEATURE_OVERRIDES["ridge"])
    old_row = pd.DataFrame([[0.1] * len(columns)], columns=columns, index=[pd.Timestamp("2026-06-30")])
    readiness = health.assess_data_readiness(
        live_row=old_row,
        feed_readiness={"stale_required_feeds": [], "checks": []},
        as_of=date(2026, 9, 18),
        run_date=date(2026, 9, 18),
    )
    assert readiness["data_ready"] is False
    assert any("2026-08" in feed for feed in readiness["stale_required_feeds"])
    assert readiness["readiness_basis"] == "live"


# ---------------------------------------------------------------------------
# End to end: the production path never reaches CPCV; every surface agrees
# ---------------------------------------------------------------------------

_LOG_TEMPLATE = "\n".join(
    [
        "# PGR Monthly Decision Log",
        "",
        "## Log",
        "",
        "| As-Of Date | Run Date | Consensus Signal | Sell % | Predicted 6M Return | Mean IC | Hit Rate | Notes |",
        "|------------|----------|-----------------|--------|---------------------|---------|----------|-------|",
        "| 2026-03-20 | 2026-03-20 | NEUTRAL | 50% | +1.00% | 0.0500 | 60.0% |  |",
        "",
        "---",
        "",
    ]
)


def _patch_peripherals(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Temp DB, output folder and decision log; tax/portfolio/shadow stubbed."""
    root.mkdir(parents=True, exist_ok=True)
    out_root = root / "monthly_decisions"
    log_path = root / "decision_log.md"
    log_path.write_text(_LOG_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(config, "DB_PATH", str(root / "pipeline.db"))
    monkeypatch.setattr(config, "RECOMMENDATION_LAYER_MODE", "live_only")
    monkeypatch.setattr(config, "DECISION_LOG_PATH", str(log_path))
    monkeypatch.setattr(artifacts, "output_dir", lambda as_of: out_root / as_of.strftime("%Y-%m"))
    monkeypatch.setattr(artifacts, "already_ran", lambda as_of: False)
    monkeypatch.setattr(artifacts, "load_previous_decision_summary", lambda *a, **k: None)
    monkeypatch.setattr(refresh, "fetch_fred_step", lambda *a, **k: None)
    monkeypatch.setattr(tax_lots, "build_provisional_vest_scenario", lambda *a, **k: None)
    monkeypatch.setattr(tax_lots, "build_existing_holdings_guidance", lambda *a, **k: [])
    monkeypatch.setattr(portfolio, "build_redeploy_guidance", lambda *a, **k: [])
    monkeypatch.setattr(portfolio, "build_redeploy_portfolio", lambda *a, **k: None)
    monkeypatch.setattr(diagnostic_report, "plot_calibration_curve", lambda *a, **k: None)
    monkeypatch.setattr(db_client, "warn_if_db_behind", lambda *a, **k: [])

    def _no_shadow(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("classifier shadow disabled in this test")

    monkeypatch.setattr(pipeline, "build_classification_shadow_summary", _no_shadow)
    return out_root


def _surfaces(out_dir: Path, log_path: Path) -> dict[str, str]:
    """Every output surface of one run, as text."""
    from src.reporting.email_sender import build_email_summary

    recommendation = (out_dir / "recommendation.md").read_text(encoding="utf-8")
    summary_text = (out_dir / "monthly_summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    return {
        "recommendation.md": recommendation,
        "diagnostic.md": (out_dir / "diagnostic.md").read_text(encoding="utf-8"),
        "monthly_summary.json": summary_text,
        "run_manifest.json": (out_dir / "run_manifest.json").read_text(encoding="utf-8"),
        "dashboard.html": (out_dir / "dashboard.html").read_text(encoding="utf-8"),
        "email": build_email_summary(recommendation, summary_payload=summary),
        "decision_log.md": [
            line for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith("| 20")
        ][-1],
    }


def _run_stubbed_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    fields: dict[str, Any],
    as_of: str = "2026-04-20",
    stub_readiness: bool = True,
) -> Path:
    """Run ``pipeline.main`` with the model stubbed to the given gate fields.

    The metric fields go into the aggregate health; the readiness fields are
    what the readiness step reports (``stub_readiness``).
    """
    out_root = _patch_peripherals(monkeypatch, root)
    signals = pd.DataFrame(
        {
            "predicted_relative_return": [-0.05, -0.03],
            "ic": [0.09, 0.09],
            "hit_rate": [0.70, 0.70],
            "signal": ["UNDERPERFORM", "UNDERPERFORM"],
            "prob_outperform": [0.35, 0.40],
            "confidence_tier": ["MODERATE", "MODERATE"],
            "calibrated_prob_outperform": [0.35, 0.40],
        },
        index=pd.Index(["VOO", "BND"], name="benchmark"),
    )
    from src.models.calibration import CalibrationResult

    cal = CalibrationResult(n_obs=48, method="platt", ece=0.04, ece_ci_lower=0.02, ece_ci_upper=0.07)
    monkeypatch.setattr(
        signal_generation,
        "generate_signals",
        lambda *a, **k: (
            signals.copy(),
            {},
            {
                "obs_feature_report": None,
                "missing_live_features": list(fields.get("missing_live_features") or []),
                "nan_live_features": list(fields.get("missing_live_features") or []),
            },
        ),
    )
    monkeypatch.setattr(
        signal_generation,
        "calibrate_signals",
        lambda signals, ensemble_results, target_horizon_months=6, panel=None: (
            signals.copy(), cal, np.array([0.4, 0.5]), np.array([1, 0]),
        ),
    )
    monkeypatch.setattr(
        signal_generation,
        "compute_conformal_intervals",
        lambda signals, ensemble_results, panel=None, target_horizon_months=6: signals.copy(),
    )
    monkeypatch.setattr(
        signal_generation,
        "consensus_signal",
        lambda signals: ("UNDERPERFORM", -0.04, MEAN_IC, 0.70, 0.375, "MODERATE"),
    )
    def _stub_diagnostic(out_dir: Path, *args: Any, **kwargs: Any) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diagnostic.md").write_text("# Diagnostic stub\n", encoding="utf-8")

    monkeypatch.setattr(diagnostic_report, "write_diagnostic_report", _stub_diagnostic)
    metric_keys = ("oos_r2", "pt_p_value", "agg_hit", "constant_rule_hit_rate")
    metrics = {key: fields[key] for key in metric_keys if key in fields}
    metrics.update(nw_ic=0.10, nw_pval=0.02, cw_t_stat=2.0, cw_p_value=0.03, base_rate=0.5)
    monkeypatch.setattr(health, "compute_aggregate_health", lambda *a, **k: dict(metrics))
    if stub_readiness:
        readiness_keys = ("wfo_completed", "data_ready", "missing_live_features", "stale_required_feeds")
        readiness = {key: fields[key] for key in readiness_keys if key in fields}
        readiness.update(
            wfo_required_pairs=["VOO/ridge", "VOO/gbt", "BND/ridge", "BND/gbt"],
            wfo_failed_pairs=(
                [] if fields.get("wfo_completed") is True
                else [{"benchmark": "BND", "model": "gbt", "reason": "no folds"}]
            ),
            wfo_optional_excluded=[],
            readiness_basis="live",
        )
        monkeypatch.setattr(health, "build_readiness", lambda *a, **k: dict(readiness), raising=False)

    pipeline.main(as_of_date_str=as_of, dry_run=False, skip_fred=True)
    return out_root / as_of[:7]


def test_healthy_run_is_actionable_on_every_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the CPCV gone and readiness proven, the healthy fixture acts."""
    out_dir = _run_stubbed_pipeline(monkeypatch, tmp_path, dict(HEALTHY))
    surfaces = _surfaces(out_dir, tmp_path / "decision_log.md")
    summary = json.loads(surfaces["monthly_summary.json"])
    assert summary["recommendation"]["recommendation_mode"] == "ACTIONABLE"
    assert summary["recommendation"]["recommended_sell_pct"] == pytest.approx(1.0)
    assert [g["name"] for g in summary["model_health"]["gates"]] == EXPECTED_GATES
    assert [g["status"] for g in summary["model_health"]["gates"]] == ["PASS"] * 5
    manifest = json.loads(surfaces["run_manifest.json"])
    assert manifest["decision_gates"]["recommendation_mode"] == "ACTIONABLE"
    assert manifest["decision_gates"]["failed_gates"] == []
    assert manifest["decision_gates"]["gate_contract_version"] == config.DECISION_GATE_CONTRACT_VERSION
    assert "| 100% |" in surfaces["decision_log.md"]
    for name, text in surfaces.items():
        assert "CPCV" not in text, name


@pytest.mark.parametrize(
    ("field", "value", "gate", "token"),
    [(f, v, g, t) for _, f, v, g, t in NEGATIVE_FIXTURES],
    ids=[name for name, *_ in NEGATIVE_FIXTURES],
)
def test_deferral_reason_is_named_in_every_output_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    gate: str,
    token: str,
) -> None:
    """Each one-field negative fixture defers at 50 %, named everywhere."""
    out_dir = _run_stubbed_pipeline(monkeypatch, tmp_path, _with(field, value))
    surfaces = _surfaces(out_dir, tmp_path / "decision_log.md")
    summary = json.loads(surfaces["monthly_summary.json"])
    assert summary["recommendation"]["recommendation_mode"] == "DEFER-TO-TAX-DEFAULT"
    assert summary["recommendation"]["recommended_sell_pct"] == pytest.approx(0.5)
    manifest = json.loads(surfaces["run_manifest.json"])
    assert gate in manifest["decision_gates"]["failed_gates"]
    assert "| 50% |" in surfaces["decision_log.md"]
    for name, text in surfaces.items():
        if name == "diagnostic.md":
            continue  # the diagnostic report carries metrics, not the decision
        assert token in text, f"{token!r} missing from {name}"
        assert "CPCV" not in text, name


def _synthetic_model_inputs(nan_feature: str | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """150 business month-ends (2010-01 .. 2022-06) of the live feature set."""
    n = 150
    idx = pd.date_range("2010-01-29", periods=n, freq="BME")
    rng = np.random.default_rng(7)
    columns = list(dict.fromkeys([*config.MODEL_FEATURE_OVERRIDES["ridge"], *config.MODEL_FEATURE_OVERRIDES["gbt"]]))
    X = pd.DataFrame(rng.normal(size=(n, len(columns))), index=idx, columns=columns)
    if nan_feature is not None:
        X.loc[idx[-1], nan_feature] = np.nan
    targets = {
        etf: pd.Series(0.05 * X["mom_12m"].to_numpy() + rng.normal(0.0, 0.05, n), index=idx, name=etf)
        for etf in ("VOO", "BND")
    }
    return X, targets


def _run_synthetic_production_path(
    monkeypatch: pytest.MonkeyPatch, root: Path, nan_feature: str | None = None
) -> tuple[Path, dict[str, int]]:
    """``pipeline.main`` with real signal generation, WFO, calibration and
    health on synthetic inputs; CPCV entry points are spied on."""
    import skfolio.model_selection as skms

    out_root = _patch_peripherals(monkeypatch, root)
    X, targets = _synthetic_model_inputs(nan_feature)
    monkeypatch.setattr(config, "PRIMARY_FORECAST_UNIVERSE", ["VOO", "BND"])
    monkeypatch.setattr(signal_generation, "build_feature_matrix_from_db", lambda *a, **k: X.copy())
    monkeypatch.setattr(
        signal_generation,
        "load_relative_return_matrix",
        lambda conn, etf, horizon: targets[etf].copy() if etf in targets else pd.Series(dtype=float),
    )

    calls = {"run_cpcv": 0, "CombinatorialPurgedCV": 0}
    real_run_cpcv = wfo_engine.run_cpcv

    def _spy_run_cpcv(*args: Any, **kwargs: Any) -> Any:
        calls["run_cpcv"] += 1
        return real_run_cpcv(*args, **kwargs)

    real_cls = skms.CombinatorialPurgedCV

    class _SpyCPCV(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["CombinatorialPurgedCV"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(wfo_engine, "run_cpcv", _spy_run_cpcv)
    # The pre-R3 caller imported ``run_cpcv`` by name; spy on that too.
    monkeypatch.setattr(signal_generation, "run_cpcv", _spy_run_cpcv, raising=False)
    monkeypatch.setattr(skms, "CombinatorialPurgedCV", _SpyCPCV)

    pipeline.main(as_of_date_str="2022-07-20", dry_run=False, skip_fred=True)
    return out_root / "2022-07", calls


def test_live_decision_does_not_invoke_cpcv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production path attempts no CPCV call and builds no CPCV splitter.

    A raised sentinel would not do: the pre-R3 caller caught every exception
    from ``run_cpcv``. The spies count attempted calls, which must be zero.
    """
    out_dir, calls = _run_synthetic_production_path(monkeypatch, tmp_path)
    assert calls == {"run_cpcv": 0, "CombinatorialPurgedCV": 0}

    surfaces = _surfaces(out_dir, tmp_path / "decision_log.md")
    for name, text in surfaces.items():
        assert "CPCV" not in text, name
    summary = json.loads(surfaces["monthly_summary.json"])
    assert [g["name"] for g in summary["model_health"]["gates"]] == EXPECTED_GATES
    manifest = json.loads(surfaces["run_manifest.json"])
    gates = manifest["decision_gates"]
    # Real WFO on complete synthetic inputs: both required pairs per benchmark.
    assert gates["wfo_completed"] is True, gates["wfo_failed_pairs"]
    assert gates["wfo_required_pairs"] == ["VOO/ridge", "VOO/gbt", "BND/ridge", "BND/gbt"]
    # The temp DB holds no feeds: the run is not ready and defers.
    assert gates["data_ready"] is False
    assert gates["stale_required_feeds"]
    assert gates["recommendation_mode"] == "DEFER-TO-TAX-DEFAULT"


def test_missing_live_feature_is_found_before_imputation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN in the decision row reaches the manifest although the model
    median-imputes it for the forecast."""
    out_dir, calls = _run_synthetic_production_path(monkeypatch, tmp_path, nan_feature="vix")
    assert calls == {"run_cpcv": 0, "CombinatorialPurgedCV": 0}
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision_gates"]["missing_live_features"] == ["vix"]
    assert manifest["decision_gates"]["data_ready"] is False
    assert "data_ready" in manifest["decision_gates"]["failed_gates"]


def test_retired_run_cpcv_raises_without_building_a_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    import skfolio.model_selection as skms

    built = {"n": 0}
    real_cls = skms.CombinatorialPurgedCV

    class _SpyCPCV(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(skms, "CombinatorialPurgedCV", _SpyCPCV)
    idx = pd.date_range("2005-01-31", periods=120, freq="ME")
    X = pd.DataFrame({"a": np.arange(120, dtype=float)}, index=idx)
    y = pd.Series(np.arange(120, dtype=float), index=idx, name="y")
    with pytest.raises(wfo_engine.UnsupportedValidationMethodError, match="walk-forward"):
        wfo_engine.run_cpcv(X, y, model_type="ridge")
    assert built["n"] == 0


def test_manifest_records_the_gate_contract_version() -> None:
    """The contract has a stable version; the metric version is unchanged."""
    assert config.DECISION_GATE_CONTRACT_VERSION == "chronological-readiness-2026-09-27"
    assert config.MODEL_HEALTH_METRICS_VERSION == "prequential-2026-09-25"
