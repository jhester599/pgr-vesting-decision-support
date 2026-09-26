"""Review 2026-09-25, step 5 (WP7): validation and gating.

Tests named in ``docs/reviews/REPO_REVIEW_2026-09-25.md``:

- F02: for 8 folds and 2 test folds, 7 CPCV paths each cover every row exactly
  once, and a perfect predictor gets GOOD.
- F04: an oracle forecaster on an overlapping series gets OOS R^2 > 0, and the
  naive benchmark never uses a target realised after t.
- F13: reported metrics use no later-fold statistics (recomputing on the
  truncated history gives the same values); an always-positive predictor
  fails the hit-rate gate; the gate uses the equal-weight IC; IC significance
  is clustered by date.
- F20: a missing or UNKNOWN CPCV must not permit ACTIONABLE (and a FAIL
  verdict no longer gates: CPCV is diagnostic-only).
- F21: confidence tiers are not all identical across a run.

Fixtures are synthetic ``EnsembleWFOResult`` objects built from fold records,
so the production reconstruction and gating code runs unchanged.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import config
import scripts.monthly_decision as md
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.models.wfo_engine import CPCVResult, FoldResult, WFOResult, run_cpcv

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _wfo_result(
    model_type: str,
    benchmark: str,
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    fold_size: int = 6,
) -> WFOResult:
    folds: list[FoldResult] = []
    for fold_idx, start in enumerate(range(0, len(dates), fold_size)):
        stop = min(start + fold_size, len(dates))
        fold_dates = dates[start:stop]
        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=fold_dates[0] - pd.DateOffset(months=68),
            train_end=fold_dates[0] - pd.DateOffset(months=9),
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


def _ensemble(
    benchmark: str,
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    target_history: pd.Series | None = None,
) -> EnsembleWFOResult:
    model_results = {
        model: _wfo_result(model, benchmark, dates, y_true, pred) for model, pred in preds.items()
    }
    ens = EnsembleWFOResult(
        benchmark=benchmark,
        target_horizon=6,
        mean_ic=float(np.mean([r.information_coefficient for r in model_results.values()])),
        mean_hit_rate=float(np.mean([r.hit_rate for r in model_results.values()])),
        mean_mae=float(np.mean([r.mean_absolute_error for r in model_results.values()])),
        model_results=model_results,
    )
    if target_history is not None:
        # A plain attribute on older code, a dataclass field after WP7.
        ens.target_history = target_history
    return ens


def _two_model_ensemble(n_oos: int, seed: int = 0, benchmark: str = "VTI") -> EnsembleWFOResult:
    """Ridge-like (accurate) and GBT-like (noisy, biased) members on one benchmark."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-31", periods=n_oos, freq="ME")
    signal = 0.08 * np.sin(np.arange(n_oos) / 5.0)
    y_true = signal + rng.normal(0.02, 0.08, n_oos)
    ridge = signal + rng.normal(0.0, 0.04 + 0.03 * (np.arange(n_oos) > n_oos // 2), n_oos)
    gbt = 0.3 * signal + rng.normal(0.01, 0.10, n_oos)
    return _ensemble(benchmark, dates, y_true, {"ridge": ridge, "gbt": gbt})


def _truncate(ens: EnsembleWFOResult, n_keep: int) -> EnsembleWFOResult:
    """Rebuild ``ens`` from its first ``n_keep`` OOS rows (the earlier folds)."""
    ref = next(iter(ens.model_results.values()))
    dates = pd.DatetimeIndex(ref.test_dates_all)[:n_keep]
    y_true = ref.y_true_all[:n_keep]
    preds = {m: r.y_hat_all[:n_keep] for m, r in ens.model_results.items()}
    return _ensemble(ens.benchmark, dates, y_true, preds, getattr(ens, "target_history", None))


def _cpcv(path_ics: list[float]) -> CPCVResult:
    return CPCVResult(
        model_type="ridge",
        benchmark="VOO",
        n_splits=28,
        n_paths=len(path_ics),
        path_ics=list(path_ics),
        mean_ic=float(np.mean(path_ics)) if path_ics else float("nan"),
        ic_std=float(np.std(path_ics)) if len(path_ics) > 1 else float("nan"),
        split_ics=[],
    )


# 28 positive paths: GOOD under the old 28-path thresholds and the scaled ones,
# so tests of the other gates are not masked by F02 on the unfixed code.
_GOOD_CPCV = _cpcv([0.1] * 28)


def _skilled_health() -> dict:
    """Aggregate health of a predictor with real directional skill (R^2 set to pass)."""
    rng = np.random.default_rng(3)
    n = 180
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    signal = rng.normal(0.02, 0.10, n)
    y_true = signal + rng.normal(0.0, 0.03, n)
    ens = _ensemble("VTI", dates, y_true, {"ridge": signal, "gbt": signal})
    health = md._compute_aggregate_health({"VTI": ens})
    assert health is not None
    health = dict(health)
    health["oos_r2"] = 0.05
    return health


# ---------------------------------------------------------------------------
# F02 — CPCV recombination, thresholds, purge/embargo
# ---------------------------------------------------------------------------


def _linear_data(n: int = 240, n_features: int = 4, noise: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    X = pd.DataFrame(rng.normal(size=(n, n_features)), index=idx, columns=[f"f{i}" for i in range(n_features)])
    beta = np.array([0.5, -0.3, 0.2, 0.1])[:n_features]
    y = pd.Series(X.to_numpy() @ beta + noise * rng.normal(size=n), index=idx, name="target")
    return X, y


def test_cpcv_recombines_seven_paths_each_covering_every_row_once() -> None:
    """F02: C(8,2) gives 7 test paths; each path takes every fold exactly once."""
    X, y = _linear_data(n=240, noise=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_cpcv(X, y, model_type="ridge", n_folds=8, n_test_folds=2)

    assert result.n_splits == 28
    assert result.n_paths == 7
    assert len(result.path_ics) == 7, "one IC per recombined path (not per fold)"
    path_rows = getattr(result, "path_row_indices", None)
    assert path_rows is not None and len(path_rows) == 7
    for rows in path_rows:
        np.testing.assert_array_equal(np.sort(np.asarray(rows)), np.arange(len(X)))


def test_cpcv_perfect_predictor_is_good() -> None:
    """F02: a perfect predictor must pass the (diagnostic) CPCV check."""
    X, y = _linear_data(n=240, noise=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_cpcv(X, y, model_type="ridge", n_folds=8, n_test_folds=2)
    assert result.n_positive_paths == result.n_paths == 7
    assert result.stability_verdict == "GOOD"


def test_cpcv_thresholds_scale_to_the_path_count() -> None:
    """19/28 good and 9/28 marginal become ceil(19*7/28)=5 and ceil(9*7/28)=3 of 7."""
    assert _cpcv([0.1] * 5 + [-0.1] * 2).stability_verdict == "GOOD"
    assert _cpcv([0.1] * 4 + [-0.1] * 3).stability_verdict == "MARGINAL"
    assert _cpcv([0.1] * 3 + [-0.1] * 4).stability_verdict == "MARGINAL"
    assert _cpcv([0.1] * 2 + [-0.1] * 5).stability_verdict == "FAIL"
    # The 28-path reference is unchanged.
    assert _cpcv([0.1] * 19 + [-0.1] * 9).stability_verdict == "GOOD"
    assert _cpcv([0.1] * 9 + [-0.1] * 19).stability_verdict == "MARGINAL"
    assert _cpcv([0.1] * 8 + [-0.1] * 20).stability_verdict == "FAIL"


def test_cpcv_purges_the_horizon_and_embargoes_at_least_two_rows() -> None:
    import skfolio.model_selection as skms

    captured: dict[str, int] = {}
    real_cls = skms.CombinatorialPurgedCV

    class Spy(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    X, y = _linear_data(n=240, noise=0.5)
    from unittest.mock import patch

    with patch("skfolio.model_selection.CombinatorialPurgedCV", Spy), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cpcv(X, y, model_type="ridge", target_horizon_months=6)
    assert captured["purged_size"] == 6
    assert captured["embargo_size"] >= 2


# ---------------------------------------------------------------------------
# F04 — look-ahead-free OOS R^2
# ---------------------------------------------------------------------------


def _oracle_panel_ensembles(seed: int = 6, n_bench: int = 8, n_hist: int = 60, n_oos: int = 60):
    """Overlapping 6M targets sharing a common (PGR-leg) shock across benchmarks.

    y[b, t] = c[b, t] + sum_{k=1..6} (u[t+k] + v[b, t+k]) where c is known at t,
    u is common to all benchmarks and v is benchmark-specific. The forecast is
    the true conditional mean c, so every honest benchmark must lose to it.
    """
    rng = np.random.default_rng(seed)
    total = n_hist + n_oos
    dates = pd.date_range("2008-01-31", periods=total, freq="ME")
    common = rng.normal(0.0, 0.05, total + 6)
    ensembles: dict[str, EnsembleWFOResult] = {}
    for b in range(n_bench):
        idio = rng.normal(0.0, 0.02, total + 6)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        cond_mean = 0.10 * np.sin(2.0 * np.pi * np.arange(total) / 30.0 + phase)
        noise = np.array([common[t + 1 : t + 7].sum() + idio[t + 1 : t + 7].sum() for t in range(total)])
        y = cond_mean + noise
        oos = slice(n_hist, total)
        ensembles[f"B{b}"] = _ensemble(
            f"B{b}",
            dates[oos],
            y[oos],
            {"ridge": cond_mean[oos], "gbt": cond_mean[oos]},
            target_history=pd.Series(y, index=dates),
        )
    return ensembles


def test_oracle_forecaster_gets_positive_pooled_oos_r2() -> None:
    """F04: the forecast equal to the true conditional mean must beat the naive."""
    ensembles = _oracle_panel_ensembles()
    health = md._compute_aggregate_health(ensembles)
    assert health is not None
    assert health["oos_r2"] > 0.0


def test_naive_benchmark_never_uses_a_target_realised_after_t() -> None:
    """F04: naive(t) = mean of that benchmark's targets with window end <= t."""
    from src.models.prequential import prevailing_mean_forecast

    ensembles = _oracle_panel_ensembles()
    health = md._compute_aggregate_health(ensembles)
    panel = health["panel"]
    for benchmark, ens in ensembles.items():
        history = ens.target_history
        rows = panel[panel["benchmark"] == benchmark]
        for date, naive in zip(rows["date"], rows["naive"]):
            cutoff = pd.Timestamp(date).to_period("M") - 6
            realised = history[history.index.to_period("M") <= cutoff]
            assert naive == pytest.approx(float(realised.mean()), abs=1e-12)

    # Perturbing a target that is not yet realised at t leaves naive(t) alone.
    history = pd.Series(np.arange(24, dtype=float), index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    forecast_index = pd.DatetimeIndex([pd.Timestamp("2021-06-30")])
    before = prevailing_mean_forecast(forecast_index, history, horizon_months=6).iloc[0]
    bumped = history.copy()
    bumped.loc[pd.Timestamp("2021-01-31")] += 1_000.0  # window ends 2021-07: after t
    after = prevailing_mean_forecast(forecast_index, bumped, horizon_months=6).iloc[0]
    assert before == after == pytest.approx(float(history.loc[:"2020-12-31"].mean()))


def test_compute_oos_r_squared_does_not_score_against_the_current_target() -> None:
    """F04 toy case from the review: y = [1, -1, 3] used to give naive [1, 0, 1]."""
    from src.reporting.backtest_report import compute_oos_r_squared

    realized = pd.Series([1.0, -1.0, 3.0, 0.5, 2.0])
    # A forecaster that always predicts the prevailing mean of the targets
    # realised before it (one-step targets) scores exactly 0.
    naive_like = pd.Series([np.nan, 1.0, 0.0, 1.0, 0.875])
    assert compute_oos_r_squared(naive_like, realized) == pytest.approx(0.0)

    # With 6-month overlapping targets the first 6 rows have nothing realised.
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    y = pd.Series(np.linspace(-0.1, 0.2, 12), index=dates)
    prevailing = pd.Series(
        [np.nan] * 6 + [float(y.iloc[: k - 5].mean()) for k in range(6, 12)], index=dates
    )
    assert compute_oos_r_squared(prevailing, y, horizon_months=6) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F13 — honest metrics
# ---------------------------------------------------------------------------


def test_reported_ensemble_predictions_use_no_later_fold_statistics() -> None:
    """F13: appending later folds must not change earlier reconstructed OOS rows."""
    full = _two_model_ensemble(n_oos=96)
    truncated = _truncate(full, 48)

    health_full = md._compute_aggregate_health({"VTI": full})
    health_trunc = md._compute_aggregate_health({"VTI": truncated})
    early = health_trunc["agg_predicted"].index
    np.testing.assert_allclose(
        health_full["agg_predicted"].loc[early].to_numpy(dtype=float),
        health_trunc["agg_predicted"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )

    # The aggregate metrics recomputed on the truncated history are the same
    # numbers as the full run's rows restricted to that history.
    from src.models.forecast_diagnostics import summarize_panel_diagnostics

    panel_full = health_full["panel"]
    restricted = panel_full[panel_full["date"] <= early.max()]
    pooled, _ = summarize_panel_diagnostics(restricted, target_horizon_months=6)
    assert pooled["oos_r2"] == pytest.approx(health_trunc["oos_r2"], abs=1e-12)
    assert pooled["nw_ic"] == pytest.approx(health_trunc["nw_ic"], abs=1e-12)
    assert pooled["hit_rate"] == pytest.approx(health_trunc["agg_hit"], abs=1e-12)


def test_calibration_probabilities_use_no_later_fold_statistics() -> None:
    """F13: reported (ECE) probabilities come from calibrators fitted on realised rows."""
    full = _two_model_ensemble(n_oos=96, seed=5)
    truncated = _truncate(full, 60)
    signals = pd.DataFrame(
        {
            "predicted_relative_return": [0.02],
            "raw_ensemble_prediction": [0.04],
            "signal": ["OUTPERFORM"],
            "confidence_tier": ["LOW"],
        },
        index=pd.Index(["VTI"], name="benchmark"),
    )
    _, cal_full, probs_full, outcomes_full = md._calibrate_signals(signals, {"VTI": full})
    _, cal_trunc, probs_trunc, outcomes_trunc = md._calibrate_signals(signals, {"VTI": truncated})
    assert len(probs_trunc) > 0
    np.testing.assert_allclose(probs_full[: len(probs_trunc)], probs_trunc, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(outcomes_full[: len(outcomes_trunc)], outcomes_trunc)


def test_always_positive_predictor_fails_hit_rate_gate() -> None:
    """F13: a 68 % base rate must not pass the hit-rate gate on its own."""
    rng = np.random.default_rng(11)
    n = 180
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    outcome_up = rng.random(n) < 0.68
    y_true = np.where(outcome_up, rng.uniform(0.01, 0.2, n), -rng.uniform(0.01, 0.2, n))
    always_up = rng.uniform(0.02, 0.10, n)
    ens = _ensemble("VTI", dates, y_true, {"ridge": always_up, "gbt": always_up})
    health = md._compute_aggregate_health({"VTI": ens})
    assert health is not None
    health = dict(health)
    health["oos_r2"] = 0.05  # isolate the hit-rate gate
    assert health["agg_hit"] > config.DIAG_MIN_HIT_RATE  # the old absolute gate passes

    mode = md._determine_recommendation_mode(
        "OUTPERFORM", 0.10, 0.10, health["agg_hit"], health, _GOOD_CPCV
    )
    assert mode["mode"] != "actionable"


def test_skilled_directional_predictor_passes_hit_rate_gate() -> None:
    """Positive control: the gate is passable when direction is genuinely predicted."""
    health = _skilled_health()
    mode = md._determine_recommendation_mode(
        "OUTPERFORM", 0.10, 0.10, health["agg_hit"], health, _GOOD_CPCV
    )
    assert mode["mode"] == "actionable"


def test_gate_uses_the_equal_weight_ic() -> None:
    """F13: a quality-weighted IC of 0.09 must not rescue an equal-weight IC of 0.06."""
    signals = pd.DataFrame(
        {
            "predicted_relative_return": [0.06, 0.05, 0.04, 0.05],
            "ic": [0.20, 0.02, 0.01, 0.01],
            "hit_rate": [0.62, 0.55, 0.55, 0.55],
            "signal": ["OUTPERFORM"] * 4,
            "prob_outperform": [0.62] * 4,
        },
        index=pd.Index(["VOO", "BND", "GLD", "DBC"], name="benchmark"),
    )
    health = _skilled_health()
    health["benchmark_quality_df"] = pd.DataFrame(
        {"benchmark": ["VOO", "BND", "GLD", "DBC"], "nw_ic": [0.40, 0.0, 0.0, 0.0]}
    )
    live, table = md._resolve_live_consensus(signals, health, _GOOD_CPCV)
    assert table is not None
    live_row = table[table["is_live_path"]].iloc[0]
    assert live_row["variant"] == "quality_weighted"
    assert float(live_row["mean_ic"]) >= config.DIAG_MIN_IC  # the inflated number
    assert live_row["recommendation_mode"] != "ACTIONABLE"
    assert live[2] == pytest.approx(float(signals["ic"].mean()))


def test_pooled_ic_significance_is_clustered_by_date() -> None:
    """F13: the pooled IC p-value is Driscoll-Kraay by date, not Newey-West by row."""
    import statsmodels.api as sm
    from scipy.stats import rankdata
    from scipy.stats import t as t_dist

    rng = np.random.default_rng(21)
    n_oos = 72
    dates = pd.date_range("2011-01-31", periods=n_oos, freq="ME")
    common = rng.normal(0.0, 0.1, n_oos)
    ensembles = {}
    for b in range(6):
        y_true = common + rng.normal(0.0, 0.03, n_oos)
        pred = 0.5 * common + rng.normal(0.0, 0.08, n_oos)
        ensembles[f"B{b}"] = _ensemble(f"B{b}", dates, y_true, {"ridge": pred, "gbt": pred})
    health = md._compute_aggregate_health(ensembles)
    # The IC is ranked on the ensemble score before shrinkage (proportional to
    # the prediction when the shrinkage is a constant, as before WP7).
    predicted = health.get("agg_score", health["agg_predicted"])
    realized = health["agg_realized"]

    x_rank = rankdata(predicted.to_numpy(dtype=float))
    y_rank = rankdata(realized.to_numpy(dtype=float))
    months = (predicted.index.year * 12 + predicted.index.month).to_numpy()
    res = sm.OLS(y_rank, sm.add_constant(x_rank)).fit(
        cov_type="hac-groupsum", cov_kwds={"time": months - months.min(), "maxlags": 5}
    )
    n_dates = int(predicted.index.nunique())
    expected = float(2 * t_dist.sf(abs(res.params[1] / res.bse[1]), df=n_dates - 1))
    assert health["nw_pval"] == pytest.approx(expected, rel=1e-9)


def test_conformal_coverage_calibrates_on_realised_residuals_only() -> None:
    """F13: the interval at t ignores residuals whose 6M window ends after t."""
    from src.models.conformal import backtest_conformal_coverage

    rng = np.random.default_rng(4)
    n = 60
    k = 30
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    y_hat = rng.normal(0.0, 0.05, n)
    y_true = y_hat + rng.normal(0.0, 0.10, n)
    y_true[k] = y_hat[k]  # zero residual in the base run
    kwargs = dict(coverage=0.8, method="split", dates=dates, horizon_months=6, trailing_window=12)
    base = backtest_conformal_coverage(y_hat, y_true, **kwargs)
    bumped_true = y_true.copy()
    bumped_true[k] += 50.0
    bumped = backtest_conformal_coverage(y_hat, bumped_true, **kwargs)

    base_widths = dict(zip(base.evaluated_dates, base.widths))
    bumped_widths = dict(zip(bumped.evaluated_dates, bumped.widths))
    # Rows k+1 .. k+5 are scored before row k's window has ended.
    for date in dates[k + 1 : k + 6]:
        assert base_widths[date] == bumped_widths[date]
    # From row k+6 on, row k is realised and enters the calibration set.
    assert bumped_widths[dates[k + 6]] > base_widths[dates[k + 6]]


def test_monthly_signals_no_longer_report_in_sample_conformal_coverage() -> None:
    """F13: only trailing (prequential) coverage is reported."""
    ens = _two_model_ensemble(n_oos=96)
    signals = pd.DataFrame(
        {"predicted_relative_return": [0.02], "raw_ensemble_prediction": [0.04]},
        index=pd.Index(["VTI"], name="benchmark"),
    )
    out = md._compute_conformal_intervals(signals, {"VTI": ens})
    assert "ci_empirical_coverage" not in out.columns
    assert "ci_trailing_empirical_coverage" in out.columns


# ---------------------------------------------------------------------------
# F20 — CPCV fails closed; F02 — CPCV verdict no longer gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cpcv", [None, _cpcv([])], ids=["missing", "unknown"])
def test_missing_or_unknown_cpcv_does_not_permit_actionable(cpcv) -> None:
    health = _skilled_health()
    mode = md._determine_recommendation_mode(
        "UNDERPERFORM", -0.10, 0.10, 0.60, health, cpcv
    )
    assert mode["mode"] != "actionable"


def test_cpcv_verdict_is_diagnostic_only() -> None:
    """F02: a FAIL verdict no longer forces DEFER when every gate passes."""
    health = _skilled_health()
    mode = md._determine_recommendation_mode(
        "OUTPERFORM", 0.18, 0.10, 0.60, health, _cpcv([0.1] + [-0.1] * 6)
    )
    assert mode["mode"] == "actionable"
    assert mode["sell_pct"] == 0.25  # the live mapping is unchanged (step 6)


# ---------------------------------------------------------------------------
# F21 — confidence tiers come from calibrated probabilities
# ---------------------------------------------------------------------------


def test_confidence_tiers_are_not_all_identical() -> None:
    rng = np.random.default_rng(8)
    n = 120
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    strong_z = rng.normal(0.0, 0.10, n)
    strong_y = strong_z + rng.normal(0.0, 0.02, n)
    weak_z = rng.normal(0.0, 0.10, n)
    weak_y = rng.normal(0.0, 0.10, n)
    ensembles = {
        "VOO": _ensemble("VOO", dates, strong_y, {"ridge": strong_z, "gbt": strong_z}),
        "BND": _ensemble("BND", dates, weak_y, {"ridge": weak_z, "gbt": weak_z}),
    }
    signals = pd.DataFrame(
        {
            "predicted_relative_return": [0.15, 0.002],
            "raw_ensemble_prediction": [0.15, 0.002],
            "ic": [0.9, 0.0],
            "hit_rate": [0.9, 0.5],
            "signal": ["OUTPERFORM", "NEUTRAL"],
            "prob_outperform": [0.5, 0.5],
            "confidence_tier": ["LOW", "LOW"],
        },
        index=pd.Index(["VOO", "BND"], name="benchmark"),
    )
    calibrated, *_ = md._calibrate_signals(signals, ensembles)
    tiers = calibrated["confidence_tier"].tolist()
    assert len(set(tiers)) > 1, tiers
    assert calibrated.loc["VOO", "confidence_tier"] == "HIGH"
    assert calibrated.loc["VOO", "prob_outperform"] == pytest.approx(
        calibrated.loc["VOO", "calibrated_prob_outperform"]
    )
    assert calibrated.loc["VOO", "prob_outperform"] > 0.7


def test_confidence_tier_supports_the_signal_direction() -> None:
    from src.models.calibration import confidence_tier_from_probability

    assert confidence_tier_from_probability(0.75, "OUTPERFORM") == "HIGH"
    assert confidence_tier_from_probability(0.62, "OUTPERFORM") == "MODERATE"
    assert confidence_tier_from_probability(0.55, "OUTPERFORM") == "LOW"
    assert confidence_tier_from_probability(0.25, "UNDERPERFORM") == "HIGH"
    # A calibrated 62 % chance of outperforming does not support UNDERPERFORM.
    assert confidence_tier_from_probability(0.62, "UNDERPERFORM") == "LOW"
    assert confidence_tier_from_probability(0.90, "NEUTRAL") == "LOW"
    assert confidence_tier_from_probability(float("nan"), "OUTPERFORM") == "LOW"


def test_scaled_threshold_matches_report_text() -> None:
    """The diagnostic report and the verdict use the same scaled thresholds."""
    from src.models.wfo_engine import cpcv_path_thresholds

    good, marginal = cpcv_path_thresholds(7)
    assert (good, marginal) == (
        math.ceil(config.DIAG_CPCV_MIN_POSITIVE_PATHS * 7 / 28),
        math.ceil(config.DIAG_CPCV_MARGINAL_POSITIVE_PATHS * 7 / 28),
    )


# ---------------------------------------------------------------------------
# model_performance_log: corrected values are stored and never mixed with the
# pre-2026-09-25 definitions (migration 008)
# ---------------------------------------------------------------------------


def _health_record(month_end: str, ic: float, ece: float, **extra) -> dict:
    return {
        "month_end": month_end,
        "aggregate_oos_r2": 0.05,
        "aggregate_nw_ic": ic,
        "aggregate_hit_rate": 0.6,
        "ece": ece,
        "ece_ci_lower": ece - 0.01,
        "ece_ci_upper": ece + 0.01,
        "conformal_target_coverage": 0.8,
        "conformal_empirical_coverage": 0.7,
        "conformal_trailing_empirical_coverage": 0.7,
        "conformal_trailing_coverage_gap": -0.1,
        **extra,
    }


def test_migration_008_tags_existing_rows_as_pre_review(tmp_path) -> None:
    import sqlite3

    from src.database import db_client, migration_runner

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    migration_runner.ensure_migration_table(conn)
    for migration in migration_runner.list_migrations():
        if migration.migration_id.startswith("008"):
            break
        if migration.path.suffix == ".py":
            migration_runner._run_python_migration(conn, migration.path)
        else:
            conn.executescript(migration.path.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO schema_migrations (migration_id, applied_at) VALUES (?, 'x')",
            (migration.migration_id,),
        )
    conn.execute(
        "INSERT INTO model_performance_log (month_end, aggregate_oos_r2, ece) VALUES ('2026-09-30', -0.011, 0.034)"
    )
    conn.commit()
    conn.close()

    conn = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn)
    log = db_client.get_model_performance_log(conn)
    conn.close()
    assert log.loc[pd.Timestamp("2026-09-30"), "metrics_version"] == "pre-2026-09-25"
    assert log.loc[pd.Timestamp("2026-09-30"), "aggregate_oos_r2"] == pytest.approx(-0.011)


def test_new_health_rows_carry_the_current_metrics_version(tmp_path) -> None:
    from src.database import db_client

    conn = db_client.get_connection(str(tmp_path / "fresh.db"))
    db_client.initialize_schema(conn)
    db_client.upsert_model_performance_log(conn, [_health_record("2026-10-31", 0.12, 0.11)])
    log = db_client.get_model_performance_log(conn)
    conn.close()
    assert log["metrics_version"].tolist() == [config.MODEL_HEALTH_METRICS_VERSION]


def test_drift_summary_does_not_mix_metric_definitions() -> None:
    from src.models.drift_monitor import summarize_latest_model_drift

    history = pd.DataFrame(
        [
            _health_record("2026-07-31", 0.02, 0.02, metrics_version="pre-2026-09-25"),
            _health_record("2026-08-31", 0.02, 0.02, metrics_version="pre-2026-09-25"),
            _health_record("2026-09-30", 0.02, 0.03, metrics_version="pre-2026-09-25"),
            _health_record("2026-10-31", 0.12, 0.11, metrics_version=config.MODEL_HEALTH_METRICS_VERSION),
        ]
    )
    summary = summarize_latest_model_drift(history)
    assert summary is not None
    assert summary.history_months == 1
    assert summary.rolling_ece == pytest.approx(0.11)
    assert summary.ic_below_threshold_streak == 0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_grid_shrinkage_follows_v38_rule() -> None:
    from src.models.prequential import best_grid_shrinkage, least_squares_shrinkage

    rng = np.random.default_rng(2)
    z = rng.normal(0.0, 0.1, 400)
    y = 0.42 * z + rng.normal(0.0, 0.05, 400)
    alpha = least_squares_shrinkage(z, y, min_obs=10)
    grid = np.asarray(config.ENSEMBLE_SHRINKAGE_ALPHA_GRID)
    sse = [float(np.sum((y - a * z) ** 2)) for a in grid]
    assert alpha == pytest.approx(float(grid[int(np.argmin(sse))]))
    assert alpha == pytest.approx(0.40)
    # A record with no value never switches the forecast off: v38's floor.
    assert best_grid_shrinkage(sum_zy=-1.0, sum_zz=2.0) == pytest.approx(min(grid))
    # Too few realised rows: no shrinkage.
    assert least_squares_shrinkage(z[:5], y[:5], min_obs=10) == 1.0


def test_prequential_alpha_uses_only_realised_rows() -> None:
    from src.models.prequential import prequential_shrinkage_alphas

    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(9)
    z = rng.normal(0.0, 0.1, 60)
    y = 0.5 * z + rng.normal(0.0, 0.05, 60)
    alphas = prequential_shrinkage_alphas(dates, z, y, horizon_months=6, min_obs=12)
    # Row i has i - 5 realised rows (month(j) + 6 <= month(i)): 12 from row 17.
    assert np.all(alphas[:17] == 1.0)
    bumped = y.copy()
    bumped[40:] *= -5.0  # rows realised only from month 46 on
    alphas_bumped = prequential_shrinkage_alphas(dates, z, bumped, horizon_months=6, min_obs=12)
    np.testing.assert_array_equal(alphas[:46], alphas_bumped[:46])


def test_pesaran_timmermann_edge_cases() -> None:
    from src.models.robust_inference import pesaran_timmermann_test

    rng = np.random.default_rng(12)
    y = rng.normal(0.02, 0.1, 120)
    perfect = pesaran_timmermann_test(y, y, None, lags=5)
    assert perfect.hit_rate == 1.0 and perfect.pt_p_value < 1e-6
    constant = pesaran_timmermann_test(np.full(120, 0.03), y, None, lags=5)
    assert np.isnan(constant.pt_p_value)
    assert constant.excess_over_base_rate <= 0.0
    # Zero forecasts make no call and are left out of every directional rate.
    with_zeros = pesaran_timmermann_test(np.r_[np.zeros(20), y[20:]], y, None, lags=5)
    assert with_zeros.n_obs == 120 and with_zeros.n_calls == 100
    assert with_zeros.hit_rate == 1.0


def test_prequential_platt_ignores_unrealised_outcomes() -> None:
    from src.models.calibration import prequential_platt_probabilities

    rng = np.random.default_rng(13)
    dates = pd.date_range("2012-01-31", periods=80, freq="ME")
    scores = rng.normal(0.0, 0.1, 80)
    outcomes = (scores + rng.normal(0.0, 0.08, 80) > 0).astype(int)
    probs = prequential_platt_probabilities(scores, outcomes, dates, horizon_months=6, min_obs=20)
    # Row i has i - 5 realised rows: 20 (the Platt minimum) from row 25.
    assert np.all(np.isnan(probs[:25]))
    assert np.all(np.isfinite(probs[25:]))
    flipped = outcomes.copy()
    flipped[50:] = 1 - flipped[50:]
    probs_flipped = prequential_platt_probabilities(scores, flipped, dates, horizon_months=6, min_obs=20)
    np.testing.assert_allclose(probs[:56], probs_flipped[:56])


def test_date_block_bootstrap_keeps_same_date_rows_together() -> None:
    from src.models.calibration import block_bootstrap_ece_ci

    rng = np.random.default_rng(14)
    dates = pd.DatetimeIndex(np.repeat(pd.date_range("2015-01-31", periods=48, freq="ME"), 8))
    probs = rng.uniform(0.3, 0.8, len(dates))
    outcomes = (rng.random(len(dates)) < probs).astype(int)
    lo, hi = block_bootstrap_ece_ci(probs, outcomes, block_len=6, n_bootstrap=100, dates=dates)
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isnan(block_bootstrap_ece_ci(probs, outcomes, n_bootstrap=0)[0])


def test_backdated_health_snapshot_ignores_later_months(tmp_path) -> None:
    """A dry run as of 2026-02 must not summarise drift from months after it."""
    from datetime import date

    from src.database import db_client
    from src.models.calibration import CalibrationResult

    conn = db_client.get_connection(str(tmp_path / "log.db"))
    db_client.initialize_schema(conn)
    db_client.upsert_model_performance_log(
        conn,
        [
            _health_record(f"2026-0{month}-28", 0.15, 0.02, metrics_version="pre-2026-09-25")
            for month in range(2, 10)
        ],
    )
    summary = md._record_model_health_snapshot(
        conn,
        date(2026, 2, 28),
        {"oos_r2": 0.06, "nw_ic": 0.16, "agg_hit": 0.64},
        CalibrationResult(n_obs=900, method="platt", ece=0.146, ece_ci_lower=0.1, ece_ci_upper=0.2),
        None,
        dry_run=True,
    )
    conn.close()
    assert summary is not None
    assert summary.as_of_month == "2026-02-28"
    assert summary.history_months == 1
    assert summary.rolling_ece == pytest.approx(0.146)
