"""
Black-Litterman portfolio construction for the v4.0 PGR vesting decision engine.

The Black-Litterman model combines equilibrium market returns (from a reference
portfolio's market cap weights) with investor views (from the WFO ensemble
signals) to produce optimal portfolio weights.

Key design choices:
  - Equilibrium returns (π) estimated from market-cap-weighted ETF returns
    via the reverse optimization formula: π = δΣw_mkt.
  - View matrix P is one row per ETF with an active signal; Q = point
    predictions from the ensemble.
  - View uncertainty Ω is diagonal with Ω_ii = model_rmse² × scalar.
    Models with lower RMSE get proportionally higher view confidence.
  - Covariance matrix is Ledoit-Wolf shrunk for stability (scikit-learn).
  - Final weights are further capped at KELLY_MAX_POSITION per ETF.

Uses PyPortfolioOpt's BlackLittermanModel.  Requires:
    PyPortfolioOpt>=1.5.5
    scikit-learn (Ledoit-Wolf)

CPCV/WFO inputs:
  - ``ensemble_signals``: output of ``run_ensemble_benchmarks()`` from
    multi_benchmark_wfo.py — one EnsembleWFOResult per ETF.
  - ``covariance_matrix``: historical monthly return covariance across
    ETFs (DatetimeIndex × ETF tickers); shrinkage applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import config
from src.models.multi_benchmark_wfo import EnsembleWFOResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BLDiagnostics:
    """Operational metadata about whether BL used a fallback allocation."""

    fallback_used: bool
    fallback_reason: str | None
    n_active_tickers: int
    n_view_tickers: int


def _ledoit_wolf_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Ledoit-Wolf shrunk covariance matrix of monthly returns.

    Args:
        returns_df: DataFrame of monthly returns (rows = dates, cols = tickers).

    Returns:
        Shrunk covariance as a symmetric DataFrame with same columns/index.
    """
    lw = LedoitWolf()
    lw.fit(returns_df.dropna())
    cov_arr = lw.covariance_
    return pd.DataFrame(cov_arr, index=returns_df.columns, columns=returns_df.columns)


def build_bl_weights(
    ensemble_signals: dict[str, EnsembleWFOResult],
    returns_df: pd.DataFrame,
    risk_aversion: float | None = None,
    view_confidence_scalar: float | None = None,
    market_weights: dict[str, float] | None = None,
    tau: float | None = None,
    risk_free_rate: float | None = None,
    return_diagnostics: bool = False,
) -> dict[str, float] | tuple[dict[str, float], BLDiagnostics]:
    """
    Compute Black-Litterman optimal ETF weights from ensemble WFO signals.

    Steps:
      1. Compute Ledoit-Wolf shrunk covariance of ETF returns.
      2. Estimate equilibrium returns π via reverse optimization.
      3. Build view matrix P (one row per active signal) and view returns Q.
      4. Build diagonal uncertainty matrix Ω = diag(RMSE² × scalar).
      5. Run PyPortfolioOpt's BlackLittermanModel.
      6. Return posterior-optimal weights, capped at KELLY_MAX_POSITION.

    Args:
        ensemble_signals:       Dict of ETF ticker → EnsembleWFOResult from
                                ``run_ensemble_benchmarks()``.
        returns_df:             Monthly ETF returns (rows = dates, cols = tickers).
                                Must cover the full backtest period.
        risk_aversion:          Market risk aversion δ (default: config.BL_RISK_AVERSION).
        view_confidence_scalar: Scales Ω = MAE² × scalar.  Higher → less confident
                                in views (default: config.BL_VIEW_CONFIDENCE_SCALAR).
        market_weights:         Equal-weight market cap proxy if not provided.
                                Dict mapping ETF ticker → weight (must sum to 1.0).
        tau:                    BL prior uncertainty scalar τ (default: config.BL_TAU).
                                Controls how much the model trusts the views vs the
                                equilibrium prior. Higher τ → more weight on views.
        risk_free_rate:         Risk-free rate for max-Sharpe optimisation (default: 0.04).
                                Must be in the same time units as the returns in returns_df.
                                Pass 0.0 to optimise without a risk-free floor.
        return_diagnostics:     If True, return ``(weights, diagnostics)`` where
                                diagnostics includes whether a fallback path was
                                used and why.

    Returns:
        Dict mapping ETF ticker → portfolio weight (0–KELLY_MAX_POSITION), or
        ``(weights, diagnostics)`` when ``return_diagnostics=True``. Weights may
        not sum exactly to 1.0 because per-ETF caps are applied before
        re-normalisation. Returns equal weights if optimization fails.

    Raises:
        ImportError: If PyPortfolioOpt is not installed.
        ValueError:  If ``returns_df`` has fewer than 12 rows.
    """
    from pypfopt import BlackLittermanModel, risk_models, expected_returns

    if risk_aversion is None:
        risk_aversion = config.BL_RISK_AVERSION
    if view_confidence_scalar is None:
        view_confidence_scalar = config.BL_VIEW_CONFIDENCE_SCALAR

    if returns_df.shape[0] < 12:
        raise ValueError(
            f"Need at least 12 months of return data, got {returns_df.shape[0]}."
        )

    def _finish(
        weights: dict[str, float],
        *,
        fallback_used: bool,
        fallback_reason: str | None,
        n_active_tickers: int,
        n_view_tickers: int,
    ) -> dict[str, float] | tuple[dict[str, float], BLDiagnostics]:
        diagnostics = BLDiagnostics(
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            n_active_tickers=n_active_tickers,
            n_view_tickers=n_view_tickers,
        )
        if return_diagnostics:
            return weights, diagnostics
        return weights

    # Restrict to ETFs present in both signals and returns
    active_tickers = [
        t for t in ensemble_signals
        if t in returns_df.columns
    ]
    if not active_tickers:
        return _finish(
            {},
            fallback_used=False,
            fallback_reason=None,
            n_active_tickers=0,
            n_view_tickers=0,
        )

    ret_subset = returns_df[active_tickers].dropna()
    if ret_subset.shape[0] < 12:
        return _finish(
            {t: 1.0 / len(active_tickers) for t in active_tickers},
            fallback_used=True,
            fallback_reason="insufficient_overlap_history",
            n_active_tickers=len(active_tickers),
            n_view_tickers=0,
        )

    # 1. Ledoit-Wolf covariance
    cov_matrix = _ledoit_wolf_covariance(ret_subset)

    # 2. Market weights: equal-weight proxy if not provided
    if market_weights is None:
        n = len(active_tickers)
        w_mkt = pd.Series({t: 1.0 / n for t in active_tickers})
    else:
        w_mkt = pd.Series(market_weights).reindex(active_tickers).fillna(0.0)
        w_mkt = w_mkt / w_mkt.sum()

    # 3. Equilibrium returns via reverse optimization: π = δΣw_mkt
    cov_arr = cov_matrix.values
    w_arr = w_mkt.values
    pi_arr = risk_aversion * cov_arr @ w_arr
    pi = pd.Series(pi_arr, index=active_tickers)

    # 4. Build views (only from benchmarks with mean_ic > 0)
    view_tickers: list[str] = []
    Q_list: list[float] = []
    omega_diag: list[float] = []

    for ticker in active_tickers:
        sig = ensemble_signals[ticker]
        if sig.mean_ic <= 0:
            continue
        # Use the ensemble mean OOS y_hat as the view return (best available estimate).
        # If model results exist, compute the mean predicted return across all folds.
        view_return = 0.0
        if sig.model_results:
            all_preds: list[float] = []
            for res in sig.model_results.values():
                try:
                    all_preds.extend(res.y_hat_all.tolist())
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Could not extract BL view predictions for ticker %s. Error=%r",
                        ticker,
                        exc,
                    )
            if all_preds:
                view_return = float(np.mean(all_preds))
        view_tickers.append(ticker)
        Q_list.append(view_return)
        mae = max(sig.mean_mae, 1e-6) if not np.isnan(sig.mean_mae) else 1e-4
        omega_diag.append(mae ** 2 * view_confidence_scalar)

    if not view_tickers:
        # No positive-IC views — return equal weights
        return _finish(
            {t: 1.0 / len(active_tickers) for t in active_tickers},
            fallback_used=True,
            fallback_reason="no_positive_views",
            n_active_tickers=len(active_tickers),
            n_view_tickers=0,
        )

    # P matrix: identity rows for each view ticker
    n_views = len(view_tickers)
    n_assets = len(active_tickers)
    P = np.zeros((n_views, n_assets))
    for row_i, vt in enumerate(view_tickers):
        col_j = active_tickers.index(vt)
        P[row_i, col_j] = 1.0

    Q = np.array(Q_list)
    omega = np.diag(omega_diag)

    # 5. Black-Litterman model
    try:
        bl = BlackLittermanModel(
            cov_matrix=cov_matrix,
            pi=pi,
            Q=Q,
            P=P,
            omega=omega,
            tau=tau if tau is not None else config.BL_TAU,
            risk_aversion=risk_aversion,
        )
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        # Efficient frontier: max Sharpe with BL inputs
        from pypfopt import EfficientFrontier
        ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(0, config.KELLY_MAX_POSITION))
        _rf = risk_free_rate if risk_free_rate is not None else 0.04
        ef.max_sharpe(risk_free_rate=_rf)
        raw_weights = ef.clean_weights()
        return _finish(
            dict(raw_weights),
            fallback_used=False,
            fallback_reason=None,
            n_active_tickers=len(active_tickers),
            n_view_tickers=len(view_tickers),
        )

    except Exception as exc:  # noqa: BLE001 — fall back to equal weight
        logger.exception(
            "Black-Litterman optimization failed; falling back to equal weights. Error=%r",
            exc,
        )
        return _finish(
            {t: 1.0 / len(active_tickers) for t in active_tickers},
            fallback_used=True,
            fallback_reason="optimization_failure",
            n_active_tickers=len(active_tickers),
            n_view_tickers=len(view_tickers),
        )


def compute_equilibrium_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float | None = None,
) -> pd.Series:
    """
    Compute implied equilibrium returns via reverse optimization.

    π = δ × Σ × w_mkt

    Args:
        cov_matrix:     Covariance matrix (symmetric DataFrame, tickers × tickers).
        market_weights: Market cap weights (Series, same tickers as cov_matrix).
        risk_aversion:  Market risk aversion δ (default config.BL_RISK_AVERSION).

    Returns:
        Series of implied equilibrium returns per ticker.
    """
    if risk_aversion is None:
        risk_aversion = config.BL_RISK_AVERSION
    cov_arr = cov_matrix.values
    w_arr = market_weights.reindex(cov_matrix.index).fillna(0.0).values
    pi_arr = risk_aversion * cov_arr @ w_arr
    return pd.Series(pi_arr, index=cov_matrix.index)
