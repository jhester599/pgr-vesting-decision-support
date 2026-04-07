"""
Monte Carlo tax scenario analysis using Geometric Brownian Motion.

Tier 4.5 (v35): extends the deterministic three-scenario framework (v7.1)
with stochastic price paths to quantify outcome uncertainty in the
HOLD_TO_LTCG scenario.

Key question answered:
  "Given my model's return forecast and historical volatility, what is the
   probability that waiting 366 days for LTCG treatment produces MORE
   after-tax net proceeds than selling immediately at STCG rates?"

Model: Geometric Brownian Motion (risk-neutral with drift = model forecast)
    dS = S * (mu * dt + sigma * dW)
    S(T) = S(0) * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)

N=1000 independent paths are simulated (reproducible via fixed seed).
Each path produces a terminal price; net proceeds are computed after
applying the appropriate capital gains tax rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Distribution of after-tax net proceeds across simulated price paths."""

    scenario_label: str        # Scenario this covers (e.g. "HOLD_TO_LTCG")
    n_paths: int               # Number of simulated paths
    annual_drift: float        # Annualized drift (model forecast)
    annual_vol: float          # Annualized volatility used

    # Net-proceeds distribution (after tax, across all paths)
    net_proceeds_mean: float
    net_proceeds_p10: float
    net_proceeds_p25: float
    net_proceeds_p50: float
    net_proceeds_p75: float
    net_proceeds_p90: float

    # Key probabilities
    prob_beats_sell_now: float   # P(HOLD_TO_LTCG net > SELL_NOW_STCG net)
    prob_positive_gain: float    # P(terminal price > cost_basis)


@dataclass
class MonteCarloTaxAnalysis:
    """Full Monte Carlo tax-sensitivity analysis for the next vesting decision."""

    sell_now_net: float                # Deterministic SELL_NOW_STCG net proceeds
    sell_now_tax_rate: float           # STCG rate applied to sell-now scenario
    hold_ltcg: MonteCarloResult        # MC outcome distribution for HOLD_TO_LTCG
    current_price: float               # S0 used in simulation
    cost_basis_per_share: float        # Cost basis per share
    shares: float                      # Shares analysed
    horizon_days: int                  # Holding period simulated (typically 366)


# ---------------------------------------------------------------------------
# GBM simulator
# ---------------------------------------------------------------------------


def simulate_gbm_terminal_prices(
    s0: float,
    annual_drift: float,
    annual_vol: float,
    days: int,
    n_paths: int,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate terminal prices under Geometric Brownian Motion.

    Uses the exact GBM solution (not Euler discretisation) for the terminal
    price distribution, which is log-normal and has no discretisation error.

    Args:
        s0:           Starting price.
        annual_drift: Annualised drift (model's expected return, e.g. 0.08).
        annual_vol:   Annualised volatility (e.g. 0.22).
        days:         Number of calendar days in the holding period.
        n_paths:      Number of independent paths to draw.
        seed:         Random seed for reproducibility (None for entropy).

    Returns:
        1-D numpy array of shape (n_paths,) with terminal prices.
    """
    if s0 <= 0:
        raise ValueError(f"Starting price must be positive, got {s0}")
    if annual_vol < 0:
        raise ValueError(f"Volatility must be non-negative, got {annual_vol}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")

    rng = np.random.default_rng(seed)
    t = days / 365.0  # convert calendar days to years

    # log-normal terminal price: S(T) = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    z = rng.standard_normal(n_paths)
    log_returns = (annual_drift - 0.5 * annual_vol ** 2) * t + annual_vol * math.sqrt(t) * z
    return s0 * np.exp(log_returns)


# ---------------------------------------------------------------------------
# Historical volatility estimator
# ---------------------------------------------------------------------------


def estimate_annual_vol(prices: "np.ndarray | list[float]", trading_days_per_year: int = 252) -> float:
    """
    Estimate annualised volatility from a price series using log-returns.

    Args:
        prices:                 Sequence of prices (at least 2 observations).
        trading_days_per_year:  Annualisation factor.

    Returns:
        Annualised volatility as a decimal (e.g. 0.22 for 22%).

    Raises:
        ValueError: If fewer than 2 prices are provided.
    """
    arr = np.asarray(prices, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"Need at least 2 prices, got shape {arr.shape}")
    log_rets = np.diff(np.log(arr))
    daily_vol = float(np.std(log_rets, ddof=1))
    return daily_vol * math.sqrt(trading_days_per_year)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def run_monte_carlo_tax_analysis(
    current_price: float,
    cost_basis_per_share: float,
    shares: float,
    annual_vol: float,
    annual_drift: float,
    horizon_days: int = 366,
    n_paths: int = 1000,
    seed: int | None = 42,
    ltcg_rate: float | None = None,
    stcg_rate: float | None = None,
) -> MonteCarloTaxAnalysis:
    """
    Run a Monte Carlo tax-sensitivity analysis for a HOLD_TO_LTCG decision.

    Simulates N GBM price paths over ``horizon_days`` and computes the
    distribution of after-tax net proceeds under the LTCG scenario.
    Compares to the deterministic SELL_NOW_STCG net as the reference outcome.

    Args:
        current_price:         Current market price per share (S0).
        cost_basis_per_share:  RSU cost basis per share (FMV on vest date).
        shares:                Number of shares being evaluated.
        annual_vol:            Annualised return volatility (e.g. from recent history).
        annual_drift:          Annualised drift = model's expected return for the horizon.
        horizon_days:          Calendar days in the HOLD_TO_LTCG holding period (default 366).
        n_paths:               Number of Monte Carlo paths (default 1000).
        seed:                  RNG seed for reproducibility (default 42).
        ltcg_rate:             LTCG tax rate override. Default: config.LTCG_RATE.
        stcg_rate:             STCG tax rate override. Default: config.STCG_RATE.

    Returns:
        MonteCarloTaxAnalysis with sell-now reference and HOLD_TO_LTCG distribution.
    """
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE

    # --- Deterministic SELL_NOW_STCG reference ---
    sell_now_gain_per_share = current_price - cost_basis_per_share
    sell_now_tax = max(0.0, sell_now_gain_per_share) * stcg_rate * shares
    sell_now_gross = shares * current_price
    sell_now_net = sell_now_gross - sell_now_tax

    # --- Monte Carlo HOLD_TO_LTCG distribution ---
    terminal_prices = simulate_gbm_terminal_prices(
        s0=current_price,
        annual_drift=annual_drift,
        annual_vol=annual_vol,
        days=horizon_days,
        n_paths=n_paths,
        seed=seed,
    )

    # Compute net proceeds for each path under LTCG tax treatment
    gross_ltcg = shares * terminal_prices
    gain_per_share_ltcg = terminal_prices - cost_basis_per_share
    # Only gains are taxed; losses produce a tax benefit (negative tax_liability)
    tax_ltcg = gain_per_share_ltcg * ltcg_rate * shares
    net_ltcg = gross_ltcg - tax_ltcg

    # Key probabilities
    prob_beats = float(np.mean(net_ltcg > sell_now_net))
    prob_gain = float(np.mean(terminal_prices > cost_basis_per_share))

    mc_result = MonteCarloResult(
        scenario_label="HOLD_TO_LTCG",
        n_paths=n_paths,
        annual_drift=annual_drift,
        annual_vol=annual_vol,
        net_proceeds_mean=float(np.mean(net_ltcg)),
        net_proceeds_p10=float(np.percentile(net_ltcg, 10)),
        net_proceeds_p25=float(np.percentile(net_ltcg, 25)),
        net_proceeds_p50=float(np.percentile(net_ltcg, 50)),
        net_proceeds_p75=float(np.percentile(net_ltcg, 75)),
        net_proceeds_p90=float(np.percentile(net_ltcg, 90)),
        prob_beats_sell_now=prob_beats,
        prob_positive_gain=prob_gain,
    )

    return MonteCarloTaxAnalysis(
        sell_now_net=sell_now_net,
        sell_now_tax_rate=stcg_rate,
        hold_ltcg=mc_result,
        current_price=current_price,
        cost_basis_per_share=cost_basis_per_share,
        shares=shares,
        horizon_days=horizon_days,
    )
