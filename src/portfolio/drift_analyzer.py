"""
Portfolio drift analyzer: sector overlap between the concentrated PGR
position and the target diversified allocation.

Because PGR is a P&C Insurance stock (Financials sector), selling a
tranche and immediately buying a broad market ETF like VTI creates
redundant sector exposure (the ETF already holds ~14% Financials).

This module quantifies the overlap and recommends which target ETFs to
buy with the proceeds to drive the aggregate portfolio toward true
total-market sector equilibrium.

Target allocation (user-configurable):
  These are the low-cost index fund targets from the project brief.
  Proceeds are directed away from sectors where PGR creates overlap
  and toward underweighted sectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sector weight reference data
# ---------------------------------------------------------------------------

# Approximate sector weights for target ETFs (as of 2026-Q1).
# Source: fund fact sheets / sponsor websites. Update annually.
# Expense ratios are stored separately in ETF_EXPENSE_RATIOS below.
ETF_SECTOR_WEIGHTS: dict[str, dict[str, float]] = {

    # ------------------------------------------------------------------
    # US Broad Market (core diversifiers — highest priority for proceeds)
    # ------------------------------------------------------------------
    "FZROX": {  # Fidelity Zero Total Market Index Fund  | ER: 0.00%
        "Information Technology": 0.313,
        "Health Care": 0.123,
        "Financials": 0.136,
        "Consumer Discretionary": 0.107,
        "Industrials": 0.087,
        "Communication Services": 0.085,
        "Consumer Staples": 0.058,
        "Energy": 0.039,
        "Real Estate": 0.025,
        "Materials": 0.023,
        "Utilities": 0.022,
    },
    "FNILX": {  # Fidelity Zero Large Cap Index Fund     | ER: 0.00%
        "Information Technology": 0.323,
        "Health Care": 0.122,
        "Financials": 0.135,
        "Consumer Discretionary": 0.109,
        "Communication Services": 0.090,
        "Industrials": 0.083,
        "Consumer Staples": 0.059,
        "Energy": 0.038,
        "Real Estate": 0.023,
        "Materials": 0.021,
        "Utilities": 0.020,
    },
    "FSKAX": {  # Fidelity Total Market Index Fund       | ER: 0.015%
        "Information Technology": 0.310,
        "Health Care": 0.124,
        "Financials": 0.137,
        "Consumer Discretionary": 0.107,
        "Industrials": 0.088,
        "Communication Services": 0.084,
        "Consumer Staples": 0.057,
        "Energy": 0.039,
        "Real Estate": 0.025,
        "Materials": 0.023,
        "Utilities": 0.022,
    },
    "VTI": {  # Vanguard Total Stock Market ETF          | ER: 0.03%
        "Information Technology": 0.312,
        "Health Care": 0.124,
        "Financials": 0.134,
        "Consumer Discretionary": 0.106,
        "Industrials": 0.088,
        "Communication Services": 0.085,
        "Consumer Staples": 0.057,
        "Energy": 0.038,
        "Real Estate": 0.025,
        "Materials": 0.023,
        "Utilities": 0.023,
    },
    "ITOT": {  # iShares Core S&P Total US Stock Market  | ER: 0.03%
        "Information Technology": 0.316,
        "Health Care": 0.122,
        "Financials": 0.131,
        "Consumer Discretionary": 0.109,
        "Communication Services": 0.086,
        "Industrials": 0.086,
        "Consumer Staples": 0.056,
        "Energy": 0.038,
        "Real Estate": 0.025,
        "Materials": 0.022,
        "Utilities": 0.022,
    },
    "SCHB": {  # Schwab US Broad Market ETF              | ER: 0.03%
        "Information Technology": 0.312,
        "Health Care": 0.123,
        "Financials": 0.133,
        "Consumer Discretionary": 0.107,
        "Industrials": 0.089,
        "Communication Services": 0.085,
        "Consumer Staples": 0.058,
        "Energy": 0.039,
        "Real Estate": 0.025,
        "Materials": 0.023,
        "Utilities": 0.023,
    },

    # ------------------------------------------------------------------
    # International (developed + emerging — geographic diversification)
    # ------------------------------------------------------------------
    "VXUS": {  # Vanguard Total International Stock ETF  | ER: 0.07%
        "Financials": 0.218,
        "Industrials": 0.143,
        "Information Technology": 0.131,
        "Consumer Discretionary": 0.105,
        "Health Care": 0.096,
        "Materials": 0.079,
        "Consumer Staples": 0.074,
        "Energy": 0.057,
        "Communication Services": 0.054,
        "Utilities": 0.032,
        "Real Estate": 0.021,
    },
    "FZILX": {  # Fidelity Zero International Index Fund | ER: 0.00%
        "Financials": 0.220,
        "Industrials": 0.140,
        "Information Technology": 0.133,
        "Consumer Discretionary": 0.104,
        "Health Care": 0.094,
        "Materials": 0.080,
        "Consumer Staples": 0.073,
        "Energy": 0.056,
        "Communication Services": 0.053,
        "Utilities": 0.031,
        "Real Estate": 0.022,
    },
    "VEA": {  # Vanguard Developed Markets Index ETF     | ER: 0.05%
        "Financials": 0.214,
        "Industrials": 0.151,
        "Health Care": 0.114,
        "Consumer Discretionary": 0.113,
        "Consumer Staples": 0.086,
        "Information Technology": 0.085,
        "Materials": 0.082,
        "Energy": 0.048,
        "Communication Services": 0.047,
        "Utilities": 0.037,
        "Real Estate": 0.028,
    },
    "IXUS": {  # iShares Core MSCI Total Intl Stock ETF  | ER: 0.07%
        "Financials": 0.216,
        "Industrials": 0.146,
        "Information Technology": 0.130,
        "Consumer Discretionary": 0.106,
        "Health Care": 0.098,
        "Materials": 0.079,
        "Consumer Staples": 0.073,
        "Energy": 0.055,
        "Communication Services": 0.053,
        "Utilities": 0.033,
        "Real Estate": 0.022,
    },
    "VWO": {  # Vanguard Emerging Markets Stock Index    | ER: 0.08%
        "Financials": 0.220,
        "Information Technology": 0.210,
        "Consumer Discretionary": 0.130,
        "Materials": 0.082,
        "Energy": 0.072,
        "Industrials": 0.070,
        "Communication Services": 0.068,
        "Consumer Staples": 0.062,
        "Health Care": 0.042,
        "Utilities": 0.038,
        "Real Estate": 0.012,
    },

    # ------------------------------------------------------------------
    # Sector tilts — directly counter PGR's Financials overweight
    # ------------------------------------------------------------------
    "VGT": {  # Vanguard Information Technology ETF      | ER: 0.10%
        "Information Technology": 0.991,
        "Communication Services": 0.009,
    },
    "FTEC": {  # Fidelity MSCI Information Technology ETF | ER: 0.084%
        "Information Technology": 0.990,
        "Communication Services": 0.010,
    },
    "VHT": {  # Vanguard Health Care ETF                 | ER: 0.10%
        "Health Care": 1.000,
    },
    "FHLC": {  # Fidelity MSCI Health Care ETF            | ER: 0.084%
        "Health Care": 1.000,
    },
    "VIS": {  # Vanguard Industrials ETF                 | ER: 0.10%
        "Industrials": 1.000,
    },

    # ------------------------------------------------------------------
    # Dividend / income factor
    # ------------------------------------------------------------------
    "SCHD": {  # Schwab US Dividend Equity ETF           | ER: 0.06%
        "Financials": 0.182,
        "Industrials": 0.172,
        "Health Care": 0.148,
        "Consumer Staples": 0.131,
        "Energy": 0.099,
        "Information Technology": 0.091,
        "Materials": 0.052,
        "Consumer Discretionary": 0.043,
        "Utilities": 0.038,
        "Communication Services": 0.030,
        "Real Estate": 0.020,
    },
    "VYM": {  # Vanguard High Dividend Yield ETF         | ER: 0.06%
        "Financials": 0.210,
        "Health Care": 0.148,
        "Industrials": 0.142,
        "Consumer Staples": 0.119,
        "Energy": 0.099,
        "Information Technology": 0.083,
        "Consumer Discretionary": 0.050,
        "Materials": 0.050,
        "Utilities": 0.042,
        "Communication Services": 0.033,
        "Real Estate": 0.020,
    },

    # ------------------------------------------------------------------
    # Fixed income
    # ------------------------------------------------------------------
    "FXNAX": {  # Fidelity US Bond Index Fund            | ER: 0.025%
        "Government": 0.450,
        "Mortgage-Backed": 0.280,
        "Corporate": 0.220,
        "Other": 0.050,
    },
    "BND": {  # Vanguard Total Bond Market ETF           | ER: 0.03%
        "Government": 0.440,
        "Mortgage-Backed": 0.260,
        "Corporate": 0.275,
        "Other": 0.025,
    },
    "AGG": {  # iShares Core US Aggregate Bond ETF       | ER: 0.03%
        "Government": 0.430,
        "Mortgage-Backed": 0.270,
        "Corporate": 0.270,
        "Other": 0.030,
    },
    "BNDX": {  # Vanguard Total International Bond ETF  | ER: 0.07%
        "Government": 0.580,
        "Corporate": 0.260,
        "Mortgage-Backed": 0.080,
        "Other": 0.080,
    },
}

# Expense ratios (annual %) for reference and display.
# Used only for output / reporting — not in optimization math.
ETF_EXPENSE_RATIOS: dict[str, float] = {
    # US Broad
    "FZROX": 0.000, "FNILX": 0.000, "FSKAX": 0.015, "VTI": 0.030,
    "ITOT": 0.030, "SCHB": 0.030,
    # International
    "VXUS": 0.070, "FZILX": 0.000, "VEA": 0.050, "IXUS": 0.070, "VWO": 0.080,
    # Sector tilts
    "VGT": 0.100, "FTEC": 0.084, "VHT": 0.100, "FHLC": 0.084, "VIS": 0.100,
    # Dividend
    "SCHD": 0.060, "VYM": 0.060,
    # Fixed income
    "FXNAX": 0.025, "BND": 0.030, "AGG": 0.030, "BNDX": 0.070,
}

# Default fund universe used by recommend_reallocation() when no override given.
# Ordered from most tax-efficient / lowest-ER to most targeted.
DEFAULT_ETF_UNIVERSE: list[str] = [
    # Core US (zero/near-zero ER)
    "FZROX", "FNILX", "FSKAX",
    # Core US alternatives (portable to any brokerage)
    "VTI", "ITOT", "SCHB",
    # International
    "VXUS", "FZILX", "VEA",
    # Sector tilts (for Financials counterweight)
    "VGT", "FTEC", "VHT", "FHLC",
    # Dividend
    "SCHD", "VYM",
    # Fixed income
    "FXNAX", "BND", "AGG",
]

# PGR's sector classification
PGR_SECTOR = "Financials"
PGR_SUBSECTOR = "P&C Insurance"

# True total-market equilibrium sector weights (MSCI ACWI approximate)
MARKET_EQUILIBRIUM_WEIGHTS: dict[str, float] = {
    "Information Technology": 0.250,
    "Financials": 0.155,
    "Health Care": 0.115,
    "Consumer Discretionary": 0.108,
    "Industrials": 0.102,
    "Communication Services": 0.080,
    "Consumer Staples": 0.063,
    "Energy": 0.046,
    "Materials": 0.047,
    "Real Estate": 0.023,
    "Utilities": 0.028,
    "Bonds/Other": 0.000,  # handled via FXNAX allocation
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """
    Snapshot of the current aggregate portfolio sector weights.
    """
    pgr_value: float                   # Market value of PGR position
    etf_holdings: dict[str, float]     # ETF ticker -> market value
    total_value: float | None = None   # Computed in __post_init__

    def __post_init__(self) -> None:
        self.total_value = self.pgr_value + sum(self.etf_holdings.values())


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_sector_weights(state: PortfolioState) -> dict[str, float]:
    """
    Compute the aggregate sector weights of the entire portfolio.

    The PGR position is treated as 100% Financials (P&C Insurance sub-sector).
    ETF holdings are weighted by their sector breakdown tables.

    Args:
        state: Current portfolio snapshot.

    Returns:
        Dict mapping sector name to portfolio weight (sums to ~1.0).
    """
    sector_values: dict[str, float] = {}

    # PGR contribution (100% Financials)
    sector_values[PGR_SECTOR] = sector_values.get(PGR_SECTOR, 0.0) + state.pgr_value

    # ETF contributions
    for ticker, etf_value in state.etf_holdings.items():
        weights = ETF_SECTOR_WEIGHTS.get(ticker, {})
        for sector, weight in weights.items():
            sector_values[sector] = sector_values.get(sector, 0.0) + etf_value * weight

    # Normalize to portfolio weight
    total = sum(sector_values.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(sector_values.items())}


def compute_sector_deviation(state: PortfolioState) -> dict[str, float]:
    """
    Compute the deviation of the current portfolio sector weights from
    the market equilibrium weights.

    Positive deviation = overweight vs. equilibrium.
    Negative deviation = underweight vs. equilibrium.

    Args:
        state: Current portfolio snapshot.

    Returns:
        Dict mapping sector to deviation in percentage points.
    """
    current = compute_sector_weights(state)
    deviations: dict[str, float] = {}

    all_sectors = set(current.keys()) | set(MARKET_EQUILIBRIUM_WEIGHTS.keys())
    for sector in all_sectors:
        curr_w = current.get(sector, 0.0)
        target_w = MARKET_EQUILIBRIUM_WEIGHTS.get(sector, 0.0)
        deviations[sector] = curr_w - target_w

    return dict(sorted(deviations.items(), key=lambda x: x[1]))


def recommend_reallocation(
    state: PortfolioState,
    proceeds: float,
    available_etfs: list[str] | None = None,
) -> dict[str, float]:
    """
    Recommend how to allocate sale proceeds to minimize sector deviation.

    Directs proceeds toward the most underweighted sectors by purchasing
    the ETF(s) with the highest concentration in those sectors.

    Args:
        state:          Current portfolio snapshot.
        proceeds:       Cash proceeds available for reallocation (after tax).
        available_etfs: ETFs to score and consider. Defaults to
                        DEFAULT_ETF_UNIVERSE (18 funds). Pass a shorter list
                        to restrict choices to funds held at a specific broker.

    Returns:
        Dict mapping ETF ticker to recommended allocation amount ($).
        Zero-score funds are omitted. At most the top 5 funds receive
        a non-zero allocation to keep the recommendation actionable.
    """
    # Fallback core funds used when the portfolio is already at equilibrium.
    _CORE_FALLBACK = ["FZROX", "VXUS", "FXNAX"]

    if available_etfs is None:
        available_etfs = DEFAULT_ETF_UNIVERSE

    deviations = compute_sector_deviation(state)

    # Find the most underweighted sectors (negative deviations)
    underweight = {k: -v for k, v in deviations.items() if v < 0}

    if not underweight:
        # Portfolio already at equilibrium — split equally across core 3
        core = [e for e in _CORE_FALLBACK if e in available_etfs] or available_etfs[:3]
        return {etf: proceeds / len(core) for etf in core}

    # Score each available ETF by its coverage of underweighted sectors
    scores: dict[str, float] = {}
    for etf in available_etfs:
        if etf not in ETF_SECTOR_WEIGHTS:
            continue
        etf_weights = ETF_SECTOR_WEIGHTS[etf]
        score = sum(
            etf_weights.get(sector, 0.0) * underweight_amount
            for sector, underweight_amount in underweight.items()
        )
        if score > 0:
            scores[etf] = score

    if not scores:
        core = [e for e in _CORE_FALLBACK if e in available_etfs] or available_etfs[:3]
        return {etf: proceeds / len(core) for etf in core}

    # Retain only the top-5 scoring funds to keep allocation actionable
    top_etfs = sorted(scores, key=lambda e: scores[e], reverse=True)[:5]
    top_scores = {e: scores[e] for e in top_etfs}
    total_score = sum(top_scores.values())

    return {
        etf: (score / total_score) * proceeds
        for etf, score in top_scores.items()
    }
