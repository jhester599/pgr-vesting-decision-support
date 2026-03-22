"""
Portfolio rebalancer: integrates ML signal, tax optimization, and
sector drift analysis into a unified vesting date recommendation.

This module is the top-level decision support output layer. It:
  1. Reads the WFO signal (predicted 6-month forward return + confidence)
  2. Reads the current tax lot position (LTCG/STCG breakdown)
  3. Reads the sector drift analysis
  4. Outputs a structured recommendation for each vesting event:
     - Recommended sale percentage (0–100%)
     - Estimated net after-tax proceeds
     - Recommended ETF reallocation by ticker
     - Blackout window warning (if applicable)
     - Tax lot selection strategy (which lots to sell)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.models.wfo_engine import WFOResult
from src.tax.capital_gains import TaxLot, optimize_sale, compute_position_summary
from src.portfolio.drift_analyzer import (
    PortfolioState,
    recommend_reallocation,
    compute_sector_deviation,
)
import config


@dataclass
class VestingRecommendation:
    """Full recommendation for a single vesting date."""
    vest_date: date
    rsu_type: str
    predicted_6m_return: float
    ic_confidence: float
    wfo_hit_rate: float
    recommended_sell_pct: float        # 0.0–1.0
    shares_to_sell: float
    estimated_gross_proceeds: float
    estimated_tax_liability: float
    estimated_net_proceeds: float
    effective_tax_rate: float
    etf_allocation: dict[str, float]   # ticker -> $ amount
    tax_lot_strategy: str              # e.g. "Sell 50 LTCG lots, 0 STCG lots"
    signal_rationale: str              # Human-readable explanation
    # v2 multi-benchmark fields (optional; populated when multi-benchmark WFO
    # results are available)
    benchmark_signals: dict[str, dict] | None = None   # ETF -> signal dict
    target_horizon: int = 6                            # months (6 or 12)


def generate_recommendation(
    vest_date: date,
    rsu_type: str,
    current_price: float,
    lots: list[TaxLot],
    wfo_result: WFOResult,
    portfolio_state: PortfolioState,
    sell_pct_override: float | None = None,
    multi_benchmark_results: dict[str, WFOResult] | None = None,
    target_horizon: int = 6,
) -> VestingRecommendation:
    """
    Generate a vesting recommendation for a single event.

    Args:
        vest_date:               The RSU vesting date.
        rsu_type:                "time" (January) or "performance" (July).
        current_price:           Current PGR market price per share.
        lots:                    All available tax lots.
        wfo_result:              Completed WFOResult from run_wfo().
        portfolio_state:         Current aggregate portfolio (PGR + ETF holdings).
        sell_pct_override:       If provided, overrides the signal-derived percentage.
        multi_benchmark_results: Optional dict from run_all_benchmarks() mapping
                                 ETF ticker → WFOResult.  When provided, populates
                                 ``benchmark_signals`` on the returned recommendation.
        target_horizon:          Forward return horizon in months (default 6).

    Returns:
        VestingRecommendation data structure.
    """
    # --- Signal extraction ---
    ic = wfo_result.information_coefficient
    hit_rate = wfo_result.hit_rate

    # Use last fold's out-of-sample mean as a directional proxy
    # (predict_current() provides a live refit; this path uses the WFO
    # result directly when a current observation is not yet available)
    last_fold = wfo_result.folds[-1]
    predicted_return = float(last_fold.y_hat.mean())

    # --- Derive sell percentage from signal ---
    if sell_pct_override is not None:
        recommended_sell_pct = float(sell_pct_override)
        rationale = f"Manual override: {recommended_sell_pct:.0%} sale."
    else:
        recommended_sell_pct, rationale = _compute_sell_pct(
            predicted_return, ic, hit_rate
        )

    # --- Tax calculation ---
    total_shares = sum(lot.shares_remaining for lot in lots if lot.shares_remaining > 0)
    shares_to_sell = total_shares * recommended_sell_pct

    if shares_to_sell > 0:
        sale = optimize_sale(lots, shares_to_sell, current_price, vest_date)
        gross = sale.total_gross
        tax = sale.total_tax
        net = sale.total_net
        eff_rate = sale.effective_tax_rate

        # Build lot strategy summary
        ltcg_count = sum(1 for r in sale.lots if r.holding_type == "LTCG")
        stcg_count = sum(1 for r in sale.lots if r.holding_type == "STCG")
        loss_count = sum(1 for r in sale.lots if r.holding_type == "LOSS")
        lot_strategy = (
            f"Sell from {ltcg_count} LTCG lot(s), "
            f"{stcg_count} STCG lot(s), {loss_count} loss lot(s)."
        )
    else:
        gross = tax = 0.0
        net = 0.0
        eff_rate = 0.0
        lot_strategy = "No shares sold."

    # --- Reallocation recommendation ---
    etf_allocation = recommend_reallocation(portfolio_state, net) if net > 0 else {}

    # --- Multi-benchmark signal summary (v2) ---
    benchmark_signals: dict[str, dict] | None = None
    if multi_benchmark_results is not None:
        benchmark_signals = {
            etf: {
                "ic":        res.information_coefficient,
                "hit_rate":  res.hit_rate,
                "n_folds":   len(res.folds),
                "benchmark": etf,
            }
            for etf, res in multi_benchmark_results.items()
        }

    return VestingRecommendation(
        vest_date=vest_date,
        rsu_type=rsu_type,
        predicted_6m_return=predicted_return,
        ic_confidence=ic,
        wfo_hit_rate=hit_rate,
        recommended_sell_pct=recommended_sell_pct,
        shares_to_sell=shares_to_sell,
        estimated_gross_proceeds=gross,
        estimated_tax_liability=tax,
        estimated_net_proceeds=net,
        effective_tax_rate=eff_rate,
        etf_allocation=etf_allocation,
        tax_lot_strategy=lot_strategy,
        signal_rationale=rationale,
        benchmark_signals=benchmark_signals,
        target_horizon=target_horizon,
    )


def _compute_sell_pct(
    predicted_return: float,
    ic: float,
    hit_rate: float,
) -> tuple[float, str]:
    """
    Derive a recommended sale percentage from the WFO signal.

    Logic:
      - If the model predicts a significantly positive 6M return (>15%) AND
        IC is positive: suggest holding (sell 0–25%).
      - If the model predicts a near-flat or negative return OR IC is low:
        suggest selling (50–100%).
      - IC < 0.05 (model is not informative): default to 50% sell.
    """
    if ic < 0.05:
        return 0.50, (
            f"WFO Information Coefficient ({ic:.3f}) is below the confidence "
            "threshold of 0.05 — model not sufficiently predictive. "
            "Defaulting to 50% diversification sale."
        )

    if predicted_return > 0.15 and ic > 0.10:
        pct = 0.25
        rationale = (
            f"Model predicts +{predicted_return:.1%} 6-month return "
            f"(IC={ic:.3f}, Hit Rate={hit_rate:.1%}). "
            f"Holding majority; recommending {pct:.0%} sale for diversification."
        )
    elif predicted_return > 0.05:
        pct = 0.50
        rationale = (
            f"Model predicts modest +{predicted_return:.1%} 6-month return "
            f"(IC={ic:.3f}). Recommending balanced {pct:.0%} sale."
        )
    else:
        pct = 1.00
        rationale = (
            f"Model predicts {predicted_return:.1%} 6-month return "
            f"(IC={ic:.3f}). Recommending full {pct:.0%} sale for diversification."
        )

    return pct, rationale


def print_recommendation(rec: VestingRecommendation) -> None:
    """Print a formatted recommendation to stdout."""
    separator = "=" * 70
    print(separator)
    print(f"  PGR VESTING RECOMMENDATION — {rec.vest_date} ({rec.rsu_type.upper()} RSU)")
    print(separator)
    print(f"  Predicted 6M Return  : {rec.predicted_6m_return:+.2%}")
    print(f"  Model IC             : {rec.ic_confidence:.4f}")
    print(f"  WFO Hit Rate         : {rec.wfo_hit_rate:.1%}")
    print(f"  Recommended Sale     : {rec.recommended_sell_pct:.0%} ({rec.shares_to_sell:.1f} shares)")
    print(f"  Gross Proceeds       : ${rec.estimated_gross_proceeds:,.2f}")
    print(f"  Tax Liability        : ${rec.estimated_tax_liability:,.2f} "
          f"({rec.effective_tax_rate:.1%} effective rate)")
    print(f"  Net Proceeds         : ${rec.estimated_net_proceeds:,.2f}")
    print(f"  Lot Strategy         : {rec.tax_lot_strategy}")
    print(f"  Rationale            : {rec.signal_rationale}")
    if rec.etf_allocation:
        print("  Recommended Reallocation:")
        for ticker, amount in sorted(rec.etf_allocation.items()):
            print(f"    {ticker:<10} ${amount:>12,.2f}")
    print(separator)
