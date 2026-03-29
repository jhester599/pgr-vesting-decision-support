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
    # v3.1 Kelly sizing fields (optional; populated when BayesianRidge uncertainty
    # is available from the ensemble runner)
    prediction_std: float = 0.0          # BayesianRidge posterior std (0 if unavailable)
    kelly_fraction_used: float = 0.0     # Effective Kelly fraction applied
    # v4.4 STCG boundary guard (None when no lots are in the 6–12M zone, or
    # when predicted alpha exceeds the breakeven threshold)
    stcg_warning: str | None = None


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

    # --- v4.4: STCG boundary guard ---
    stcg_warning = _check_stcg_boundary(lots, vest_date, predicted_return, current_price)

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
        stcg_warning=stcg_warning,
    )


def _check_stcg_boundary(
    lots: list[TaxLot],
    sell_date: date,
    predicted_alpha: float,
    current_price: float,
    breakeven_threshold: float | None = None,
    zone_min_days: int | None = None,
    zone_max_days: int | None = None,
) -> str | None:
    """
    Warn when lots in the 6–12 month STCG zone would be better held for LTCG.

    Identifies lots held between ``zone_min_days`` and ``zone_max_days`` at the
    time of ``sell_date``.  These lots are in the "STCG boundary zone": they
    have incurred short-term capital gains treatment but are close enough to the
    365-day LTCG threshold that the tax savings from waiting may exceed the
    expected alpha from diversifying now.

    The breakeven logic:
        Selling STCG vs. LTCG costs ~17–22pp in effective tax rate for most
        high-income earners (37% ordinary − 20% LTCG = 17pp; add 3.8% NIIT
        for an upper bound near 21pp; mid-point ≈ 18pp = ``STCG_BREAKEVEN_THRESHOLD``).
        If ``predicted_alpha < breakeven_threshold``, the warning fires.

    Args:
        lots:                 All available tax lots.
        sell_date:            Proposed sale date (typically the vesting date).
        predicted_alpha:      Model's predicted 6M PGR excess return.
        current_price:        Current PGR share price (for gain estimation).
        breakeven_threshold:  Override for config.STCG_BREAKEVEN_THRESHOLD.
        zone_min_days:        Override for config.STCG_ZONE_MIN_DAYS (default 180).
        zone_max_days:        Override for config.STCG_ZONE_MAX_DAYS (default 365).

    Returns:
        A human-readable warning string if boundary lots exist AND
        ``predicted_alpha < breakeven_threshold``; otherwise ``None``.
    """
    if breakeven_threshold is None:
        breakeven_threshold = config.STCG_BREAKEVEN_THRESHOLD
    if zone_min_days is None:
        zone_min_days = config.STCG_ZONE_MIN_DAYS
    if zone_max_days is None:
        zone_max_days = config.STCG_ZONE_MAX_DAYS

    boundary_lots = []
    for lot in lots:
        if lot.shares_remaining is None or lot.shares_remaining <= 0:
            continue
        holding_days = (sell_date - lot.vest_date).days
        if zone_min_days < holding_days <= zone_max_days:
            days_to_ltcg = zone_max_days - holding_days
            gain_per_share = current_price - lot.cost_basis_per_share
            boundary_lots.append({
                "lot": lot,
                "holding_days": holding_days,
                "days_to_ltcg": days_to_ltcg,
                "gain_per_share": gain_per_share,
                "embedded_gain": lot.shares_remaining * gain_per_share,
            })

    if not boundary_lots:
        return None

    if predicted_alpha >= breakeven_threshold:
        return None

    # Build warning message
    total_boundary_shares = sum(
        info["lot"].shares_remaining for info in boundary_lots
    )
    closest = min(boundary_lots, key=lambda x: x["days_to_ltcg"])
    days_to_ltcg_min = closest["days_to_ltcg"]
    total_embedded_gain = sum(info["embedded_gain"] for info in boundary_lots)

    lot_lines = []
    for info in sorted(boundary_lots, key=lambda x: x["days_to_ltcg"]):
        lot = info["lot"]
        lot_lines.append(
            f"  • {lot.vest_date} lot: {lot.shares_remaining:.1f} shares, "
            f"{info['holding_days']}d held, "
            f"{info['days_to_ltcg']}d to LTCG, "
            f"gain/share ${info['gain_per_share']:+.2f}"
        )

    warning = (
        f"⚠️  STCG BOUNDARY WARNING: {len(boundary_lots)} lot(s) "
        f"({total_boundary_shares:.1f} shares, "
        f"total embedded gain ${total_embedded_gain:+,.2f}) "
        f"are in the 6–12 month STCG zone. "
        f"The nearest lot qualifies for LTCG in {days_to_ltcg_min} day(s).\n"
        f"  Predicted alpha ({predicted_alpha:+.1%}) is below the STCG-to-LTCG "
        f"breakeven threshold ({breakeven_threshold:.0%}). "
        f"Consider delaying sale of boundary lots until LTCG qualification.\n"
        + "\n".join(lot_lines)
    )
    return warning


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


def _compute_sell_pct_kelly(
    predicted_excess_return: float,
    prediction_variance: float,
    kelly_fraction: float | None = None,
    max_position: float | None = None,
) -> tuple[float, float, str]:
    """
    Compute recommended sell percentage using fractional Kelly criterion.

    The Kelly formula sizes the optimal position as:
        f* = kelly_fraction × predicted_excess_return / prediction_variance

    A high signal (positive predicted return) → larger position in PGR
    → sell LESS.  A high uncertainty (large variance) → smaller position
    → sell MORE.

    The sell percentage is ``1 - f*`` (clipped to [0, 1]).

    Args:
        predicted_excess_return: Model's predicted PGR excess return (can be
                                 negative).
        prediction_variance:     BayesianRidge posterior variance (std²).
                                 Must be positive; falls back to legacy logic
                                 if zero.
        kelly_fraction:          Kelly multiplier (< 1 for fractional Kelly).
                                 Defaults to ``config.KELLY_FRACTION`` (0.25).
        max_position:            Maximum single-stock allocation fraction.
                                 Defaults to ``config.KELLY_MAX_POSITION`` (0.30).

    Returns:
        Tuple of (sell_pct, kelly_fraction_used, rationale_str).
    """
    if kelly_fraction is None:
        kelly_fraction = config.KELLY_FRACTION
    if max_position is None:
        max_position = config.KELLY_MAX_POSITION

    if prediction_variance <= 0.0:
        # No uncertainty estimate — fall back to legacy logic
        sell_pct, rationale = _compute_sell_pct(
            predicted_excess_return, ic=0.06, hit_rate=0.5
        )
        return sell_pct, 0.0, rationale + " [Kelly fallback: no variance estimate]"

    raw_kelly = kelly_fraction * predicted_excess_return / prediction_variance
    position_fraction = min(max(raw_kelly, 0.0), max_position)
    sell_pct = 1.0 - position_fraction
    sell_pct = min(max(sell_pct, 0.0), 1.0)

    rationale = (
        f"Fractional Kelly sizing: predicted excess return {predicted_excess_return:+.2%}, "
        f"variance {prediction_variance:.4f}, raw f* = {raw_kelly:.3f}, "
        f"capped at {max_position:.0%} max position → hold {position_fraction:.0%}, "
        f"sell {sell_pct:.0%}. "
        f"(Kelly fraction = {kelly_fraction}, max position = {max_position:.0%})"
    )
    return sell_pct, kelly_fraction, rationale


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
    if rec.stcg_warning:
        print()
        print(rec.stcg_warning)
    print(separator)


# ---------------------------------------------------------------------------
# v4.0 — Per-Benchmark Weighting
# ---------------------------------------------------------------------------

def compute_benchmark_weights(
    monthly_stability_results: list,
    window_months: int = 24,
    min_ic: float = 0.0,
) -> dict[str, float]:
    """
    Compute reliability weights for each ETF benchmark based on rolling IC.

    Benchmarks are weighted by their recent predictive skill:
        raw_weight = rolling_ic × rolling_hit_rate

    Benchmarks with mean IC ≤ ``min_ic`` over the window receive zero weight.
    Weights are normalised to sum to 1.0 across active benchmarks.

    This allows the final recommendation to favour benchmarks where the WFO
    model has demonstrated consistent predictive power, rather than equal-
    weighting all 20 ETFs regardless of model quality.

    Args:
        monthly_stability_results: List of BacktestEventResult from
                                   ``run_monthly_stability_backtest()``.
        window_months:             Rolling look-back window (default 24).
        min_ic:                    Minimum mean IC over the window for a
                                   benchmark to receive non-zero weight.

    Returns:
        Dict mapping ETF ticker → float weight.  Weights sum to 1.0.
        Returns equal weights if no benchmark exceeds ``min_ic``.
    """
    import numpy as np

    if not monthly_stability_results:
        return {}

    # Aggregate IC and hit_rate by benchmark
    from collections import defaultdict
    data: dict[str, list] = defaultdict(list)
    for r in monthly_stability_results:
        data[r.benchmark].append({
            "ic":       r.ic_at_event,
            "correct":  float(r.correct_direction),
        })

    weights_raw: dict[str, float] = {}
    for benchmark, rows in data.items():
        # Take the most recent window_months observations
        recent = rows[-window_months:]
        ics = [r["ic"] for r in recent]
        hits = [r["correct"] for r in recent]
        mean_ic = float(np.mean(ics)) if ics else 0.0
        mean_hit = float(np.mean(hits)) if hits else 0.0

        if mean_ic <= min_ic:
            weights_raw[benchmark] = 0.0
        else:
            weights_raw[benchmark] = mean_ic * mean_hit

    total = sum(weights_raw.values())
    if total <= 0.0:
        # All benchmarks below threshold — return equal weights
        n = len(weights_raw)
        return {k: 1.0 / n for k in weights_raw} if n > 0 else {}

    return {k: v / total for k, v in weights_raw.items()}
