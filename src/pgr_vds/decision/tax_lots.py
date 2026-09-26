"""Tax context and position lots for the monthly decision run.

The next-vest date, the STCG/LTCG breakeven section, the provisional
three-scenario vest view with its Monte Carlo, and the tax-bucketed guidance
for shares already held (``data/processed/position_lots.csv``).
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path

import config
from src.database import db_client
from src.portfolio.redeploy_buckets import summarize_existing_holdings_actions
from src.tax.capital_gains import compute_three_scenarios, load_position_lots
from src.tax.monte_carlo import (
    MonteCarloTaxAnalysis,
    estimate_annual_vol_weekly,
    run_monte_carlo_tax_analysis,
)

logger = logging.getLogger(__name__)


def get_next_vest_info(as_of: date) -> tuple[date, str]:
    """Return the next vest date and RSU type after the as-of date."""
    candidates = [
        (date(as_of.year, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY), "time"),
        (date(as_of.year, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY), "performance"),
        (date(as_of.year + 1, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY), "time"),
        (date(as_of.year + 1, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY), "performance"),
    ]
    future = sorted((d, rsu_type) for d, rsu_type in candidates if d >= as_of)
    return future[0]


def build_tax_context_lines(
    predicted_6m_return: float,
    prob_outperform: float,
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
    as_of: date | None = None,
) -> list[str]:
    """Build the ## Tax Context section for recommendation.md.

    Shows the absolute PGR return at which holding a lot to LTCG ties selling
    it now at STCG, ``-g * (S - L) / (1 - L)`` for a lot whose gain is ``g``
    of its price (``compute_stcg_ltcg_breakeven``). No lot data is needed.

    The model forecast is PGR's return *relative to the benchmarks*. It is
    reported for context only: it is not a PGR price forecast, so it is not
    compared with the breakeven, and a negative value does not imply a loss
    on the shares (review 2026-09-25, F19).

    Args:
        predicted_6m_return: Ensemble mean 6M relative return prediction.
        prob_outperform:      Mean P(outperform) across benchmarks.
        stcg_rate:            STCG rate override (default: config.STCG_RATE).
        ltcg_rate:            LTCG rate override (default: config.LTCG_RATE).
        as_of:                Reporting as-of date for next-vest calculations.

    Returns:
        List of markdown lines (without leading blank line separator).
    """
    from src.tax.capital_gains import compute_stcg_ltcg_breakeven

    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    if as_of is None:
        as_of = date.today()

    breakeven_full = compute_stcg_ltcg_breakeven(stcg_rate, ltcg_rate, gain_fraction=1.0)
    breakeven_half = compute_stcg_ltcg_breakeven(stcg_rate, ltcg_rate, gain_fraction=0.5)
    tax_differential = stcg_rate - ltcg_rate

    verdict = (
        f"**Holding a vested lot to its LTCG date gives more after-tax cash than selling it now "
        f"at STCG unless PGR's own price falls by more than {abs(breakeven_full):.2%} "
        f"(a lot that is all gain) or {abs(breakeven_half):.2%} (a lot whose gain is half its "
        f"price) before that date.** At the vest itself the shares carry no gain, so the tax rate "
        f"does not change the proceeds of an immediate sale."
    )
    if predicted_6m_return < 0:
        forecast_note = (
            f"The model expects PGR to lag the benchmarks by {abs(predicted_6m_return):.1%} over "
            f"6 months. That is a relative forecast, not a PGR price forecast: it is an argument "
            f"for diversifying, not a tax loss. A lot has a harvestable loss only when PGR trades "
            f"below its cost basis, and selling it at a loss within 30 days of a vest is a wash "
            f"sale (the loss is disallowed)."
        )
    else:
        forecast_note = (
            f"The model expects PGR to beat the benchmarks by {predicted_6m_return:.1%} over "
            f"6 months. That is a relative forecast, not a PGR price forecast, so it is not "
            f"compared with the breakeven above."
        )

    # Next vest dates from config
    this_year = as_of.year
    next_time_vest = date(this_year, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY)
    next_perf_vest = date(this_year, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY)
    if next_time_vest < as_of:
        next_time_vest = date(this_year + 1, config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY)
    if next_perf_vest < as_of:
        next_perf_vest = date(this_year + 1, config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY)

    lines = [
        "",
        "---",
        "",
        "## Tax Context",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| STCG Rate (federal) | {stcg_rate:.0%} |",
        f"| LTCG Rate (federal) | {ltcg_rate:.0%} |",
        f"| Tax-rate differential | {tax_differential:.0%} |",
        f"| **LTCG breakeven PGR return (lot all gain)** | **{breakeven_full:+.2%}** |",
        f"| LTCG breakeven PGR return (gain = half the price) | {breakeven_half:+.2%} |",
        f"| Model forecast, PGR vs benchmarks (6M, relative) | {predicted_6m_return:+.2%} |",
        f"| P(outperform) | {prob_outperform:.1%} |",
        f"| Next time-based vest | {next_time_vest} |",
        f"| Next performance vest | {next_perf_vest} |",
        "",
        verdict,
        "",
        forecast_note,
        "",
        "> **Breakeven formula:** `-g × (STCG − LTCG) / (1 − LTCG)`, where `g` is the",
        "> lot's unrealised gain as a fraction of the current price. Below this absolute",
        "> PGR return, selling now at STCG beats holding to the LTCG date (the day after",
        "> the one-year anniversary of the vest); above it, holding wins. It compares",
        "> cash at the LTCG date and ignores what sale proceeds would earn meanwhile.",
    ]

    return lines


def build_provisional_vest_scenario(
    conn,
    as_of: date,
    mean_predicted: float,
    prob_outperform: float,
) -> dict | None:
    """Build a provisional three-scenario view for the next vest using current lots.

    Only lots vested by ``as_of`` are held. ``mean_predicted`` is the relative
    forecast and is not used as a PGR price return (review 2026-09-25, F19);
    the scenarios and the Monte Carlo use ``config.TAX_SCENARIO_PGR_ANNUAL_RETURN``.
    """
    del mean_predicted
    lots_path = Path("data/processed/position_lots.csv")
    if not lots_path.exists():
        return None

    lots = load_position_lots(str(lots_path), as_of=as_of)
    if not lots:
        return None

    prices = db_client.get_prices(conn, "PGR", end_date=str(as_of))
    if prices.empty:
        return None

    current_price = float(prices["close"].iloc[-1])
    total_shares = sum(lot.shares_remaining for lot in lots if lot.shares_remaining and lot.shares_remaining > 0)
    if total_shares <= 0:
        return None

    avg_basis = sum(
        lot.shares_remaining * lot.cost_basis_per_share
        for lot in lots
        if lot.shares_remaining and lot.shares_remaining > 0
    ) / total_shares
    vest_date, rsu_type = get_next_vest_info(as_of)
    # The scenarios need PGR's absolute price return. The model forecasts the
    # return relative to benchmarks (mean_predicted), which is not a price
    # forecast, so the configured absolute assumption is used (F19).
    annual_return = config.TAX_SCENARIO_PGR_ANNUAL_RETURN
    scenario = compute_three_scenarios(
        vest_date=vest_date,
        rsu_type=rsu_type,
        shares=total_shares,
        cost_basis_per_share=avg_basis,
        current_price=current_price,
        predicted_6m_return=(1.0 + annual_return) ** 0.5 - 1.0,
        predicted_12m_return=annual_return,
        prob_outperform_6m=prob_outperform,
        prob_outperform_12m=prob_outperform,
    )

    # v35: Monte Carlo tax-sensitivity analysis
    mc_analysis: MonteCarloTaxAnalysis | None = None
    try:
        close_prices = prices["close"].dropna()
        if len(close_prices) >= 30:
            # Recent split-adjusted weekly returns x sqrt(52) (review F19).
            annual_vol = estimate_annual_vol_weekly(
                close_prices, db_client.get_splits(conn, "PGR")
            )
            # Absolute drift assumption, not the relative forecast (F19).
            annual_drift = math.log1p(annual_return)
            mc_analysis = run_monte_carlo_tax_analysis(
                current_price=current_price,
                cost_basis_per_share=avg_basis,
                shares=total_shares,
                annual_vol=annual_vol,
                annual_drift=annual_drift,
            )
    except Exception:
        logger.debug("Monte Carlo tax analysis skipped", exc_info=True)

    return {
        "vest_date": vest_date,
        "rsu_type": rsu_type,
        "current_price": current_price,
        "avg_basis": avg_basis,
        "shares": total_shares,
        "scenario": scenario,
        "mc_analysis": mc_analysis,
    }


def build_existing_holdings_guidance(conn, as_of: date) -> list[dict[str, object]]:
    """Build tax-bucketed guidance for already-held PGR shares."""
    lots_path = Path("data/processed/position_lots.csv")
    if not lots_path.exists():
        return []
    lots = load_position_lots(str(lots_path), as_of=as_of)
    if not lots:
        return []
    prices = db_client.get_prices(conn, "PGR", end_date=str(as_of))
    if prices.empty:
        return []
    current_price = float(prices["close"].iloc[-1])
    return [
        {
            "vest_date": action.vest_date,
            "shares": action.shares,
            "cost_basis_per_share": action.cost_basis_per_share,
            "tax_bucket": action.tax_bucket,
            "unrealized_gain": action.unrealized_gain,
            "unrealized_return": action.unrealized_return,
            "rationale": action.rationale,
        }
        for action in summarize_existing_holdings_actions(
            lots,
            current_price=current_price,
            sell_date=as_of,
        )
    ]
