"""
Tax lot optimizer for taxable brokerage account positions.

NUA is out of scope — stock is held in a taxable account, not a 401k/ESOP.

Strategy: minimize total tax liability on a given sale amount by selecting
the optimal combination of tax lots using the following priority rules:

  1. Sell lots with embedded losses first (tax-loss harvesting, reduces
     ordinary income or offsets other capital gains).
  2. Among gain lots, sell LTCG-eligible lots before STCG lots (LTCG rate
     is materially lower than ordinary income rates).
  3. Among LTCG gain lots, sell highest cost-basis lots first (minimize
     the taxable gain on each lot, known as "specific identification").

Tax rates are configured in config.py and can be overridden via .env
to reflect state tax additions (e.g., CA adds 13.3% for a combined
LTCG rate of ~33.3% for high earners).

All lot-level calculations use exact per-share arithmetic to avoid
rounding errors that accumulate over multi-year sale programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaxLot:
    """A single RSU vesting tranche or acquired lot."""
    vest_date: date
    rsu_type: str          # "time" or "performance"
    shares: float          # Total shares in this lot
    cost_basis_per_share: float   # Fair market value on vest date
    shares_remaining: float | None = None  # Tracks partial sales; defaults to shares

    def __post_init__(self) -> None:
        if self.shares_remaining is None:
            self.shares_remaining = self.shares

    @property
    def total_cost_basis(self) -> float:
        return self.shares_remaining * self.cost_basis_per_share

    def is_ltcg_eligible(self, sell_date: date) -> bool:
        """Return True if this lot has been held > 365 days from vest_date."""
        return (sell_date - self.vest_date).days > 365

    def embedded_gain(self, current_price: float) -> float:
        """Current unrealized gain (negative = loss)."""
        return self.shares_remaining * (current_price - self.cost_basis_per_share)


@dataclass
class SaleResult:
    """Outcome of a single lot's contribution to a sale."""
    lot: TaxLot
    shares_sold: float
    sale_price: float
    gross_proceeds: float
    cost_basis_used: float
    taxable_gain: float
    tax_rate: float
    tax_liability: float
    net_proceeds: float
    holding_type: str      # "LTCG", "STCG", or "LOSS"


@dataclass
class TotalSaleResult:
    """Aggregated result of a full sale transaction across multiple lots."""
    lots: list[SaleResult]
    total_gross: float
    total_tax: float
    total_net: float
    effective_tax_rate: float


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

def optimize_sale(
    lots: list[TaxLot],
    shares_to_sell: float,
    sale_price: float,
    sell_date: date,
    ltcg_rate: float | None = None,
    stcg_rate: float | None = None,
) -> TotalSaleResult:
    """
    Select the optimal tax lots to minimize total tax liability on a sale.

    Implements the three-priority lot selection strategy:
      1. Loss lots first (tax-loss harvesting)
      2. LTCG gain lots before STCG gain lots
      3. Highest basis first within each LTCG bucket

    Args:
        lots:            All available tax lots (with shares_remaining > 0).
        shares_to_sell:  Number of shares to liquidate in this transaction.
        sale_price:      Per-share sale price (unadjusted market price).
        sell_date:       Date of the sale transaction.
        ltcg_rate:       Override for long-term capital gains tax rate.
                         Defaults to config.LTCG_RATE.
        stcg_rate:       Override for short-term capital gains tax rate.
                         Defaults to config.STCG_RATE.

    Returns:
        TotalSaleResult with per-lot breakdown and aggregate tax metrics.

    Raises:
        ValueError: If shares_to_sell exceeds total shares available.
    """
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE

    total_available = sum(lot.shares_remaining for lot in lots if lot.shares_remaining > 0)
    if shares_to_sell > total_available + 1e-9:
        raise ValueError(
            f"Requested {shares_to_sell:.2f} shares to sell but only "
            f"{total_available:.2f} available across all lots."
        )

    # Classify and sort lots by priority
    loss_lots: list[TaxLot] = []
    ltcg_gain_lots: list[TaxLot] = []
    stcg_lots: list[TaxLot] = []

    for lot in lots:
        if lot.shares_remaining <= 0:
            continue
        if lot.embedded_gain(sale_price) < 0:
            loss_lots.append(lot)
        elif lot.is_ltcg_eligible(sell_date):
            ltcg_gain_lots.append(lot)
        else:
            stcg_lots.append(lot)

    # Sort: loss lots (largest loss first), LTCG gain lots (highest basis first),
    # STCG lots (smallest gain first to defer tax)
    loss_lots.sort(key=lambda lot: lot.embedded_gain(sale_price))       # most negative first
    ltcg_gain_lots.sort(key=lambda lot: lot.cost_basis_per_share, reverse=True)
    stcg_lots.sort(key=lambda lot: lot.embedded_gain(sale_price))       # smallest gain first

    ordered_lots = loss_lots + ltcg_gain_lots + stcg_lots

    sale_results: list[SaleResult] = []
    shares_remaining_to_sell = shares_to_sell

    for lot in ordered_lots:
        if shares_remaining_to_sell <= 1e-9:
            break

        shares_from_lot = min(lot.shares_remaining, shares_remaining_to_sell)
        gross = shares_from_lot * sale_price
        basis = shares_from_lot * lot.cost_basis_per_share
        gain = gross - basis

        if gain < 0:
            # Loss lot: tax benefit (negative liability)
            rate = ltcg_rate if lot.is_ltcg_eligible(sell_date) else stcg_rate
            tax = gain * rate  # negative number = tax benefit
            holding_type = "LOSS"
        elif lot.is_ltcg_eligible(sell_date):
            rate = ltcg_rate
            tax = gain * rate
            holding_type = "LTCG"
        else:
            rate = stcg_rate
            tax = gain * rate
            holding_type = "STCG"

        sale_results.append(SaleResult(
            lot=lot,
            shares_sold=shares_from_lot,
            sale_price=sale_price,
            gross_proceeds=gross,
            cost_basis_used=basis,
            taxable_gain=gain,
            tax_rate=rate,
            tax_liability=tax,
            net_proceeds=gross - tax,
            holding_type=holding_type,
        ))
        shares_remaining_to_sell -= shares_from_lot

    total_gross = sum(r.gross_proceeds for r in sale_results)
    total_tax = sum(r.tax_liability for r in sale_results)
    total_net = total_gross - total_tax
    effective_rate = total_tax / total_gross if total_gross > 0 else 0.0

    return TotalSaleResult(
        lots=sale_results,
        total_gross=total_gross,
        total_tax=total_tax,
        total_net=total_net,
        effective_tax_rate=effective_rate,
    )


# ---------------------------------------------------------------------------
# Position loader
# ---------------------------------------------------------------------------

def load_position_lots(csv_path: str) -> list[TaxLot]:
    """
    Load tax lots from the user-provided position_lots.csv.

    Expected columns: vest_date, rsu_type, shares, cost_basis_per_share.
    The file is gitignored as it contains personal financial data.

    Args:
        csv_path: Path to position_lots.csv.

    Returns:
        List of TaxLot objects sorted ascending by vest_date.
    """
    df = pd.read_csv(csv_path)
    df["vest_date"] = pd.to_datetime(df["vest_date"]).dt.date

    lots = []
    for _, row in df.iterrows():
        lot = TaxLot(
            vest_date=row["vest_date"],
            rsu_type=str(row.get("rsu_type", "time")),
            shares=float(row["shares"]),
            cost_basis_per_share=float(row["cost_basis_per_share"]),
        )
        lots.append(lot)

    lots.sort(key=lambda x: x.vest_date)
    return lots


def compute_position_summary(
    lots: list[TaxLot],
    current_price: float,
    sell_date: date,
) -> dict:
    """
    Return a summary of the current position: total shares, total cost basis,
    unrealized gain, and LTCG-eligible vs. STCG breakdown.

    Args:
        lots:          List of TaxLot objects.
        current_price: Current market price per share.
        sell_date:     Date used to evaluate LTCG eligibility.

    Returns:
        Dict with position summary statistics.
    """
    total_shares = sum(lot.shares_remaining for lot in lots)
    total_basis = sum(lot.total_cost_basis for lot in lots)
    total_value = total_shares * current_price
    total_gain = total_value - total_basis

    ltcg_shares = sum(
        lot.shares_remaining for lot in lots if lot.is_ltcg_eligible(sell_date)
    )
    stcg_shares = total_shares - ltcg_shares

    return {
        "total_shares": total_shares,
        "total_cost_basis": total_basis,
        "current_value": total_value,
        "total_unrealized_gain": total_gain,
        "unrealized_gain_pct": total_gain / total_basis if total_basis > 0 else 0.0,
        "ltcg_eligible_shares": ltcg_shares,
        "stcg_shares": stcg_shares,
        "ltcg_pct_of_position": ltcg_shares / total_shares if total_shares > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# v4.0 Tax-Loss Harvesting
# ---------------------------------------------------------------------------

def identify_tlh_candidates(
    lots: list[TaxLot],
    current_price: float,
    loss_threshold: float | None = None,
) -> list[TaxLot]:
    """
    Identify tax lots eligible for tax-loss harvesting.

    A lot is a TLH candidate when its unrealized return falls below
    ``loss_threshold`` (default: config.TLH_LOSS_THRESHOLD = -10%).

    Harvesting a loss lets it offset capital gains elsewhere (or up to
    $3,000/year of ordinary income), reducing the after-tax cost of
    diversification.

    Args:
        lots:           List of TaxLot objects (current position).
        current_price:  Current price per share.
        loss_threshold: Unrealized return below which harvest is triggered.
                        Default: config.TLH_LOSS_THRESHOLD (-0.10).

    Returns:
        List of TaxLot objects with unrealized return < loss_threshold,
        sorted by unrealized return ascending (largest loss first).
    """
    if loss_threshold is None:
        loss_threshold = config.TLH_LOSS_THRESHOLD

    candidates: list[TaxLot] = []
    for lot in lots:
        if lot.shares_remaining is None or lot.shares_remaining <= 0:
            continue
        unrealized_return = (
            (current_price - lot.cost_basis_per_share) / lot.cost_basis_per_share
        )
        if unrealized_return < loss_threshold:
            candidates.append(lot)

    return sorted(
        candidates,
        key=lambda lot: (current_price - lot.cost_basis_per_share) / lot.cost_basis_per_share,
    )


def compute_after_tax_expected_return(
    predicted_return: float,
    unrealized_gain_fraction: float,
    tax_rate: float | None = None,
) -> float:
    """
    Compute the after-tax expected return, accounting for embedded capital gains.

    After-tax return = predicted_return − max(0, unrealized_gain_fraction × tax_rate)

    This creates a natural asymmetry:
      - Lots with embedded losses increase the effective sell incentive (no tax drag).
      - Lots with large embedded gains raise the hurdle rate (tax drag must be overcome).

    Args:
        predicted_return:        Model's predicted 6M forward return.
        unrealized_gain_fraction: Embedded gain as a fraction of cost basis
                                  (positive = gain, negative = loss).
        tax_rate:                Applicable capital gains tax rate.
                                 Defaults to config.LTCG_RATE.

    Returns:
        After-tax expected return as a float.
    """
    if tax_rate is None:
        tax_rate = config.LTCG_RATE
    tax_drag = max(0.0, unrealized_gain_fraction * tax_rate)
    return predicted_return - tax_drag


def suggest_tlh_replacement(harvested_ticker: str) -> str | None:
    """
    Suggest a correlated-but-not-substantially-identical replacement ETF.

    Uses ``config.TLH_REPLACEMENT_MAP`` to find the replacement.  The IRS
    wash-sale rule prohibits repurchasing a "substantially identical" security
    within 30 days before or after the sale.  ETF pairs in the map are chosen
    to be correlated (tracking similar indices) but legally distinct (different
    index methodologies or providers).

    Wash-sale compliance: wait at least ``config.TLH_WASH_SALE_DAYS`` (31) days
    before re-purchasing the original ticker.

    Args:
        harvested_ticker: The ticker being sold for a tax loss.

    Returns:
        Replacement ticker from TLH_REPLACEMENT_MAP, or None if no replacement
        is defined (e.g., for individual stocks like PGR).
    """
    return config.TLH_REPLACEMENT_MAP.get(harvested_ticker)


def wash_sale_clear_date(harvest_date: "date", wash_sale_days: int | None = None) -> "date":
    """
    Return the earliest date on which the original security can be repurchased.

    Per IRS rules, this is 31 days after the sale date.

    Args:
        harvest_date:    Date of the tax-loss sale.
        wash_sale_days:  Days to wait (default: config.TLH_WASH_SALE_DAYS = 31).

    Returns:
        First safe repurchase date.
    """
    if wash_sale_days is None:
        wash_sale_days = config.TLH_WASH_SALE_DAYS
    return harvest_date + timedelta(days=wash_sale_days)


# ---------------------------------------------------------------------------
# v7.1 — Three-Scenario Tax Framework
# ---------------------------------------------------------------------------

@dataclass
class TaxScenario:
    """One of three tax scenarios for a vesting event."""
    label: str                       # "SELL_NOW_STCG", "HOLD_TO_LTCG", "HOLD_FOR_LOSS"
    sell_date: date                  # When the sale would occur
    tax_rate: float                  # Applicable tax rate (STCG or LTCG)
    holding_period_days: int         # Days from vest_date to sell_date
    predicted_return: float          # Model's predicted return over the holding period
    predicted_price: float           # current_price * (1 + predicted_return)
    gross_proceeds: float            # shares * predicted_price
    tax_liability: float             # (predicted_price - cost_basis) * shares * tax_rate
    net_proceeds: float              # gross_proceeds - tax_liability
    breakeven_return: float          # Minimum return needed to prefer this scenario
    probability: float               # Model-assigned probability (0.0–1.0)
    rationale: str                   # Human-readable explanation


@dataclass
class ThreeScenarioResult:
    """Complete three-scenario analysis for a vesting event."""
    vest_date: date
    rsu_type: str
    current_price: float
    cost_basis_per_share: float
    shares: float
    scenarios: list[TaxScenario]     # Always exactly 3 scenarios
    recommended_scenario: str        # Label of the highest-utility scenario
    stcg_ltcg_breakeven: float       # Return needed for LTCG to beat STCG
    days_to_ltcg: int                # Days from vest_date until LTCG-eligible


def compute_stcg_ltcg_breakeven(
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
) -> float:
    """
    Compute the minimum return needed for holding to LTCG to beat selling
    immediately at STCG.

    The breakeven is the return R such that:
        (1+R) * (1 - ltcg_rate) = (1 - stcg_rate) + R
    Simplified: R_breakeven ≈ (stcg_rate - ltcg_rate) / (1 - ltcg_rate)

    With default rates (37% STCG, 20% LTCG): ~21.25%.

    Args:
        stcg_rate: Short-term capital gains tax rate. Default: config.STCG_RATE.
        ltcg_rate: Long-term capital gains tax rate. Default: config.LTCG_RATE.

    Returns:
        Breakeven return as a decimal (e.g., 0.2125 for 21.25%).
    """
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    return (stcg_rate - ltcg_rate) / (1.0 - ltcg_rate)


def compute_three_scenarios(
    vest_date: date,
    rsu_type: str,
    shares: float,
    cost_basis_per_share: float,
    current_price: float,
    predicted_6m_return: float,
    predicted_12m_return: float,
    prob_outperform_6m: float,
    prob_outperform_12m: float,
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
) -> ThreeScenarioResult:
    """
    Compute three tax-aware scenarios for a vesting event.

    Scenario A — Sell at Vest (STCG):
        Sell immediately on vest_date.  Tax rate = STCG.
        Proceeds are certain (no holding period risk).

    Scenario B — Hold to LTCG Eligibility:
        Hold for 366 days post-vest, then sell at LTCG rate.
        Uses predicted_12m_return.  Tax rate = LTCG.  Proceeds are uncertain.

    Scenario C — Hold for Capital Loss:
        Relevant when predicted return is negative.  Harvest the loss to
        offset other capital gains at the higher STCG rate plus up to
        $3,000/yr of ordinary income.  Tax benefit = |loss| × STCG rate.

    Args:
        vest_date:             Date shares vest (holding period starts).
        rsu_type:              "time" or "performance".
        shares:                Number of shares in this lot.
        cost_basis_per_share:  FMV on vest date (= cost basis for RSUs).
        current_price:         Current market price per share.
        predicted_6m_return:   Model's predicted 6-month PGR relative return.
        predicted_12m_return:  Model's predicted 12-month PGR return.
                               If unavailable, pass predicted_6m_return * 2
                               as a rough annualization.
        prob_outperform_6m:    Calibrated P(PGR outperforms) at 6M.
        prob_outperform_12m:   Calibrated P(PGR outperforms) at 12M.
                               If unavailable, pass prob_outperform_6m.
        stcg_rate:             Override STCG rate.  Default: config.STCG_RATE.
        ltcg_rate:             Override LTCG rate.  Default: config.LTCG_RATE.

    Returns:
        ThreeScenarioResult with exactly 3 scenarios and a recommendation.
    """
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE

    breakeven = compute_stcg_ltcg_breakeven(stcg_rate, ltcg_rate)
    ltcg_date = vest_date + timedelta(days=366)
    days_to_ltcg = (ltcg_date - vest_date).days  # always 366

    # --- Scenario A: Sell Now at STCG ---
    gain_a = (current_price - cost_basis_per_share) * shares
    if gain_a >= 0:
        tax_a = gain_a * stcg_rate
    else:
        tax_a = gain_a * stcg_rate  # negative number = tax benefit
    gross_a = shares * current_price
    net_a = gross_a - tax_a
    prob_a = 1.0  # certain outcome
    scenario_a = TaxScenario(
        label="SELL_NOW_STCG",
        sell_date=vest_date,
        tax_rate=stcg_rate,
        holding_period_days=0,
        predicted_return=0.0,
        predicted_price=current_price,
        gross_proceeds=gross_a,
        tax_liability=tax_a,
        net_proceeds=net_a,
        breakeven_return=0.0,
        probability=prob_a,
        rationale=(
            f"Sell {shares:.0f} shares immediately at ${current_price:.2f}. "
            f"Tax rate {stcg_rate:.0%} (STCG). "
            f"Net proceeds: ${net_a:,.2f}."
        ),
    )

    # --- Scenario B: Hold to LTCG Eligibility (366 days) ---
    predicted_price_b = current_price * (1.0 + predicted_12m_return)
    gain_b = (predicted_price_b - cost_basis_per_share) * shares
    if gain_b >= 0:
        tax_b = gain_b * ltcg_rate
    else:
        tax_b = gain_b * ltcg_rate  # negative number = tax benefit
    gross_b = shares * predicted_price_b
    net_b = gross_b - tax_b
    prob_b = prob_outperform_12m
    scenario_b = TaxScenario(
        label="HOLD_TO_LTCG",
        sell_date=ltcg_date,
        tax_rate=ltcg_rate,
        holding_period_days=days_to_ltcg,
        predicted_return=predicted_12m_return,
        predicted_price=predicted_price_b,
        gross_proceeds=gross_b,
        tax_liability=tax_b,
        net_proceeds=net_b,
        breakeven_return=breakeven,
        probability=prob_b,
        rationale=(
            f"Hold until {ltcg_date} (LTCG-eligible). "
            f"Predicted price: ${predicted_price_b:.2f} "
            f"({predicted_12m_return:+.1%} 12M return). "
            f"Tax rate {ltcg_rate:.0%} (LTCG). "
            f"Net proceeds: ${net_b:,.2f} (P={prob_b:.0%})."
        ),
    )

    # --- Scenario C: Hold for Capital Loss ---
    if predicted_6m_return < 0:
        predicted_loss_price = current_price * (1.0 + predicted_6m_return)
        loss_gain = (predicted_loss_price - cost_basis_per_share) * shares  # negative
        # Tax benefit: loss offsets gains at the higher STCG rate
        tax_c = loss_gain * stcg_rate  # negative = tax benefit
        gross_c = shares * predicted_loss_price
        net_c = gross_c - tax_c  # net_c > gross_c because tax_c is negative
        prob_c = 1.0 - prob_outperform_6m
        rationale_c = (
            f"Hold and harvest capital loss. "
            f"Predicted price: ${predicted_loss_price:.2f} "
            f"({predicted_6m_return:+.1%} 6M return). "
            f"Loss tax benefit at {stcg_rate:.0%} STCG rate. "
            f"Net (incl. benefit): ${net_c:,.2f} (P={prob_c:.0%})."
        )
    else:
        # Positive predicted return: loss scenario is degenerate.
        predicted_loss_price = current_price
        tax_c = 0.0
        gross_c = shares * current_price
        net_c = gross_c
        prob_c = 0.0
        rationale_c = (
            "Capital loss harvest not applicable: model predicts positive return."
        )

    scenario_c = TaxScenario(
        label="HOLD_FOR_LOSS",
        sell_date=vest_date + timedelta(days=180),  # approximate 6M
        tax_rate=stcg_rate,
        holding_period_days=180,
        predicted_return=predicted_6m_return,
        predicted_price=predicted_loss_price,
        gross_proceeds=gross_c,
        tax_liability=tax_c,
        net_proceeds=net_c,
        breakeven_return=0.0,
        probability=prob_c,
        rationale=rationale_c,
    )

    # --- Select recommended scenario (highest probability-weighted utility) ---
    utility_a = prob_a * net_a
    utility_b = prob_b * net_b
    # Scenario C is only competitive when predicted return is negative.
    utility_c = prob_c * net_c if predicted_6m_return < 0 else float("-inf")

    candidates = [
        ("SELL_NOW_STCG", utility_a),
        ("HOLD_TO_LTCG", utility_b),
        ("HOLD_FOR_LOSS", utility_c),
    ]
    recommended = max(candidates, key=lambda x: x[1])[0]

    return ThreeScenarioResult(
        vest_date=vest_date,
        rsu_type=rsu_type,
        current_price=current_price,
        cost_basis_per_share=cost_basis_per_share,
        shares=shares,
        scenarios=[scenario_a, scenario_b, scenario_c],
        recommended_scenario=recommended,
        stcg_ltcg_breakeven=breakeven,
        days_to_ltcg=days_to_ltcg,
    )
