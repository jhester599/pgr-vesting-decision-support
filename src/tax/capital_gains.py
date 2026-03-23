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
from datetime import date
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
    from datetime import timedelta
    if wash_sale_days is None:
        wash_sale_days = config.TLH_WASH_SALE_DAYS
    return harvest_date + timedelta(days=wash_sale_days)
