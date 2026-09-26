"""
Tax lot optimizer for taxable brokerage account positions.

NUA is out of scope — stock is held in a taxable account, not a 401k/ESOP.

Strategy: minimize total tax liability on a given sale amount by selecting
the optimal combination of tax lots using the following priority rules:

  1. Sell lots with embedded losses first, largest loss per share first
     (tax-loss harvesting, reduces ordinary income or offsets other gains).
  2. Among gain lots, sell LTCG-eligible lots before STCG lots (LTCG rate
     is materially lower than ordinary income rates).
  3. Within each bucket, smallest gain per share (highest basis) first
     ("specific identification"). Ordering is per share, not per lot total.
  4. Loss lots within 30 days of a vest (a wash sale) last: the loss would
     be disallowed now.

Only vested lots are held. A sale is long-term when it is later than the
one-year calendar anniversary of the vest (``ltcg_eligible_date``).

Tax rates are configured in config.py and can be overridden via .env
to reflect state tax additions (e.g., CA adds 13.3% for a combined
LTCG rate of ~33.3% for high earners).

All lot-level calculations use exact per-share arithmetic to avoid
rounding errors that accumulate over multi-year sale programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

import pandas as pd
from dateutil.relativedelta import relativedelta

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
        """True when a sale on ``sell_date`` is long-term: held more than one year.

        The holding period starts the day after vesting, so the sale must be
        later than the one-year anniversary (``vest_date + relativedelta(years=1)``).
        Counting 366 days instead is one day early whenever the year spans
        29 February (review 2026-09-25, F19).
        """
        return sell_date >= ltcg_eligible_date(self.vest_date)

    def is_vested(self, as_of: date) -> bool:
        """True once the lot has vested (on or before ``as_of``)."""
        return self.vest_date <= as_of

    def embedded_gain(self, current_price: float) -> float:
        """Current unrealized gain (negative = loss)."""
        return self.shares_remaining * (current_price - self.cost_basis_per_share)


# ---------------------------------------------------------------------------
# Holding period, vested lots and the wash-sale window
# ---------------------------------------------------------------------------

WASH_SALE_WINDOW_DAYS: int = 30
"""A loss sale is a wash sale when substantially identical shares are
acquired within 30 days before or after it (IRC §1091). RSU vests are
acquisitions."""


def ltcg_eligible_date(vest_date: date) -> date:
    """First sale date that is long-term: the day after the one-year anniversary.

    ``relativedelta`` keeps the calendar anniversary across leap years; a
    29 February vest has its anniversary on 28 February.
    """
    return vest_date + relativedelta(years=1) + timedelta(days=1)


def vested_lots(lots: Iterable[TaxLot], as_of: date) -> list[TaxLot]:
    """Lots vested on or before ``as_of``; future (unvested) lots are not held."""
    return [lot for lot in lots if lot.is_vested(as_of)]


def scheduled_vest_dates(start: date, end: date) -> list[date]:
    """Scheduled RSU vest dates (time and performance) between ``start`` and ``end`` inclusive."""
    dates: list[date] = []
    for year in range(start.year, end.year + 1):
        for month, day in (
            (config.TIME_RSU_VEST_MONTH, config.TIME_RSU_VEST_DAY),
            (config.PERF_RSU_VEST_MONTH, config.PERF_RSU_VEST_DAY),
        ):
            vest = date(year, month, day)
            if start <= vest <= end:
                dates.append(vest)
    return sorted(dates)


def wash_sale_conflicts(
    sell_date: date,
    acquisition_dates: Iterable[date] | None = None,
    window_days: int = WASH_SALE_WINDOW_DAYS,
) -> list[date]:
    """Acquisitions within ``window_days`` before or after ``sell_date``.

    ``acquisition_dates`` defaults to the scheduled vest dates
    (``scheduled_vest_dates``). A loss realised on ``sell_date`` is disallowed
    (deferred into the replacement shares' basis) when this is non-empty.
    """
    if acquisition_dates is None:
        acquisition_dates = scheduled_vest_dates(
            sell_date - timedelta(days=window_days),
            sell_date + timedelta(days=window_days),
        )
    return sorted(
        {d for d in acquisition_dates if abs((d - sell_date).days) <= window_days}
    )


def first_wash_free_sale_date(
    earliest: date,
    acquisition_dates: Iterable[date] | None = None,
    window_days: int = WASH_SALE_WINDOW_DAYS,
) -> date:
    """First date on or after ``earliest`` with no acquisition within the window."""
    candidate = earliest
    for _ in range(12):  # each step clears one vest; two vests a year
        conflicts = wash_sale_conflicts(candidate, acquisition_dates, window_days)
        if not conflicts:
            return candidate
        candidate = max(conflicts) + timedelta(days=window_days + 1)
    raise ValueError(f"No wash-sale-free date found after {earliest}")


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
    holding_type: str      # "LTCG", "STCG", "LOSS", or "WASH_SALE" (loss disallowed)


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
    acquisition_dates: Sequence[date] | None = None,
) -> TotalSaleResult:
    """
    Select the optimal tax lots to minimize total tax liability on a sale.

    Only lots vested on or before ``sell_date`` are available. Lot priority:
      1. Loss lots, largest loss per share first (tax-loss harvesting).
      2. LTCG gain lots, smallest gain per share (highest basis) first.
      3. STCG gain lots, smallest gain per share (highest basis) first.
      4. Loss lots inside the wash-sale window, last: their loss is
         disallowed now (deferred into the replacement shares' basis), so
         they are kept to be harvested outside the window.

    Ordering is per share, because a sale needs a number of shares, not a
    dollar amount: a small lot with a large gain per share costs more tax per
    share sold than a large lot with a small one (review 2026-09-25, F19).

    Args:
        lots:              All tax lots; unvested lots are ignored.
        shares_to_sell:    Number of shares to liquidate in this transaction.
        sale_price:        Per-share sale price (unadjusted market price).
        sell_date:         Date of the sale transaction.
        ltcg_rate:         Override for config.LTCG_RATE.
        stcg_rate:         Override for config.STCG_RATE.
        acquisition_dates: Acquisitions of PGR shares that can trigger a wash
                           sale. Defaults to the scheduled RSU vests within
                           30 days of ``sell_date`` plus the vest dates of the
                           other lots.

    Returns:
        TotalSaleResult with per-lot breakdown and aggregate tax metrics. A
        wash-sale lot has ``holding_type`` "WASH_SALE" and zero tax.

    Raises:
        ValueError: If shares_to_sell exceeds the vested shares available.
    """
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE

    held = [
        lot for lot in vested_lots(lots, sell_date)
        if lot.shares_remaining is not None and lot.shares_remaining > 0
    ]
    total_available = sum(lot.shares_remaining for lot in held)
    if shares_to_sell > total_available + 1e-9:
        raise ValueError(
            f"Requested {shares_to_sell:.2f} shares to sell but only "
            f"{total_available:.2f} available across all lots."
        )

    if acquisition_dates is None:
        schedule = wash_sale_conflicts(sell_date)
    else:
        schedule = list(acquisition_dates)

    def _in_wash_window(lot: TaxLot) -> bool:
        replacements = [d for d in schedule if d != lot.vest_date]
        replacements += [
            other.vest_date for other in held
            if other is not lot and other.vest_date != lot.vest_date
        ]
        return bool(wash_sale_conflicts(sell_date, replacements))

    def _gain_per_share(lot: TaxLot) -> float:
        return sale_price - lot.cost_basis_per_share

    loss_lots: list[TaxLot] = []
    wash_lots: list[TaxLot] = []
    ltcg_gain_lots: list[TaxLot] = []
    stcg_lots: list[TaxLot] = []

    for lot in held:
        if _gain_per_share(lot) < 0:
            (wash_lots if _in_wash_window(lot) else loss_lots).append(lot)
        elif lot.is_ltcg_eligible(sell_date):
            ltcg_gain_lots.append(lot)
        else:
            stcg_lots.append(lot)

    loss_lots.sort(key=_gain_per_share)        # largest loss per share first
    ltcg_gain_lots.sort(key=_gain_per_share)   # smallest gain per share first
    stcg_lots.sort(key=_gain_per_share)        # smallest gain per share first
    wash_lots.sort(key=_gain_per_share, reverse=True)  # smallest disallowed loss first

    ordered_lots = loss_lots + ltcg_gain_lots + stcg_lots + wash_lots
    wash_ids = {id(lot) for lot in wash_lots}

    sale_results: list[SaleResult] = []
    shares_remaining_to_sell = shares_to_sell

    for lot in ordered_lots:
        if shares_remaining_to_sell <= 1e-9:
            break

        shares_from_lot = min(lot.shares_remaining, shares_remaining_to_sell)
        gross = shares_from_lot * sale_price
        basis = shares_from_lot * lot.cost_basis_per_share
        gain = gross - basis

        if id(lot) in wash_ids:
            # Loss disallowed now; it is added to the replacement shares' basis.
            rate = 0.0
            tax = 0.0
            holding_type = "WASH_SALE"
        elif gain < 0:
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

def load_position_lots(csv_path: str, as_of: date | None = None) -> list[TaxLot]:
    """
    Load tax lots from the user-provided position_lots.csv.

    Expected columns: vest_date, rsu_type, shares, cost_basis_per_share.
    The file is gitignored as it contains personal financial data.

    Args:
        csv_path: Path to position_lots.csv.
        as_of:    When given, lots vesting after this date are dropped: an
                  unvested lot is not held and cannot be sold (F19).

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
    if as_of is not None:
        lots = vested_lots(lots, as_of)
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
        lots:          List of TaxLot objects; lots vesting after ``sell_date``
                       are not held and are ignored.
        current_price: Current market price per share.
        sell_date:     Date used to evaluate vesting and LTCG eligibility.

    Returns:
        Dict with position summary statistics.
    """
    lots = vested_lots(lots, sell_date)
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
    breakeven_return: float          # Absolute return below which selling now wins (B only)
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
    recommended_scenario: str        # Label of the highest expected-proceeds scenario
    stcg_ltcg_breakeven: float       # Absolute PGR return at which LTCG ties STCG (this lot)
    days_to_ltcg: int                # Days from vest_date until LTCG-eligible


def embedded_gain_fraction(current_price: float, cost_basis_per_share: float) -> float:
    """Unrealised gain as a fraction of the current price: ``(P - B) / P``.

    1.0 for a zero-basis (fully appreciated) lot, 0.0 at vest (basis = FMV),
    negative for a loss lot.
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")
    return (current_price - cost_basis_per_share) / current_price


def compute_stcg_ltcg_breakeven(
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
    gain_fraction: float = 1.0,
) -> float:
    """
    Absolute PGR return at which holding to LTCG ties selling now at STCG.

    With price ``P``, basis ``B`` and gain fraction ``g = (P - B) / P``:

        sell now:          P - S * (P - B)                 = P * (1 - S * g)
        hold, return r:    P(1+r) - L * (P(1+r) - B)       = P * ((1+r)(1-L) + L(1-g))

    Setting them equal gives ``r* = -g * (S - L) / (1 - L)``. Holding to the
    LTCG date gives more after-tax cash unless PGR's own price return (not a
    return relative to benchmarks) is below ``r*``. For a fully appreciated
    lot (g = 1) at 37 % / 20 % that is a fall of more than 21.25 %; for a lot
    at its vest price (g = 0) it is 0 %. Loss lots (g < 0) have a positive
    threshold, because selling now takes the loss at the higher STCG rate.

    The comparison is cash at the LTCG date; it ignores what sale proceeds
    earn meanwhile. The old version returned ``+(S - L) / (1 - L)`` as the
    return needed to *hold*, which inverted the sign (review 2026-09-25, F19).

    Args:
        stcg_rate: Short-term capital gains tax rate. Default: config.STCG_RATE.
        ltcg_rate: Long-term capital gains tax rate. Default: config.LTCG_RATE.
        gain_fraction: ``g`` above. Default 1.0 (fully appreciated lot).

    Returns:
        Breakeven absolute return as a decimal (e.g. -0.2125).
    """
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    return -gain_fraction * (stcg_rate - ltcg_rate) / (1.0 - ltcg_rate)


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
    acquisition_dates: Sequence[date] | None = None,
) -> ThreeScenarioResult:
    """
    Compute three tax-aware scenarios for a vesting event.

    The return inputs are **absolute PGR price returns**. The production
    models forecast PGR *relative to benchmarks*, which is not a price
    forecast; callers must not pass it here (review 2026-09-25, F19).

    Scenario A — Sell at Vest (STCG):
        Sell immediately on vest_date.  Tax rate = STCG.

    Scenario B — Hold to LTCG Eligibility:
        Hold until the first long-term date (the day after the one-year
        anniversary), then sell at the LTCG rate. Uses predicted_12m_return.

    Scenario C — Hold for Capital Loss:
        Relevant only when the expected 6-month price return is negative.
        Sell six months after vest (STCG). The sale date is moved past the
        30-day wash-sale window of any scheduled vest (``acquisition_dates``),
        otherwise the loss would be disallowed.

    The recommended scenario is the one with the highest expected after-tax
    proceeds. The probabilities are reported only: weighting proceeds by a
    probability of outperforming benchmarks picked SELL_NOW on a +30 %
    forecast.

    Args:
        vest_date:             Date shares vest (holding period starts).
        rsu_type:              "time" or "performance".
        shares:                Number of shares in this lot.
        cost_basis_per_share:  FMV on vest date (= cost basis for RSUs).
        current_price:         Current market price per share.
        predicted_6m_return:   Expected absolute 6-month PGR price return.
        predicted_12m_return:  Expected absolute PGR price return to the LTCG date.
        prob_outperform_6m:    Reported probability for scenario C (1 - p).
        prob_outperform_12m:   Reported probability for scenario B.
        stcg_rate:             Override STCG rate.  Default: config.STCG_RATE.
        ltcg_rate:             Override LTCG rate.  Default: config.LTCG_RATE.
        acquisition_dates:     Vests that can trigger a wash sale. Default:
                               the scheduled vests (config).

    Returns:
        ThreeScenarioResult with exactly 3 scenarios and a recommendation.
        ``stcg_ltcg_breakeven`` is this lot's breakeven absolute return.
    """
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE

    breakeven = compute_stcg_ltcg_breakeven(
        stcg_rate,
        ltcg_rate,
        gain_fraction=embedded_gain_fraction(current_price, cost_basis_per_share),
    )
    ltcg_date = ltcg_eligible_date(vest_date)
    days_to_ltcg = (ltcg_date - vest_date).days  # 366, or 367 across 29 February

    # --- Scenario A: Sell Now at STCG ---
    gain_a = (current_price - cost_basis_per_share) * shares
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

    # --- Scenario B: Hold to LTCG Eligibility ---
    predicted_price_b = current_price * (1.0 + predicted_12m_return)
    gain_b = (predicted_price_b - cost_basis_per_share) * shares
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
            f"Expected price: ${predicted_price_b:.2f} "
            f"({predicted_12m_return:+.1%} PGR price return). "
            f"Tax rate {ltcg_rate:.0%} (LTCG). "
            f"Net proceeds: ${net_b:,.2f}. Holding beats selling now unless "
            f"PGR returns less than {breakeven:+.2%}."
        ),
    )

    # --- Scenario C: Hold for Capital Loss (six months, outside wash-sale windows) ---
    loss_sell_date = first_wash_free_sale_date(
        vest_date + relativedelta(months=6), acquisition_dates
    )
    if predicted_6m_return < 0:
        predicted_loss_price = current_price * (1.0 + predicted_6m_return)
        loss_gain = (predicted_loss_price - cost_basis_per_share) * shares
        tax_c = loss_gain * stcg_rate  # negative = tax benefit when a real loss
        gross_c = shares * predicted_loss_price
        net_c = gross_c - tax_c
        prob_c = 1.0 - prob_outperform_6m
        if loss_gain < 0:
            rationale_c = (
                f"Sell on {loss_sell_date} and realise a short-term loss. "
                f"Expected price: ${predicted_loss_price:.2f} "
                f"({predicted_6m_return:+.1%} PGR price return), below the "
                f"${cost_basis_per_share:.2f} basis. The loss offsets gains at up to "
                f"{stcg_rate:.0%}. Net (incl. benefit): ${net_c:,.2f}."
            )
        else:
            rationale_c = (
                f"Sell on {loss_sell_date}. Expected price ${predicted_loss_price:.2f} "
                f"({predicted_6m_return:+.1%}) is still above the "
                f"${cost_basis_per_share:.2f} basis: a short-term gain, not a loss. "
                f"Net: ${net_c:,.2f}."
            )
    else:
        # Non-negative expected return: the loss scenario is degenerate.
        predicted_loss_price = current_price
        tax_c = 0.0
        gross_c = shares * current_price
        net_c = gross_c
        prob_c = 0.0
        rationale_c = (
            "Capital loss harvest not applicable: expected PGR price return is not negative."
        )

    scenario_c = TaxScenario(
        label="HOLD_FOR_LOSS",
        sell_date=loss_sell_date,
        tax_rate=stcg_rate,
        holding_period_days=(loss_sell_date - vest_date).days,
        predicted_return=predicted_6m_return,
        predicted_price=predicted_loss_price,
        gross_proceeds=gross_c,
        tax_liability=tax_c,
        net_proceeds=net_c,
        breakeven_return=0.0,
        probability=prob_c,
        rationale=rationale_c,
    )

    # --- Recommended scenario: highest expected after-tax proceeds ---
    candidates = [
        ("SELL_NOW_STCG", net_a),
        ("HOLD_TO_LTCG", net_b),
        ("HOLD_FOR_LOSS", net_c if predicted_6m_return < 0 else float("-inf")),
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
