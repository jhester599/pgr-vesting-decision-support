"""Review 2026-09-25, step 6 (WP8, F19): hand-computed tax cases.

Each test states the arithmetic it checks. Rates are 37 % STCG / 20 % LTCG
unless a test says otherwise. Scheduled vests (config): 19 January (time) and
17 July (performance).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.tax.capital_gains import (
    TaxLot,
    compute_position_summary,
    compute_stcg_ltcg_breakeven,
    compute_three_scenarios,
    load_position_lots,
    optimize_sale,
)

S, L = 0.37, 0.20


# ---------------------------------------------------------------------------
# Breakeven on the absolute PGR return: -g (S - L) / (1 - L)
# ---------------------------------------------------------------------------


def test_fully_appreciated_lot_breakeven_is_a_21_25_pct_fall() -> None:
    """Basis 0, price 100: sell now nets 100 x 0.63 = 63.00. Holding to LTCG
    through a -21.25 % move: 78.75 x 0.80 = 63.00. Equal, so r* = -21.25 %."""
    r_star = compute_stcg_ltcg_breakeven(S, L)  # default: fully appreciated (g = 1)
    assert r_star == pytest.approx(-0.2125)
    sell_now = 100.0 * (1 - S)
    hold = 100.0 * (1 + r_star) * (1 - L)
    assert sell_now == pytest.approx(63.0)
    assert hold == pytest.approx(63.0)


def test_partially_appreciated_lot_breakeven() -> None:
    """Basis 60, price 100 (g = 0.4): sell now 100 - 0.37 x 40 = 85.20.
    r* = -0.4 x 0.2125 = -8.5 %: 91.50 - 0.20 x 31.50 = 85.20."""
    r_star = compute_stcg_ltcg_breakeven(S, L, gain_fraction=0.4)
    assert r_star == pytest.approx(-0.085)
    assert 100.0 - S * 40.0 == pytest.approx(85.2)
    assert 91.5 - L * (91.5 - 60.0) == pytest.approx(85.2)


def test_scenarios_tie_exactly_at_the_breakeven() -> None:
    """compute_three_scenarios: at r = r* for the lot, A and B net the same."""
    price, basis, shares = 100.0, 60.0, 10.0
    r_star = compute_stcg_ltcg_breakeven(S, L, gain_fraction=(price - basis) / price)
    result = compute_three_scenarios(
        vest_date=date(2026, 7, 17), rsu_type="performance", shares=shares,
        cost_basis_per_share=basis, current_price=price,
        predicted_6m_return=0.0, predicted_12m_return=r_star,
        prob_outperform_6m=0.5, prob_outperform_12m=0.5, stcg_rate=S, ltcg_rate=L,
    )
    a, b, _ = result.scenarios
    assert a.net_proceeds == pytest.approx(852.0)
    assert b.net_proceeds == pytest.approx(852.0)
    assert result.stcg_ltcg_breakeven == pytest.approx(-0.085)


def test_positive_forecast_recommends_holding_to_ltcg() -> None:
    """Review F19: the old utility (probability x proceeds) picked SELL_NOW on
    a +30 % forecast. 100 shares, basis 50, price 100:
    A = 10,000 - 0.37 x 5,000 = 8,150; B = 13,000 - 0.20 x 8,000 = 11,400."""
    result = compute_three_scenarios(
        vest_date=date(2026, 7, 17), rsu_type="performance", shares=100.0,
        cost_basis_per_share=50.0, current_price=100.0,
        predicted_6m_return=0.15, predicted_12m_return=0.30,
        prob_outperform_6m=0.6, prob_outperform_12m=0.6, stcg_rate=S, ltcg_rate=L,
    )
    a, b, _ = result.scenarios
    assert a.net_proceeds == pytest.approx(8150.0)
    assert b.net_proceeds == pytest.approx(11400.0)
    assert result.recommended_scenario == "HOLD_TO_LTCG"


# ---------------------------------------------------------------------------
# LTCG iff sold after vest + 1 year (calendar), across 29 February
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vest", "last_short", "first_long"),
    [
        # 2028 is a leap year: 2027-07-17 -> 2028-07-17 is 366 days and still short-term.
        (date(2027, 7, 17), date(2028, 7, 17), date(2028, 7, 18)),
        (date(2028, 1, 19), date(2029, 1, 19), date(2029, 1, 20)),
        # No 29 February in between: 365 days to the anniversary.
        (date(2026, 7, 17), date(2027, 7, 17), date(2027, 7, 18)),
        # A 29 February vest: the anniversary is 28 February.
        (date(2028, 2, 29), date(2029, 2, 28), date(2029, 3, 1)),
    ],
)
def test_ltcg_starts_the_day_after_the_calendar_anniversary(
    vest: date, last_short: date, first_long: date
) -> None:
    lot = TaxLot(vest_date=vest, rsu_type="time", shares=1.0, cost_basis_per_share=1.0)
    assert lot.is_ltcg_eligible(last_short) is False
    assert lot.is_ltcg_eligible(first_long) is True


def test_hold_to_ltcg_date_across_a_leap_year() -> None:
    """Vest 2027-07-17: the old code sold on vest + 366 days = 2028-07-17, still short-term."""
    result = compute_three_scenarios(
        vest_date=date(2027, 7, 17), rsu_type="performance", shares=1.0,
        cost_basis_per_share=100.0, current_price=100.0,
        predicted_6m_return=0.0, predicted_12m_return=0.0,
        prob_outperform_6m=0.5, prob_outperform_12m=0.5,
    )
    b = result.scenarios[1]
    assert b.sell_date == date(2028, 7, 18)
    assert result.days_to_ltcg == 367


def test_sale_on_the_anniversary_is_taxed_short_term() -> None:
    """Sell 10 shares at 200 on 2028-07-17 from a 2027-07-17 lot at basis 100:
    gain 1,000 at 37 % = 370 (the old day count said LTCG: 200)."""
    lot = TaxLot(vest_date=date(2027, 7, 17), rsu_type="performance", shares=10.0,
                 cost_basis_per_share=100.0)
    sale = optimize_sale([lot], 10.0, 200.0, date(2028, 7, 17), ltcg_rate=L, stcg_rate=S)
    assert sale.lots[0].holding_type == "STCG"
    assert sale.total_tax == pytest.approx(370.0)


# ---------------------------------------------------------------------------
# Wash sales against the vest schedule
# ---------------------------------------------------------------------------


def test_hold_for_loss_sale_date_clears_the_next_vest_window() -> None:
    """Old: vest 2027-01-19 + 180 days = 2027-07-18, one day after the
    2027-07-17 vest (a wash sale). Now: 2027-07-19 (six months) is 2 days
    after that vest, so the sale moves to 2027-07-17 + 31 = 2027-08-17."""
    result = compute_three_scenarios(
        vest_date=date(2027, 1, 19), rsu_type="time", shares=100.0,
        cost_basis_per_share=100.0, current_price=100.0,
        predicted_6m_return=-0.10, predicted_12m_return=-0.10,
        prob_outperform_6m=0.4, prob_outperform_12m=0.4, stcg_rate=S, ltcg_rate=L,
    )
    c = result.scenarios[2]
    assert c.sell_date == date(2027, 8, 17)
    for vest in (date(2027, 1, 19), date(2027, 7, 17), date(2028, 1, 19)):
        assert abs((c.sell_date - vest).days) > 30
    # Loss: 100 x (90 - 100) = -1,000 at 37 % -> benefit 370; net 9,000 + 370.
    assert c.tax_liability == pytest.approx(-370.0)
    assert c.net_proceeds == pytest.approx(9370.0)


def test_performance_vest_loss_sale_clears_the_january_vest() -> None:
    """Vest 2026-07-17 + 6 months = 2027-01-17, two days before the
    2027-01-19 vest; moved to 2027-02-19."""
    result = compute_three_scenarios(
        vest_date=date(2026, 7, 17), rsu_type="performance", shares=1.0,
        cost_basis_per_share=100.0, current_price=100.0,
        predicted_6m_return=-0.05, predicted_12m_return=-0.05,
        prob_outperform_6m=0.4, prob_outperform_12m=0.4,
    )
    assert result.scenarios[2].sell_date == date(2027, 2, 19)


@pytest.mark.parametrize(
    ("sell_date", "is_wash"),
    [
        (date(2027, 6, 16), False),  # 31 days before the 2027-07-17 vest
        (date(2027, 6, 17), True),   # 30 days before
        (date(2027, 8, 16), True),   # 30 days after
        (date(2027, 8, 17), False),  # 31 days after
    ],
)
def test_loss_lot_inside_the_vest_window_is_a_wash_sale(sell_date: date, is_wash: bool) -> None:
    """Sell 10 shares at 80 from a lot at basis 100 (loss 200). Inside the
    window the loss is disallowed (tax 0); outside, the benefit is 200 x 0.20 = 40
    (the lot is long-term)."""
    lot = TaxLot(vest_date=date(2025, 1, 21), rsu_type="time", shares=10.0,
                 cost_basis_per_share=100.0)
    sale = optimize_sale([lot], 10.0, 80.0, sell_date, ltcg_rate=L, stcg_rate=S)
    result = sale.lots[0]
    if is_wash:
        assert result.holding_type == "WASH_SALE"
        assert sale.total_tax == pytest.approx(0.0)
    else:
        assert result.holding_type == "LOSS"
        assert sale.total_tax == pytest.approx(-40.0)


def test_wash_sale_lots_are_sold_last() -> None:
    """Selling on a vest date: an old loss lot would be a wash sale, so the
    optimiser takes the gain lot first. Price 100; gain lot 10 @ 90 (LTCG,
    tax 10 x 10 x 0.20 = 20); loss lot 10 @ 120."""
    gain_lot = TaxLot(vest_date=date(2024, 1, 19), rsu_type="time", shares=10.0,
                      cost_basis_per_share=90.0)
    loss_lot = TaxLot(vest_date=date(2025, 1, 21), rsu_type="time", shares=10.0,
                      cost_basis_per_share=120.0)
    sale = optimize_sale([loss_lot, gain_lot], 10.0, 100.0, date(2026, 7, 17),
                         ltcg_rate=L, stcg_rate=S)
    assert [r.holding_type for r in sale.lots] == ["LTCG"]
    assert sale.total_tax == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Per-share lot ordering
# ---------------------------------------------------------------------------


def test_stcg_lots_are_ordered_by_gain_per_share() -> None:
    """Price 200. Lot A: 10 @ 150 (gain 50/share, 500 total). Lot B: 100 @ 190
    (gain 10/share, 1,000 total). Selling 10 shares from B costs
    10 x 10 x 0.37 = 37; from A (the old smallest-total order) 185."""
    sell = date(2026, 4, 15)
    lot_a = TaxLot(vest_date=date(2026, 1, 20), rsu_type="time", shares=10.0,
                   cost_basis_per_share=150.0)
    lot_b = TaxLot(vest_date=date(2025, 7, 18), rsu_type="performance", shares=100.0,
                   cost_basis_per_share=190.0)
    sale = optimize_sale([lot_a, lot_b], 10.0, 200.0, sell, ltcg_rate=L, stcg_rate=S)
    assert sale.lots[0].lot is lot_b
    assert sale.total_tax == pytest.approx(37.0)


def test_loss_lots_are_ordered_by_loss_per_share() -> None:
    """Price 200. Lot A: 10 @ 300 (loss 100/share, 1,000 total). Lot B: 100 @ 220
    (loss 20/share, 2,000 total). Selling 10 shares harvests 1,000 from A;
    the old total-loss order took B and harvested 200."""
    sell = date(2026, 4, 15)
    lot_a = TaxLot(vest_date=date(2023, 1, 19), rsu_type="time", shares=10.0,
                   cost_basis_per_share=300.0)
    lot_b = TaxLot(vest_date=date(2023, 7, 17), rsu_type="performance", shares=100.0,
                   cost_basis_per_share=220.0)
    sale = optimize_sale([lot_b, lot_a], 10.0, 200.0, sell, ltcg_rate=L, stcg_rate=S)
    assert sale.lots[0].lot is lot_a
    assert sale.lots[0].taxable_gain == pytest.approx(-1000.0)
    assert sale.total_tax == pytest.approx(-200.0)


# ---------------------------------------------------------------------------
# Unvested lots are not held
# ---------------------------------------------------------------------------


def _lots() -> list[TaxLot]:
    return [
        TaxLot(vest_date=date(2025, 7, 17), rsu_type="performance", shares=100.0,
               cost_basis_per_share=150.0),
        TaxLot(vest_date=date(2027, 1, 19), rsu_type="time", shares=50.0,
               cost_basis_per_share=200.0),  # vests after the sale date
    ]


def test_unvested_lot_cannot_be_sold() -> None:
    with pytest.raises(ValueError, match="only 100.00 available"):
        optimize_sale(_lots(), 120.0, 200.0, date(2026, 9, 1))


def test_position_summary_excludes_unvested_lots() -> None:
    """100 held shares at 200 = 20,000 value; basis 15,000; gain 5,000."""
    summary = compute_position_summary(_lots(), 200.0, date(2026, 9, 1))
    assert summary["total_shares"] == pytest.approx(100.0)
    assert summary["current_value"] == pytest.approx(20_000.0)
    assert summary["total_unrealized_gain"] == pytest.approx(5_000.0)


def test_load_position_lots_drops_lots_after_as_of(tmp_path: Path) -> None:
    csv = tmp_path / "position_lots.csv"
    pd.DataFrame(
        {
            "vest_date": ["2025-07-17", "2026-01-19", "2027-01-19"],
            "rsu_type": ["performance", "time", "time"],
            "shares": [100, 80, 50],
            "cost_basis_per_share": [150.0, 180.0, 0.0],
        }
    ).to_csv(csv, index=False)
    assert len(load_position_lots(str(csv))) == 3
    held = load_position_lots(str(csv), as_of=date(2026, 9, 21))
    assert [lot.vest_date for lot in held] == [date(2025, 7, 17), date(2026, 1, 19)]


def test_existing_holdings_guidance_skips_unvested_lots() -> None:
    from src.portfolio.redeploy_buckets import summarize_existing_holdings_actions

    actions = summarize_existing_holdings_actions(_lots(), current_price=200.0,
                                                  sell_date=date(2026, 9, 1))
    assert [a.vest_date for a in actions] == [date(2025, 7, 17)]


# ---------------------------------------------------------------------------
# The monthly report's tax inputs
# ---------------------------------------------------------------------------


def test_provisional_scenario_does_not_use_the_relative_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relative forecast (-30 % here) is not a PGR price forecast: the
    scenarios and the Monte Carlo use TAX_SCENARIO_PGR_ANNUAL_RETURN (0 %),
    and only vested lots count (100 shares, not 150)."""
    import config
    import scripts.monthly_decision as md

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    pd.DataFrame(
        {
            "vest_date": ["2025-07-17", "2027-01-19"],
            "rsu_type": ["performance", "time"],
            "shares": [100, 50],
            "cost_basis_per_share": [150.0, 999.0],
        }
    ).to_csv(tmp_path / "data" / "processed" / "position_lots.csv", index=False)
    weeks = pd.date_range("2025-06-06", "2026-09-18", freq="W-FRI")
    prices = pd.DataFrame({"close": [200.0 + (i % 3) for i in range(len(weeks))]}, index=weeks)
    monkeypatch.setattr(md.db_client, "get_prices", lambda *a, **k: prices)
    monkeypatch.setattr(md.db_client, "get_splits", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(config, "TAX_SCENARIO_PGR_ANNUAL_RETURN", 0.0)

    summary = md._build_provisional_vest_scenario(None, date(2026, 9, 21), -0.30, 0.2)
    assert summary is not None
    assert summary["shares"] == pytest.approx(100.0)
    a, b, c = summary["scenario"].scenarios
    assert b.predicted_return == pytest.approx(0.0)
    assert c.predicted_return == pytest.approx(0.0)
    assert summary["scenario"].recommended_scenario == "HOLD_TO_LTCG"
    assert summary["mc_analysis"].hold_ltcg.annual_drift == pytest.approx(0.0)
