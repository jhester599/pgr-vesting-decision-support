"""Compare P/B and P/E as predictors of PGR's forward returns.

Writes tables to ``research/studies/pb_vs_pe_analysis/outputs/``:
    signals_and_targets.csv   as-of signals and forward log returns by month
    one_way_full.csv          one-variable statistics, full sample
    one_way_subperiods.csv    the same for 2004-2014 and 2015+ start dates
    oos.csv                   walk-forward out-of-sample R^2 by feature set
    encompassing.csv          both signals in one regression
    noise.csv                 month-to-month behaviour of each multiple
    rolling_ic.csv            IC inside rolling 7-year windows
    split_sensitivity.csv     early/late IC for every split year 2009-2017
    annual_sample.csv         12-month IC using one observation per year
    return_decomposition.csv  book growth vs re-rating vs dividends by era
    implied_roe.csv           long-run ROE implied by today's P/B
    relative_valuation.csv    PGR P/E vs S&P 500 P/E and the 10y Treasury

Usage:
    python research/studies/pb_vs_pe_analysis/pb_vs_pe_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db_client
from src.processing.multi_total_return import build_etf_monthly_returns
from src.processing.total_return import build_position_series
from src.processing.valuation_multiples import build_monthly_valuation_multiples
from src.research.pb_vs_pe import (
    HORIZONS,
    SIGNALS,
    annual_sample_ic,
    build_asof_signals,
    encompassing_table,
    forward_log_returns,
    noise_diagnostics,
    implied_roe,
    one_way_stats,
    oos_table,
    return_decomposition,
    window_ic,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "research" / "studies" / "pb_vs_pe_analysis" / "outputs"
MARKET_TICKER = "VTI"
SPLIT_DATE = "2014-12-31"
ROLLING_YEARS = 7
ROBUSTNESS_TARGETS = ["abs_12m", "rel_12m", "abs_24m", "rel_24m"]
# PGR earnings yield relative to the S&P 500's (log S&P P/E / PGR P/E);
# the S&P series starts in 2008, so it is kept out of the core signal list.
MARKET_RELATIVE_SIGNAL = "ep_rel_mkt"
ALL_SIGNALS = SIGNALS + [MARKET_RELATIVE_SIGNAL]
ERAS: list[tuple[str, str]] = [
    ("2004-12-31", "2012-12-31"),
    ("2012-12-31", "2019-12-31"),
    ("2019-12-31", "LATEST"),
    ("2004-12-31", "LATEST"),
]
COST_OF_EQUITY = [0.08, 0.09, 0.10]
LONG_RUN_GROWTH = [0.03, 0.04, 0.05, 0.06]

FEATURE_SETS: dict[str, list[str]] = {
    **{name: [name] for name in ALL_SIGNALS},
    "bp+ep": ["bp", "ep"],
    "bp_z60+ep_z60": ["bp_z60", "ep_z60"],
    "pb_roe_adj+ep": ["pb_roe_adj", "ep"],
    "bp+roe_gap": ["bp", "roe_gap"],
}
ENCOMPASSING_PAIRS: list[tuple[str, str]] = [
    ("bp", "ep"),
    ("bp_z60", "ep_z60"),
    ("pb_roe_adj", "ep"),
    ("pb_roe_adj", "ep_z60"),
    ("bp", "roe_gap"),
    ("ep", "roe_gap"),
]


def main() -> None:
    conn = db_client.get_connection(config.DB_PATH)
    try:
        prices = db_client.get_prices(conn, "PGR", exclude_proxy=True)
        splits = db_client.get_splits(conn, "PGR")
        valuation = build_monthly_valuation_multiples(
            prices,
            db_client.get_pgr_edgar_monthly(conn),
            splits,
            db_client.get_pgr_fundamentals(conn),
        )
        pgr_fwd = {h: build_etf_monthly_returns(conn, "PGR", h) for h in HORIZONS}
        mkt_fwd = {h: build_etf_monthly_returns(conn, MARKET_TICKER, h) for h in HORIZONS}
        dividends = db_client.get_dividends(conn, "PGR").rename(columns={"amount": "dividend"})
        macro = db_client.get_fred_macro(conn, ["SP500_PE_RATIO_MULTPL", "GS10"])
    finally:
        conn.close()

    targets = forward_log_returns(pgr_fwd, mkt_fwd)
    start = pd.Timestamp(valuation["month_end"].min())
    month_ends = pd.DatetimeIndex(targets.index[targets.index >= start])
    signals = build_asof_signals(valuation, prices, splits, month_ends)
    frame = signals.join(targets)

    # FRED rows are stamped at calendar month-end; the frame uses business
    # month-ends, so align on the calendar month.
    macro.index = pd.DatetimeIndex(macro.index).to_period("M")
    macro = macro.groupby(level=0).last()
    month_key = frame.index.to_period("M")
    frame["sp500_pe"] = macro["SP500_PE_RATIO_MULTPL"].reindex(month_key).to_numpy()
    frame["gs10"] = macro["GS10"].reindex(month_key).to_numpy() / 100.0
    frame[MARKET_RELATIVE_SIGNAL] = np.log(frame["sp500_pe"] / frame["pe"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "signals_and_targets.csv", float_format="%.6f")

    full = one_way_stats(frame, frame, ALL_SIGNALS, HORIZONS)
    full.to_csv(OUTPUT_DIR / "one_way_full.csv", index=False, float_format="%.6f")

    early = one_way_stats(frame, frame, ALL_SIGNALS, HORIZONS, end=SPLIT_DATE)
    late = one_way_stats(
        frame, frame, ALL_SIGNALS, HORIZONS, start=str(pd.Timestamp(SPLIT_DATE) + pd.DateOffset(days=1))
    )
    subperiods = pd.concat(
        [early.assign(period="2004-2014"), late.assign(period="2015+")], ignore_index=True
    )
    subperiods.to_csv(OUTPUT_DIR / "one_way_subperiods.csv", index=False, float_format="%.6f")

    oos = oos_table(frame, FEATURE_SETS, HORIZONS)
    oos.to_csv(OUTPUT_DIR / "oos.csv", index=False, float_format="%.6f")

    enc = encompassing_table(frame, ENCOMPASSING_PAIRS, HORIZONS)
    enc.to_csv(OUTPUT_DIR / "encompassing.csv", index=False, float_format="%.6f")

    noise = noise_diagnostics(frame)
    noise.to_csv(OUTPUT_DIR / "noise.csv", index=False, float_format="%.6f")

    last_year = frame["abs_12m"].last_valid_index().year
    rolling_windows = [
        (f"{y}-01-01", f"{y + ROLLING_YEARS - 1}-12-31")
        for y in range(2005, last_year - ROLLING_YEARS + 2)
    ]
    rolling = window_ic(frame, ALL_SIGNALS, ROBUSTNESS_TARGETS, rolling_windows)
    rolling.to_csv(OUTPUT_DIR / "rolling_ic.csv", index=False, float_format="%.6f")

    split_parts = []
    for year in range(2009, 2018):
        halves = [("2004-01-01", f"{year}-12-31"), (f"{year + 1}-01-01", "2100-01-01")]
        part = window_ic(frame, ["bp", "ep"], ROBUSTNESS_TARGETS, halves)
        part["split_year"] = year
        part["half"] = np.where(part["window_start"] == "2004-01-01", "early", "late")
        split_parts.append(part)
    pd.concat(split_parts, ignore_index=True).to_csv(
        OUTPUT_DIR / "split_sensitivity.csv", index=False, float_format="%.6f"
    )

    annual = pd.concat(
        [annual_sample_ic(frame, ALL_SIGNALS, target) for target in ("abs_12m", "rel_12m")],
        ignore_index=True,
    )
    annual.to_csv(OUTPUT_DIR / "annual_sample.csv", index=False, float_format="%.6f")

    position = build_position_series(prices, dividends, splits)
    tri = position["portfolio_value"].resample("BME").last()
    latest = str(frame["pb"].last_valid_index().date())
    eras = [(a, latest if b == "LATEST" else b) for a, b in ERAS]
    decomposition = return_decomposition(tri, frame["price"], frame["bvps"], eras)
    decomposition.to_csv(
        OUTPUT_DIR / "return_decomposition.csv", index=False, float_format="%.6f"
    )

    today = frame.dropna(subset=["pb"]).iloc[-1]
    implied = pd.DataFrame(
        [
            {
                "pb": today["pb"],
                "cost_of_equity": r,
                "growth": g,
                "implied_roe": implied_roe(today["pb"], r, g),
            }
            for r in COST_OF_EQUITY
            for g in LONG_RUN_GROWTH
        ]
    )
    implied.to_csv(OUTPUT_DIR / "implied_roe.csv", index=False, float_format="%.6f")

    relative = frame[["pe", "sp500_pe", "ep", "gs10"]].copy()
    relative["pe_vs_sp500"] = relative["pe"] / relative["sp500_pe"]
    relative["ep_minus_gs10"] = relative["ep"] - relative["gs10"]
    relative.dropna(subset=["sp500_pe"]).to_csv(
        OUTPUT_DIR / "relative_valuation.csv", float_format="%.6f"
    )

    summary = {
        "market_benchmark": MARKET_TICKER,
        "first_signal_month": str(frame["pb"].first_valid_index().date()),
        "last_signal_month": str(frame["pb"].last_valid_index().date()),
        "horizons_months": HORIZONS,
        "signals": ALL_SIGNALS,
        "today": {
            "date": str(frame.index[-1].date()),
            "pb": float(today["pb"]),
            "pe": float(today["pe"]),
            "roe": float(today["roe"]),
            "roe_avg60": float(today["roe_avg60"]),
            "roe_gap": float(today["roe_gap"]),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote results to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
