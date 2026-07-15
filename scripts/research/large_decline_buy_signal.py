"""Research: is a large PGR down-move a forward buying signal?

Motivation
----------
PGR fell sharply (~-9%) intraday on 2026-07-15.  This study asks whether
historically large PGR declines have been followed by above-market forward
returns (a "buy the dip" signal) at 3, 6, and 12 month horizons.

Method
------
1. Identify every "large-decline" bar where PGR fell by >= 8% versus the prior
   bar (split-adjusted price return).
2. From each event bar, compute the forward DRIP total return of PGR at +3, +6,
   and +12 calendar months.
3. Compute the same-window forward total return of the S&P 500 proxy and the
   relative (PGR - benchmark) return.

IMPORTANT DATA CAVEAT
---------------------
The v2 database (``daily_prices``) is sourced from Alpha Vantage
``TIME_SERIES_WEEKLY`` (free tier).  Despite the table name, the bars are
**weekly Friday closes**, not daily.  True daily -8% day detection is not
possible with the on-hand data (premium AV subscription + API key required).
This study therefore detects **weekly** declines of >= 8%.  A weekly -8% move
is a strictly higher bar than a daily one, so the event set is a conservative
subset of "big down move" episodes.  Interpret accordingly.

Benchmark
---------
The repo's canonical market benchmark is VTI (Vanguard Total Stock Market),
used here as the S&P 500 proxy (~0.99 correlation with the S&P 500).  VOO (the
exact S&P 500 ETF) only has history from 2010-09, so it is reported as a
secondary cross-check where available.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from src.database import db_client  # noqa: E402
from src.processing.total_return import build_position_series  # noqa: E402

DECLINE_THRESHOLD = -0.08          # -8% or worse
HORIZONS_MONTHS = [3, 6, 12]
BENCHMARK = "VTI"                  # S&P 500 proxy (full history from 2001)
BENCHMARK_SECONDARY = "VOO"       # exact S&P 500 ETF, data from 2010-09
OUT_DIR = os.path.join(_PROJECT_ROOT, "results", "research")


def _empty_div_index() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["dividend", "source"], index=pd.DatetimeIndex([], name="ex_date")
    )


def _empty_split_index() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"],
        index=pd.DatetimeIndex([], name="split_date"),
    )


def _load(conn, ticker):
    """Return (price_only_position, drip_position) for a ticker.

    price_only_position: split-adjusted price (dividends OFF) -> used for event
        detection so week-over-week moves reflect pure price declines.
    drip_position: split + dividend reinvested -> used for forward total return.
    """
    prices = db_client.get_prices(conn, ticker)
    if prices.empty:
        return None, None

    divs = db_client.get_dividends(conn, ticker)
    divs = divs.rename(columns={"amount": "dividend"}) if not divs.empty else _empty_div_index()

    splits = db_client.get_splits(conn, ticker)
    if splits.empty:
        splits = _empty_split_index()

    price_pos = build_position_series(prices, _empty_div_index(), splits, initial_shares=1.0)
    drip_pos = build_position_series(prices, divs, splits, initial_shares=1.0)
    return price_pos, drip_pos


def _fwd_return(position: pd.DataFrame, start: pd.Timestamp, target: pd.Timestamp):
    """DRIP total return from `start` to the last bar at/after `start` and
    on/before `target`. Returns (ret, realized_date) or (nan, None) if the
    forward window extends past available data."""
    values = position["portfolio_value"]
    last_date = values.index.max()
    if target > last_date:
        return np.nan, None  # window not yet complete
    start_val = values.asof(start)
    end_val = values.asof(target)
    if pd.isna(start_val) or start_val == 0 or pd.isna(end_val):
        return np.nan, None
    realized = values.loc[:target].index.max()
    return float(end_val / start_val) - 1.0, realized


def main() -> None:
    conn = db_client.get_connection()
    try:
        pgr_price, pgr_drip = _load(conn, "PGR")
        _, bench_drip = _load(conn, BENCHMARK)
        _, bench2_drip = _load(conn, BENCHMARK_SECONDARY)
    finally:
        conn.close()

    # --- Event detection: weekly split-adjusted price decline >= 8% ---------
    price = pgr_price["portfolio_value"].copy()
    # collapse to the observed weekly closes (position series is daily ffilled)
    weekly = price.reindex(pgr_price.index).dropna()
    weekly = weekly[pgr_price["close_price"].reindex(weekly.index).notna()]
    # Use the raw weekly close dates directly:
    closes = pgr_price["close_price"].dropna()
    # split-adjusted price = portfolio_value from the dividends-off position
    adj = pgr_price.loc[closes.index, "portfolio_value"]
    wow = adj.pct_change()
    events = wow[wow <= DECLINE_THRESHOLD]

    rows = []
    for evt_date, decline in events.items():
        rec = {
            "event_date": evt_date.date().isoformat(),
            "decline_pct": round(decline * 100, 2),
            "adj_price": round(float(adj.loc[evt_date]), 4),
        }
        for m in HORIZONS_MONTHS:
            target = evt_date + relativedelta(months=m)
            pgr_ret, realized = _fwd_return(pgr_drip, evt_date, target)
            bench_ret, _ = _fwd_return(bench_drip, evt_date, target)
            bench2_ret, _ = _fwd_return(bench2_drip, evt_date, target)
            rel = pgr_ret - bench_ret if not (np.isnan(pgr_ret) or np.isnan(bench_ret)) else np.nan
            rel_voo = pgr_ret - bench2_ret if not (np.isnan(pgr_ret) or np.isnan(bench2_ret)) else np.nan
            rec[f"pgr_{m}m"] = None if np.isnan(pgr_ret) else round(pgr_ret * 100, 2)
            rec[f"spx_{m}m"] = None if np.isnan(bench_ret) else round(bench_ret * 100, 2)
            rec[f"rel_{m}m"] = None if np.isnan(rel) else round(rel * 100, 2)
            rec[f"voo_{m}m"] = None if np.isnan(bench2_ret) else round(bench2_ret * 100, 2)
            rec[f"relvoo_{m}m"] = None if np.isnan(rel_voo) else round(rel_voo * 100, 2)
        rows.append(rec)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "large_decline_buy_signal_detail.csv")
    df.to_csv(csv_path, index=False)

    # --- Summary stats ------------------------------------------------------
    def summarize(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return {}
        return {
            "n": int(s.count()),
            "mean": round(s.mean(), 2),
            "median": round(s.median(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
            "pct_positive": round((s > 0).mean() * 100, 1),
        }

    print(f"\nEvents (weekly PGR decline <= {DECLINE_THRESHOLD*100:.0f}%): {len(df)}")
    print(f"Detail CSV: {csv_path}\n")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))

    print("\n=== SUMMARY: forward DRIP total return from event bar ===")
    summ_rows = []
    for m in HORIZONS_MONTHS:
        for label, col in [
            ("PGR abs", f"pgr_{m}m"),
            ("vs S&P (VTI proxy)", f"rel_{m}m"),
            ("vs S&P (VOO, 2010+)", f"relvoo_{m}m"),
        ]:
            st = summarize(col)
            if st:
                summ_rows.append({"horizon": f"{m}m", "metric": label, **st})
    summ = pd.DataFrame(summ_rows)
    print(summ.to_string(index=False))
    summ_path = os.path.join(OUT_DIR, "large_decline_buy_signal_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(f"\nSummary CSV: {summ_path}")


if __name__ == "__main__":
    main()
