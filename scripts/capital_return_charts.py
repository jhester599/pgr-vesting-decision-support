"""
PGR Annual Capital Return: Share Repurchases vs. Dividends
Chart 1 — total dollars returned ($B), stacked bar.
Chart 2 — same dollars as % of year-end market cap, stacked bar.
Chart 3 — combined ratio vs. capital returned, two scatter panels.
Saves the files listed in ``CHART_FILES`` to results/research/.

The plotted data comes from ``build_chart_frames``, which the tests check.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import db_client  # noqa: E402
from src.reporting.capital_return_data import (  # noqa: E402
    annual_capital_return,
    build_monthly_frame,
    dividend_dollars,
    load_chart_inputs,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pgr_financials.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "research")

CHART_FILES: tuple[str, ...] = (
    "pgr_repurchase_dividend_annual.png",
    "pgr_capital_return_pct_marketcap.png",
    "pgr_cr_vs_capital_return.png",
)

BAR_WIDTH = 0.65


def build_chart_frames(conn) -> dict[str, pd.DataFrame]:
    """Return the monthly, dividend and annual frames the charts plot."""
    inputs = load_chart_inputs(conn)
    monthly = build_monthly_frame(inputs)
    dividends = dividend_dollars(inputs.dividends, monthly, inputs.splits)
    annual = annual_capital_return(monthly, dividends, inputs.splits)
    return {"monthly": monthly, "dividends": dividends, "annual": annual}


def scatter_years(annual: pd.DataFrame) -> pd.DataFrame:
    """Full years (12 months of combined ratio) with a market cap."""
    return annual[
        (annual["cr_months"] == 12) & annual["partial_label"].isna() & annual["market_cap"].notna()
    ]


def _annotate_partial(ax, annual: pd.DataFrame, values: pd.Series, offset: float) -> None:
    for year, label in annual["partial_label"].dropna().items():
        if year in values.index:
            total = float(values[year])
            ax.annotate(
                f"★ {label}",
                xy=(year, total),
                xytext=(year, total + offset),
                ha="center", va="bottom", fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
            )


def _note(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=7.5, color="#555555", ha="left", va="bottom",
             wrap=True)


def _style_bar_axis(ax, years, title, ylabel, formatter) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=9, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="upper left")


def scatter_panels(annual: pd.DataFrame) -> list[tuple[str, list[float], list[float]]]:
    """(panel, combined ratio, value) for the two scatter panels."""
    sc = scatter_years(annual)
    cr = sc["combined_ratio"].to_list()
    return [
        ("dollars", cr, (sc["total_dollars"] / 1000).to_list()),
        ("pct_market_cap", cr, sc["total_pct_market_cap"].to_list()),
    ]


def plot_cr_scatter(annual: pd.DataFrame):
    """Combined ratio vs capital returned: $B (left) and % of market cap (right)."""
    sc = scatter_years(annual)
    sc_years = [str(y) for y in sc.index]
    sc_cr = sc["combined_ratio"].to_list()
    cmap = plt.get_cmap("plasma")
    n = len(sc)
    colors = [cmap(0.15 + 0.70 * i / (n - 1)) for i in range(n)]
    fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
    for ax_s, y_vals, y_label, y_fmt in [
        (ax_l, (sc["total_dollars"] / 1000).to_list(), "Total Capital Returned ($B)",
         lambda x, _: f"${x:.1f}B"),
        (ax_r, sc["total_pct_market_cap"].to_list(),
         "Total Capital Returned (% of Year-End Mkt Cap)", lambda x, _: f"{x:.1f}%"),
    ]:
        for y, cr, val, c in zip(sc_years, sc_cr, y_vals, colors):
            ax_s.text(cr, val, y, fontsize=8.5, color=c, fontweight="bold",
                      ha="center", va="center", zorder=3)
        # ax.text does not update the data limits; an invisible marker per
        # year does, so no year falls outside the axes.
        ax_s.scatter(sc_cr, y_vals, s=0, alpha=0)
        ax_s.margins(x=0.06, y=0.08)
        coef = np.polyfit(sc_cr, y_vals, 1)
        x_fit = np.linspace(min(sc_cr), max(sc_cr), 200)
        ax_s.plot(x_fit, np.polyval(coef, x_fit), color="#888888", linewidth=1.2,
                  linestyle="--", alpha=0.7, zorder=1)
        ax_s.invert_xaxis()
        ax_s.set_xlabel("Combined Ratio  ← Better underwriting", fontsize=11)
        ax_s.set_ylabel(y_label, fontsize=10)
        ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
        ax_s.grid(linestyle="--", alpha=0.35)
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)
        ax_s.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_s.axvline(96, color="#cc4444", linewidth=0.9, linestyle=":", alpha=0.6, zorder=0)
        ylim = ax_s.get_ylim()
        ax_s.text(96.1, ylim[0] + (ylim[1] - ylim[0]) * 0.97,
                  "96%", fontsize=7, color="#cc4444", va="top")
    ax_l.set_title("Combined Ratio vs. Capital Returned ($B)", fontsize=12,
                   fontweight="bold", pad=10)
    ax_r.set_title("Combined Ratio vs. Capital Returned (% of Mkt Cap)", fontsize=12,
                   fontweight="bold", pad=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(int(sc_years[0]), int(sc_years[-1])))
    sm.set_array([])
    cbar = fig3.colorbar(sm, ax=ax_r, orientation="vertical", fraction=0.04, pad=0.03, shrink=0.85)
    cbar.set_label("Year", fontsize=9)
    first, last = int(sc_years[0]), int(sc_years[-1])
    cbar.set_ticks(sorted({first, last} | {y for y in (2010, 2015, 2020) if first < y < last}))
    fig3.tight_layout()
    return fig3


def plot_charts(frames: dict[str, pd.DataFrame], out_dir: str) -> list[str]:
    """Write every chart in ``CHART_FILES`` to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    annual = frames["annual"]
    written = []

    def _save(fig, name, **kwargs):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, **kwargs)
        plt.close(fig)
        print(f"Saved: {path}")
        written.append(path)

    # Chart 1: annual dollars
    years = list(annual.index)
    rep_b = annual["repurchase_dollars"] / 1000
    div_b = annual["dividend_dollars"] / 1000
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(years, rep_b.to_list(), width=BAR_WIDTH, color="#2ca02c", alpha=0.85,
           label="Share Repurchases")
    ax.bar(years, div_b.to_list(), width=BAR_WIDTH, color="#1f77b4", alpha=0.85,
           label="Dividends", bottom=rep_b.to_list())
    _annotate_partial(ax, annual, rep_b + div_b, 0.25)
    _style_bar_axis(ax, years,
                    "PGR — Annual Capital Returned to Shareholders: Repurchases vs. Dividends",
                    "$ Billions", lambda x, _: f"${x:.1f}B")
    _note(fig, "Repurchases = Σ monthly shares repurchased × average cost (monthly 8-K; the "
               "2006-05 split month from the Q2 2006 10-Q). Dividends = amount "
               "per ex-date × shares outstanding that month; Q1 ex-dates count toward the "
               "prior year.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "pgr_repurchase_dividend_annual.png", bbox_inches="tight")

    # Chart 2: % of year-end market cap
    with_cap = annual[annual["market_cap"].notna() & (annual["market_cap"] != 0)]
    years_mc = list(with_cap.index)
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.bar(years_mc, with_cap["repurchase_pct_market_cap"].to_list(), width=BAR_WIDTH,
            color="#1f77b4", alpha=0.85, label="Share Repurchases")
    ax2.bar(years_mc, with_cap["dividend_pct_market_cap"].to_list(), width=BAR_WIDTH,
            color="#ff7f0e", alpha=0.85, label="Dividends",
            bottom=with_cap["repurchase_pct_market_cap"].to_list())
    _annotate_partial(ax2, with_cap, with_cap["total_pct_market_cap"], 0.3)
    _style_bar_axis(ax2, years_mc, "PGR — Annual Capital Returned as % of Year-End Market Cap",
                    "% of Market Cap", lambda x, _: f"{x:.1f}%")
    _note(fig2, "Year-end market cap = last weekly close of December (latest reported month "
                "in a partial year) × common shares outstanding that month.")
    fig2.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig2, "pgr_capital_return_pct_marketcap.png", bbox_inches="tight")

    # Chart 3: combined ratio vs capital returned
    _save(plot_cr_scatter(annual), "pgr_cr_vs_capital_return.png")
    return written


def print_summary(frames: dict[str, pd.DataFrame]) -> None:
    annual = frames["annual"]
    print()
    print("─" * 60)
    print(f"{'Year':<6}  {'Repurchases ($B)':>17}  {'Dividends ($B)':>15}  {'Total ($B)':>11}")
    print("─" * 60)
    for year, row in annual.iterrows():
        marker = " ★" if pd.notna(row["partial_label"]) else ""
        r, d = row["repurchase_dollars"] / 1000, row["dividend_dollars"] / 1000
        print(f"{year:<6}  {r:>17.2f}  {d:>15.2f}  {r + d:>11.2f}{marker}")
    print("─" * 60)
    print()
    print("─" * 72)
    print(f"{'Year':<6}  {'Mkt Cap ($B)':>12}  {'Repurch %':>10}  {'Div %':>8}  {'Total %':>8}")
    print("─" * 72)
    for year, row in annual.dropna(subset=["market_cap"]).iterrows():
        marker = " ★" if pd.notna(row["partial_label"]) else ""
        print(f"{year:<6}  {row['market_cap'] / 1000:>12.1f}  "
              f"{row['repurchase_pct_market_cap']:>10.2f}  "
              f"{row['dividend_pct_market_cap']:>8.2f}  "
              f"{row['total_pct_market_cap']:>8.2f}{marker}")
    print("─" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)
    conn = db_client.get_connection(args.db_path, read_only=True)
    try:
        frames = build_chart_frames(conn)
    finally:
        conn.close()
    plot_charts(frames, args.out_dir)
    print_summary(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
