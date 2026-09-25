"""
PGR Monthly Time Series: Book Value Per Share, Share Repurchases, Share Price
Produces the monthly charts listed in ``CHART_FILES`` in results/research/.

The plotted data comes from ``build_chart_frames``, which the tests check.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import db_client  # noqa: E402
from src.reporting.capital_return_data import (  # noqa: E402
    AVG_COST_ESTIMATED,
    build_monthly_frame,
    load_chart_inputs,
    split_markers,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pgr_financials.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "research")

CHART_FILES: tuple[str, ...] = (
    "pgr_book_value_per_share.png",
    "pgr_book_value_per_share_split_adjusted.png",
    "pgr_share_repurchase_volume.png",
    "pgr_repurchase_dollar_amount.png",
    "pgr_repurchase_dollar_amount_capped.png",
    "pgr_share_price.png",
    "pgr_share_price_split_adjusted.png",
    "pgr_price_to_book.png",
    "pgr_price_to_book_split_adjusted.png",
)

REPURCHASE_AXIS_CAP = 400.0  # $M, capped dollar chart

# Months whose repurchase is a known one-off event, for the chart label.
REPURCHASE_EVENTS: dict[str, str] = {"2004-10": "Dutch auction tender offer"}

PRICE_NOTE = "Monthly price = last weekly close in each calendar month (unadjusted weekly bars)."

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
PURPLE = "#9467bd"


def build_chart_frames(conn) -> dict[str, object]:
    """Return the monthly frame and the split markers the charts plot."""
    inputs = load_chart_inputs(conn)
    monthly = build_monthly_frame(inputs)
    start = min(monthly.index.min(), monthly["price_date"].min())
    markers = split_markers(inputs.splits, start, monthly.index.max())
    return {"monthly": monthly, "split_markers": markers}


def style_ax(ax, title, ylabel, color):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_split_annotation(ax, split_date, split_label, ymax_frac=0.92):
    """Draw a single vertical dashed split-line with label."""
    ax.axvline(split_date, color="#888888", linewidth=1.0, linestyle="--", alpha=0.7)
    ylim = ax.get_ylim()
    y_pos = ylim[0] + (ylim[1] - ylim[0]) * ymax_frac
    ax.text(split_date, y_pos, split_label,
            ha="left", va="top", fontsize=7.5, color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8))


def add_split_lines(ax, markers, start, ymax_frac=0.92):
    """Annotate every split on or after ``start``."""
    for split_date, label in markers:
        if split_date >= start:
            _add_split_annotation(ax, split_date.date(), label, ymax_frac=ymax_frac)


def _note(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=7.5, color="#555555", ha="left", va="bottom")


def _dates(index) -> list[datetime.date]:
    return [pd.Timestamp(d).date() for d in index]


def plot_charts(frames: dict[str, object], out_dir: str) -> list[str]:
    """Write every chart in ``CHART_FILES`` to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    monthly: pd.DataFrame = frames["monthly"]
    markers = frames["split_markers"]
    edgar = monthly[monthly["edgar_month"]]
    first_edgar = edgar.index.min()
    written = []

    def _save(fig, name):
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        written.append(path)

    def _line_chart(values, x, title, ylabel, color, fmt, name, note, split_start, mean=False):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, values.to_list(), color=color, linewidth=1.8)
        ax.fill_between(x, values.to_list(), alpha=0.10, color=color)
        if mean:
            avg = float(values.mean())
            ax.axhline(avg, color=color, linewidth=1.0, linestyle="--", alpha=0.6,
                       label=f"Period mean  {avg:.2f}×")
            ax.legend(fontsize=9, frameon=False)
        style_ax(ax, title, ylabel, color)
        ax.yaxis.set_major_formatter(fmt)
        add_split_lines(ax, markers, split_start)
        _note(fig, note)
        _save(fig, name)

    dollar_fmt = mticker.FormatStrFormatter("$%.0f")
    multiple_fmt = mticker.FuncFormatter(lambda x, _: f"{x:.1f}×")
    basis_note = "Restated onto the share basis after the latest split in split_history."

    # Book value per share: as reported and split-adjusted
    bvps = edgar["book_value_per_share"].dropna()
    _line_chart(bvps, _dates(bvps.index),
                "PGR — Book Value Per Share (Monthly, As-Reported)", "$ per share", BLUE,
                dollar_fmt, "pgr_book_value_per_share.png",
                "Month-end book value per share from the monthly 8-K, on the share basis "
                "in effect at the time.", first_edgar)
    bvps_adj = edgar["book_value_per_share_split_adjusted"].dropna()
    _line_chart(bvps_adj, _dates(bvps_adj.index),
                "PGR — Book Value Per Share (Monthly, Split-Adjusted)",
                "$ per share (current basis)", BLUE, dollar_fmt,
                "pgr_book_value_per_share_split_adjusted.png",
                f"Month-end book value per share from the monthly 8-K. {basis_note}",
                first_edgar)

    # Share repurchase volume (as reported)
    volume = edgar["shares_repurchased"].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(_dates(volume.index), volume.to_list(), width=20, color=ORANGE, alpha=0.85)
    style_ax(ax, "PGR — Share Repurchase Volume (Monthly, As-Reported)",
             "Shares Repurchased (millions)", ORANGE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    add_split_lines(ax, markers, first_edgar, ymax_frac=0.88)
    _note(fig, "Shares repurchased in the month, on the share basis in effect at the time. "
               "2006-05 (split month): 10-Q pre-split shares × 4 + post-split shares.")
    _save(fig, "pgr_share_repurchase_volume.png")

    # Repurchase dollars: full axis and capped axis
    dollars = edgar["repurchase_dollars"].dropna()
    estimated = edgar.loc[dollars.index, "avg_cost_source"] == AVG_COST_ESTIMATED
    peak_date = dollars.idxmax()
    peak_val = float(dollars.max())
    event = REPURCHASE_EVENTS.get(peak_date.strftime("%Y-%m"), "largest month")
    dollar_note = (
        "Repurchase $ = shares repurchased × average cost per share (monthly 8-K; "
        "2006-05 from the Q2 2006 10-Q)."
    )
    if estimated.any():
        dollar_note += (" Hatched: average cost not reported, estimated as the mean of "
                        "the month's weekly closes on the month-end share basis.")

    def _dollar_bars(ax):
        ax.bar(_dates(dollars.index[~estimated]), dollars[~estimated].to_list(), width=20,
               color=GREEN, alpha=0.85, label="Reported average cost")
        if estimated.any():
            ax.bar(_dates(dollars.index[estimated]), dollars[estimated].to_list(), width=20,
                   color="white", edgecolor=GREEN, hatch="////", linewidth=0.8,
                   label="Estimated average cost")
            ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig, ax = plt.subplots(figsize=(12, 5))
    _dollar_bars(ax)
    style_ax(ax, "PGR — Share Repurchase Dollar Amount (Monthly)", "Repurchase $ (millions)", GREEN)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    add_split_lines(ax, markers, first_edgar, ymax_frac=0.88)
    ax.annotate(f"{peak_date:%b %Y}\n{event}\n${peak_val:,.0f}M",
                xy=(peak_date.date(), peak_val),
                xytext=(peak_date.date() + datetime.timedelta(days=1100), peak_val * 0.70),
                fontsize=7.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
    _note(fig, dollar_note)
    _save(fig, "pgr_repurchase_dollar_amount.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    _dollar_bars(ax)
    style_ax(ax, f"PGR — Share Repurchase Dollar Amount (Monthly, axis capped at "
                 f"${REPURCHASE_AXIS_CAP:,.0f}M)", "Repurchase $ (millions)", GREEN)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.set_ylim(0, REPURCHASE_AXIS_CAP)
    add_split_lines(ax, markers, first_edgar, ymax_frac=0.88)
    for clipped_date, clipped_val in dollars[dollars > REPURCHASE_AXIS_CAP].items():
        label = REPURCHASE_EVENTS.get(clipped_date.strftime("%Y-%m"), "")
        ax.annotate(
            f"{clipped_date:%b %Y} {label}: ${clipped_val:,.0f}M\n(bar clipped — exceeds axis)",
            xy=(clipped_date.date(), REPURCHASE_AXIS_CAP),
            xytext=(clipped_date.date() + datetime.timedelta(days=1100),
                    REPURCHASE_AXIS_CAP * 0.70),
            fontsize=8, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, alpha=0.85),
        )
    _note(fig, dollar_note)
    _save(fig, "pgr_repurchase_dollar_amount_capped.png")

    # Share price: as reported and split-adjusted, plotted at the bar date
    priced = monthly.dropna(subset=["price"])
    x_price = _dates(priced["price_date"])
    _line_chart(priced["price"], x_price,
                "PGR — Share Price (Last Weekly Close of Each Month, As-Reported)",
                "$ per share", RED, dollar_fmt, "pgr_share_price.png",
                f"{PRICE_NOTE} Not split-adjusted.", priced["price_date"].min())
    _line_chart(priced["price_split_adjusted"], x_price,
                "PGR — Share Price (Last Weekly Close of Each Month, Split-Adjusted)",
                "$ per share (current basis)", RED, dollar_fmt,
                "pgr_share_price_split_adjusted.png",
                f"{PRICE_NOTE} {basis_note} Dividends not included.",
                priced["price_date"].min())

    # Price / book: as reported and split-adjusted
    pb_note = "P/B = last weekly close of the month ÷ month-end book value per share."
    pb = edgar["price_to_book"].dropna()
    _line_chart(pb, _dates(pb.index),
                "PGR — Price / Book Value Multiple (Monthly, As-Reported)", "P/B multiple",
                PURPLE, multiple_fmt, "pgr_price_to_book.png",
                f"{pb_note} Both on the share basis in effect at the time.", first_edgar,
                mean=True)
    pb_adj = edgar["price_to_book_split_adjusted"].dropna()
    _line_chart(pb_adj, _dates(pb_adj.index),
                "PGR — Price / Book Value Multiple (Monthly, Split-Adjusted)", "P/B multiple",
                PURPLE, multiple_fmt, "pgr_price_to_book_split_adjusted.png",
                f"{pb_note} Price and book value both restated onto the latest share basis.",
                first_edgar, mean=True)

    return written


def print_summary(frames: dict[str, object]) -> None:
    monthly: pd.DataFrame = frames["monthly"]
    edgar = monthly[monthly["edgar_month"]]
    series = {
        "Book value per share": edgar["book_value_per_share"].dropna(),
        "Repurchase volume": edgar["shares_repurchased"].dropna(),
        "Repurchase $ amount": edgar["repurchase_dollars"].dropna(),
        "Share price": monthly["price"].dropna(),
        "Price / Book": edgar["price_to_book"].dropna(),
    }
    print()
    print("─" * 60)
    for label, values in series.items():
        print(f"{label:<21}: {len(values):>4} obs  "
              f"{values.index.min().date()} → {values.index.max().date()}")
    pb = series["Price / Book"]
    print()
    print(f"Latest BVPS          : ${series['Book value per share'].iloc[-1]:.2f}")
    print(f"Latest repurchase vol: {series['Repurchase volume'].iloc[-1]:.3f}M shares")
    print(f"Latest repurchase $  : ${series['Repurchase $ amount'].iloc[-1]:.1f}M  "
          f"(avg cost ${edgar['avg_cost_per_share'].iloc[-1]:.2f})")
    print(f"Latest share price   : ${series['Share price'].iloc[-1]:.2f}")
    print(f"Latest P/B           : {pb.iloc[-1]:.2f}×  "
          f"(mean {pb.mean():.2f}×, min {pb.min():.2f}×, max {pb.max():.2f}×)")
    print("─" * 60)


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
