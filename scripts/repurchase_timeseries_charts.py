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
    build_monthly_frame,
    load_chart_inputs,
    split_markers,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pgr_financials.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "research")

CHART_FILES: tuple[str, ...] = (
    "pgr_book_value_per_share.png",
    "pgr_share_repurchase_volume.png",
    "pgr_repurchase_dollar_amount.png",
    "pgr_repurchase_dollar_amount_capped.png",
    "pgr_share_price.png",
    "pgr_price_to_book.png",
)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
PURPLE = "#9467bd"


def build_chart_frames(conn) -> dict[str, object]:
    """Return the monthly frame and the split markers the charts plot."""
    inputs = load_chart_inputs(conn)
    monthly = build_monthly_frame(inputs)
    markers = split_markers(inputs.splits, monthly.index.min(), monthly.index.max())
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


def _dates(index) -> list[datetime.date]:
    return [d.date() for d in index]


def plot_charts(frames: dict[str, object], out_dir: str) -> list[str]:
    """Write every chart in ``CHART_FILES`` to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    monthly: pd.DataFrame = frames["monthly"]
    markers = frames["split_markers"]
    edgar = monthly[monthly["edgar_month"]]
    first_edgar = edgar.index.min()
    written = []

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        written.append(path)

    # Chart 1: Book Value Per Share
    bvps = edgar["book_value_per_share"].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(_dates(bvps.index), bvps.to_list(), color=BLUE, linewidth=1.8)
    ax.fill_between(_dates(bvps.index), bvps.to_list(), alpha=0.10, color=BLUE)
    style_ax(ax, "PGR — Book Value Per Share (Monthly, As-Reported)", "$ per share", BLUE)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    add_split_lines(ax, markers, first_edgar)
    fig.tight_layout()
    _save(fig, "pgr_book_value_per_share.png")

    # Chart 2: Share Repurchase Volume
    volume = edgar["shares_repurchased"].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(_dates(volume.index), volume.to_list(), width=20, color=ORANGE, alpha=0.85)
    style_ax(ax, "PGR — Share Repurchase Volume (Monthly, As-Reported)",
             "Shares Repurchased (millions)", ORANGE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    add_split_lines(ax, markers, first_edgar, ymax_frac=0.88)
    fig.tight_layout()
    _save(fig, "pgr_share_repurchase_volume.png")

    # Chart 3: Repurchase Dollar Amount
    dollars = edgar["repurchase_dollars"].dropna()
    peak_date = dollars.idxmax()
    peak_val = float(dollars.max())
    asr_date = peak_date.date()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(_dates(dollars.index), dollars.to_list(), width=20, color=GREEN, alpha=0.85)
    style_ax(ax, "PGR — Share Repurchase Dollar Amount (Monthly)", "Repurchase $ (millions)", GREEN)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    add_split_lines(ax, markers, first_edgar, ymax_frac=0.88)
    ax.annotate("Oct 2004\nASR $1.49B",
                xy=(asr_date, peak_val),
                xytext=(asr_date + datetime.timedelta(days=600), peak_val * 0.90),
                fontsize=7.5, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
    fig.tight_layout()
    _save(fig, "pgr_repurchase_dollar_amount.png")

    # Chart 3b: Repurchase Dollar Amount — capped at $400M
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(_dates(dollars.index), dollars.to_list(), width=20, color=GREEN, alpha=0.85)
    style_ax(ax, "PGR — Share Repurchase Dollar Amount (Monthly, axis capped at $400M)",
             "Repurchase $ (millions)", GREEN)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.set_ylim(0, 400)
    add_split_lines(ax, markers, first_edgar, ymax_frac=0.88)
    ax.annotate(
        "Oct 2004 ASR: $1,487M\n(bar clipped — exceeds axis)",
        xy=(asr_date, 400), xytext=(asr_date + datetime.timedelta(days=500), 355),
        fontsize=8, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.85),
    )
    ax.annotate("", xy=(asr_date, 400), xytext=(asr_date, 385),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
    fig.tight_layout()
    _save(fig, "pgr_repurchase_dollar_amount_capped.png")

    # Chart 4: Share Price (EDGAR months at month-end, earlier months at bar date)
    price = monthly["price"].dropna()
    x_price = [
        m.date() if monthly.at[m, "edgar_month"] else monthly.at[m, "price_date"].date()
        for m in price.index
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_price, price.to_list(), color=RED, linewidth=1.8)
    ax.fill_between(x_price, price.to_list(), alpha=0.08, color=RED)
    style_ax(ax, "PGR — Share Price (Monthly, As-Reported / Not Split-Adjusted)",
             "$ per share", RED)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    add_split_lines(ax, markers, price.index.min())
    fig.tight_layout()
    _save(fig, "pgr_share_price.png")

    # Chart 5: Price / Book Value Multiple
    pb = edgar["price_to_book"].dropna()
    pb_mean = float(pb.mean())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(_dates(pb.index), pb.to_list(), color=PURPLE, linewidth=1.8)
    ax.fill_between(_dates(pb.index), pb.to_list(), alpha=0.10, color=PURPLE)
    ax.axhline(pb_mean, color=PURPLE, linewidth=1.0, linestyle="--", alpha=0.6,
               label=f"Period mean  {pb_mean:.2f}×")
    ax.legend(fontsize=9, frameon=False)
    style_ax(ax, "PGR — Price / Book Value Multiple (Monthly)", "P/B multiple", PURPLE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    add_split_lines(ax, markers, first_edgar)
    fig.tight_layout()
    _save(fig, "pgr_price_to_book.png")

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
