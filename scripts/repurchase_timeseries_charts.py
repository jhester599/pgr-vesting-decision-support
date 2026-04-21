"""
PGR Monthly Time Series: Book Value Per Share, Share Repurchases, Share Price
Produces four line charts saved to results/research/.

Data quality note
─────────────────
Three months in pgr_edgar_monthly have a known unit error in shares_repurchased:
2024-10-31 (195.0), 2024-11-30 (51.0), and 2025-08-31 (87.0) appear to store
the dollar amount in $M rather than the share count in millions. The ingestion
parser hit a code path that treated a "$195M" text token as a raw share count.
These rows are corrected below by dividing by avg_cost_per_share to recover the
implied share count, so the dollar series remains internally consistent.
"""

import sqlite3
import datetime
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# Threshold above which shares_repurchased is almost certainly a dollar amount
# (max plausible monthly buyback for PGR is ~20M shares; 25 gives headroom)
_SHARE_UNIT_ERROR_THRESHOLD = 25.0

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pgr_financials.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "research")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load edgar monthly data ─────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT
        month_end,
        book_value_per_share,
        shares_repurchased,        -- millions of shares
        avg_cost_per_share         -- dollars per share
    FROM pgr_edgar_monthly
    ORDER BY month_end
""")
edgar_rows = cur.fetchall()

# ── 2. Load month-end share prices from daily_prices ──────────────────────────
# "Month-end" = the last available price record in each calendar month.
cur.execute("""
    SELECT
        strftime('%Y-%m', date) AS ym,
        date,
        close
    FROM daily_prices
    WHERE ticker = 'PGR'
    ORDER BY date
""")
all_prices = cur.fetchall()
conn.close()

# Collapse to one price per month (last trading date of each month)
price_by_month = {}
for ym, date_str, close in all_prices:
    price_by_month[ym] = (date_str, close)

# ── 3. Parse and compute series ───────────────────────────────────────────────
def parse_date(s):
    return datetime.date.fromisoformat(s)

dates_bvps, bvps_vals = [], []
dates_repvol, repvol_vals = [], []
dates_repdol, repdol_vals = [], []
dates_price, price_vals = [], []

for month_end, bvps, shares_repurch, avg_cost in edgar_rows:
    d = parse_date(month_end)
    ym = month_end[:7]

    if bvps is not None:
        dates_bvps.append(d)
        bvps_vals.append(bvps)

    if shares_repurch is not None and avg_cost is not None:
        # Correct rows where the dollar amount was stored in the shares column.
        # When shares_repurch > threshold and avg_cost is a plausible per-share
        # price (>$10), the value is treated as $M and converted back to shares.
        if shares_repurch > _SHARE_UNIT_ERROR_THRESHOLD and avg_cost > 10:
            corrected_shares = shares_repurch / avg_cost   # implied M shares
            corrected_dollars = shares_repurch             # already in $M
        else:
            corrected_shares = shares_repurch
            corrected_dollars = shares_repurch * avg_cost  # $M

        dates_repvol.append(d)
        repvol_vals.append(corrected_shares)

        dates_repdol.append(d)
        repdol_vals.append(corrected_dollars)

    elif shares_repurch is not None:
        # avg_cost missing; record shares only
        dates_repvol.append(d)
        repvol_vals.append(shares_repurch)

    if ym in price_by_month:
        dates_price.append(d)
        price_vals.append(price_by_month[ym][1])

# Also add price points for months before edgar coverage (pre-2004)
edgar_ym_set = {r[0][:7] for r in edgar_rows}
for ym, (date_str, close) in sorted(price_by_month.items()):
    if ym not in edgar_ym_set:
        d = parse_date(date_str)
        if d not in dates_price:
            dates_price.append(d)
            price_vals.append(close)

dates_price_sorted = sorted(zip(dates_price, price_vals))
dates_price = [x[0] for x in dates_price_sorted]
price_vals  = [x[1] for x in dates_price_sorted]


# ── 4. Shared style helpers ───────────────────────────────────────────────────
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"

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


# PGR stock split history
SPLIT_2002 = (datetime.date(2002, 4, 23), "3-for-1 split\n(Apr 2002)")
SPLIT_2006 = (datetime.date(2006, 5, 1),  "4-for-1 split\n(May 2006)")

def _add_split_annotation(ax, split_date, split_label, ymax_frac=0.92):
    """Draw a single vertical dashed split-line with label."""
    ax.axvline(split_date, color="#888888", linewidth=1.0, linestyle="--", alpha=0.7)
    ylim = ax.get_ylim()
    y_pos = ylim[0] + (ylim[1] - ylim[0]) * ymax_frac
    ax.text(split_date, y_pos, split_label,
            ha="left", va="top", fontsize=7.5, color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8))

def add_split_line(ax, ymax_frac=0.92):
    """Annotate the May 2006 4-for-1 split (used on charts whose data starts post-2002)."""
    _add_split_annotation(ax, *SPLIT_2006, ymax_frac=ymax_frac)

def add_both_split_lines(ax, ymax_frac=0.92):
    """Annotate both the Apr 2002 3-for-1 and May 2006 4-for-1 splits."""
    _add_split_annotation(ax, *SPLIT_2002, ymax_frac=ymax_frac)
    _add_split_annotation(ax, *SPLIT_2006, ymax_frac=ymax_frac)


# ── 5. Chart 1: Book Value Per Share ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_bvps, bvps_vals, color=BLUE, linewidth=1.8)
ax.fill_between(dates_bvps, bvps_vals, alpha=0.10, color=BLUE)
style_ax(ax, "PGR — Book Value Per Share (Monthly, As-Reported)", "$ per share", BLUE)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
add_split_line(ax)
fig.tight_layout()
out1 = os.path.join(OUT_DIR, "pgr_book_value_per_share.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# ── 6. Chart 2: Share Repurchase Volume ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(dates_repvol, repvol_vals, width=20, color=ORANGE, alpha=0.85)
style_ax(ax, "PGR — Share Repurchase Volume (Monthly, As-Reported)", "Shares Repurchased (millions)", ORANGE)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
add_split_line(ax, ymax_frac=0.88)
fig.tight_layout()
out2 = os.path.join(OUT_DIR, "pgr_share_repurchase_volume.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")


# ── 7. Chart 3: Repurchase Dollar Amount ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(dates_repdol, repdol_vals, width=20, color=GREEN, alpha=0.85)
style_ax(ax, "PGR — Share Repurchase Dollar Amount (Monthly)", "Repurchase $ (millions)", GREEN)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
add_split_line(ax, ymax_frac=0.88)
# Annotate the Oct-2004 large ASR
asr_date = datetime.date(2004, 10, 31)
asr_val  = next(v for d, v in zip(dates_repdol, repdol_vals) if d == asr_date)
ax.annotate("Oct 2004\nASR $1.49B",
            xy=(asr_date, asr_val), xytext=(asr_date + datetime.timedelta(days=600), asr_val * 0.90),
            fontsize=7.5, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
fig.tight_layout()
out3 = os.path.join(OUT_DIR, "pgr_repurchase_dollar_amount.png")
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")


# ── 7b. Chart 3b: Repurchase Dollar Amount — capped at $400M ─────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(dates_repdol, repdol_vals, width=20, color=GREEN, alpha=0.85)
style_ax(ax, "PGR — Share Repurchase Dollar Amount (Monthly, axis capped at $400M)", "Repurchase $ (millions)", GREEN)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax.set_ylim(0, 400)
add_split_line(ax, ymax_frac=0.88)

# Outlier bar (Oct 2004, $1.49B) is clipped — mark it explicitly
ax.annotate(
    "Oct 2004 ASR: $1,487M\n(bar clipped — exceeds axis)",
    xy=(asr_date, 400), xytext=(asr_date + datetime.timedelta(days=500), 355),
    fontsize=8, color="#2ca02c",
    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.85),
)
# Draw a small upward-pointing arrow stub above the clipped bar to signal truncation
ax.annotate("", xy=(asr_date, 400), xytext=(asr_date, 385),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))

fig.tight_layout()
out3b = os.path.join(OUT_DIR, "pgr_repurchase_dollar_amount_capped.png")
fig.savefig(out3b, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3b}")


# ── 8. Chart 4: Share Price ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_price, price_vals, color=RED, linewidth=1.8)
ax.fill_between(dates_price, price_vals, alpha=0.08, color=RED)
style_ax(ax, "PGR — Share Price (Monthly, As-Reported / Not Split-Adjusted)", "$ per share", RED)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
add_both_split_lines(ax)  # price data starts 1999 — both 2002 and 2006 splits visible
fig.tight_layout()
out4 = os.path.join(OUT_DIR, "pgr_share_price.png")
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4}")


# ── 9. Console summary ────────────────────────────────────────────────────────
print()
print("─" * 60)
print(f"Book value per share : {len(bvps_vals):>4} obs  "
      f"{dates_bvps[0]} → {dates_bvps[-1]}")
print(f"Repurchase volume    : {len(repvol_vals):>4} obs  "
      f"{dates_repvol[0]} → {dates_repvol[-1]}")
print(f"Repurchase $ amount  : {len(repdol_vals):>4} obs  "
      f"{dates_repdol[0]} → {dates_repdol[-1]}")
print(f"Share price          : {len(price_vals):>4} obs  "
      f"{dates_price[0]} → {dates_price[-1]}")
print()
print(f"Latest BVPS          : ${bvps_vals[-1]:.2f}")
print(f"Latest repurchase vol: {repvol_vals[-1]:.3f}M shares")
print(f"Latest repurchase $  : ${repdol_vals[-1]:.1f}M  "
      f"(avg cost ${edgar_rows[-1][3]:.2f})")
print(f"Latest share price   : ${price_vals[-1]:.2f}")
print("─" * 60)
