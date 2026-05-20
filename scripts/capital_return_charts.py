"""
PGR Annual Capital Return: Share Repurchases vs. Dividends
Stacked bar chart comparing total dollars returned to shareholders per calendar year.
Saves to results/research/.
"""

import sqlite3
import os
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_SHARE_UNIT_ERROR_THRESHOLD = 25.0  # same guard used in repurchase_timeseries_charts.py

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pgr_financials.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "research")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load EDGAR monthly data ────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT
        month_end,
        COALESCE(common_shares_outstanding,
                 CASE WHEN book_value_per_share > 0
                      THEN shareholders_equity / book_value_per_share
                      ELSE NULL END) AS shares_M,
        shares_repurchased,
        avg_cost_per_share
    FROM pgr_edgar_monthly
    ORDER BY month_end
""")
edgar_rows = cur.fetchall()

# Build ordered list of (month_str, shares_M) for nearest-neighbor lookup
shares_timeline = [(r[0][:7], r[1]) for r in edgar_rows if r[1] is not None]

def shares_at(ym):
    """Return shares outstanding (M) for the EDGAR month closest to ym (YYYY-MM)."""
    for i, (m, s) in enumerate(shares_timeline):
        if m >= ym:
            # Use this month, or the previous one if it's closer
            if i > 0:
                prev_m, prev_s = shares_timeline[i - 1]
                if abs(int(ym.replace("-", "")) - int(prev_m.replace("-", ""))) < \
                   abs(int(m.replace("-", "")) - int(ym.replace("-", ""))):
                    return prev_s
            return s
    # ym is beyond all known months — use the last available
    return shares_timeline[-1][1]

# ── 2. Compute annual repurchase dollars ──────────────────────────────────────
from collections import defaultdict

annual_repurchase = defaultdict(float)  # year -> $M

for month_end, _shares, shares_repurch, avg_cost in edgar_rows:
    year = month_end[:4]
    if shares_repurch is None or avg_cost is None:
        continue
    if shares_repurch > _SHARE_UNIT_ERROR_THRESHOLD and avg_cost > 10:
        dollars_M = shares_repurch  # stored as $M
    else:
        dollars_M = shares_repurch * avg_cost
    annual_repurchase[year] += dollars_M

# ── 3. Compute annual dividend dollars ────────────────────────────────────────
cur.execute("""
    SELECT ex_date, amount
    FROM daily_dividends
    WHERE ticker = 'PGR'
    ORDER BY ex_date
""")
div_rows = cur.fetchall()
conn.close()

annual_dividend = defaultdict(float)  # year -> $M

for ex_date, amount in div_rows:
    year = ex_date[:4]
    ym = ex_date[:7]
    shares = shares_at(ym)
    if shares is None:
        continue
    annual_dividend[year] += amount * shares  # $M

# ── 4. Align years to repurchase data range (2004 onwards) ───────────────────
years = sorted(set(annual_repurchase.keys()) | set(annual_dividend.keys()))
years = [y for y in years if y >= "2004"]

rep_vals  = [annual_repurchase.get(y, 0.0) / 1000 for y in years]  # convert to $B
div_vals  = [annual_dividend.get(y, 0.0)   / 1000 for y in years]  # convert to $B
year_ints = [int(y) for y in years]

# Identify partial years: 2004 (data starts Aug) and 2026 (ongoing)
PARTIAL_YEARS = {"2004": "Aug–Dec 2004", "2026": "Jan–Apr 2026"}

# ── 5. Build chart ────────────────────────────────────────────────────────────
GREEN  = "#2ca02c"   # repurchases — matches existing chart style
BLUE   = "#1f77b4"   # dividends

BAR_WIDTH = 0.65

fig, ax = plt.subplots(figsize=(14, 6))

bars_rep = ax.bar(year_ints, rep_vals,  width=BAR_WIDTH, color=GREEN, alpha=0.85, label="Share Repurchases")
bars_div = ax.bar(year_ints, div_vals,  width=BAR_WIDTH, color=BLUE,  alpha=0.85, label="Dividends",
                  bottom=rep_vals)

# ── 6. Annotate partial years ─────────────────────────────────────────────────
for y, label in PARTIAL_YEARS.items():
    if y in years:
        xi = int(y)
        total = (annual_repurchase.get(y, 0.0) + annual_dividend.get(y, 0.0)) / 1000
        ax.annotate(
            f"★ {label}",
            xy=(xi, total),
            xytext=(xi, total + 0.25),
            ha="center", va="bottom", fontsize=7.5, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        )

# ── 7. Style ──────────────────────────────────────────────────────────────────
ax.set_title("PGR — Annual Capital Returned to Shareholders: Repurchases vs. Dividends",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("$ Billions", fontsize=11)
ax.set_xlabel("")
ax.set_xticks(year_ints)
ax.set_xticklabels(years, fontsize=9, rotation=45, ha="right")
ax.tick_params(axis="y", labelsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}B"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, frameon=False, loc="upper left")

fig.tight_layout()

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "pgr_repurchase_dividend_annual.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ── 9. Console summary ────────────────────────────────────────────────────────
print()
print("─" * 60)
print(f"{'Year':<6}  {'Repurchases ($B)':>17}  {'Dividends ($B)':>15}  {'Total ($B)':>11}")
print("─" * 60)
for y, r, d in zip(years, rep_vals, div_vals):
    marker = " ★" if y in PARTIAL_YEARS else ""
    print(f"{y:<6}  {r:>17.2f}  {d:>15.2f}  {r+d:>11.2f}{marker}")
print("─" * 60)
