"""
PGR Annual Capital Return: Share Repurchases vs. Dividends
Chart 1 — total dollars returned ($B), stacked bar.
Chart 2 — same dollars as % of year-end market cap, stacked bar.
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

# Year-end (December) prices for market-cap denominator.
# For years with no December record (e.g. partial 2026), the last available
# month is used instead — handled below after grouping.
cur.execute("""
    SELECT strftime('%Y', date) AS yr,
           date,
           close
    FROM daily_prices
    WHERE ticker = 'PGR'
    ORDER BY date
""")
all_price_rows = cur.fetchall()
conn.close()

# Collapse to the last trading date of December for each year.
# Fall back to the last available date in any month for years without December.
from collections import defaultdict as _dd
_by_year_all   = _dd(list)   # year -> all (date, close)
_by_year_dec   = _dd(list)   # year -> December (date, close) only
for yr, date_str, close in all_price_rows:
    _by_year_all[yr].append((date_str, close))
    if date_str[5:7] == "12":
        _by_year_dec[yr].append((date_str, close))

yearend_price = {}   # year -> close price
for yr in sorted(set(_by_year_all.keys())):
    bucket = _by_year_dec[yr] if _by_year_dec[yr] else _by_year_all[yr]
    yearend_price[yr] = sorted(bucket)[-1][1]  # last date in bucket

annual_dividend = defaultdict(float)  # year -> $M

for ex_date, amount in div_rows:
    # Q1 dividends (Jan–Mar) are distributions of the prior year's earnings;
    # attribute them to the prior calendar year so the chart aligns with the
    # performance period rather than the payment date.
    year_int = int(ex_date[:4])
    month    = int(ex_date[5:7])
    if month <= 3:
        year_int -= 1
    year = str(year_int)
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
PARTIAL_YEARS = {"2004": "Aug–Dec 2004", "2026": "Jan–Apr 2026 (repurchases only)"}

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

# ── 9. Console summary (Chart 1) ─────────────────────────────────────────────
print()
print("─" * 60)
print(f"{'Year':<6}  {'Repurchases ($B)':>17}  {'Dividends ($B)':>15}  {'Total ($B)':>11}")
print("─" * 60)
for y, r, d in zip(years, rep_vals, div_vals):
    marker = " ★" if y in PARTIAL_YEARS else ""
    print(f"{y:<6}  {r:>17.2f}  {d:>15.2f}  {r+d:>11.2f}{marker}")
print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2: Capital Return as % of Year-End Market Cap
# ══════════════════════════════════════════════════════════════════════════════

# ── 10. Compute year-end market caps ─────────────────────────────────────────
# market_cap_M = year-end close price × shares outstanding (M)
# For the partial year 2026 use the latest available price & EDGAR month.
mktcap_M = {}   # year -> $M

for y in years:
    price = yearend_price.get(y)
    # shares_at expects YYYY-MM; use December for full years, latest for 2026
    ym_key = f"{y}-12" if y != "2026" else shares_timeline[-1][0]
    shares = shares_at(ym_key)
    if price is None or shares is None:
        continue
    mktcap_M[y] = price * shares  # $M

# ── 11. Express returns as % of year-end market cap ──────────────────────────
rep_pct = []
div_pct = []
years_mktcap = []   # only years where market cap is available

for y in years:
    mc = mktcap_M.get(y)
    if mc is None or mc == 0:
        continue
    rep_pct.append(annual_repurchase.get(y, 0.0) / mc * 100)
    div_pct.append(annual_dividend.get(y, 0.0)   / mc * 100)
    years_mktcap.append(y)

year_ints_mc = [int(y) for y in years_mktcap]

# ── 12. Build chart ───────────────────────────────────────────────────────────
BLUE2   = "#1f77b4"   # repurchases
ORANGE2 = "#ff7f0e"   # dividends

fig2, ax2 = plt.subplots(figsize=(14, 6))

ax2.bar(year_ints_mc, rep_pct, width=BAR_WIDTH, color=BLUE2,   alpha=0.85, label="Share Repurchases")
ax2.bar(year_ints_mc, div_pct, width=BAR_WIDTH, color=ORANGE2, alpha=0.85, label="Dividends",
        bottom=rep_pct)

# Partial-year annotations
for y, label in PARTIAL_YEARS.items():
    if y in years_mktcap:
        xi = int(y)
        mc = mktcap_M.get(y, 1)
        total_pct = (annual_repurchase.get(y, 0.0) + annual_dividend.get(y, 0.0)) / mc * 100
        ax2.annotate(
            f"★ {label}",
            xy=(xi, total_pct),
            xytext=(xi, total_pct + 0.3),
            ha="center", va="bottom", fontsize=7.5, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        )

ax2.set_title("PGR — Annual Capital Returned as % of Year-End Market Cap",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("% of Market Cap", fontsize=11)
ax2.set_xlabel("")
ax2.set_xticks(year_ints_mc)
ax2.set_xticklabels(years_mktcap, fontsize=9, rotation=45, ha="right")
ax2.tick_params(axis="y", labelsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=10, frameon=False, loc="upper left")

fig2.tight_layout()

# ── 13. Save ──────────────────────────────────────────────────────────────────
out2_path = os.path.join(OUT_DIR, "pgr_capital_return_pct_marketcap.png")
fig2.savefig(out2_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2_path}")

# ── 14. Console summary (Chart 2) ────────────────────────────────────────────
print()
print("─" * 72)
print(f"{'Year':<6}  {'Mkt Cap ($B)':>12}  {'Repurch %':>10}  {'Div %':>8}  {'Total %':>8}")
print("─" * 72)
for y, rp, dp in zip(years_mktcap, rep_pct, div_pct):
    mc_b = mktcap_M[y] / 1000
    marker = " ★" if y in PARTIAL_YEARS else ""
    print(f"{y:<6}  {mc_b:>12.1f}  {rp:>10.2f}  {dp:>8.2f}  {rp+dp:>8.2f}{marker}")
print("─" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3: Combined Ratio vs. Capital Returned — XY Scatter (two panels)
#
# Left panel  — raw dollars ($B): shows the absolute scale of returns; the
#               time-color gradient reveals that recent years cluster high on Y
#               primarily because the company is much larger, not just because
#               CR improved.
# Right panel — % of year-end market cap: size-normalized, cleaner signal on
#               the CR→return relationship independent of company growth.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np  # for regression line

# ── 15. Build annual combined ratio (NPE-weighted average of monthly CRs) ────
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    SELECT
        strftime('%Y', month_end) AS yr,
        COUNT(*) AS months,
        SUM(combined_ratio * net_premiums_earned) / SUM(net_premiums_earned) AS cr_wtd
    FROM pgr_edgar_monthly
    WHERE combined_ratio IS NOT NULL AND net_premiums_earned IS NOT NULL
    GROUP BY yr
    ORDER BY yr
""")
annual_cr = {r[0]: (r[1], r[2]) for r in cur.fetchall()}  # year -> (months, cr)
conn.close()

# ── 16. Align scatter data (full years only: 2005–2025) ──────────────────────
scatter_years = [y for y in years if y >= "2005" and y <= "2025"
                 and annual_cr.get(y, (0, None))[0] == 12]

sc_cr    = [annual_cr[y][1]                                           for y in scatter_years]
sc_tot_B = [(annual_repurchase.get(y, 0) + annual_dividend.get(y, 0)) / 1000
             for y in scatter_years]
sc_pct   = [(annual_repurchase.get(y, 0) + annual_dividend.get(y, 0)) / mktcap_M[y] * 100
             if mktcap_M.get(y) else None for y in scatter_years]

# Remove any year where % is unavailable
valid = [(y, cr, tb, pct) for y, cr, tb, pct in
         zip(scatter_years, sc_cr, sc_tot_B, sc_pct) if pct is not None]
scatter_years, sc_cr, sc_tot_B, sc_pct = map(list, zip(*valid))

year_ints_sc = [int(y) for y in scatter_years]

# ── 17. Color gradient (dark blue → orange, older → newer) ───────────────────
cmap = plt.get_cmap("plasma")
n = len(scatter_years)
colors = [cmap(0.15 + 0.70 * i / (n - 1)) for i in range(n)]

# ── 18. Regression helpers ────────────────────────────────────────────────────
def _reg_line(xs, ys):
    coef = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(min(xs), max(xs), 200)
    return x_fit, np.polyval(coef, x_fit), coef

# ── 19. Build chart ───────────────────────────────────────────────────────────
fig3, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
fig3.subplots_adjust(wspace=0.10)

MARKER_SIZE = 90

# Label nudge table: (dx_cr, dy) in data units to avoid overlap
# Computed manually for legibility
_nudge_B = {
    "2007": (+0.3, +0.10), "2020": (+0.3, +0.05), "2025": (+0.3, +0.12),
    "2022": (-0.5, -0.12), "2005": (+0.3, +0.05),
}
_nudge_pct = {
    "2007": (-0.6, +0.5),  "2020": (+0.3, +0.2), "2025": (+0.3, +0.3),
    "2022": (-0.5, -0.3),  "2021": (+0.3, -0.2),
}

for ax_s, y_vals, y_label, y_fmt, nudge_map in [
    (axL, sc_tot_B, "Total Capital Returned ($B)",
     lambda x, _: f"${x:.1f}B", _nudge_B),
    (axR, sc_pct,  "Total Capital Returned (% of Year-End Mkt Cap)",
     lambda x, _: f"{x:.1f}%", _nudge_pct),
]:
    # Scatter points, colored by time
    for i, (y, cr, val, c) in enumerate(zip(scatter_years, sc_cr, y_vals, colors)):
        ax_s.scatter(cr, val, s=MARKER_SIZE, color=c, zorder=3,
                     edgecolors="white", linewidths=0.6)

        # Label placement with optional nudge
        dx, dy = nudge_map.get(y, (0.18, 0.0))
        ax_s.annotate(
            y[2:],   # last two digits: '05', '07' etc.
            xy=(cr, val),
            xytext=(cr + dx, val + dy),
            fontsize=7.5, color="#333333", va="center",
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.5)
            if (dx**2 + dy**2) > 0.1 else None,
        )

    # Regression line
    x_fit, y_fit, coef = _reg_line(sc_cr, y_vals)
    ax_s.plot(x_fit, y_fit, color="#888888", linewidth=1.2,
              linestyle="--", alpha=0.7, zorder=1)

    # Axes + style
    ax_s.invert_xaxis()
    ax_s.set_xlabel("Combined Ratio  ← Better underwriting", fontsize=11)
    ax_s.set_ylabel(y_label, fontsize=10)
    ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
    ax_s.grid(linestyle="--", alpha=0.35)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Reference line at 96% CR
    ax_s.axvline(96, color="#cc4444", linewidth=0.9, linestyle=":", alpha=0.6,
                 zorder=0)
    ylim = ax_s.get_ylim()
    ax_s.text(96.1, ylim[0] + (ylim[1]-ylim[0])*0.97,
              "96%", fontsize=7, color="#cc4444", va="top")

axL.set_title("Combined Ratio vs. Capital Returned ($B)", fontsize=12,
              fontweight="bold", pad=10)
axR.set_title("Combined Ratio vs. Capital Returned (% of Mkt Cap)", fontsize=12,
              fontweight="bold", pad=10)

# Colorbar (year gradient legend)
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(int(scatter_years[0]),
                                               int(scatter_years[-1])))
sm.set_array([])
cbar = fig3.colorbar(sm, ax=[axL, axR], orientation="vertical",
                     fraction=0.015, pad=0.02)
cbar.set_label("Year", fontsize=9)
cbar.set_ticks([int(scatter_years[0]), 2010, 2015, 2020, int(scatter_years[-1])])

fig3.tight_layout(rect=[0, 0, 0.97, 1])

# ── 20. Save ──────────────────────────────────────────────────────────────────
out3_path = os.path.join(OUT_DIR, "pgr_cr_vs_capital_return.png")
fig3.savefig(out3_path, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3_path}")
