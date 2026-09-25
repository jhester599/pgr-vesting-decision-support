# Review 2026-09-25, step 4 — price-feature rewrite (WP2)

Findings F01, F15, the Monte Carlo part of F19 and the TA part of F24 in
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md). No data changed:
this step is code only, and the committed DB is untouched.

## How the September decision was replayed

Both runs used `python scripts/monthly_decision.py --dry-run --as-of 2026-09-21 --skip-fred`,
the as-of date of the committed September run. Each ran in its own checkout
with its own copy of the committed DB (sha256 `c032cf5f…d1ed`, identical before
and after each run; dry runs open the DB read-only).

- **Before:** `master` at `89dc3e8` (steps 1–3b merged).
- **After:** this branch.

Both runs are on the same data, which already carries the step 2/3a/3b
rebuilds. The review's own replay predates those rebuilds, so its baseline
numbers are different (see the last section).

## Result

The recommendation does not change: **DEFER-TO-TAX-DEFAULT, sell 50 %**,
consensus NEUTRAL (LOW confidence). The OOS-R² and CPCV gates fail in both
runs (F04 and F02, step 7).

### Aggregate health and gates

| Metric | Before | After |
|---|---|---|
| Pooled (aggregate) IC | 0.1528 | 0.1424 |
| Pooled (aggregate) hit rate | 68.7 % | 63.8 % |
| Quality-weighted mean IC (gate ≥ 0.07) | 0.1042 PASS | 0.0845 PASS |
| Quality-weighted mean hit rate (gate ≥ 55 %) | 66.8 % PASS | 64.5 % PASS |
| Aggregate OOS R² (gate ≥ 2 %) | −0.44 % FAIL | −2.16 % FAIL |
| Representative CPCV | FAIL | FAIL |
| Mean predicted 6M relative return | −2.37 % | −1.64 % |
| P(outperform), calibrated | 59.1 % | 60.3 % |
| Consensus signal | NEUTRAL (LOW) | NEUTRAL (LOW) |
| Signals OUTPERFORM / NEUTRAL / UNDERPERFORM | 3 / 2 / 3 | 2 / 4 / 2 |
| Recommendation mode / sell % | DEFER-TO-TAX-DEFAULT / 50 % | DEFER-TO-TAX-DEFAULT / 50 % |
| Classification shadow P(actionable sell) | 45.7 % (LOW, aligned) | 48.1 % (LOW, aligned) |

### Per benchmark

| Benchmark | Predicted before | Predicted after | Signal before | Signal after | IC before | IC after | Hit before | Hit after |
|---|---|---|---|---|---|---|---|---|
| VOO | +3.63 % | +0.61 % | OUTPERFORM | NEUTRAL | 0.051 | 0.086 | 62.3 % | 60.5 % |
| VXUS | −3.80 % | +3.02 % | NEUTRAL | NEUTRAL | 0.003 | −0.010 | 69.0 % | 65.3 % |
| VWO | −2.67 % | −0.98 % | NEUTRAL | NEUTRAL | 0.021 | 0.005 | 62.2 % | 61.9 % |
| VMBS | +4.26 % | +4.56 % | OUTPERFORM | NEUTRAL | 0.063 | −0.009 | 75.0 % | 68.7 % |
| BND | +5.15 % | +2.71 % | OUTPERFORM | OUTPERFORM | 0.174 | 0.134 | 70.8 % | 67.9 % |
| GLD | −1.02 % | +2.26 % | UNDERPERFORM | OUTPERFORM | 0.183 | 0.104 | 61.3 % | 60.5 % |
| DBC | −10.62 % | −8.95 % | UNDERPERFORM | UNDERPERFORM | 0.187 | 0.167 | 72.0 % | 67.9 % |
| VDE | −11.99 % | −9.29 % | UNDERPERFORM | UNDERPERFORM | 0.059 | 0.089 | 61.0 % | 61.8 % |

VXUS and GLD forecasts change sign; VOO and VMBS drop from OUTPERFORM to
NEUTRAL, and GLD moves from UNDERPERFORM to OUTPERFORM.

## What changed in the inputs

### Live decision row (2026-08-31)

| Feature | Live model | Before | After |
|---|---|---|---|
| `mom_12m` | Ridge, GBT | +1.304 | −0.115 |
| `mom_6m` | GBT | +0.057 | +0.023 |
| `mom_3m` | GBT | −0.184 | +0.148 |
| `vol_63d` | Ridge, GBT | 0.556 | 0.346 |
| `high_52w` | — | 0.764 | 0.881 |

`book_value_per_share_growth_yoy`, `pb_ratio`, `pe_ratio`, `buyback_yield` and
the synthetic spreads are unchanged on this row: no split falls in their
windows.

### History (321 common monthly rows, 1999-12 → 2026-09)

| Feature | Rows changed | Correlation, before vs after | Max abs change |
|---|---|---|---|
| `mom_3m` | 320 | 0.393 | 1.24 |
| `mom_6m` | 317 | 0.402 | 1.31 |
| `mom_12m` | 311 | 0.515 | 2.24 |
| `vol_63d` | 321 | 0.140 | 2.65 |
| `high_52w` | 157 | 0.510 | 0.85 |
| `book_value_per_share_growth_yoy` | 12 (2006-07 → 2007-06) | 0.606 | 0.91 |
| `pb_ratio` | 2 (2006-05, 2006-06) | 0.981 | 2.53 |
| `buyback_yield` | 2 (2006-05, 2006-06) | 0.980 | 0.20 |
| `pgr_vs_kie_6m` | 12 | 0.764 | 0.71 |
| `pgr_vs_peers_6m` | 13 | 0.787 | 0.81 |
| `pgr_vs_vfh_6m` | 6 | 0.860 | 0.71 |
| `commodity_equity_momentum` | 6 | 0.627 | 1.15 |

`vwo_vxus_spread_6m` and `gold_vs_treasury_6m` do not change: VXUS starts in
2011, after VWO's 2008 split, and GLD and BND have no splits. The matrix also
gains one early row (1999-12), because calendar momentum needs less burn-in
than a 63-row shift.

At the 2006 split (2006-05-31): `mom_12m` −0.797 → +0.140, `vol_63d` 2.774 →
0.138, and `book_value_per_share_growth_yoy` for 2006-07 is −0.704 → +0.183.

### TA shadow features and Monte Carlo

- `ta_pgr_natr_63d` at 2006-05: 0.195 → 0.033 (the review's corrected value is
  about 0.037). `ta_ratio_roc_6m_vwo` at 2008-06: −0.733 → +0.155.
  - The weekly bar that contains a split mixes share bases: PGR 2006-05-19 has
    high 108.63 (pre-split) and low 26.79 (post-split).
  - `split_adjusted_ohlcv` restates each of open, high and low on the basis
    nearest the bar's own range. Adjusting by bar date alone had left
    NATR at 0.264.
- TA shadow aggregate P(actionable sell):
  - `ta_minimal_replacement`: 48.1 % → 48.7 %;
  - `ta_minimal_plus_vwo_pct_b`: 46.6 % → 47.9 %.
- Monte Carlo volatility for PGR as of 2026-09-21: **0.949 → 0.265**. The new
  value uses the last 52 split-adjusted weekly returns × √52; the old one used
  the full history × √252. The replay has no lot file, so the Monte Carlo
  section itself did not run.

## Comparison with the review's estimate

The review estimated that correcting the four momentum and volatility
features would move pooled IC from 0.165 to 0.104 and quality-weighted IC
from 0.113 to 0.052, failing the 0.07 gate.

Measured here on the current data:
- pooled IC 0.153 → 0.142;
- quality-weighted IC 0.104 → 0.085, which still passes the gate.

The two replays differ in two ways:
- The review's baseline predates the target rebuild (step 2: VOO/VGT splits,
  BME windows), the FRED lag fix (3a) and the EDGAR repair (3b). All three
  changed the training data.
- This step also restates BVPS growth (a live Ridge feature) and the
  synthetic spreads.

The review saw the consensus flip on 3 of 6 past vest dates. That was not
re-measured here, because the brief was the September replay only.

Model selection is unchanged. The v18/v20 feature sets were chosen with the
mis-specified features and are re-evaluated in step 8a.
