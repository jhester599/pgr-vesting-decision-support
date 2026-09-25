# Review 2026-09-25, step 2 — target rebuild diff (WP1 + WP4)

Produced by `scripts/rebuild_relative_returns.py --attribute` on a copy of the
committed DB (sha256 `a6c3c99b…efe0` before, `ab5651bc…9260` after). No API
calls were made. The rebuilt copy is the committed `data/pgr_financials.db`.
Row-level data: [`2026-09-25_step2_target_rebuild_rows.csv`](2026-09-25_step2_target_rebuild_rows.csv)
(one row per changed target: `before`, `after`, `delta`, and the split into
`delta_data` and `delta_window`).

## What the rebuild did

1. Seeded `split_history` from `config.KNOWN_SPLITS`. This adds VOO
   2013-10-24 (1-for-2, ratio 0.5) and VGT 2026-04-21 (8-for-1). Both were
   verified against issuer notices; the evidence is in `config/splits.py`.
2. Deleted 31 superseded partial-week bars from `daily_prices` (29,797 →
   29,766 rows), keeping the latest bar per ticker-ISO-week:
   - 2026-03-24 (18 tickers);
   - 2026-03-30 (the four peers);
   - 2026-04-30 (9 ETFs).
3. Rebuilt both horizons with windows ending at the last bar on or before
   `BMonthEnd(t + h)`, replacing the whole table.

Only `daily_prices`, `split_history` and `monthly_relative_returns` changed.
Every other table is identical, and `PRAGMA integrity_check` is `ok`.

## Headline

- **Rows:** 9,658 before and 9,658 after. None were added or removed, and
  **863 changed** (790 at 6M, 73 at 12M).
- **Split rows (F03/F05):**
  - VOO has 18 rows (6 at 6M, 12 at 12M) off by +106 to +134 pp (the stored values were −1.14 to −1.47).
  - VGT has 10 rows (6M 2025-10-31 … 2026-02-27, 12M 2025-04-30 …
    2025-08-29) off by −92 to −140 pp. **All 10 change sign.** Two more VGT
    rows (6M 2025-09-30, 12M 2025-03-31) move by less than 0.002 pp through
    the duplicate-bar fix. Both match the review.
- **Duplicate bars (F22):** removing them changes rows at every benchmark
  whose window started or ended in 2025-09 … 2026-02 (6M) or
  2025-03 … 2025-08 (12M), by at most 1.65 pp.
  - Before, the ETF window could end on the partial Thursday 2026-04-30 bar
    while PGR's ended 2026-04-24.
  - Before, a 2026-03-24 ex-dividend could be reinvested at the partial
    Tuesday close.
- **Window change (F22):** 770 rows move (757 at 6M, 13 at 12M), by a mean of
  1.3–2.5 pp. 23 of them change sign.
  - 6M rows move when `t + 6 months` lands before the month's last weekly
    bar. Most start in February (Feb → Aug) or November (Nov → May).
  - 12M windows almost never moved: `t + 12 months` lands on the same
    month-end except around leap days.

## Per benchmark and horizon (changed rows only)

Benchmark/horizon pairs not listed have no changed rows. "Split/bar" is the
change from steps 1–2 under the legacy window. "Window" is the change from the
new window end, given the corrected data.

| Benchmark | h | Changed | > 1 pp | Split/bar rows | Max split/bar Δ (pp) | Window rows | Mean window Δ (pp) | Max Δ (pp) | Max Δ date | Sign flips |
|---|---|---|---|---|---|---|---|---|---|---|
| BND | 6 | 36 | 20 | 0 | 0.00 | 36 | 1.91 | 12.08 | 2008-04-30 | 0 |
| BNDX | 6 | 25 | 14 | 0 | 0.00 | 25 | 1.70 | 6.14 | 2024-02-29 | 0 |
| DBC | 6 | 38 | 28 | 0 | 0.00 | 38 | 2.29 | 7.86 | 2021-02-26 | 1 |
| DBC | 12 | 1 | 1 | 0 | 0.00 | 1 | 5.79 | 5.79 | 2007-02-28 | 0 |
| GLD | 6 | 39 | 29 | 0 | 0.00 | 39 | 2.51 | 11.78 | 2008-04-30 | 2 |
| GLD | 12 | 1 | 1 | 0 | 0.00 | 1 | 6.66 | 6.66 | 2007-02-28 | 0 |
| KIE | 6 | 38 | 23 | 0 | 0.00 | 38 | 1.85 | 8.99 | 2009-02-27 | 2 |
| KIE | 12 | 1 | 1 | 0 | 0.00 | 1 | 1.21 | 1.21 | 2007-02-28 | 0 |
| SCHD | 6 | 28 | 14 | 0 | 0.00 | 28 | 1.34 | 4.41 | 2024-02-29 | 0 |
| VCIT | 6 | 32 | 18 | 1 | 0.49 | 31 | 1.69 | 6.25 | 2024-02-29 | 1 |
| VCIT | 12 | 1 | 0 | 1 | 0.51 | 0 | 0.00 | 0.51 | 2025-04-30 | 0 |
| VDE | 6 | 42 | 31 | 6 | 0.02 | 39 | 2.22 | 8.04 | 2021-02-26 | 0 |
| VDE | 12 | 7 | 1 | 6 | 0.02 | 1 | 4.66 | 4.66 | 2007-02-28 | 0 |
| VEA | 6 | 36 | 22 | 0 | 0.00 | 36 | 1.88 | 6.41 | 2025-11-28 | 1 |
| VFH | 6 | 43 | 30 | 6 | 1.24 | 40 | 1.92 | 6.01 | 2009-01-30 | 3 |
| VFH | 12 | 7 | 1 | 6 | 1.38 | 1 | 0.55 | 1.38 | 2025-04-30 | 0 |
| VGT | 6 | 43 | 27 | 6 | 114.19 | 40 | 2.24 | 117.84 | 2025-11-28 | 5 |
| VGT | 12 | 7 | 6 | 6 | 140.36 | 1 | 1.79 | 140.36 | 2025-05-30 | 5 |
| VHT | 6 | 43 | 23 | 6 | 0.00 | 40 | 1.60 | 6.03 | 2008-04-30 | 2 |
| VHT | 12 | 7 | 1 | 6 | 0.00 | 1 | 1.65 | 1.65 | 2007-02-28 | 0 |
| VIG | 6 | 39 | 20 | 1 | 0.87 | 38 | 1.49 | 4.94 | 2025-11-28 | 2 |
| VIG | 12 | 2 | 2 | 1 | 1.01 | 1 | 1.38 | 1.38 | 2007-02-28 | 0 |
| VIS | 6 | 42 | 27 | 6 | 1.34 | 39 | 1.94 | 5.78 | 2025-11-28 | 1 |
| VIS | 12 | 7 | 2 | 6 | 1.65 | 1 | 1.11 | 1.65 | 2025-04-30 | 0 |
| VMBS | 6 | 32 | 17 | 1 | 0.45 | 31 | 1.66 | 6.23 | 2024-02-29 | 0 |
| VMBS | 12 | 1 | 0 | 1 | 0.47 | 0 | 0.00 | 0.47 | 2025-04-30 | 0 |
| VNQ | 6 | 42 | 24 | 6 | 0.01 | 39 | 1.97 | 6.72 | 2009-02-27 | 0 |
| VNQ | 12 | 7 | 1 | 6 | 0.01 | 1 | 1.80 | 1.80 | 2007-02-28 | 0 |
| VOO | 6 | 34 | 21 | 6 | 116.04 | 28 | 1.68 | 116.04 | 2013-06-28 | 2 |
| VOO | 12 | 12 | 12 | 12 | 134.09 | 0 | 0.00 | 134.09 | 2012-12-31 | 1 |
| VPU | 6 | 43 | 26 | 6 | 1.38 | 40 | 1.58 | 5.58 | 2008-04-30 | 1 |
| VPU | 12 | 7 | 1 | 6 | 1.60 | 1 | 0.83 | 1.60 | 2025-04-30 | 0 |
| VTI | 6 | 46 | 26 | 1 | 0.64 | 45 | 1.65 | 5.95 | 2025-11-28 | 0 |
| VTI | 12 | 2 | 1 | 1 | 0.79 | 1 | 1.51 | 1.51 | 2007-02-28 | 0 |
| VWO | 6 | 40 | 27 | 1 | 0.17 | 39 | 2.41 | 7.28 | 2023-11-30 | 4 |
| VWO | 12 | 2 | 1 | 1 | 0.21 | 1 | 1.68 | 1.68 | 2007-02-28 | 0 |
| VXUS | 6 | 29 | 16 | 1 | 0.78 | 28 | 1.78 | 6.42 | 2025-11-28 | 1 |
| VXUS | 12 | 1 | 0 | 1 | 0.93 | 0 | 0.00 | 0.93 | 2025-04-30 | 0 |

## VOO and VGT split rows (|Δ| > 1 pp from splits)

| Benchmark | h | Date | Before | After | Δ (pp) |
|---|---|---|---|---|---|
| VGT | 6 | 2025-10-31 | +0.907 | -0.014 | -92.1 |
| VGT | 6 | 2025-11-28 | +0.774 | -0.404 | -117.8 |
| VGT | 6 | 2025-12-31 | +0.903 | -0.137 | -104.0 |
| VGT | 6 | 2026-01-30 | +0.877 | -0.195 | -107.2 |
| VGT | 6 | 2026-02-27 | +0.864 | -0.299 | -116.3 |
| VGT | 12 | 2025-04-30 | +0.616 | -0.724 | -134.0 |
| VGT | 12 | 2025-05-30 | +0.511 | -0.892 | -140.4 |
| VGT | 12 | 2025-06-30 | +0.731 | -0.486 | -121.8 |
| VGT | 12 | 2025-07-31 | +0.736 | -0.418 | -115.4 |
| VGT | 12 | 2025-08-29 | +0.770 | -0.440 | -121.0 |
| VOO | 6 | 2013-04-30 | -1.198 | -0.074 | +112.4 |
| VOO | 6 | 2013-05-31 | -1.141 | -0.022 | +111.8 |
| VOO | 6 | 2013-06-28 | -1.256 | -0.095 | +116.0 |
| VOO | 6 | 2013-07-31 | -1.200 | -0.135 | +106.5 |
| VOO | 6 | 2013-08-30 | -1.287 | -0.136 | +115.2 |
| VOO | 6 | 2013-09-30 | -1.303 | -0.195 | +110.8 |
| VOO | 12 | 2012-10-31 | -1.308 | -0.035 | +127.3 |
| VOO | 12 | 2012-11-30 | -1.275 | +0.028 | +130.3 |
| VOO | 12 | 2012-12-31 | -1.367 | -0.026 | +134.1 |
| VOO | 12 | 2013-01-31 | -1.352 | -0.142 | +121.0 |
| VOO | 12 | 2013-02-28 | -1.473 | -0.221 | +125.2 |
| VOO | 12 | 2013-03-29 | -1.432 | -0.223 | +121.0 |
| VOO | 12 | 2013-04-30 | -1.422 | -0.220 | +120.2 |
| VOO | 12 | 2013-05-31 | -1.385 | -0.182 | +120.3 |
| VOO | 12 | 2013-06-28 | -1.462 | -0.214 | +124.8 |
| VOO | 12 | 2013-07-31 | -1.419 | -0.226 | +119.4 |
| VOO | 12 | 2013-08-30 | -1.468 | -0.215 | +125.3 |
| VOO | 12 | 2013-09-30 | -1.420 | -0.225 | +119.5 |

## Not in this rebuild

- **2026 dividends are still missing** (F08). Every ETF has had no ex-date
  since 2025-12 or 2026-03, and PGR since 2026-04-02.
  `scripts/check_data_integrity.py` reports 21 stale tickers.
  - This session had no `AV_API_KEY`, so the backfill could not run here.
  - The first Wednesday `--dividend-refresh` run (or a `workflow_dispatch`
    with `dividend_refresh: true`) re-fetches all 22 due tickers within one
    day's budget, then rebuilds these targets.
  - Recent 6M/12M rows will then move by up to the ~2 pp the review estimated.
- Price features (F01) still use raw weekly closes; that is WP2.
