# PGR v2 Initial Fetch — Status Log

Appended each time `initial_fetch.py` runs (via GitHub Actions or locally).
Most-recent run appears **last**.

---

## How to read this log

| Status | Meaning |
|--------|---------|
| ✅ OK | Rows were fetched from Alpha Vantage and upserted to the DB |
| ⏭️ SKIPPED | Ticker already had fresh data; no new rows written |
| ❌ ERROR | HTTP or parse failure — check the Actions log for detail |

---

## Scheduled runs

| Workflow | Date | Time (local) | Mode |
|----------|------|--------------|------|
| `initial_fetch_prices.yml` | Mon 2026-03-23 | 6:00 AM EDT | `--prices` |
| `initial_fetch_dividends.yml` | Tue 2026-03-24 | 6:00 AM EDT | `--dividends` |

Results will be appended below automatically after each run.

## 2026-03-22 23:17 UTC — `prices` *(dry run)*  ✅ SUCCESS

- **AV calls used:** 21 / 25
- **Duration:** 0s
- **Tickers attempted:** 21
- **Loaded new data:** 0
- **Skipped (no new data):** 21
- **Errors:** 0

| Ticker | Mode | Rows | Status | Detail |
|--------|------|-----:|--------|--------|
| `BND` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `BNDX` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `DBC` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `GLD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `PGR` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `SCHD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VCIT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VDE` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VEA` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VFH` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VGT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VHT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIG` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VMBS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VNQ` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VOO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VPU` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VTI` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VWO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VXUS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |

## 2026-03-22 23:17 UTC — `prices` *(dry run)*  ✅ SUCCESS

- **AV calls used:** 21 / 25
- **Duration:** 0s
- **Tickers attempted:** 21
- **Loaded new data:** 0
- **Skipped (no new data):** 21
- **Errors:** 0

| Ticker | Mode | Rows | Status | Detail |
|--------|------|-----:|--------|--------|
| `BND` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `BNDX` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `DBC` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `GLD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `PGR` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `SCHD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VCIT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VDE` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VEA` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VFH` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VGT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VHT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIG` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VMBS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VNQ` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VOO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VPU` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VTI` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VWO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VXUS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |

## 2026-03-22 23:17 UTC — `prices` *(dry run)*  ✅ SUCCESS

- **AV calls used:** 21 / 25
- **Duration:** 0s
- **Tickers attempted:** 21
- **Loaded new data:** 0
- **Skipped (no new data):** 21
- **Errors:** 0

| Ticker | Mode | Rows | Status | Detail |
|--------|------|-----:|--------|--------|
| `BND` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `BNDX` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `DBC` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `GLD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `PGR` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `SCHD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VCIT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VDE` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VEA` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VFH` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VGT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VHT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIG` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VMBS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VNQ` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VOO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VPU` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VTI` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VWO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VXUS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |

## 2026-03-22 23:18 UTC — `prices` *(dry run)*  ✅ SUCCESS

- **AV calls used:** 21 / 25
- **Duration:** 0s
- **Tickers attempted:** 21
- **Loaded new data:** 0
- **Skipped (no new data):** 21
- **Errors:** 0

| Ticker | Mode | Rows | Status | Detail |
|--------|------|-----:|--------|--------|
| `BND` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `BNDX` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `DBC` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `GLD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `PGR` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `SCHD` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VCIT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VDE` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VEA` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VFH` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VGT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VHT` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIG` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VIS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VMBS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VNQ` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VOO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VPU` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VTI` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VWO` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |
| `VXUS` | prices | 0 | ⏭️ DRY-RUN | no HTTP call made |

## 2026-03-25 15:02 UTC — `prices + FRED`  ✅ SUCCESS

- **AV calls used:** 22 / 25
- **Duration:** 280s
- **Tickers attempted:** 22
- **Loaded new data:** 18
- **Skipped (no new data):** 4

- **Errors:** 0

| Ticker | Mode | Rows | Status | Detail |
|--------|------|-----:|--------|--------|
| `BND` | prices | 989 | ✅ OK | 989 rows upserted |
| `BNDX` | prices | 668 | ✅ OK | 668 rows upserted |
| `DBC` | prices | 1,050 | ✅ OK | 1050 rows upserted |
| `GLD` | prices | 0 | ⏭️ SKIPPED | already fresh or no new data |
| `KIE` | prices | 1,062 | ✅ OK | 1062 rows upserted |
| `PGR` | prices | 1,377 | ✅ OK | 1377 rows upserted |
| `SCHD` | prices | 0 | ⏭️ SKIPPED | already fresh or no new data |
| `VCIT` | prices | 852 | ✅ OK | 852 rows upserted |
| `VDE` | prices | 1,121 | ✅ OK | 1121 rows upserted |
| `VEA` | prices | 0 | ⏭️ SKIPPED | already fresh or no new data |
| `VFH` | prices | 1,156 | ✅ OK | 1156 rows upserted |
| `VGT` | prices | 1,156 | ✅ OK | 1156 rows upserted |
| `VHT` | prices | 1,156 | ✅ OK | 1156 rows upserted |
| `VIG` | prices | 0 | ⏭️ SKIPPED | already fresh or no new data |
| `VIS` | prices | 1,121 | ✅ OK | 1121 rows upserted |
| `VMBS` | prices | 852 | ✅ OK | 852 rows upserted |
| `VNQ` | prices | 1,121 | ✅ OK | 1121 rows upserted |
| `VOO` | prices | 811 | ✅ OK | 811 rows upserted |
| `VPU` | prices | 1,156 | ✅ OK | 1156 rows upserted |
| `VTI` | prices | 1,295 | ✅ OK | 1295 rows upserted |
| `VWO` | prices | 1,098 | ✅ OK | 1098 rows upserted |
| `VXUS` | prices | 791 | ✅ OK | 791 rows upserted |

## 2026-03-26 16:07 UTC — `dividends`  ✅ SUCCESS

- **AV calls used:** 22 / 25
- **Duration:** 278s
- **Tickers attempted:** 22
- **Loaded new data:** 21
- **Skipped (no new data):** 1

- **Errors:** 0

| Ticker | Mode | Rows | Status | Detail |
|--------|------|-----:|--------|--------|
| `BND` | dividends | 227 | ✅ OK | 227 rows upserted |
| `BNDX` | dividends | 153 | ✅ OK | 153 rows upserted |
| `DBC` | dividends | 9 | ✅ OK | 9 rows upserted |
| `GLD` | dividends | 0 | ⏭️ SKIPPED | already fresh or no new data |
| `KIE` | dividends | 82 | ✅ OK | 82 rows upserted |
| `PGR` | dividends | 76 | ✅ OK | 76 rows upserted |
| `SCHD` | dividends | 58 | ✅ OK | 58 rows upserted |
| `VCIT` | dividends | 196 | ✅ OK | 196 rows upserted |
| `VDE` | dividends | 54 | ✅ OK | 54 rows upserted |
| `VEA` | dividends | 65 | ✅ OK | 65 rows upserted |
| `VFH` | dividends | 86 | ✅ OK | 86 rows upserted |
| `VGT` | dividends | 54 | ✅ OK | 54 rows upserted |
| `VHT` | dividends | 54 | ✅ OK | 54 rows upserted |
| `VIG` | dividends | 79 | ✅ OK | 79 rows upserted |
| `VIS` | dividends | 54 | ✅ OK | 54 rows upserted |
| `VMBS` | dividends | 196 | ✅ OK | 196 rows upserted |
| `VNQ` | dividends | 86 | ✅ OK | 86 rows upserted |
| `VOO` | dividends | 62 | ✅ OK | 62 rows upserted |
| `VPU` | dividends | 86 | ✅ OK | 86 rows upserted |
| `VTI` | dividends | 99 | ✅ OK | 99 rows upserted |
| `VWO` | dividends | 61 | ✅ OK | 61 rows upserted |
| `VXUS` | dividends | 56 | ✅ OK | 56 rows upserted |
