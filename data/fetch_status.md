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
