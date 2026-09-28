# R2-lite: dividend refresh check — v187 (2026-09-28)

Scope: R2-lite only, per the
[amended execution plan](PRE_V200_FIX_PROMPTS_codex.md#r2-lite--verify-the-dividend-refresh-offline).
Evidence: [verification V03 / N1](VERIFICATION_2026-09-26.md) and the
original findings F03, F05, F08 and F22 in the
[repository review](REPO_REVIEW_2026-09-25.md). This is a docs-only record.
No code, test, workflow, config or DB file changes.

**Result: the exit gate holds.**

- No required ticker (PGR and the 8 primary benchmarks) is STALE. GLD is a
  non-payer and reports NO_HISTORY.
- Five worked hand checks match the stored targets to at most 6.7e-16. An
  independent recompute of all 9,658 stored rows matches to at most
  1.6e-15, on both the before and the after DB.
- Every one of the 273 changed target rows is explained:
  - 223 by a dividend the refresh added or revised inside the row's window;
  - 50 DBC rows by float noise of at most 4.4e-16, from DBC's revised
    2018–2025 amounts.
- One caveat, below: VWO has no March 2026 ex-date. The provider's full
  VWO history does not contain one either. The owner accepted the gap as
  provider data on 2026-09-28.

## Pins and isolation

- Latest `master` when the session started:
  `ed7997f6f540f664e59dd44a4079616d74a8e8cb`. This is the refresh commit
  itself.
- Branch: `claude/confident-ritchie-6y711i`.
- Linux, Python 3.11.15, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1,
  pytest 9.0.2.
- DB work used two read-only copies, extracted with `git show` into the
  session scratch directory outside the repository:
  - `before.db`, from `1ff7d7c:data/pgr_financials.db`;
  - `after.db`, from `ed7997f:data/pgr_financials.db`.
- Every connection to a copy was
  `sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)`.
- No fetcher, provider call, monthly decision or e-mail ran. The only
  provider calls were the owner's dispatched workflow run described below.
- Nothing in this session wrote to the tracked DB. Its sha256 was the same
  before and after the full suite (see "Commands and results").
- No holdout outcome was opened.
- The replay and the new governance baseline are R3b's job
  ([D5](PRE_V200_FIX_PROMPTS_codex.md#owner-decisions-and-execution-plan-2026-09-27)),
  so `docs/model-governance.md` is unchanged.

## 1. Commits, hashes and the integrity step

Exactly one commit changed `data/pgr_financials.db` after 2026-09-27:

| Item | Value |
|---|---|
| Refresh commit | `ed7997f6f540f664e59dd44a4079616d74a8e8cb`, "chore: weekly data update 2026-09-28 [skip ci]", by `github-actions[bot]` at 2026-09-28 10:38:02 UTC |
| First parent (DB before) | `1ff7d7cad43d8b45d28604f5b11187777fba46f0`, the merge of PR #143 (v192) |
| Workflow run | "Weekly Data Accumulation" run #29, id `36410317252`, `workflow_dispatch` by the owner; head `1ff7d7c`; conclusion **success** |
| Mode | `dividend_refresh=true`, `dry_run=false`; step "Run weekly fetch" ran `python scripts/weekly_fetch.py --dividend-refresh` |
| DB before, sha256 | `f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d` (git blob `115a321a32d950aaa86c17403f2a8f0b131977b2`) |
| DB after, sha256 | `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6` (git blob `3f32d1571ebd7a6efd179e967495c49f07cf9d20`) |
| Tracked DB on `master` now | same as the after DB, `38991c76…` |

The before hash matches the pre-refresh DB pins of R3 (in
[`docs/model-governance.md`](../model-governance.md)) and of
[the R4 closeout](R4_shadow_closeout.md).

What the run's log reports, quoted from job `108888719045`:

- `AV used today 0/25; refreshing 21 due dividend tickers: ['GLD', 'VTI', 'VOO', 'VGT', 'VHT', 'VFH', 'VIS', 'VDE', 'VPU', 'KIE', 'VXUS', 'VEA', 'VWO', 'VIG', 'SCHD', 'BND', 'BNDX', 'VCIT', 'VMBS', 'VNQ', 'DBC']`
- `Dividends - 1874 rows upserted`
- `Split history seeded: 11 records upserted.` The split rows did not
  change (see section 5).
- `6M: 4892 rows written across 21 benchmarks`
- `12M: 4766 rows written across 21 benchmarks`
- `Dividend freshness: all tickers OK.`

The `api_request_log` table confirms the 21 Alpha Vantage calls, one
`DIVIDENDS/<ticker>` call per ticker dated 2026-09-28. That leaves 4 of the
25-call daily budget.

The step "Check price, split and dividend integrity" passed. It ran
`scripts/check_data_integrity.py --fail-on-stale-dividends` and printed:

```text
## Data integrity

- Price jumps: OK (every jump outside [0.6, 1.7] has a split row)
- One bar per ticker-week: OK
- Dividend freshness: OK
```

## 2. Diffs

### Whole-DB diff

The table below counts, for each table, the rows present only in the
before DB and only in the after DB (`EXCEPT` over all columns). The schema
(`sqlite_master`) is identical, and `PRAGMA integrity_check` is `ok` on both
copies.

| Table | Rows before | Rows after | Only before | Only after |
|---|---:|---:|---:|---:|
| `api_request_log` | 889 | 910 | 0 | 21 |
| `daily_dividends` | 2393 | 2450 | 5 | 62 |
| `daily_prices` | 29792 | 29792 | 0 | 0 |
| `fred_macro_monthly` | 7574 | 7574 | 0 | 0 |
| `ingestion_metadata` | 52 | 53 | 20 | 21 |
| `model_performance_log` | 8 | 8 | 0 | 0 |
| `model_retrain_log` | 18 | 18 | 0 | 0 |
| `monthly_relative_returns` | 9658 | 9658 | 273 | 273 |
| `pgr_edgar_filing_parses` | 266 | 266 | 0 | 0 |
| `pgr_edgar_monthly` | 265 | 265 | 0 | 0 |
| `pgr_edgar_monthly_raw` | 15329 | 15329 | 0 | 0 |
| `pgr_fundamentals_quarterly` | 73 | 73 | 0 | 0 |
| `schema_migrations` | 8 | 8 | 0 | 0 |
| `split_history` | 11 | 11 | 0 | 0 |
| `sqlite_sequence` | 2 | 2 | 0 | 0 |

Two tables changed only as bookkeeping:

- `api_request_log`: the 21 request counts.
- `ingestion_metadata`: `last_fetched` and `rows_stored` for the 20 tickers
  that already had a row, plus a new row for GLD, with `rows_stored` 0.
  `rows_stored` equals each ticker's full row count after the refresh (for
  example VWO 63), so each response was the ticker's whole history.

### `daily_dividends`

- 57 new rows. Every new row has `source = 'av'`.
- 5 rows had their amount revised in place.
- No rows removed.
- No PGR, peer (ALL, TRV, CB, HIG) or DBC row was added. GLD returned no
  rows.

New rows:

| Ticker | New rows | Ex-dates | Amounts (per share) |
|---|---:|---|---|
| BND | 6 | 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-03, 2026-09-01 | 0.250016, 0.241713, 0.247259, 0.2445, 0.2516, 0.252886 |
| BNDX | 6 | 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-03, 2026-09-01 | 0.1117, 0.1126, 0.1141, 0.1112, 0.1185, 0.1206 |
| KIE | 2 | 2026-06-22, 2026-09-21 | 0.214529, 0.184122 |
| SCHD | 2 | 2026-06-24, 2026-09-23 | 0.2525, 0.2665 |
| VCIT | 6 | 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-03, 2026-09-01 | 0.3438, 0.3262, 0.3412, 0.3319, 0.3421, 0.3401 |
| VDE | 2 | 2026-06-24, 2026-09-23 | 1.0321, 1.0519 |
| VEA | 2 | 2026-06-18, 2026-09-18 | 0.3772, 0.1775 |
| VFH | 2 | 2026-06-24, 2026-09-23 | 0.8519, 0.4461 |
| VGT | 2 | 2026-06-24, 2026-09-23 | 0.1384, 0.1465 |
| VHT | 2 | 2026-06-24, 2026-09-23 | 0.9868, 0.9685 |
| VIG | 3 | 2026-03-27, 2026-06-26, 2026-09-28 | 0.8334, 0.9988, 0.9295 |
| VIS | 2 | 2026-06-24, 2026-09-23 | 0.7599, 0.7022 |
| VMBS | 6 | 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-03, 2026-09-01 | 0.1617, 0.1609, 0.1634, 0.1645, 0.168, 0.1666 |
| VNQ | 2 | 2026-06-24, 2026-09-23 | 0.8554, 0.8046 |
| VOO | 3 | 2026-03-27, 2026-06-26, 2026-09-28 | 1.8724, 1.9622, 1.8226 |
| VPU | 2 | 2026-06-24, 2026-09-23 | 1.3009, 1.258 |
| VTI | 3 | 2026-03-27, 2026-06-26, 2026-09-28 | 0.9982, 1.0437, 0.9555 |
| VWO | 2 | 2026-06-18, 2026-09-18 | 0.071, 0.1137 |
| VXUS | 2 | 2026-06-18, 2026-09-18 | 0.3861, 0.1562 |

Amounts revised in place. These are provider revisions to existing keys,
not new events:

| Ticker | Ex-date | Amount before | Amount after |
|---|---|---:|---:|
| DBC | 2018-12-24 | 0.189 | 0.1885 |
| DBC | 2019-12-23 | 0.254 | 0.25383 |
| DBC | 2023-12-18 | 1.0893 | 1.08926 |
| DBC | 2025-12-22 | 0.7442 | 0.74424 |
| VXUS | 2026-03-20 | 0.08 | 0.0795 |

The new rows fall into two groups:

- **38 new rows lie inside a mature window.** They have ex-dates up to
  2026-08-03. The last end bar of any stored window is 2026-08-28: the 6M
  window anchored 2026-02-27 and the 12M window anchored 2025-08-29 both end
  there.
- **19 new rows lie inside no stored window yet.** They have ex-dates from
  2026-09-01 to 2026-09-28. They will enter targets as windows mature.
  - VOO, VTI and VIG go ex on 2026-09-28. That date is after every ticker's
    last price bar (2026-09-25), so no current window can contain it.
  - PGR 2026-10-01 is an already-stored scheduled payment, not part of this
    refresh.

### `monthly_relative_returns`

- The key set is identical: 9,658 `(date, benchmark, target_horizon)` rows
  before and after. No row was added or dropped.
- `pgr_return` and `proxy_fill` are unchanged on every row.
- 273 rows changed `benchmark_return`, and so `relative_return`: 131 at 6M
  and 142 at 12M.
- No `relative_return` changed sign.
- `relative_return = pgr_return − benchmark_return` and `pgr_return` did not
  move, so each row's `relative_return` change is exactly minus its
  `benchmark_return` change.

| Horizon | Benchmark | Rows changed | Anchor dates | Max abs change `benchmark_return` | Max abs change `relative_return` | Sign flips | Of which float-noise only (< 1e-14) |
|---:|---|---:|---|---:|---:|---:|---:|
| 6M | BND | 5 | 2025-10-31 to 2026-02-27 | 1.6400e-02 | 1.6400e-02 | 0 | 0 |
| 6M | BNDX | 5 | 2025-10-31 to 2026-02-27 | 1.1518e-02 | 1.1518e-02 | 0 | 0 |
| 6M | DBC | 54 | 2018-06-29 to 2025-12-31 | 3.3566e-05 | 3.3566e-05 | 0 | 30 |
| 6M | KIE | 3 | 2025-12-31 to 2026-02-27 | 4.0719e-03 | 4.0719e-03 | 0 | 0 |
| 6M | SCHD | 3 | 2025-12-31 to 2026-02-27 | 9.2790e-03 | 9.2790e-03 | 0 | 0 |
| 6M | VCIT | 5 | 2025-10-31 to 2026-02-27 | 1.9847e-02 | 1.9847e-02 | 0 | 0 |
| 6M | VDE | 3 | 2025-12-31 to 2026-02-27 | 8.3509e-03 | 8.3509e-03 | 0 | 0 |
| 6M | VEA | 3 | 2025-12-31 to 2026-02-27 | 5.8704e-03 | 5.8704e-03 | 0 | 0 |
| 6M | VFH | 3 | 2025-12-31 to 2026-02-27 | 7.3532e-03 | 7.3532e-03 | 0 | 0 |
| 6M | VGT | 3 | 2025-12-31 to 2026-02-27 | 1.5256e-03 | 1.5256e-03 | 0 | 0 |
| 6M | VHT | 3 | 2025-12-31 to 2026-02-27 | 3.8363e-03 | 3.8363e-03 | 0 | 0 |
| 6M | VIG | 6 | 2025-09-30 to 2026-02-27 | 8.7864e-03 | 8.7864e-03 | 0 | 0 |
| 6M | VIS | 3 | 2025-12-31 to 2026-02-27 | 2.5023e-03 | 2.5023e-03 | 0 | 0 |
| 6M | VMBS | 5 | 2025-10-31 to 2026-02-27 | 1.7105e-02 | 1.7105e-02 | 0 | 0 |
| 6M | VNQ | 3 | 2025-12-31 to 2026-02-27 | 1.0046e-02 | 1.0046e-02 | 0 | 0 |
| 6M | VOO | 6 | 2025-09-30 to 2026-02-27 | 6.8913e-03 | 6.8913e-03 | 0 | 0 |
| 6M | VPU | 3 | 2025-12-31 to 2026-02-27 | 7.2668e-03 | 7.2668e-03 | 0 | 0 |
| 6M | VTI | 6 | 2025-09-30 to 2026-02-27 | 6.8071e-03 | 6.8071e-03 | 0 | 0 |
| 6M | VWO | 3 | 2025-12-31 to 2026-02-27 | 1.2670e-03 | 1.2670e-03 | 0 | 0 |
| 6M | VXUS | 6 | 2025-09-30 to 2026-02-27 | 4.9538e-03 | 4.9538e-03 | 0 | 0 |
| 12M | BND | 5 | 2025-04-30 to 2025-08-29 | 1.7034e-02 | 1.7034e-02 | 0 | 0 |
| 12M | BNDX | 5 | 2025-04-30 to 2025-08-29 | 1.1847e-02 | 1.1847e-02 | 0 | 0 |
| 12M | DBC | 65 | 2017-12-29 to 2025-08-29 | 3.3966e-05 | 3.3966e-05 | 0 | 20 |
| 12M | KIE | 3 | 2025-06-30 to 2025-08-29 | 4.1581e-03 | 4.1581e-03 | 0 | 0 |
| 12M | SCHD | 3 | 2025-06-30 to 2025-08-29 | 1.0187e-02 | 1.0187e-02 | 0 | 0 |
| 12M | VCIT | 5 | 2025-04-30 to 2025-08-29 | 2.0636e-02 | 2.0636e-02 | 0 | 0 |
| 12M | VDE | 3 | 2025-06-30 to 2025-08-29 | 9.6671e-03 | 9.6671e-03 | 0 | 0 |
| 12M | VEA | 3 | 2025-06-30 to 2025-08-29 | 6.6440e-03 | 6.6440e-03 | 0 | 0 |
| 12M | VFH | 3 | 2025-06-30 to 2025-08-29 | 7.0533e-03 | 7.0533e-03 | 0 | 0 |
| 12M | VGT | 3 | 2025-06-30 to 2025-08-29 | 1.6047e-03 | 1.6047e-03 | 0 | 0 |
| 12M | VHT | 3 | 2025-06-30 to 2025-08-29 | 4.4889e-03 | 4.4889e-03 | 0 | 0 |
| 12M | VIG | 6 | 2025-03-31 to 2025-08-29 | 9.5842e-03 | 9.5842e-03 | 0 | 0 |
| 12M | VIS | 3 | 2025-06-30 to 2025-08-29 | 2.7371e-03 | 2.7371e-03 | 0 | 0 |
| 12M | VMBS | 5 | 2025-04-30 to 2025-08-29 | 1.7927e-02 | 1.7927e-02 | 0 | 0 |
| 12M | VNQ | 3 | 2025-06-30 to 2025-08-29 | 1.0276e-02 | 1.0276e-02 | 0 | 0 |
| 12M | VOO | 6 | 2025-03-31 to 2025-08-29 | 7.3740e-03 | 7.3740e-03 | 0 | 0 |
| 12M | VPU | 3 | 2025-06-30 to 2025-08-29 | 7.7730e-03 | 7.7730e-03 | 0 | 0 |
| 12M | VTI | 6 | 2025-03-31 to 2025-08-29 | 7.3221e-03 | 7.3221e-03 | 0 | 0 |
| 12M | VWO | 3 | 2025-06-30 to 2025-08-29 | 1.4200e-03 | 1.4200e-03 | 0 | 0 |
| 12M | VXUS | 6 | 2025-03-31 to 2025-08-29 | 5.5811e-03 | 5.5811e-03 | 0 | 0 |
| 6M | **total** | **131** | | | | **0** | **30** |
| 12M | **total** | **142** | | | | **0** | **20** |

Notes on the table:

- **Primary benchmarks** affected: VOO, VXUS, VWO, VMBS, BND, VDE and DBC.
  GLD has no dividends, so none of its rows changed.
- **Size.** All 148 non-DBC rows whose window contains a new dividend rose
  in `benchmark_return`, and fell in `relative_return`, by 0.12 to 2.06
  percentage points. Before the refresh, the newest targets understated
  these benchmarks' returns, as V03/N1 expected.
- **VXUS.** Six VXUS rows (three per horizon) contain only the revised
  2026-03-20 amount, and fell by 6.7e-6.
- **DBC** moved by at most 3.4e-5, from its four revised amounts. Section 5
  explains its float-noise rows.

## 3. Freshness

`db_client.check_dividend_freshness(conn)` was run with default arguments
(interval multiple 1.5, last 8 gaps) on both copies. This is the check that
`scripts/check_data_integrity.py` and the refresh itself run.

| Ticker | Required | Before: status (last ex-date) | After: status (last ex-date) | After: interval (d) / due by |
|---|---|---|---|---|
| PGR | yes | OK (2026-10-01) | OK (2026-10-01) | 91 / 2026-05-12 |
| VOO | yes | **STALE** (2025-12-22) | OK (2026-09-28) | 92.5 / 2026-05-10 |
| VXUS | yes | **STALE** (2026-03-20) | OK (2026-09-18) | 91 / 2026-05-12 |
| VWO | yes | **STALE** (2025-12-19) | OK (2026-09-18) | 91 / 2026-05-12 |
| VMBS | yes | **STALE** (2026-03-02) | OK (2026-09-01) | 30 / 2026-08-11 |
| BND | yes | **STALE** (2026-03-02) | OK (2026-09-01) | 30 / 2026-08-11 |
| GLD | yes | NO_HISTORY | NO_HISTORY | known non-payer |
| DBC | yes | OK (2025-12-22) | OK (2025-12-22) | 364 / 2025-03-28 |
| VDE | yes | **STALE** (2026-03-24) | OK (2026-09-23) | 91.5 / 2026-05-11 |
| VTI | no | **STALE** (2025-12-22) | OK (2026-09-28) | 92.5 / 2026-05-10 |
| VGT, VHT, VFH, VIS, VPU, VNQ | no | **STALE** (2026-03-24) | OK (2026-09-23) | 91.5 / 2026-05-11 |
| KIE | no | **STALE** (2026-03-23) | OK (2026-09-21) | 91 / 2026-05-12 |
| VEA | no | **STALE** (2026-03-20) | OK (2026-09-18) | 91 / 2026-05-12 |
| VIG | no | **STALE** (2025-12-22) | OK (2026-09-28) | 92.5 / 2026-05-10 |
| SCHD | no | **STALE** (2026-03-25) | OK (2026-09-23) | 91 / 2026-05-12 |
| BNDX, VCIT | no | **STALE** (2026-03-02) | OK (2026-09-01) | 30 / 2026-08-11 |
| ALL | no | OK (2026-08-31) | OK (2026-08-31) | 91 / 2026-05-12 |
| TRV | no | OK (2026-09-10) | OK (2026-09-10) | 91.5 / 2026-05-11 |
| CB | no | OK (2026-09-11) | OK (2026-09-11) | 91 / 2026-05-12 |
| HIG | no | OK (2026-09-01) | OK (2026-09-01) | 91 / 2026-05-12 |

Every ticker's last price date is 2026-09-25.

- **Before:** 6 required tickers were STALE (VOO, VXUS, VWO, VMBS, BND,
  VDE), and 13 others.
- **After:** no ticker is STALE. The only NO_HISTORY ticker is GLD, a known
  non-payer.
- **ALL.** V03 listed ALL as stale, but it was already OK in the before DB:
  the 2026-09-27 peer run (`e215be9`) had refreshed it. The dividend refresh
  did not fetch it.
- **Dependence on 2026-09-28 rows.** The OK results do not depend on the
  2026-09-28 ex-dates. With `as_of=date(2026, 9, 25)`, the last price date,
  the check still reports OK for every ticker except GLD (NO_HISTORY).

This before/after pair is the check's red/green evidence. The same code
reports the 6 required STALE tickers on the before DB and none on the after
DB. No code changed, so no counterfactual reversal is needed.

## 4. Hand check

The recompute uses Appendix C of
[`VERIFICATION_2026-09-26.md`](VERIFICATION_2026-09-26.md), with the
horizon `h` as a parameter. It uses only standard-library SQLite, calendar,
datetime and math calls, and no production return helper.

- **Start bar:** the last close on or before the month-end anchor.
- **End bar:** the last close on or before the last weekday of the month
  `h` months later.
- **Splits:** each `split_history` ratio with start < split date <= end
  multiplies the share count.
- **Dividends:** each `daily_dividends` amount with start < ex-date <= end
  is reinvested at the last close on or before its ex-date. It multiplies
  the share count by `1 + amount / close`.
- **Return:** `prod(factors) * end_close / start_close − 1`.

The inputs are raw closes (`daily_prices.close`), the canonical
`split_history` and the stored amounts. None of these five windows contains
a split. VOO's 2013 and VGT's 2024 splits were checked in Appendix C, and
they are covered again by the full-table recompute below.

All five windows contain a dividend that the refresh added. VXUS also
contains one it revised. Values are from the after DB.

**VOO, 6M, anchor 2026-02-27** (primary):

- Start bar 2026-02-27, close 631.04. End bar 2026-08-28, close 707.24.
- New dividend 2026-03-27, 1.8724, reinvested at the 2026-03-27 close
  582.96: factor 1.0032118841773021.
- New dividend 2026-06-26, 1.9622, reinvested at the 2026-06-26 close
  670.26: factor 1.002927520663623.

| | Manual | Stored after | Gap |
|---|---:|---:|---:|
| benchmark_return | 0.1276443375324725 | 0.12764433753247273 | −2.2e-16 |
| pgr_return | 0.024273049854432793 | 0.02427304985443257 | 2.2e-16 |
| relative_return | −0.10337128767803971 | −0.10337128767804016 | 4.4e-16 |

Before the refresh: benchmark 0.120753042596349, relative
−0.09647999274191643.

**BND, 12M, anchor 2025-08-29** (primary):

- Start bar 2025-08-29, close 73.8. End bar 2026-08-28, close 72.31.
- Seven existing monthly dividends, 2025-09-02 to 2026-03-02.
- Five new dividends:

| Ex-date | Amount | Reinvested at (date, close) | Factor |
|---|---:|---|---:|
| 2026-04-01 | 0.250016 | 2026-03-27, 73.11 | 1.0034197237040077 |
| 2026-05-01 | 0.241713 | 2026-05-01, 73.36 | 1.0032948882224646 |
| 2026-06-01 | 0.247259 | 2026-05-29, 73.46 | 1.0033658998094201 |
| 2026-07-01 | 0.2445 | 2026-06-26, 73.67 | 1.0033188543504818 |
| 2026-08-03 | 0.2516 | 2026-07-31, 72.23 | 1.0034833171812267 |

| | Manual | Stored after | Gap |
|---|---:|---:|---:|
| benchmark_return | 0.019217543135879422 | 0.019217543135878756 | 6.7e-16 |
| pgr_return | −0.05701824692633317 | −0.05701824692633317 | 0 |
| relative_return | −0.0762357900622126 | −0.07623579006221193 | −6.7e-16 |

Before the refresh: benchmark 0.002183365261110737, relative
−0.05920161218744391.

**VWO, 6M, anchor 2025-12-31** (primary):

- Start bar 2025-12-26, close 54.02. End bar 2026-06-26, close 58.58.
- New dividend 2026-06-18, 0.071, reinvested at the 2026-06-18 close 60.77.
- By hand: (58.58 / 54.02) × (1 + 0.071 / 60.77) − 1 = 1.08441318… ×
  1.00116834… − 1 = 0.0856801432…

| | Manual | Stored after | Gap |
|---|---:|---:|---:|
| benchmark_return | 0.08568014320965589 | 0.08568014320965589 | 0 |
| pgr_return | 0.05158167831521632 | 0.05158167831521632 | 0 |
| relative_return | −0.03409846489443957 | −0.03409846489443957 | 0 |

Before the refresh: benchmark 0.08441318030359124 (= 58.58 / 54.02 − 1),
relative −0.03283150198837492.

**VXUS, 6M, anchor 2025-12-31** (primary; one new and one revised dividend):

- Start bar 2025-12-26, close 75.85. End bar 2026-06-26, close 84.48.
- Revised dividend 2026-03-20, 0.0795 (was 0.08), at the close 74.71:
  factor 1.001064114576362.
- New dividend 2026-06-18, 0.3861, at the close 86.77: factor
  1.0044496945949062.

| | Manual | Stored after | Gap |
|---|---:|---:|---:|
| benchmark_return | 0.11992362043917337 | 0.11992362043917315 | 2.2e-16 |
| pgr_return | 0.05158167831521632 | 0.05158167831521632 | 0 |
| relative_return | −0.06834194212395706 | −0.06834194212395683 | −2.2e-16 |

Before the refresh: benchmark 0.11496983237403935, relative
−0.06338815405882303.

**VDE, 12M, anchor 2025-08-29** (primary):

- Start bar 2025-08-29, close 126.89. End bar 2026-08-28, close 176.56.
- Three existing dividends, 2025-09-24, 2025-12-17 and 2026-03-24.
- New dividend 2026-06-24, 1.0321, at the 2026-06-18 close 151.82: factor
  1.0067981820576999.

| | Manual | Stored after | Gap |
|---|---:|---:|---:|
| benchmark_return | 0.43167776457350193 | 0.4316777645735017 | 2.2e-16 |
| pgr_return | −0.05701824692633317 | −0.05701824692633317 | 0 |
| relative_return | −0.4886960114998351 | −0.4886960114998349 | −2.2e-16 |

Before the refresh: benchmark 0.42201067710256557, relative
−0.47902892402889874.

All five match to at most 6.7e-16, below the 1e-12 requirement.

**Full-table recompute.** The same function was run on every one of the
9,658 stored rows, for PGR, the benchmark and the relative return, on both
copies. The largest gap between the manual value and the stored value is:

- after DB: 1.554e-15;
- before DB: 1.554e-15.

So both the pre-refresh targets (with their missing dividends) and the
post-refresh targets are exactly what the stored raw prices, splits and
dividends imply.

## 5. Nothing else moved

- **Prices and splits.**
  - `daily_prices` and `split_history` are row-for-row identical. The
    refresh's "11 records upserted" re-seeded the same canonical splits.
  - `find_unexplained_price_jumps` returns 0 rows on both copies, and
    `find_duplicate_week_bars` returns 0 rows on both copies.
  - Every required ticker's last bar is 2026-09-25 on both copies.
- **FRED.** `fred_macro_monthly` is row-for-row identical. There are 0
  duplicate `(series_id, calendar month)` groups on both copies.
- **EDGAR.** `pgr_edgar_monthly`, `pgr_edgar_monthly_raw`,
  `pgr_edgar_filing_parses` and `pgr_fundamentals_quarterly` are
  row-for-row identical. The identities therefore cannot have changed. The
  full suite also re-ran them on the tracked DB, which is now the after DB,
  in `tests/integration/data/test_pgr_edgar_integrity.py`:
  - revenue − expenses = pretax;
  - CR = LR + ER;
  - equity ≈ BVPS × shares;
  - monthly NI reconciles to the XBRL quarters;
  - PIF;
  - accession tracing.
- **Every changed target row is explained.** Prices and splits did not
  change. So for every row, (1 + benchmark_return after) / (1 +
  benchmark_return before) must equal the product, over the refresh's new
  or revised dividends inside the window, of (1 + new / p) / (1 + old / p).
  Here p is the reinvestment close, and old = 0 for a new row.
  - Checked on all 9,658 rows, the largest deviation is 4.4e-16.
  - 223 changed rows have at least one delta dividend in their window:
    - 142 rows: a new dividend only;
    - 75 rows: a revised amount only (69 DBC, and 6 VXUS, three per
      horizon);
    - 6 rows: both (VXUS, three per horizon).
  - No unchanged row has a new or revised dividend in its window.
  - **The 50 remaining changed rows are all DBC.** No delta dividend falls
    inside their windows, and each changed by at most 4.4e-16 (1–2 ulp).
    - Cause: production (`build_etf_monthly_returns`) computes each window
      as a ratio of one cumulative DRIP share-count series that starts at
      the ticker's first bar. Revising DBC's 2018-12-24 and later amounts
      rescales that series from 2018-12-24 on.
    - That leaves every later window's ratio unchanged in exact arithmetic,
      but not in binary floating point.
    - All 50 anchors fall after 2018-12-24. The ratio predicted for these
      rows is exactly 1.

## Caveats and open items

1. **VWO has no March 2026 ex-date.**
   - VWO has a March ex-date every year from 2013 to 2025. For example,
     2025-03-21 paid 0.0468. The stored history now goes from 2025-12-19
     straight to 2026-06-18, a 181-day gap against a 91-day cadence.
   - The provider's response is the whole VWO history (63 rows, all
     stored), and it has no 2026-03 row. So the refresh stored everything
     it received. It was not a partial fetch.
   - The freshness check looks only at the latest ex-date. It cannot see a
     gap in the middle, so VWO passes.
   - No provider or other source was queried to confirm whether VWO paid
     in March 2026; that is outside R2-lite.
   - Affected rows: the 12 VWO targets whose windows span 2026-03-20:
     - 6M anchors 2025-09-30 to 2026-02-27;
     - 12M anchors 2025-03-31 to 2025-08-29.
   - Illustration only: a March 2026 payment the size of March 2025's
     (0.0468 at the 2026-03-20 close of 52.56) would raise those
     benchmark returns by about 0.09 percentage points.
   - **Owner decision (2026-09-28): accept the gap as-is, as provider
     data.** The 12 VWO targets stay as stored. No source will be queried
     and no row added, so R3b uses this DB unchanged.
   - The audit flagged every gap since 2014 longer than 1.5 times the
     ticker's median gap. For PGR, the ETF benchmarks and DBC, the only
     one without a known cause is VWO's. The others are cadence changes or
     known patterns:
     - PGR paid annually until 2019;
     - VGT, VHT, VIS and VDE paid annually until 2014 and quarterly from
       2015-09;
     - DBC paid nothing from 2009 to 2017;
     - the bond funds' mid-December to February gap is their usual
       year-end pattern.
2. **Provider revisions.** DBC's four amounts and VXUS's 2026-03-20 amount
   were revised by the provider. The DBC changes are at the fourth or fifth
   decimal. They are recorded above and cause the DBC row changes of up to
   3.4e-5.
3. **No mature target contains the 19 new ex-dates from 2026-09-01 to
   2026-09-28 yet.** The 2026-09-28 rows for VOO, VTI and VIG fall after
   the last price bar. They are realised on their ex-date, but no stored
   window reaches them yet.
4. **R3b** runs the refreshed-DB replay and sets the new governance
   baseline, on this after DB (`38991c76…`), or on the blob at `ed7997f`
   if a later weekly update changes the tracked DB.

These are data checks. They do not show any change in forecast skill or
investment performance.

## Commands and results

Commands, run from the repository root unless noted. `$S` is the session
scratch directory outside the repository:
`/tmp/claude-0/-home-user-pgr-vesting-decision-support/8b6ccdb7-721f-5218-86e4-0288da281b88/scratchpad`.

| Command | Exit | Result |
|---|---:|---|
| `git fetch origin master`; `git log origin/master --since=2026-09-26` | 0 | the refresh commit `ed7997f` is the only DB commit after 2026-09-27 |
| GitHub Actions: list runs of `weekly_data_fetch.yml`; jobs and log of run `36410317252` | — | run #29, workflow_dispatch, success; integrity step success (section 1) |
| `git show 1ff7d7c:data/pgr_financials.db > $S/db/before.db`; `git show ed7997f:data/pgr_financials.db > $S/db/after.db`; `chmod 444` | 0 | read-only copies |
| `sha256sum $S/db/*.db data/pgr_financials.db` | 0 | before `f453ab98…`, after `38991c76…`; tracked DB equals after |
| `python3 $S/diff_tables.py $S/db/before.db $S/db/after.db` | 0 | whole-DB diff table (section 2) |
| `python3 $S/diff_dividends.py $S/db/before.db $S/db/after.db` | 0 | 57 new, 5 revised, 0 removed |
| `python3 $S/diff_targets.py $S/db/before.db $S/db/after.db $S/changed_targets.csv` | 0 | 273 changed rows, 0 sign flips, key sets identical |
| `python $S/checks.py $S/db/before.db $S/db/after.db` | 0 | freshness before and after; 0 jumps; 0 duplicate weeks; 0 duplicate FRED months |
| `python $S/gaps.py $S/db/after.db` | 0 | as-of 2026-09-25 freshness; gap audit (VWO March 2026) |
| `python3 $S/recompute.py $S/db/before.db $S/db/after.db` | 0 | all 9,658 rows within 1.554e-15 on both DBs |
| `python3 $S/attribute.py $S/db/before.db $S/db/after.db` | 0 | attribution identity within 4.4e-16; five worked checks |
| `pip install -e ".[dev]"` | error (output piped to `tail`, so the exit code was not captured) | the container lacked `python-dotenv` and pytest; a Debian-installed PyYAML could not be uninstalled |
| `pip install --ignore-installed pyyaml==6.0.2`; `pip install -e ".[dev]"` | no error (also piped) | dev dependencies installed; `import dotenv, pandas, sklearn, pytest` then worked |
| Full suite, run 1: `python -m pytest -o addopts="--tb=short" -q` on the uncommitted tree | killed at about 19% | started before this record existed and stopped so the suite could run on the committed tree; DB sha256 afterwards still `38991c76…` |
| Full suite, run 2: the same command at `e32d15e` | 1 | `1 failed, 2584 passed, 1 skipped, 124 warnings in 835.05s (0:13:55)`. The failure, `test_monthly_decision_dry_run_leaves_db_and_tracked_files_unchanged`, reported `CHANGELOG.md` and this record as changed tracked files: I edited both, to record the VWO decision, while that test ran. It was not a product failure. DB sha256 `38991c76…` before and after |
| `sha256sum data/pgr_financials.db` before run 3 | 0 | `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6` |
| Full suite, run 3: `python -m pytest -o addopts="--tb=short" -q` at `efd75ae`, with the tree left untouched throughout | **0** | **`2585 passed, 1 skipped, 125 warnings in 808.75s (0:13:28)`** |
| `sha256sum data/pgr_financials.db` after run 3 | 0 | `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`, unchanged; `git status --short` empty |

Run 3's tree differs from this final record only in this table. CI on
`efd75ae` also passed: `test`, `research`, `artifacts` and
`windows-regressions`.

## Appendix — scripts

The scripts were run from the scratch directory, and they are reproduced
here so the check can be re-run on any two DB copies. `checks.py` and
`gaps.py` import `config` and `src.database.db_client` from the repository
root, which was their working directory. The others use only the standard
library.

### `recompute.py`

```python
"""R2-lite: independent split/fractional-DRIP recompute of stored targets.

Standard library only; no production return helpers. The algorithm is that
of Appendix C of VERIFICATION_2026-09-26.md, generalised to 6M and 12M:
start bar = last weekly close on/before the month-end anchor; end bar = last
close on/before the last weekday of the month `h` months later; every split
ratio and every dividend (reinvested fractionally at the last close on/before
its ex-date) with start < date <= end multiplies the share count.
"""
import calendar
import datetime
import math
import sqlite3
import sys
from pathlib import Path


def connect(p: str) -> sqlite3.Connection:
    return sqlite3.connect(Path(p).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def window(c: sqlite3.Connection, ticker: str, anchor: str, h: int):
    t = datetime.date.fromisoformat(anchor)
    m = t.month + h
    yr = t.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    end = datetime.date(yr, m, calendar.monthrange(yr, m)[1])
    while end.weekday() > 4:
        end -= datetime.timedelta(days=1)
    q = "SELECT date, close FROM daily_prices WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1"
    return c.execute(q, (ticker, anchor)).fetchone(), c.execute(q, (ticker, end.isoformat())).fetchone()


def manual(c: sqlite3.Connection, ticker: str, anchor: str, h: int):
    start_bar, end_bar = window(c, ticker, anchor, h)
    events = [(d, "split", v) for d, v in c.execute(
        "SELECT split_date, split_ratio FROM split_history WHERE ticker=? AND split_date>? AND split_date<=?",
        (ticker, start_bar[0], end_bar[0]))]
    events += [(d, "div", v) for d, v in c.execute(
        "SELECT ex_date, amount FROM daily_dividends WHERE ticker=? AND ex_date>? AND ex_date<=?",
        (ticker, start_bar[0], end_bar[0]))]
    factors = []
    for date, kind, value in sorted(events):
        if kind == "split":
            factors.append(value)
        else:
            price = c.execute(
                "SELECT close FROM daily_prices WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
                (ticker, date)).fetchone()[0]
            factors.append(1 + value / price)
    return math.prod(factors) * end_bar[1] / start_bar[1] - 1, start_bar[0], end_bar[0]


if __name__ == "__main__":
    before, after = connect(sys.argv[1]), connect(sys.argv[2])
    after.execute("ATTACH DATABASE ? AS b", (Path(sys.argv[1]).resolve().as_uri() + "?mode=ro&immutable=1",))
    # New or amount-changed dividend keys (the refresh's economic input changes).
    delta = {}
    for t, d, kind in after.execute("""
        SELECT a.ticker, a.ex_date, CASE WHEN o.ticker IS NULL THEN 'new' ELSE 'revised' END
        FROM main.daily_dividends a LEFT JOIN b.daily_dividends o USING (ticker, ex_date)
        WHERE o.ticker IS NULL OR o.amount IS NOT a.amount"""):
        delta.setdefault(t, []).append((d, kind))
    rows_a = {(d, b, h): (p, br, rr) for d, b, h, p, br, rr in after.execute(
        "SELECT date, benchmark, target_horizon, pgr_return, benchmark_return, relative_return FROM main.monthly_relative_returns")}
    rows_b = {(d, b, h): (p, br, rr) for d, b, h, p, br, rr in after.execute(
        "SELECT date, benchmark, target_horizon, pgr_return, benchmark_return, relative_return FROM b.monthly_relative_returns")}
    worst_a = worst_b = 0.0
    changed_unexplained, unchanged_but_touched = [], []
    n_changed = 0
    attribution = {"new": 0, "revised": 0, "both": 0}
    for key in sorted(rows_a):
        d, bm, h = key
        ra, rb = rows_a[key], rows_b[key]
        mb_a, s, e = manual(after, bm, d, h)
        mp_a, _, _ = manual(after, "PGR", d, h)
        worst_a = max(worst_a, abs(mb_a - ra[1]), abs(mp_a - ra[0]), abs((mp_a - mb_a) - ra[2]))
        mb_b, _, _ = manual(before, bm, d, h)
        mp_b, _, _ = manual(before, "PGR", d, h)
        worst_b = max(worst_b, abs(mb_b - rb[1]), abs(mp_b - rb[0]), abs((mp_b - mb_b) - rb[2]))
        touched = {k for x, k in delta.get(bm, []) if s < x <= e}
        changed = ra != rb
        if changed:
            n_changed += 1
            if not touched:
                changed_unexplained.append(key)
            else:
                attribution["both" if len(touched) == 2 else touched.pop()] += 1
        elif touched:
            unchanged_but_touched.append(key)
    print("rows recomputed:", len(rows_a))
    print("max |manual - stored| on AFTER DB (all rows, pgr/bench/rel):", worst_a)
    print("max |manual - stored| on BEFORE DB (all rows, pgr/bench/rel):", worst_b)
    print("changed rows:", n_changed, "attribution:", attribution)
    print("changed rows with no new/revised dividend in window:", changed_unexplained)
    print("unchanged rows with a new/revised dividend in window:", unchanged_but_touched)
```

### `attribute.py`

```python
"""R2-lite: per-row attribution identity and worked hand checks.

Prices and splits are identical before/after, so for every target row
(1 + R_after) / (1 + R_before) must equal the product, over the refresh's
new or revised dividends inside the window, of
(1 + new/p) / (1 + old/p), where p is the last close on/before the ex-date
and old = 0 for a new row. Standard library only.
"""
import sqlite3
import sys
from pathlib import Path

from recompute import connect, manual

before, after = connect(sys.argv[1]), connect(sys.argv[2])
old = {(t, d): a for t, d, a in before.execute("SELECT ticker, ex_date, amount FROM daily_dividends")}
new = {(t, d): a for t, d, a in after.execute("SELECT ticker, ex_date, amount FROM daily_dividends")}
delta = {k: (old.get(k, 0.0), v) for k, v in new.items() if old.get(k) != v}


def px(t: str, d: str) -> tuple[str, float]:
    return after.execute("SELECT date, close FROM daily_prices WHERE ticker=? AND date<=? "
                         "ORDER BY date DESC LIMIT 1", (t, d)).fetchone()


rows_b = {(d, b, h): br for d, b, h, br in before.execute(
    "SELECT date, benchmark, target_horizon, benchmark_return FROM monthly_relative_returns")}
worst = 0.0
per = {}
for d, bm, h, br_a in after.execute(
        "SELECT date, benchmark, target_horizon, benchmark_return FROM monthly_relative_returns"):
    _, s, e = manual(after, bm, d, h)
    pred = 1.0
    for (t, x), (o, n) in delta.items():
        if t == bm and s < x <= e:
            p = px(t, x)[1]
            pred *= (1 + n / p) / (1 + o / p)
    ratio = (1 + br_a) / (1 + rows_b[(d, bm, h)])
    worst = max(worst, abs(ratio - pred))
    if pred != 1.0:
        per.setdefault((bm, h), []).append(pred - 1)
print("rows:", len(rows_b), "max |observed ratio - predicted ratio|:", worst)
print("benchmark | horizon | rows with a delta dividend | min / max predicted uplift")
for (bm, h), v in sorted(per.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"{bm} | {h} | {len(v)} | {min(v):+.6e} / {max(v):+.6e}")

print("\n## Worked hand checks")
for bm, anchor, h in [("VOO", "2026-02-27", 6), ("BND", "2025-08-29", 12),
                      ("VWO", "2025-12-31", 6), ("VXUS", "2025-12-31", 6),
                      ("VDE", "2025-08-29", 12)]:
    ret, s, e = manual(after, bm, anchor, h)
    pgr, _, _ = manual(after, "PGR", anchor, h)
    sb, eb = px(bm, s)[1], px(bm, e)[1]
    print(f"\n{bm} {h}M anchor {anchor}: start bar {s} close {sb!r}; end bar {e} close {eb!r}")
    for (x, a) in after.execute("SELECT ex_date, amount FROM daily_dividends WHERE ticker=? AND ex_date>? "
                                "AND ex_date<=? ORDER BY ex_date", (bm, s, e)):
        pd_, p = px(bm, x)
        tag = "new" if (bm, x) not in old else ("revised from %r" % old[(bm, x)] if old[(bm, x)] != a else "existing")
        print(f"  div {x} {a!r} reinvested at {pd_} close {p!r}: factor {1 + a / p!r} ({tag})")
    for (x, r) in after.execute("SELECT split_date, split_ratio FROM split_history WHERE ticker=? AND "
                                "split_date>? AND split_date<=?", (bm, s, e)):
        print(f"  split {x} ratio {r}")
    st = after.execute("SELECT pgr_return, benchmark_return, relative_return FROM monthly_relative_returns "
                       "WHERE benchmark=? AND date=? AND target_horizon=?", (bm, anchor, h)).fetchone()
    stb = before.execute("SELECT benchmark_return, relative_return FROM monthly_relative_returns "
                         "WHERE benchmark=? AND date=? AND target_horizon=?", (bm, anchor, h)).fetchone()
    print(f"  manual benchmark {ret!r}  stored {st[1]!r}  gap {ret - st[1]:.3e}")
    print(f"  manual PGR {pgr!r}  stored {st[0]!r}  gap {pgr - st[0]:.3e}")
    print(f"  manual relative {pgr - ret!r}  stored {st[2]!r}  gap {(pgr - ret) - st[2]:.3e}")
    print(f"  before: benchmark {stb[0]!r} relative {stb[1]!r}")
```

### `diff_tables.py`

```python
"""R2-lite: full-table diff of the before/after DB copies (read-only, immutable)."""
import sqlite3
import sys
from pathlib import Path

before, after = (Path(p).resolve() for p in sys.argv[1:3])
c = sqlite3.connect(after.as_uri() + "?mode=ro&immutable=1", uri=True)
c.execute("ATTACH DATABASE ? AS b", (before.as_uri() + "?mode=ro&immutable=1",))
tables = [r[0] for r in c.execute(
    "SELECT name FROM main.sqlite_master WHERE type='table' ORDER BY name")]
print("table | rows before | rows after | only before | only after")
for t in tables:
    nb = c.execute(f'SELECT count(*) FROM b."{t}"').fetchone()[0]
    na = c.execute(f'SELECT count(*) FROM main."{t}"').fetchone()[0]
    ob = c.execute(f'SELECT count(*) FROM (SELECT * FROM b."{t}" EXCEPT SELECT * FROM main."{t}")').fetchone()[0]
    oa = c.execute(f'SELECT count(*) FROM (SELECT * FROM main."{t}" EXCEPT SELECT * FROM b."{t}")').fetchone()[0]
    print(f"{t} | {nb} | {na} | {ob} | {oa}")
schema_b = c.execute("SELECT type,name,sql FROM b.sqlite_master ORDER BY name").fetchall()
schema_a = c.execute("SELECT type,name,sql FROM main.sqlite_master ORDER BY name").fetchall()
print("schema identical:", schema_a == schema_b)
```

### `diff_dividends.py`

```python
"""R2-lite: daily_dividends diff (new, removed, changed rows)."""
import sqlite3
import sys
from pathlib import Path

before, after = (Path(p).resolve() for p in sys.argv[1:3])
c = sqlite3.connect(after.as_uri() + "?mode=ro&immutable=1", uri=True)
c.execute("ATTACH DATABASE ? AS b", (before.as_uri() + "?mode=ro&immutable=1",))
print("## changed rows (same key, different amount/source)")
for r in c.execute("""SELECT a.ticker,a.ex_date,o.amount,a.amount,o.source,a.source
    FROM main.daily_dividends a JOIN b.daily_dividends o USING(ticker,ex_date)
    WHERE a.amount IS NOT o.amount OR a.source IS NOT o.source ORDER BY 1,2"""):
    print(r)
print("## removed keys")
for r in c.execute("""SELECT * FROM b.daily_dividends o WHERE NOT EXISTS
    (SELECT 1 FROM main.daily_dividends a WHERE a.ticker=o.ticker AND a.ex_date=o.ex_date)"""):
    print(r)
print("## new rows per ticker: n | first ex | last ex | min amt | max amt | source")
for r in c.execute("""SELECT ticker,count(*),min(ex_date),max(ex_date),min(amount),max(amount),
    group_concat(DISTINCT source) FROM main.daily_dividends a WHERE NOT EXISTS
    (SELECT 1 FROM b.daily_dividends o WHERE a.ticker=o.ticker AND a.ex_date=o.ex_date)
    GROUP BY ticker ORDER BY ticker"""):
    print(r)
print("## every new row")
for r in c.execute("""SELECT ticker,ex_date,amount,source FROM main.daily_dividends a WHERE NOT EXISTS
    (SELECT 1 FROM b.daily_dividends o WHERE a.ticker=o.ticker AND a.ex_date=o.ex_date)
    ORDER BY ticker,ex_date"""):
    print(r)
```

### `diff_targets.py`

```python
"""R2-lite: monthly_relative_returns diff per benchmark and horizon."""
import sqlite3
import sys
from pathlib import Path

before, after = (Path(p).resolve() for p in sys.argv[1:3])
c = sqlite3.connect(after.as_uri() + "?mode=ro&immutable=1", uri=True)
c.execute("ATTACH DATABASE ? AS b", (before.as_uri() + "?mode=ro&immutable=1",))
keys_b = set(c.execute("SELECT date,benchmark,target_horizon FROM b.monthly_relative_returns"))
keys_a = set(c.execute("SELECT date,benchmark,target_horizon FROM main.monthly_relative_returns"))
print("key sets identical:", keys_a == keys_b, len(keys_a))
rows = c.execute("""
SELECT a.benchmark, a.target_horizon, a.date,
       o.pgr_return, a.pgr_return, o.benchmark_return, a.benchmark_return,
       o.relative_return, a.relative_return, o.proxy_fill, a.proxy_fill
FROM main.monthly_relative_returns a
JOIN b.monthly_relative_returns o USING (date, benchmark, target_horizon)
WHERE a.pgr_return IS NOT o.pgr_return OR a.benchmark_return IS NOT o.benchmark_return
   OR a.relative_return IS NOT o.relative_return OR a.proxy_fill IS NOT o.proxy_fill
ORDER BY 1, 2, 3""").fetchall()
print("changed rows:", len(rows))
print("pgr_return changed:", sum(r[3] != r[4] for r in rows),
      "proxy_fill changed:", sum(r[9] != r[10] for r in rows))
summary = {}
for r in rows:
    k = (r[0], r[1])
    s = summary.setdefault(k, {"n": 0, "first": r[2], "last": r[2], "db": (0, None),
                               "dr": (0, None), "flips": []})
    s["n"] += 1
    s["last"] = r[2]
    db = abs(r[6] - r[5])
    dr = abs(r[8] - r[7])
    if db > s["db"][0]:
        s["db"] = (db, r[2])
    if dr > s["dr"][0]:
        s["dr"] = (dr, r[2])
    if (r[7] > 0) != (r[8] > 0) or r[7] == 0 or r[8] == 0:
        s["flips"].append((r[2], r[7], r[8]))
print("benchmark | horizon | rows changed | first date | last date | max |d bench| (date) | max |d rel| (date) | sign flips")
for (b, h), s in sorted(summary.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"{b} | {h} | {s['n']} | {s['first']} | {s['last']} | {s['db'][0]:.3e} ({s['db'][1]}) | "
          f"{s['dr'][0]:.3e} ({s['dr'][1]}) | {s['flips']}")
for h in (6, 12):
    n = sum(s["n"] for (b, hh), s in summary.items() if hh == h)
    print("horizon", h, "rows changed", n)
import csv
with open(sys.argv[3], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["benchmark", "target_horizon", "date", "benchmark_return_before",
                "benchmark_return_after", "relative_return_before", "relative_return_after"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], repr(r[5]), repr(r[6]), repr(r[7]), repr(r[8])])
```

### `checks.py`

```python
"""R2-lite: freshness and integrity checks on the immutable before/after copies."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import config  # noqa: E402
from src.database import db_client  # noqa: E402
from src.processing.price_integrity import (  # noqa: E402
    find_duplicate_week_bars,
    find_unexplained_price_jumps,
)

REQUIRED = ["PGR", *config.PRIMARY_FORECAST_UNIVERSE]
for label, p in (("before", sys.argv[1]), ("after", sys.argv[2])):
    c = sqlite3.connect(Path(p).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    print(f"=== {label} ===")
    fr = db_client.check_dividend_freshness(c)
    for r in fr:
        req = "required" if r["ticker"] in REQUIRED else "other"
        print(f"{r['ticker']} | {req} | {r['status']} | {r['last_ex_date']} | {r['last_price_date']} | "
              f"{r['interval_days']} | {r['due_by']}")
    print("required STALE:", [r["ticker"] for r in fr if r["ticker"] in REQUIRED and r["status"] == "STALE"])
    print("other STALE:", [r["ticker"] for r in fr if r["ticker"] not in REQUIRED and r["status"] == "STALE"])
    print("NO_HISTORY:", [r["ticker"] for r in fr if r["status"] == "NO_HISTORY"])
    print("unexplained price jumps:", len(find_unexplained_price_jumps(c)))
    print("duplicate ticker-week bars:", len(find_duplicate_week_bars(c)))
    print("duplicate FRED series-months:", c.execute(
        "SELECT count(*) FROM (SELECT series_id, substr(month_end,1,7) m, count(*) n "
        "FROM fred_macro_monthly GROUP BY 1,2 HAVING n>1)").fetchone()[0])
    print("max price date per required ticker:", dict(c.execute(
        "SELECT ticker, max(date) FROM daily_prices WHERE ticker IN (%s) GROUP BY ticker"
        % ",".join("?" * len(REQUIRED)), REQUIRED).fetchall()))
    print("latest target date per horizon:", c.execute(
        "SELECT target_horizon, max(date), count(*) FROM monthly_relative_returns GROUP BY 1").fetchall())
    c.close()
```

### `gaps.py`

```python
"""R2-lite: as-of freshness and a mid-history gap audit (after DB)."""
import datetime
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import config  # noqa: E402
from src.database import db_client  # noqa: E402

c = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
print("latest target per horizon:", c.execute(
    "SELECT target_horizon, max(date), count(*) FROM monthly_relative_returns GROUP BY 1").fetchall())
fr = db_client.check_dividend_freshness(c, as_of=datetime.date(2026, 9, 25))
print("as_of 2026-09-25 non-OK:", [(r["ticker"], r["status"], r["last_ex_date"]) for r in fr if r["status"] != "OK"])
print("ex-dates after the last price bar (2026-09-25):",
      c.execute("SELECT ticker, ex_date, amount FROM daily_dividends WHERE ex_date > '2026-09-25' ORDER BY 1").fetchall())
# Gaps > 1.5x the ticker's whole-history median gap, ex-dates since 2014-01-01.
tickers = ["PGR", *config.ETF_BENCHMARK_UNIVERSE]
for t in tickers:
    ds = [datetime.date.fromisoformat(r[0]) for r in c.execute(
        "SELECT ex_date FROM daily_dividends WHERE ticker=? ORDER BY ex_date", (t,))]
    if len(ds) < 3:
        continue
    gaps = [(b - a).days for a, b in zip(ds, ds[1:])]
    med = statistics.median(gaps)
    long = [(a.isoformat(), b.isoformat(), g) for a, b, g in zip(ds, ds[1:], gaps)
            if g > 1.5 * med and b >= datetime.date(2014, 1, 1)]
    print(t, "first", ds[0], "median gap", med, "long gaps since 2014:", long)
```
