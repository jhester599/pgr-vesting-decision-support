# R3b closeout — refreshed-DB replay and current baseline (v191)

Session R3b of [the pre-v200 fix prompts](PRE_V200_FIX_PROMPTS_codex.md#r3b--refreshed-db-replay-and-current-baseline)
(D5). Builder: Claude (Linux, cloud). Reviewer: Codex. CHANGELOG **v191**.
Evidence read: verification V08
([`VERIFICATION_2026-09-26.md`](VERIFICATION_2026-09-26.md)), the
[R3 closeout](R3_validation_closeout.md) ("What is left", item 1) with its
pre-refresh rows [`R3_smoke_replay_rows.csv`](R3_smoke_replay_rows.csv), and
the [R2-lite record](2026-09-28_R2_dividend_refresh_check.md).

**This changes no code, feature, threshold, parameter, target, consensus
weighting or gate.** The replay did not appear to need any such change. It is
a pre-v200 smoke comparison of already-inspected history, not promotion
evidence, and it promises no better return.

## Result

- The replay of the eight committed decisions exited 0 on the pinned
  refreshed DB. Every DB, ledger and artifact hash is unchanged, and the
  clone's `git status` is clean.
- A second run of the latest as-of date (2026-09-21) gives a byte-identical
  `monthly_summary.json`.
- **Every month still defers at 50 %** (DEFER-TO-TAX-DEFAULT, LOW
  confidence), because directional skill fails. No mode or sell percentage
  changed.
- `data_ready` now passes in 7 of 8 months. 2026-05-22 still fails on
  **VWO dividends**: VWO has no March 2026 ex-date, the provider gap the
  owner accepted in R2-lite.
- One consensus label changed: 2026-09-21 is NEUTRAL instead of
  UNDERPERFORM, because VOO's signal went to NEUTRAL (section 3.5).
- **Every difference from R3's pre-refresh replay is the data change.**
  The pinned code, run on the pre-refresh DB, reproduces all 47 columns of
  R3's `R3 1b94bf0` rows exactly (section 3.1).
- `docs/model-governance.md` now pins the
  [current baseline](../model-governance.md#current-baseline). ADR 0008's
  Consequences point to it.

## 1. Pins

| Item | Value |
|---|---|
| Code | `master` `c3b4798b90a43f9dbd616981861e9ff01fa2de72` (2026-09-28, merge of PR #144). It contains R3 v188 (`73a3d09`, `4cd875a`), R2-lite v187 (`e32d15e`…`c037860`), R1 v186, R4 v189, R5 v190 and v192 (D6). |
| DB | R2-lite's "after" DB, sha256 `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`, git blob `3f32d1571ebd7a6efd179e967495c49f07cf9d20`, committed by the refresh `ed7997f`. No later commit changed `data/pgr_financials.db`: `git log ed7997f^..c3b4798 -- data/pgr_financials.db` lists only `ed7997f`, and `HEAD:data/pgr_financials.db` is the same blob. `git show ed7997f:data/pgr_financials.db` extracted to the scratch directory has the same sha256. |
| Pre-refresh DB (for the control) | sha256 `f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d`, from `git show 1ff7d7c:data/pgr_financials.db`: the DB of R3's replay |
| Runtime | Linux 6.18.44 x86_64, glibc 2.39; Python 3.11.15; pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, statsmodels 0.15.0, xgboost 3.2.0, matplotlib 3.11.2, MAPIE 1.5.0, skfolio 1.4.3, PyPortfolioOpt 1.6.0, pyarrow 25.0.1, requests 2.33.1, pytest 9.0.2 |
| `MODEL_HEALTH_METRICS_VERSION` | `prequential-2026-09-25` (`config/model.py:130`) |
| `DECISION_GATE_CONTRACT_VERSION` | `chronological-readiness-2026-09-27` (`config/model.py:83`) |

The container had no project packages installed. They were installed with
`pip install -c constraints.txt -e ".[dev]"`, where the constraints file pins
pandas, numpy, scikit-learn, scipy, statsmodels and xgboost to R3's recorded
versions, so the runtime matches R3's replay. The Debian PyYAML first had to
be replaced (`pip install --ignore-installed PyYAML==6.0.2`), as in R2-lite.

**Isolation.**

- No provider call, fetcher or e-mail. The dry runs pass `--skip-fred`, and
  the environment holds no Alpha Vantage, FRED, FMP or SMTP credential.
- The tracked DB was never opened by the replay. The source checkout's
  artifacts and ledgers were never written.
- Every DB read outside a clone used
  `sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)` on
  copies in the scratch directory.
- No holdout outcome beyond the eight already-inspected committed decisions
  was opened.
- Subagents used: none.

## 2. Replay

`$S` is the session scratch directory, outside the repository:
`/tmp/claude-0/-home-user-pgr-vesting-decision-support/932e520f-2d34-52f7-9ce0-c01682ab9202/scratchpad`.

| Path | What |
|---|---|
| `$S/r3b_clone` | the replay clone: `git clone` of the source checkout, checked out at `c3b4798`. Its tracked `data/pgr_financials.db` is its own copy of the pinned blob. |
| `$S/r3b_replay.csv` | replay output, committed unchanged as [`R3b_refreshed_replay_rows.csv`](R3b_refreshed_replay_rows.csv) (sha256 `110b54d3bc186a2bdfcd2853a95f915409eb9f32388cf54c5f400c04ff2cda93`) |
| `$S/r3b_rerun_2026-09-21.csv` | the second run of 2026-09-21 |
| `$S/r3b_control` | control clone at `c3b4798`, with its DB replaced by the pre-refresh copy `f453ab98…` |
| `$S/control_prerefresh_replay.csv` | control output (sha256 `9c4d5456…4987`) |
| `$S/r3b_hybrid` | clone at `c3b4798` with the DBC-only hybrid DB (section 3.3) |

Command, with cwd `$S/r3b_clone`:

```text
PYTHONPATH=$S/r3b_clone:$S/r3b_clone/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/replay_monthly_decisions.py --committed-dates --out $S/r3b_replay.csv
```

Exit **0**, real time 15 m 06 s. The script checked the clone DB's sha256
before and after each of the eight dry runs (all `38991c7653f6…`). Each dry
run logged a Black-Litterman fallback: "at least one of the assets must have
an expected return exceeding the risk-free rate". That is the redeploy
diagnostic's existing equal-weight fallback, and it does not enter the
decision, the gates or any replay column. The control run on the
pre-refresh DB logged it too.

**Hashes** (from `$S/hashes.sh`, appendix):

| | Before (2026-09-28 11:51 UTC) | After (12:08 UTC) |
|---|---|---|
| source DB `data/pgr_financials.db` | `38991c76…a94e6` | `38991c76…a94e6` |
| clone DB `$S/r3b_clone/data/pgr_financials.db` | `38991c76…a94e6` | `38991c76…a94e6` |
| source `decision_log.md` | `abcb3f73…22437` | `abcb3f73…22437` |
| source `classification_shadow_history.csv` | `eb98d134…e3c8b` | `eb98d134…e3c8b` |
| source `ta_shadow_variant_history.csv` | `d952e566…e39f1` | `d952e566…e39f1` |
| source ledgers, combined | `79b4a9f0…0b66b` | `79b4a9f0…0b66b` |
| source `artifacts/` tree | `a7f8ac6a…4d4b0` | `a7f8ac6a…4d4b0` |
| clone `artifacts/` tree | `a7f8ac6a…4d4b0` | `a7f8ac6a…4d4b0` |

No `-wal` or `-shm` sidecar appeared in either `data/` directory.
`git status --short` in `$S/r3b_clone` printed nothing after the replay and
after the rerun. The dry-run outputs live in the ignored
`results/dry_run/`.

**Determinism.** A second run with cwd `$S/r3b_clone`:

```text
PYTHONPATH=… OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/replay_monthly_decisions.py --as-of 2026-09-21 --out $S/r3b_rerun_2026-09-21.csv
```

Exit 0; clone DB still `38991c76…`. Compared with the first run's
2026-09 outputs, saved aside before the rerun:

- `monthly_summary.json` is byte-identical: 281 flattened fields, 0
  differing.
- The replay CSV row is identical.
- `signals.csv`, `benchmark_quality.csv`, `consensus_shadow.csv`,
  `classification_shadow.csv`, `decision_overlays.csv`, `recommendation.md`,
  `diagnostic.md`, `dashboard.html` and the calibration plot are
  byte-identical.
- `run_manifest.json` differs only in `run_timestamp_utc`.

## 3. Attribution against R3's pre-refresh replay

### 3.1 Control: the differences are data only

The prompt states that R3's `R3 1b94bf0` rows used the same gates and model
code. Between `1b94bf0` and `c3b4798`, the code changes are R4 (TA ledger and
classifier monitoring), R5 (backtest and rebalancer), v192 (`capital_gains`)
and R3's wording commit (deferral summary text). None is a gate, metric,
feature, target or model change.

Rather than rely on that reading, the pinned code was replayed on the
pre-refresh DB (`$S/r3b_control`, same command, exit 0, 15 m 30 s, DB copy
`f453ab98…` unchanged). Its rows equal R3's `R3 1b94bf0` rows in **all 47
shared columns**, compared as text, cell for cell. That covers every mode,
sell percentage, consensus, gate, reason, readiness field, `db_sha256` and
full-precision metric. R3's four extra columns are its retired CPCV columns,
which the current script no longer emits.

The pinned code therefore gives R3's numbers on R3's DB. Every difference
below comes from the DB change `f453ab98…` → `38991c76…`. R2-lite shows that
this change touched only `daily_dividends` (57 new rows, 5 revised amounts),
the targets derived from them (273 rows) and bookkeeping tables.

### 3.2 Decisions and gates, before → after

Before is the `R3 1b94bf0` rows (pre-refresh DB). After is the R3b rows
(refreshed DB). DEFER = DEFER-TO-TAX-DEFAULT.

| As-of | R3 pre-refresh: mode / sell / consensus | R3 failed gates | R3b refreshed: mode / sell / consensus | R3b failed gates | Stale required feeds, R3 → R3b |
|---|---|---|---|---|---|
| 2026-02-28 | DEFER / 50% / UNDERPERFORM | directional_skill | DEFER / 50% / UNDERPERFORM | directional_skill | none → none |
| 2026-03-31 | DEFER / 50% / NEUTRAL | directional_skill | DEFER / 50% / NEUTRAL | directional_skill | none → none |
| 2026-04-22 | DEFER / 50% / NEUTRAL | directional_skill; data_ready | DEFER / 50% / NEUTRAL | directional_skill | VMBS, BND → none |
| 2026-05-22 | DEFER / 50% / UNDERPERFORM | directional_skill; data_ready | DEFER / 50% / UNDERPERFORM | directional_skill; data_ready | VOO, VWO, VMBS, BND → VWO |
| 2026-06-22 | DEFER / 50% / UNDERPERFORM | directional_skill; data_ready | DEFER / 50% / UNDERPERFORM | directional_skill | VOO, VWO, VMBS, BND → none |
| 2026-07-22 | DEFER / 50% / NEUTRAL | directional_skill; data_ready | DEFER / 50% / NEUTRAL | directional_skill | VOO, VWO, VMBS, BND → none |
| 2026-08-20 | DEFER / 50% / NEUTRAL | directional_skill; data_ready | DEFER / 50% / NEUTRAL | directional_skill | VOO, VXUS, VWO, VMBS, BND, VDE → none |
| 2026-09-21 | DEFER / 50% / UNDERPERFORM | directional_skill; data_ready | DEFER / 50% / NEUTRAL | directional_skill | VOO, VXUS, VWO, VMBS, BND, VDE → none |

Other gate statuses are the same before and after in every month:

- `oos_r2`: PASS, except 2026-08-20 MARGINAL.
- `mean_ic`: PASS, except 2026-07-22 and 2026-08-20 MARGINAL.
- `directional_skill`: FAIL in every month.
- `wfo_completed`: PASS in every month; all 16 pairs complete, no failed
  pair.
- No live feature is missing in any month.
- Every month is LOW confidence.

The decision rows are unchanged. The equal-weight diagnostic consensus is
unchanged in every month.

Metrics, after (change or before value in brackets):

| As-of | OOS R² | EW IC | Pooled IC (DK p) | Hit vs base | PT p | ECE | Coverage |
|---|---|---|---|---|---|---|---|
| 2026-02-28 | +3.05 % (+0.02 pp) | 0.1005 (-0.0002) | 0.1262 (0.051; was 0.1263 (0.049)) | 64.03 % vs 69.98 % (was 64.12 %) | 0.374 (was 0.364) | 16.33 % (was 16.24 %) | 48.96 % (was 48.96 %) |
| 2026-03-31 | +5.75 % (+0.10 pp) | 0.1069 (+0.0011) | 0.1337 (0.079; was 0.1320 (0.083)) | 63.80 % vs 69.61 % (was 63.80 %) | 0.383 (was 0.383) | 17.16 % (was 17.36 %) | 44.79 % (was 44.79 %) |
| 2026-04-22 | +5.75 % (+0.10 pp) | 0.1069 (+0.0011) | 0.1337 (0.079; was 0.1320 (0.083)) | 63.80 % vs 69.61 % (was 63.80 %) | 0.383 (was 0.383) | 17.16 % (was 17.36 %) | 44.79 % (was 44.79 %) |
| 2026-05-22 | +3.03 % (-0.02 pp) | 0.0874 (-0.0005) | 0.1070 (0.162; was 0.1076 (0.159)) | 63.08 % vs 69.42 % (was 63.08 %) | 0.460 (was 0.460) | 19.66 % (was 19.29 %) | 43.75 % (was 42.71 %) |
| 2026-06-22 | +3.56 % (+0.04 pp) | 0.0735 (+0.0005) | 0.0878 (0.243; was 0.0873 (0.245)) | 61.39 % vs 68.81 % (was 61.55 %) | 0.592 (was 0.576) | 17.04 % (was 17.03 %) | 40.62 % (was 40.62 %) |
| 2026-07-22 | +5.12 % (-0.01 pp) | 0.0690 (-0.0006) | 0.0894 (0.243; was 0.0899 (0.240)) | 61.99 % vs 68.31 % (was 61.74 %) | 0.476 (was 0.502) | 17.06 % (was 16.93 %) | 42.71 % (was 42.71 %) |
| 2026-08-20 | +1.24 % (+0.03 pp) | 0.0508 (+0.0021) | 0.0736 (0.291; was 0.0730 (0.292)) | 60.38 % vs 68.22 % (was 60.46 %) | 0.622 (was 0.615) | 14.16 % (was 14.37 %) | 41.67 % (was 44.79 %) |
| 2026-09-21 | +2.85 % (-0.01 pp) | 0.0755 (-0.0007) | 0.1008 (0.132; was 0.1019 (0.125)) | 62.75 % vs 68.14 % (was 62.83 %) | 0.402 (was 0.391) | 15.49 % (was 15.41 %) | 42.71 % (was 42.71 %) |

The base rate (constant-rule hit rate) and the live prequential shrinkage
alpha (0.50) are unchanged in every month. The Clark–West p-values stay
below 0.002; the largest change is 2.0e-4, in August.

### 3.3 Which changed targets each month can see

The live model trains and evaluates on 6M relative-return targets of the
eight primary benchmarks, each hidden until its window has ended by the as-of
date (`truncate_relative_target_for_asof`). Benchmark dividends enter the
decision only through those targets and the readiness check: the live
features use PGR's own dividends, which the refresh did not change.

The table below lists, for each as-of date, the 6M targets whose value the
refresh changed and whose window had ended. Windows are anchor → end; the
change is in `relative_return`, and "r" marks a revised amount. It comes from
`$S/attrib_targets.py`, which reads both DB copies and computes window ends
independently with the standard library.

| As-of | Matured changed 6M targets (primary benchmarks) |
|---|---|
| 2026-02-28 | **DBC only**: 21 rows, Δ up to +3.4e-5, from the revised 2018-12-24, 2019-12-23, 2023-12-18 and 2025-12-22 amounts (anchors 2018-06-29 to 2025-08-29); 29 more DBC rows differ by float noise ≤ 4.4e-16 |
| 2026-03-31, 2026-04-22 | the above, plus **VOO** 2025-09-30 → 2026-03-31 (new 2026-03-27, 1.8724; Δ −0.31 pp) and **VXUS** 2025-09-30 → 2026-03-31 (revised 2026-03-20, 0.08 → 0.0795; Δ +0.001 pp) |
| 2026-05-22 | adds the 2025-10-31 → 2026-04-30 windows: **BND** (new 2026-04-01; Δ −0.34 pp), **VMBS** (new 2026-04-01; Δ −0.35 pp), **VOO** (2026-03-27; Δ −0.34 pp), **VXUS** (revised 2026-03-20) |
| 2026-06-22 | adds 2025-11-28 → 2026-05-29: **BND** (2026-04-01, 05-01; Δ −0.67 pp), **VMBS** (Δ −0.70 pp), **VOO** (Δ −0.36 pp), **VXUS** (revised) |
| 2026-07-22 | adds 2025-12-31 → 2026-06-30: **BND** and **VMBS** (2026-04-01, 05-01, 06-01; Δ −1.01 and −1.05 pp), **VOO** (2026-03-27, 06-26; Δ −0.65 pp), **VXUS** (revised 03-20, new 06-18; Δ −0.50 pp), **VWO** (new 2026-06-18; Δ −0.13 pp), **VDE** (new 2026-06-24; Δ −0.84 pp) |
| 2026-08-20 | adds 2026-01-30 → 2026-07-31: BND (four new; Δ −1.32 pp), VMBS (Δ −1.37 pp), VOO (Δ −0.66 pp), VXUS (Δ −0.47 pp), VWO (Δ −0.12 pp), VDE (Δ −0.80 pp) |
| 2026-09-21 | adds 2026-02-27 → 2026-08-31: BND (five new, 2026-04-01 to 08-03; Δ −1.64 pp), VMBS (Δ −1.71 pp), VOO (Δ −0.69 pp), VXUS (Δ −0.46 pp), VWO (Δ −0.12 pp), VDE (Δ −0.77 pp) |

GLD never has a changed target. It is an audited non-payer.

In total, 52 matured primary 6M rows change by more than 1e-12 at
2026-09-21: DBC 24, VOO 6, VXUS 6, BND 5, VMBS 5, VDE 3 and VWO 3.
New dividends raise the benchmark return and so lower `relative_return`.
The VXUS rows that hold only the revised 2026-03-20 amount rise by 6.7e-6.
The 12M targets also changed (142 rows), but the live decision uses only
the 6M horizon.

**Per-benchmark check.** Each benchmark's ensemble is trained on that
benchmark's targets only. So a benchmark whose targets did not change should
keep its live raw ensemble prediction. The table compares each month's
`signals.csv` and `benchmark_quality.csv` between the control and the
refreshed run. "Pred" is the largest change in the raw live prediction; "q"
is the largest change in `n_obs`, OOS R², NW IC and hit rate.

| As-of | Benchmarks whose live prediction moved | Metrics moved, prediction identical | Fully identical |
|---|---|---|---|
| 2026-02-28 | DBC (pred 4.5e-3, q 6.2e-3) | — | VOO, VXUS, VWO, VMBS, BND, GLD, VDE |
| 2026-03-31, 2026-04-22 | DBC, VOO (pred 6.1e-3), VXUS (pred 1.2e-3) | GLD, VWO, BND, VDE, VMBS (q ≤ 2.5e-4) | — |
| 2026-05-22 | DBC, VOO, BND, VXUS, VMBS | — | VWO, GLD, VDE |
| 2026-06-22 | DBC, VOO, VXUS, BND, VMBS | — | VWO, GLD, VDE |
| 2026-07-22 | VWO, DBC, VDE, VOO, BND, VMBS, VXUS | — | GLD |
| 2026-08-20 | VWO, DBC, VOO, VDE, BND, VMBS, VXUS | GLD (q 1.0e-3) | — |
| 2026-09-21 | VWO, DBC, VDE, VOO, BND, VMBS, VXUS | — | GLD |

In every month, a benchmark's live prediction moved exactly when it has a
matured changed target in the table above. VWO and VDE move from July, the
first month their 2026-06 dividends mature. BND and VMBS move from May.

A benchmark's quality metrics can move while its prediction stays the same
(March and April: GLD, VWO, BND, VDE, VMBS; August: GLD). This comes from the
prequential shrinkage alpha. `build_prequential_panel`
(`src/models/prequential.py`) chooses it by pooling the realised OOS rows of
all benchmarks, per historical row, and sets `y_hat = alpha × z`. A changed
target in DBC or VOO can change that historical alpha path, and so every
benchmark's shrunk OOS forecasts. That moves OOS R², while sign-based hit
rates and rank ICs are unaffected by a positive rescale. The live alpha is
0.50 in every month, before and after. The pooled statistics also couple
the benchmarks: pooled IC, the Pesaran–Timmermann test, ECE and conformal
coverage.

**February, isolated.** At 2026-02-28, DBC is the only benchmark with a
matured changed target, and only by revised amounts of at most 3.4e-5. That
is still enough to move DBC's live prediction by 4.5e-3. To confirm it
directly, a hybrid DB was built:

- Start from the pre-refresh copy.
- Replace DBC's `daily_dividends` rows and all DBC `monthly_relative_returns`
  rows with the refreshed DB's.
- Result: sha256 `dad3e226…4ef7c`. Its differences from the refreshed DB are
  confined to non-DBC tickers.

February was replayed on it (`$S/r3b_hybrid`, exit 0). All replay columns
except `db_sha256` equal the refreshed February row. `signals.csv`,
`benchmark_quality.csv` and `monthly_summary.json` are byte-identical to the
refreshed run's. So the February change is caused by DBC's four
provider-revised amounts:

- OOS R² +0.02 pp;
- hit 64.12 % → 64.03 %, i.e. 754 → 753 correct signs of 1,176 OOS rows;
- PT p 0.364 → 0.374.

### 3.4 Month by month

- **2026-02-28.** Only DBC's revised amounts (3.3). Metrics move slightly:
  OOS R² +0.02 pp, EW IC −0.0002, one fewer correct sign, PT p +0.010, ECE
  +0.09 pp. Mode, sell, consensus and gates are unchanged. `data_ready`
  passed before and after: no required dividend was overdue at 2026-02-28.
- **2026-03-31 and 2026-04-22.** Both use decision row 2026-03-31. The
  windows ending in April are not realised by 2026-04-22, so both see the
  same targets and have identical metrics.
  - The VOO 2025-09-30 window gains the 2026-03-27 dividend (−0.31 pp); the
    VXUS window has the revised 2026-03-20 amount; DBC as in February.
  - Metrics: OOS R² +0.10 pp, EW IC +0.0011, pooled IC +0.0018, ECE
    −0.20 pp. Hit rate and PT p are unchanged.
  - **2026-04-22 `data_ready` now PASSES.** Before, VMBS and BND failed:
    their last ex-date at the as-of date was 2026-03-02, and a
    30-day-cadence payer was due by 2026-03-03. The refresh added their
    2026-04-01 ex-dates.
- **2026-05-22.** Adds the 2025-10-31 windows for BND and VMBS (2026-04-01
  dividends) and for VOO and VXUS.
  - Metrics: OOS R² −0.02 pp, EW IC −0.0005. Coverage 42.71 % → 43.75 %,
    i.e. 41 → 42 of 96 trailing intervals. ECE +0.37 pp.
  - VWO, GLD and VDE are fully unchanged, as expected.
  - **`data_ready` still FAILS, now on VWO only.** VOO (2026-03-27) and
    VMBS/BND (2026-04-01, 2026-05-01) are fresh at 2026-05-22. VWO's last
    ex-date on or before 2026-05-22 is 2025-12-19. At a 91-day cadence with
    the 1.5× allowance, a payment was due by 2026-01-06
    (`check_dividend_freshness(as_of=2026-05-22)` on the refreshed copy).
  - VWO has a March ex-date in every year 2013–2025. The provider's full
    VWO history, which R2-lite stored in full, has no 2026-03 row. The owner
    accepted that gap as provider data on 2026-09-28. The feed is VWO
    dividends, and the payment that would clear it is a March 2026 ex-date
    that the provider does not report.
  - The failure does not change the decision: the month already deferred on
    directional skill.
- **2026-06-22.** Adds the 2025-11-28 windows (BND and VMBS with two new
  dividends; VOO; VXUS).
  - Hit 61.55 % → 61.39 %, i.e. two fewer correct signs of 1,212 OOS rows.
    PT p 0.576 → 0.592. OOS R² +0.04 pp.
  - **`data_ready` now PASSES.** VWO's 2026-06-18 ex-date is on or before
    the as-of date, and VOO, VMBS and BND are fresh.
- **2026-07-22.** The first month with VWO's 2026-06-18 and VDE's
  2026-06-24 dividends matured (2025-12-31 windows). BND and VMBS now have
  three new dividends in their windows, and VOO two.
  - Every benchmark except GLD moves. BND's live signal goes OUTPERFORM →
    NEUTRAL; the consensus stays NEUTRAL.
  - Mean predicted return −4.49 % → −4.98 %. Hit 61.74 % → 61.99 %, PT p
    0.502 → 0.476, EW IC −0.0006 (still MARGINAL).
  - **`data_ready` now PASSES.**
- **2026-08-20.** Adds the 2026-01-30 windows.
  - EW IC +0.0021 (still MARGINAL); OOS R² +0.03 pp (still MARGINAL).
  - Coverage 44.79 % → 41.67 %: three fewer of 96 trailing intervals
    contain the realised target. Both the realised targets and the
    prequentially fitted intervals move with the refreshed data.
  - **`data_ready` now PASSES.** VXUS (2026-06-18) and VDE (2026-06-24)
    are fresh, as are VOO, VWO, VMBS and BND.
- **2026-09-21.** Adds the 2026-02-27 windows: BND and VMBS with five new
  dividends each (Δ −1.64 and −1.71 pp), and VOO, VXUS, VWO and VDE.
  - OOS R² −0.01 pp, EW IC −0.0007, pooled IC −0.0011 (DK p 0.125 →
    0.132), hit 62.83 % → 62.75 %, PT p 0.391 → 0.402.
  - **Consensus UNDERPERFORM → NEUTRAL** (3.5). **`data_ready` now
    PASSES**, with no stale required feed.

### 3.5 The September consensus

The live consensus is UNDERPERFORM when the benchmarks signalling
UNDERPERFORM hold more than half the quality weight. The weights are
`0.75 × 1/8 + 0.25 × max(NW IC, 0) / Σ` (`build_quality_weights`). A
benchmark signals only if its IC is at least 0.05 and its |prediction| at
least 1 % (`classify_benchmark_signal`). Recomputed from each run's
`signals.csv` and `benchmark_quality.csv`:

| Benchmark | Control (pre-refresh): prediction / signal / weight | Refreshed: prediction / signal / weight |
|---|---|---|
| VOO | −1.26 % / UNDERPERFORM / 0.1331 | −0.58 % / NEUTRAL / 0.1325 |
| VXUS | −0.48 % / NEUTRAL / 0.0938 | +0.65 % / NEUTRAL / 0.0938 |
| VWO | −2.08 % / UNDERPERFORM / 0.1147 | −2.62 % / UNDERPERFORM / 0.1146 |
| VMBS | +3.26 % / NEUTRAL (IC < 0) / 0.0938 | +2.64 % / NEUTRAL / 0.0938 |
| BND | +2.80 % / OUTPERFORM / 0.1126 | +2.92 % / OUTPERFORM / 0.1117 |
| GLD | +2.08 % / OUTPERFORM / 0.1396 | +2.08 % / OUTPERFORM / 0.1397 |
| DBC | −5.55 % / UNDERPERFORM / 0.1644 | −5.20 % / UNDERPERFORM / 0.1655 |
| VDE | −6.63 % / UNDERPERFORM / 0.1482 | −6.62 % / UNDERPERFORM / 0.1485 |
| **UNDERPERFORM weight** | **0.5604 → UNDERPERFORM** | **0.4286 → NEUTRAL** |

VOO's six matured changed targets (anchors 2025-09-30 to 2026-02-27) now
contain the 2026-03-27 dividend, and the last three the 2026-06-26 one.
Refitting on them moves VOO's live prediction to −0.58 %, inside the 1 %
bar. Its weight leaves the UNDERPERFORM side, and the consensus becomes
NEUTRAL. The sell percentage is unchanged: every gate other than
directional skill passes, but that one fails, so the month defers at 50 %
either way.

### 3.6 Against the verification's replay tables

The verification's "current" table (VERIFICATION_2026-09-26, "Decision
impact") equals R3's pre-refresh rows. For example, September: R² 2.86 %, IC
0.0762, pooled 0.1019, hit 62.83 % vs 68.14 %, p 0.3914. So the differences
to it are exactly those in 3.2:

- Modes and sell percentages match the verification's current and
  historical tables: DEFER at 50 % in all eight months.
- Consensus matches the verification's current column in seven months.
  September is NEUTRAL, not UNDERPERFORM (3.5). The committed September
  decision and the step-5 re-baseline were also NEUTRAL.
- The verification reports ECE 14.37–19.29 % and coverage 40.63–48.96 %.
  The refreshed replay gives 14.16–19.66 % and 40.62–48.96 %. Both remain
  far from calibrated.
- The published step-5 table (September +5.08 %, 5/7 CPCV GOOD) is still
  superseded for the reason the verification gave: the step-6 filing-date
  timing repair (V08). The refresh moves September by only −0.01 pp.

### 3.7 Back-dated readiness

The DB does not record when each value was fetched. For every as-of date
except the latest, the replay labels readiness
`readiness_basis = backdated_reconstruction`. Those PASS results reflect the
**repaired values**, stored on 2026-09-28, not what the original decisions
had available. The committed decisions of 2026-04 to 2026-09 would have
failed `data_ready` with the dividends they had, as R3's pre-refresh replay
shows.

2026-09-21 is labelled `live` because the run date (2026-09-28) is in the
same month (`health.readiness_basis`). Its PASS also depends on dividends
fetched on 2026-09-28, after that decision. The label is the contract's
month rule and is not changed here (no code change); see "What is left".

## 4. Governance baseline

`docs/model-governance.md` replaces "Current state after R3 (pre-refresh,
not the new baseline)" with
[**Current baseline (R3b, pinned 2026-09-28)**](../model-governance.md#current-baseline).
The section records:

- the DB sha256, code commit and runtime, and the two version constants;
- the 2026-09-21 metrics with their gates;
- the recommendation, DEFER-TO-TAX-DEFAULT at 50 %;
- the calibration, coverage, reconstructed-readiness and VWO limitations.

The R3 pre-refresh table is kept below it as "Historical: R3 pre-refresh
state at 2026-09-21 (dated 2026-09-27)". The step-5 table is kept as
"Historical: step-5 health baseline", with its anchor unchanged, since ADR
0006 links to it. The "Current Governance Conclusion" now points to the
baseline. ADR 0008's Consequences gain a dated pointer.

No committed monthly artifact, ledger, performance log or DB row was edited.

## 5. Commands and results

Run from the repository root unless noted.

| Command | Exit | Result |
|---|---:|---|
| `git fetch origin master`; `git rev-parse HEAD origin/master` | 0 | both `c3b4798b90a43f9dbd616981861e9ff01fa2de72` |
| `git log --oneline ed7997f^..HEAD -- data/pgr_financials.db`; `git rev-parse ed7997f:data/pgr_financials.db HEAD:data/pgr_financials.db` | 0 | only `ed7997f`; both blobs `3f32d157…` |
| `sha256sum data/pgr_financials.db` | 0 | `38991c76…a94e6` |
| `git show ed7997f:data/pgr_financials.db > $S/r2lite_after_pinned.db`; `git show 1ff7d7c:data/pgr_financials.db > $S/before_refresh.db`; `sha256sum` | 0 | `38991c76…` and `f453ab98…` |
| `pip install --ignore-installed PyYAML==6.0.2`; `pip install -c $S/constraints.txt -e ".[dev]"` | 0 | runtime in section 1 |
| `git clone -q <source> $S/r3b_clone && git checkout c3b4798` | 0 | clean clone |
| `$S/hashes.sh before_replay` | 0 | section 2 |
| Replay (section 2), cwd `$S/r3b_clone` | **0** | 8 dry runs, 15 m 06 s |
| `$S/hashes.sh after_replay`; `git -C $S/r3b_clone status --short` | 0 | all hashes unchanged; status empty |
| Rerun of 2026-09-21, cwd `$S/r3b_clone` | **0** | byte-identical `monthly_summary.json` |
| Control replay, cwd `$S/r3b_control` (DB `f453ab98…`) | **0** | equals R3's rows in all 47 columns; DB copy unchanged |
| `python $S/attrib_targets.py $S` | 0 | section 3.3 table |
| `check_dividend_freshness(conn, tickers=required, as_of=…)` on both copies | 0 | before: VMBS/BND stale from April, VOO/VWO from May, VXUS/VDE from August; after: only VWO at 2026-05-22 |
| Hybrid DB build, then February replay, cwd `$S/r3b_hybrid` | **0** | equals the refreshed February row |
| `python $S/compare.py $S .` | 0 | sections 3.1 and 3.2 |
| `python scripts/checks/check_doc_links.py` | **0** | `[doc-links] 354 files, 0 broken links` |
| `sha256sum data/pgr_financials.db` before the full suite (12:19 UTC) | 0 | `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6` |
| `python -m pytest -o addopts="--tb=short" -q` at `8c5d43e`, with the tree left untouched throughout | **0** | **`2585 passed, 1 skipped, 131 warnings in 756.03s (0:12:36)`** |
| `sha256sum data/pgr_financials.db` after the full suite (12:32 UTC); `git status --short` | 0 | `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`, unchanged; status empty |

The full-suite tree `8c5d43e` differs from this final record only in this
table. The suite matches R2-lite's final count (2585 passed, 1 skipped),
as expected for a docs-only change.

## What is left

1. **VWO March 2026.** `data_ready` fails at 2026-05-22 on VWO dividends,
   and 12 VWO targets carry no March 2026 payment. The owner has accepted
   this as provider data. A back-dated May 2026 decision can therefore never
   pass readiness on this DB. A current decision is not affected: VWO's
   later ex-dates are stored.
2. **Readiness label of in-month back-dated runs.** `readiness_basis` says
   `live` whenever the run falls in the as-of month, even when the DB holds
   data fetched after the as-of date (2026-09-21, run 2026-09-28). The DB
   has no fetch timestamps, so no label can prove what a decision had.
   Changing the rule is a code change, outside R3b.
3. From R3: `scripts/verify_monthly_outputs.py` still checks
   `check_data_freshness` without dividends; the gate itself covers them.
4. v200 creates its clean lock by copying and hashing this same DB
   (`38991c76…`).

This replay is a pre-v200 smoke comparison of already-inspected history. It
is not promotion evidence, and it shows no improvement in forecast skill or
investment return.

## Appendix — scripts

Run from the scratch directory or, where they import project modules, with
the repository root as cwd. `compare.py` and the freshness check import
pandas and `src.database.db_client`. `attrib_targets.py` uses only the
standard library.

### `hashes.sh`

```bash
#!/bin/bash
# Usage: hashes.sh <label>
S=/tmp/claude-0/-home-user-pgr-vesting-decision-support/932e520f-2d34-52f7-9ce0-c01682ab9202/scratchpad
SRC=/home/user/pgr-vesting-decision-support
C=$S/r3b_clone
echo "== $1 $(date -u +%FT%TZ)"
sha256sum $SRC/data/pgr_financials.db $C/data/pgr_financials.db
sha256sum $SRC/artifacts/monthly_decisions/decision_log.md $SRC/artifacts/monthly_decisions/classification_shadow_history.csv $SRC/artifacts/monthly_decisions/ta_shadow_variant_history.csv
echo "source_ledgers_combined $(cat $SRC/artifacts/monthly_decisions/decision_log.md $SRC/artifacts/monthly_decisions/classification_shadow_history.csv $SRC/artifacts/monthly_decisions/ta_shadow_variant_history.csv | sha256sum | cut -c1-64)"
echo "source_artifacts_tree $(cd $SRC && find artifacts -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)"
echo "clone_artifacts_tree  $(cd $C && find artifacts -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)"
ls $SRC/data/ $C/data/ | tr '\n' ' '; echo
```

### `attrib_targets.py`

```python
"""Which changed 6M targets of the 8 primary benchmarks are matured at each as-of date.

Reads two immutable DB copies (pre-refresh f453ab98, post-refresh 38991c76).
Maturity rule: window end = last business day of the month 6 months after
the anchor, independently computed with the standard library (not the
production helper); kept if window end <= as-of.
"""
from __future__ import annotations

import calendar
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

S = Path(sys.argv[1])
PRIMARY = ["VOO", "VXUS", "VWO", "VMBS", "BND", "GLD", "DBC", "VDE"]
AS_OFS = ["2026-02-28", "2026-03-31", "2026-04-22", "2026-05-22",
          "2026-06-22", "2026-07-22", "2026-08-20", "2026-09-21"]


def ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


def bme(year: int, month: int) -> date:
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def window_end(anchor: str, h: int) -> date:
    y, m = int(anchor[:4]), int(anchor[5:7])
    m += h
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return bme(y, m)


def rows(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    q = ("SELECT date, benchmark, relative_return FROM monthly_relative_returns "
         "WHERE target_horizon = 6")
    return {(d, b): r for d, b, r in conn.execute(q)}


def divs(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    return {(t, d): a for t, d, a in conn.execute(
        "SELECT ticker, ex_date, amount FROM daily_dividends")}


before_c, after_c = ro(S / "before_refresh.db"), ro(S / "r2lite_after_pinned.db")
b, a = rows(before_c), rows(after_c)
db, da = divs(before_c), divs(after_c)
delta_divs = {k: (db.get(k), v) for k, v in da.items() if db.get(k) != v}
assert set(b) == set(a)
changed = {k: a[k] - b[k] for k in a if k[1] in PRIMARY and a[k] != b[k]}
print("changed primary 6M rows:", len(changed))
for as_of in AS_OFS:
    ao = date.fromisoformat(as_of)
    mat = sorted((k for k in changed if window_end(k[0], 6) <= ao), key=lambda k: (k[1], k[0]))
    material = [k for k in mat if abs(changed[k]) > 1e-12]
    print(f"\n== as-of {as_of}: matured changed rows {len(mat)} (|d|>1e-12: {len(material)})")
    per_b: dict[str, list[str]] = {}
    for k in material:
        anchor, bench = k
        wend = window_end(anchor, 6)
        dd = [f"{t[1]}" + ("r" if delta_divs[t][0] is not None else "")
              for t in delta_divs if t[0] == bench and anchor < t[1] <= wend.isoformat()]
        per_b.setdefault(bench, []).append(
            f"{anchor}->{wend} d_rel={changed[k]:+.5f} divs[{','.join(sorted(dd))}]")
    for bench, items in per_b.items():
        print(f"  {bench}: {len(items)} rows")
        for it in items:
            print("    ", it)
    noise = [k for k in mat if abs(changed[k]) <= 1e-12]
    if noise:
        print(f"  float-noise-only rows (|d|<=1e-12): {len(noise)} ({sorted(set(k[1] for k in noise))})")
```

### Hybrid DB (DBC changes only)

```python
import sqlite3
from pathlib import Path

S = Path("<scratch>")
# cp before_refresh.db hybrid_dbc_only.db first; the pre-refresh copy is never modified.
c = sqlite3.connect(S / "hybrid_dbc_only.db")
c.execute("ATTACH DATABASE ? AS a", ((S / "r2lite_after_pinned.db").as_uri() + "?mode=ro&immutable=1",))
c.execute("DELETE FROM main.daily_dividends WHERE ticker='DBC'")                      # 9 rows
c.execute("INSERT INTO main.daily_dividends SELECT * FROM a.daily_dividends WHERE ticker='DBC'")
c.execute("DELETE FROM main.monthly_relative_returns WHERE benchmark='DBC'")          # 476 rows
c.execute("INSERT INTO main.monthly_relative_returns SELECT * FROM a.monthly_relative_returns WHERE benchmark='DBC'")
c.commit()
```

Check: `EXCEPT` in both directions against the refreshed DB leaves only
non-DBC tickers in `daily_dividends` and `monthly_relative_returns`.

### `compare.py` (core)

```python
r3 = pd.read_csv(REPO / "docs/reviews/R3_smoke_replay_rows.csv")
r3 = r3[r3["code"] == "R3 1b94bf0"].drop(columns="code").set_index("as_of")
new = pd.read_csv(S / "r3b_replay.csv").set_index("as_of")
ctl = pd.read_csv(S / "control_prerefresh_replay.csv").set_index("as_of")
# For each shared column: numeric (non-bool) columns report max |after - before|
# (NaN == NaN); every other column is compared as text, cell by cell.
# Pairs compared: R3 vs control, control vs refreshed, R3 vs refreshed.
```

The 47-column control equality check aligns the control CSV to R3's column
order, drops R3's `code` and four retired CPCV columns, and compares every
cell as text (`fillna("NA").astype(str)`).
