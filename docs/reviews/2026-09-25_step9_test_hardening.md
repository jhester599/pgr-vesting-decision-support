# Review 2026-09-25, step 9 — test hardening sweep (WP12)

WP12 of [`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md): the parts
of finding F28 ("Test suite gaps") that steps 1–7 did not already cover.
Only tests, the test configuration, CI and one developer script change; no
production code changes, so no replay is needed.

All mutation runs used scratch clones (`git clone --no-hardlinks`); nothing
ran against the working tree's DB. No fetcher, `monthly_decision.py` or
external API was run. The committed DB hash is unchanged
(`7c68efbd…c35d`).

## 1. Mutation study (area-9 method)

`scripts/checks/mutation_study.py` (new) makes one change in a scratch
clone, runs the related test files, reverts, and reports KILLED or SURVIVED.
It refuses to run on the working tree. The F28 mutations are M01–M18 on
today's code: several sites moved in steps 1–7. M01 and M02 now remove the
filing-date placement that replaced the fixed EDGAR lag (F23). M18 counts
365 days instead of the calendar anniversary (F19).

| # | Mutation | Site | Review | Before (`master`) | After | Killed by |
|---|---|---|---|---|---|---|
| M01 | No EDGAR placement in `build_feature_matrix_from_db` | `feature_engineering.py` | survived | killed | killed | `test_edgar_filing_timing_wp10` |
| M02 | No filing-date placement for quarterly ROE | `feature_engineering.py` | survived | **survived** | killed | `test_m02_quarterly_roe_enters_on_or_after_its_filing_date` |
| M03 | `combined_ratio_ttm` window 12 → 3 | `feature_engineering.py` | survived | **survived** | killed | `test_m03_combined_ratio_ttm_is_a_twelve_month_mean` |
| M04 | BVPS YoY `pct_change(12)` → `(1)` | `feature_engineering.py` | survived | killed | killed | `test_price_features_wp2` |
| M05 | DRIP `new_shares = evt_value / div_price` | `total_return.py` | survived | killed | killed | `test_drip_closed_form`; also the new DRIP property |
| M06 | Targets drop all splits | `multi_total_return.py` | survived | killed | killed | `test_drip_closed_form` |
| M07 | `TimeSeriesSplit(max_train_size=None)` | `wfo_engine.py` | survived | **survived** | killed | WFO properties; `test_m07_…` |
| M08 | Live refit on full history | `wfo_engine.py` | survived | **survived** | killed | WFO properties; `test_m08_…` |
| M09 | CPCV `purged_size=0` | `wfo_engine.py` | survived | killed | killed | `test_validation_gating_wp7` |
| M10 | Swap the consensus λ-mix | `consensus_shadow.py` | survived | **survived** | killed | `test_m10_…` |
| M11 | Remove the IC clip | `consensus_shadow.py` | survived | **survived** | killed | `test_m11_…` |
| M12 | Remove the λ clip | `consensus_shadow.py` | survived | **survived** | killed | `test_m12_…` |
| M13 | UNDERPERFORM sells 75 % | `decision_rendering.py` | survived | **survived** | killed | `test_m13_sell_mapping_table` |
| M14 | Drop the CPCV FAIL gate | `decision_rendering.py` | survived | killed | killed | `test_validation_gating_wp7` |
| M15 | No conformal finite-sample correction | `conformal.py` | survived | **survived** | killed | `test_m15_…` |
| M16 | Flip the ACI update sign | `conformal.py` | survived | **survived** | killed | `test_m16_…` |
| M17 | Relative return sign flip | `multi_total_return.py` | killed | killed | killed | `test_multi_total_return` |
| M18 | LTCG at vest + 365 days | `capital_gains.py` | killed | killed | killed | `test_capital_gains`; also the new LTCG property |

**Survivors: review 16 of 18 → `master` 10 of 18 → 0 of 18.** Steps 1–7
had already killed M01, M04, M05, M06, M09 and M14. The new
`tests/test_mutation_kills_wp12.py` has one test per remaining survivor.
Its expected values are worked out by hand from the fixture, not
recomputed with the production formula.

Extra mutations (not in F28) show that the fixes to vacuous tests in
section 4 bite. **Survivors: 4 of 6 → 0 of 6.**

| # | Mutation | Before | After | Killed by |
|---|---|---|---|---|
| X19 | FRED `reindex(method="ffill")` → `"bfill"` (look-ahead) | survived | killed | `test_fred_features_use_only_past_data` |
| X20 | `insurance_cpi_mom3m` over 1 month | killed | killed | `test_insurance_cpi_mom3m_formula` |
| X21 | `vmt_yoy` over 1 month | survived | killed | `test_vmt_yoy_is_finite` |
| X22 | `cr_acceleration = diff(1)` | survived (test skipped) | killed | `test_cr_acceleration_is_3period_diff_of_ttm` |
| X23 | Valuation TTM availability without the 12-month max | survived | killed | `test_availability_date_covers_fills_in_ttm_window` |
| X24 | Fracdiff never accepts a candidate `d` | killed* | killed | `test_memory_preserved_via_correlation` |

\* Before, X24 was killed by `test_stationary_series`; the two tests F28
names passed vacuously.

## 2. Property tests over production code

The four v36 property files mostly checked their own arithmetic.

| File | Removed | Now checks |
|---|---|---|
| `test_property_wfo_temporal.py` | 6 properties on hand-built `FoldResult`s | `run_wfo`: every fold has an embargo of exactly `horizon + purge_buffer` rows, exactly 60 training rows, contiguous rows, 6-row disjoint increasing test windows ending on the last row, and `y_true` = target on the test dates. A fold's predictions are unchanged when targets after its training window and features outside its train/test rows are scrambled. `predict_current` ignores rows older than its window. |
| `test_property_return_calculations.py` | 6 arithmetic identities | `build_position_series`: without corporate actions value = shares × price. A split with the matching price drop leaves value unchanged at every date. DRIP ending value equals the closed form `shares × P_T × ∏(1 + d_i/P_i)`. The share count never falls. |
| `test_property_feature_engineering.py` | 4 inline formulas (VIF tests kept) | `calendar_momentum` against a loop over month-end closes. `trailing_52w_high` is the 364-day max, so the ratio is in (0, 1]. `weekly_realized_vol` is ≥ 0 and scale-free. `split_adjusted_close` removes a split from momentum and volatility. `build_feature_matrix` is causal: rows up to t do not move when later prices change. |
| `test_property_tax_boundaries.py` | 3 tautologies (constant checks kept) | `optimize_sale`: shares and dollars are conserved and tax = gain × rate per lot. Only vested lots are sold, none oversold, and only the last lot used is partly sold. Lots are used loss → LTCG → STCG. Each holding type matches an independent calendar rule. With one rate, the tax is the minimum over allocations. Overselling raises. The LTCG boundary is the day after the calendar anniversary (29 Feb → 28 Feb). |

Run with only the property files, the four mutations they target (M05,
M07, M08, M18) all survive the old files and are all killed by the new ones
(**4 of 4 → 0 of 4**).

The VIF properties failed intermittently: the first example exceeded
Hypothesis's 200 ms deadline while statsmodels was imported. Those two
tests now have `deadline=None`.

Property testing also found an edge case in `run_wfo` (fixed in v182, the
follow-up on this branch). The documented minimum was train (60) + gap +
one test window (6). But below train + gap + **two** test windows,
`TimeSeriesSplit` got `n_splits=1` and raised sklearn's "n_splits=2 or
more" `ValueError`. `_min_required_observations` is now train + gap + 2 ×
test. Below it, `run_wfo`, `evaluation.iter_wfo_splits` and the x2
research splitter raise their own "too small" error. Callers already
treated any `ValueError` as "too little data", so no output changes. Every
fold still trains on exactly 60 rows, which the WFO property now asserts.
Its generator starts at the minimum (`extra_rows` from 0).

## 3. Repository guard (autouse)

`tests/repo_guard.py` holds a `sys.addaudithook` classifier, and
`tests/conftest.py` installs it. From test setup to teardown it watches
every Python-level `open`, `sqlite3.connect`, `mkdir`, `remove`, `rename`,
`rmdir` and `rmtree`:

- **Writes inside the repository tree fail**, except in `__pycache__` and
  `.pytest_cache`. So does a SQLite connection to any repository file that
  is not a `mode=ro` URI.
- **Opening the committed `data/pgr_financials.db` fails**, whether through
  SQLite or a plain `open`. The one exception is a read-only open in a test
  marked `artifact`.
- The refused call raises `PermissionError` where it happens. The test also
  fails at teardown, so swallowing the error does not help.
- An autouse fixture points `config.DB_PATH` and
  `feature_engineering._PROCESSED_PATH` at `tmp_path`. The DB path names a
  file that does not exist, so code falling back to the default DB gets an
  empty one.
- The `committed_db_copy` fixture copies the committed DB to `tmp_path` and
  sets `config.DB_PATH` and `v37_utils.DB_PATH` to the copy. Research tests
  that run studies on real data use it; they are all `artifact` tests.
- Hypothesis's example database moves from `.hypothesis/` in the working
  directory to `$TMPDIR/pgr-vds-hypothesis`.

A discovery run on the clean clone (an audit-hook plugin that recorded
instead of refusing) found more than F28 counted:

| Access | F28 | Found |
|---|---|---|
| Tests writing `data/processed/feature_matrix.parquet` | 50 | 64 |
| Tests opening the committed DB | 4 | 27: 18 research smoke tests (18 open it read-write through `v37_utils.get_connection` or the `config.DB_PATH` default), 5 integrity tests, 3 dry-run tests (copy it), 1 hygiene test (reads its header) |
| Other writes inside the repo | — | none (the plugin ignored `.hypothesis/`, which the suite also created) |

After the change, the full suite leaves nothing in the working tree: no
`.hypothesis/` and no `feature_matrix.parquet`.
`tests/test_wp12_test_hygiene.py` runs ten probe tests
(`tests/guard_probe_wp12.py`, which is not collected by default) in a
subprocess. It checks that exactly the six probes that touch the
repository fail. Against `master`'s conftest, all six passed (the
fixture-dependent probes errored), and one wrote
`results/guard_probe_wp12.txt` into the clone.

**Limits.** Writes made inside C code are not audited (SQLite's `-wal` and
`-shm` files, for example). That is why any read-write connection to a
repository file is refused. Module-level code that runs at collection time
is not guarded.

## 4. Stored-artifact tests and other fixes

**`@pytest.mark.artifact`** (registered in `pyproject.toml`) marks 198
tests in 73 files. They are the tests the discovery run saw reading
committed data, filtered to those whose assertions are about that data:

- the committed research outputs in `results/research` and `results/v14`;
- the DB and CSV integrity tests;
- the three dry-run tests, which copy the committed DB;
- the WAL-header hygiene test;
- the 18 research smoke tests that run on real data.

CI's `test` job runs `-m "not artifact"`. A new `artifacts` job runs
`-m artifact`, and its comment says a failure there means committed outputs
changed or went stale. A plain `python -m pytest` still runs everything.

A module- or class-scoped fixture that reads committed data is recorded
only against the first test that uses it, so every test on such a fixture
is marked (the whole module for the DB and EDGAR integrity files). The
first CI run missed this. The `-m "not artifact"` job deselected the one
marked test, so the module's committed-DB connection opened in an unmarked
test and the guard refused it (24 errors). The autouse fixture also sends
`config.DATA_RAW_DIR` and `REQUEST_COUNTS_FILE` to `tmp_path`. A fresh CI
checkout has no `data/raw`, and the EDGAR client creates it before writing
its (already redirected) cache.

These tests read committed files but **stay in the main job**, because they
test production code that loads committed inputs (the F30 dependence on
`results/`):

- `test_monthly_pipeline_e2e`, `test_monthly_logging` and
  `test_shadow_followon` (the v113 and v141–v150 candidate files);
- `test_v129_feature_map` and `test_v129_dual_track_integration` (the v128
  map);
- `test_bl_fallback_monthly` and `test_policy_backtest_monthly` (the
  decision log);
- the email tests in `test_v65_p26_p27_p28` (the charts);
- the doc-link test.

Other fixes:

- **Deterministic seeds.** `tests/test_integration.py` seeded prices with
  `hash(ticker)`, which `PYTHONHASHSEED` salts per process, so every run
  used different data. It now uses `zlib.crc32(ticker)`. A test checks
  that the prices are identical under three hash seeds.
- **FRED formula guards.** The `if col in df.columns:` guards are gone.
  - `test_fred_features.py` (`yield_curvature`, `real_rate_10y`,
    `yield_slope`) and `test_pgr_fred_features.py` (`insurance_cpi_mom3m`,
    `vmt_yoy`) now assert that the column exists and has values, and check
    exact values.
  - The "only past data" test used to check only finiteness. It now
    compares with the last FRED row on or before each date. It scrambles
    FRED after a cut-off and requires unchanged features up to it. And it
    requires NaN before the first FRED row.
  - These fixtures used calendar month-ends (`ME`), but
    `build_feature_matrix` is documented to receive business-month-end rows
    (as `_apply_fred_lags` produces). On weekend month-ends that repeated a
    month, so the fixtures are now `BME`.
- **`test_v45_features.py`.** The PGR monthly fixture had 36 months from
  2022-01. That is fewer than `WFO_MIN_GAINSHARE_OBS` (60) inside the price
  range, so `combined_ratio_ttm` was dropped and the diff test skipped. The
  fixture now has 72 months from 2018-01, and the test asserts at least 60
  values instead of skipping. The suite's skips fall from 2 to 1 (the
  manual `test_classification_shadow` integration test).
- **Two more tests that could not fail** (listed in F28, not in steps 1–7):
  - `test_fracdiff.py`: on 80 monthly values, the fixed-width window
    (weights down to 1e-5) leaves one differenced value. So every
    candidate was skipped and the guarded assertions never ran. The tests
    now use 3,000 bars with no guards. The default 0.90 correlation is
    never reached together with stationarity on a random walk (the
    fallback `d = 0.5` keeps 0.45–0.58), so they use 0.40. `apply_fracdiff`
    has no production caller.
  - `test_valuation_multiples.py::test_availability_date_covers_fills_in_ttm_window`:
    its first month with a TTM value was filed after the 10-Q anyway. The
    history now starts in 2014-06, so June 2015 (its 8-K was filed
    2015-07-15) must wait for the 2015-08-14 10-Q.
- `test_ops_wp8.py`'s workflow-step parser ran the last step of a job into
  the next job. It now stops at the job boundary; the new `artifacts` job
  exposed this.

## 5. Tests: failing before, passing after

| Test | Before | After |
|---|---|---|
| `test_mutation_kills_wp12.py` (17 tests) | each fails under its mutation (M02, M03, M07, M08, M10–M13, M15, M16) | pass |
| New property files (17 properties) | the old files never failed under M05/M07/M08/M18 | pass; each mutation is killed |
| `test_wp12_test_hygiene.py::test_guard_fails_exactly_the_probes_that_touch_the_repo` | fails (the six repo-touching probes pass) | pass |
| `…::test_hypothesis_storage_is_outside_the_repo` | fails | pass |
| `…::test_ci_runs_artifact_tests_in_a_separate_job`, `…::test_artifact_marker_is_registered` | fail | pass |
| `…::test_integration_prices_do_not_depend_on_the_hash_seed` | fails (three digests) | pass |
| FRED, v45, fracdiff and valuation-multiples tests | pass (vacuously or skipped) under X19, X21, X22, X23 | fail under them; pass on the code |

Commands (on the branch, `.venv`):

- `python -m pytest -o addopts="--tb=short" -q tests/test_mutation_kills_wp12.py tests/test_wp12_test_hygiene.py tests/test_property_*.py`
- Before: the same files copied into a clone of `master`
  (`test_wp12_test_hygiene.py`: 5 failed, 20 passed).

## 6. Full suite

- `master` (clean clone): 2397 passed, 2 skipped, as in v180. In the clone,
  5 `test_entrypoint_imports` cases fail because the editable install
  points at the main checkout. They pass in the working tree.
- Branch: `python -m pytest -o addopts="--tb=short" -q` → `2437 passed,
  1 skipped, 109 warnings in 591.03s`. The skip is
  `test_classification_shadow.py:280` (manual integration test). Afterwards
  `git status --ignored` shows no new files outside caches, and the DB hash
  is unchanged.

## Judgement calls

- **Artifact tests may read the committed DB.** "Fail any test that opens
  the real DB" applies to the unit suite. The integrity and research tests
  exist to check committed data, so they are marked `artifact` and may
  open it read-only (a `mode=ro` SQLite URI or a plain read). A read-write
  open still fails; those tests use `committed_db_copy`.
- **Reads of committed CSVs are not refused** in the main job. Production
  code still loads `results/research` files (F30), so tests of that code
  have to read them.
- **M01/M02 and M18 were re-expressed** for today's code. The review's
  lag-removal and `>=365` sites no longer exist.

## Not done here

- `apply_fracdiff`: its defaults (0.90 correlation, 1e-5 weight threshold)
  never produce output on monthly-length series. It is unused in
  production (F31).
- The F30 production dependence on `results/research` files keeps 8
  production-code test files reading committed research outputs in the
  main job.
- The guard does not cover collection time or C-level file writes.
