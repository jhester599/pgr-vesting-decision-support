# Autoresearch Optimization Execution Plan — 2026-04-13

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run eight targeted autoresearch optimization loops against the pgr-vesting-decision-support project, advancing the Path B temp-scaled classifier shadow toward its 24-month promotion gate and tightening the regression ensemble's calibration and hyperparameter choices.

**Architecture:** Each section is a self-contained autoresearch loop specification. Prerequisites must be satisfied before the loop runs. Guards protect invariants that must not regress between iterations. Metrics are single-number shell extractions that autoresearch can drive without human intervention.

**Tech Stack:** Python 3.10+, pytest, pandas, numpy, scikit-learn, bash/sh. All commands assume project root is the working directory.

**Platform note:** All shell commands use POSIX syntax (forward-slash paths, bash). Compatible with Claude Code (native terminal) and Codex (bash via shell tool). In Codex, prefix each command with `cd /repo &&` if the working directory is not set to the project root automatically.

**Current baselines (as of 2026-04-13):**

| Layer | Version | Key metric |
|---|---|---|
| Production regression | v38 (shrinkage α=0.50) | Pooled OOS R² = −0.1310 |
| Production consensus | v76 quality-weighted | IC = 0.1579, hit rate = 0.7002 |
| Shadow classifier | Path B temp-scaled (v130) | Covered BA = 0.5725, Brier = 0.1917, ECE = 0.1570 |
| Abstention thresholds | v131 candidate (0.15, 0.70) | Covered BA = +6 pp vs current 0.5725; v132 temporal hold-out validates |

Promotion gate for shadow classifier (all must be met before any production role):
1. ≥ 24 matured prospective months
2. Calibrated Brier ≤ 0.18, ECE ≤ 0.08, no upward trend
3. Covered BA ≥ 0.60 at coverage ≥ 0.25
4. Actionable-sell precision ≥ 0.70
5. Agreement rate with regression baseline in [0.80, 0.97]
6. Non-negative cumulative policy uplift on disagreement months

---

## Target 1 — Per-Benchmark Feature Subsetting (v129 Shadow Integration)

### Prerequisites

The v128 research phase is complete. The following artifacts already exist and are the source of truth for this loop:

- `results/research/v128_benchmark_feature_map.csv` — per-benchmark winning feature set (4 of 10 benchmarks switched from `lean_baseline`)
- `results/research/v128_benchmark_feature_search_summary.md` — human-readable verdict per benchmark
- `results/research/v128_baseline_metrics.csv` — incumbent covered BA per benchmark (pre-subsetting)

Before the loop can run, a lightweight eval harness must be wired so autoresearch can apply a candidate feature-routing strategy and measure covered BA in a single shell command. The harness lives at:

- **`results/research/v129_feature_map_eval.py`** — accepts `--strategy {lean_baseline | v128_map | file:<path>}` and prints `covered_ba=X.XXXX` and `coverage=X.XXXX` to stdout using the v125 fold detail as the evaluation frame
- **`config/model.py`** — `V128_BENCHMARK_FEATURE_MAP_PATH` already exists; verify it points to the correct CSV

### Estimated iteration time

~45 seconds per eval (loads precomputed fold-detail CSV, applies per-benchmark routing, scores). No WFO re-training required. Overnight budget of 8 hours ≈ 640 iterations.

### Setup tasks

1. **Create `results/research/v129_feature_map_eval.py`**

   Function signature:
   ```python
   def evaluate_feature_map(
       strategy: str,                    # "lean_baseline" | "v128_map" | "file:<csv_path>"
       fold_detail_path: str,            # default: results/research/v125_portfolio_target_fold_detail.csv
       low_thresh: float = 0.30,         # abstention lower bound
       high_thresh: float = 0.70,        # abstention upper bound
   ) -> dict[str, float]:               # keys: covered_ba, coverage, brier, ece_10
   ```

   The script must:
   - Load `v125_portfolio_target_fold_detail.csv`
   - If `strategy == "lean_baseline"`, apply the shared 12-feature set to all benchmarks
   - If `strategy == "v128_map"`, load `V128_BENCHMARK_FEATURE_MAP_PATH` and route each benchmark's folds to its winning feature set; recompute per-benchmark WFO predictions using the stored fold-level coefficients (not re-training — just re-selecting features from stored logit outputs)
   - If `strategy == "file:<csv_path>"`, load the CSV (same schema as `v128_benchmark_feature_map.csv`) and apply it
   - Print exactly two lines to stdout: `covered_ba=X.XXXX` and `coverage=X.XXXX`
   - Exit 1 if coverage < 0.20

2. **Write `tests/test_research_v129_feature_map_eval.py`** with tests for:
   - `lean_baseline` strategy returns covered BA close to v128 baseline (within 0.01)
   - `v128_map` strategy returns covered BA ≥ lean baseline (the v128 winner constraint)
   - Coverage threshold exit-code behavior

3. **Verify harness produces a stable baseline reading:**
   ```bash
   python results/research/v129_feature_map_eval.py --strategy lean_baseline
   # Expected: covered_ba=0.5000 (±0.01), coverage=0.87 (±0.02)
   ```

### Autoresearch invocation

```
/autoresearch
Goal: Identify the optimal per-benchmark feature routing strategy for the Path B
      temp-scaled classifier by extending the v128 benchmark-specific feature search.
      The v128 harness found that VGT improved from BA=0.5789 to 0.9474 using only
      2 features (rate_adequacy_gap_yoy, severity_index_yoy). This loop explores
      alternative feature-map configurations — reordering, swapping, or trimming
      per-benchmark sets — to maximize pooled covered balanced accuracy without
      increasing the maximum per-benchmark feature count above 12.

Metric: python results/research/v129_feature_map_eval.py --strategy file:results/research/v129_candidate_map.csv 2>/dev/null | grep "covered_ba=" | cut -d= -f2

Scope: Contents of results/research/v129_candidate_map.csv only. The CSV has
       columns [benchmark, feature_set, feature_list]. Autoresearch may modify
       any row's feature_set label and feature_list (comma-separated feature names
       from the 72-column candidate universe in v128_feature_inventory.csv).
       Do NOT modify: classification_shadow.py, config/model.py, any test file,
       v128_benchmark_feature_map.csv (the v128 source of truth).
       Feature count per benchmark must not exceed 12.
       Only feature names appearing in v128_feature_inventory.csv are eligible.

Guard: python -m pytest tests/test_research_v129_feature_map_eval.py tests/test_classification_shadow.py -q --tb=short 2>&1 | tail -1 | grep -E "[0-9]+ passed"

Direction: higher_is_better

Iterations: 40

Log: results/research/v129_autoresearch_log.jsonl
```

### Guard design rationale

The guard runs two test files:
- `test_research_v129_feature_map_eval.py`: ensures the eval harness remains well-behaved and the candidate map is structurally valid (right columns, eligible feature names, count ≤ 12)
- `test_classification_shadow.py`: ensures no accidental modification to the live shadow inference path (v129 is research-only — wiring to production shadow requires a separate explicit decision)

These two files together protect the invariant that research exploration cannot silently corrupt the inference pipeline.

### Known risks

- **Coverage collapse:** Aggressive feature reduction on a benchmark can cause its WFO folds to have insufficient positive examples, degrading coverage. Mitigate by checking coverage ≥ 0.20 in the eval script exit code.
- **VGT overfitting:** The 2-feature VGT result (BA = 0.9474) is based on retrospective fold evaluation. If autoresearch further specializes VGT features, the retrospective gain may not generalize prospectively. Flag any VGT BA > 0.90 as a warning requiring manual review before shadow adoption.
- **Feature name drift:** The 72-column candidate universe may not match exactly between the v128 inventory and the live feature builder. Run `python results/research/v128_benchmark_feature_search.py --inventory-check` before deploying any candidate map to shadow.

### Success criteria

Pooled covered BA ≥ 0.5100 (vs incumbent 0.5000 from v128 pooled result), with per-benchmark ECE ≤ 0.06 on average and Brier ≤ 0.185. Improvement must hold in at least 6 of the 10 benchmarks individually.

### Sequencing note

Run **after** Target 5 (temperature scaling) because the optimal temperature T affects the effective probability scale, which in turn affects how feature-routing changes manifest in the BA metric. Can run concurrently with Target 3 (Ridge alpha) since they touch independent model families.

---

## Target 2 — Test Suite Runtime Reduction

### Prerequisites

No new fixtures or scripts needed. The current suite has 1,683 tests with collection time of 1.49s. The runtime harness needs only:

- **`scripts/measure_test_time.sh`** — thin wrapper that times `pytest` and prints `elapsed_seconds=XX.X` to stdout:
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  START=$(python -c "import time; print(time.time())")
  python -m pytest --tb=no -q "$@" > /dev/null 2>&1
  python -c "import time; print(f'elapsed_seconds={time.time()-$START:.1f}')"
  ```
- Run `bash scripts/measure_test_time.sh` once to establish the current baseline before the loop starts.

### Estimated iteration time

The eval is the test suite itself. Current runtime is unknown — establish the baseline in the setup task. If it is under 60 seconds, each autoresearch iteration takes ~60s plus overhead. If over 5 minutes, limit overnight iterations accordingly (see sequencing note).

### Setup tasks

1. **Profile the current suite to find the 10 slowest tests:**
   ```bash
   python -m pytest --tb=no -q --durations=10 2>&1 | tail -15
   ```
   Record the 10 slowest tests and their approximate times in a comment block at the top of `docs/superpowers/plans/2026-04-13-autoresearch-execution-plan.md` (update in place after profiling).

2. **Create `scripts/measure_test_time.sh`** as shown in Prerequisites above. Make it executable:
   ```bash
   chmod +x scripts/measure_test_time.sh
   ```

3. **Verify the baseline produces a stable number:**
   ```bash
   bash scripts/measure_test_time.sh
   # Note the elapsed_seconds value — this becomes the ceiling for the guard.
   ```

4. **Identify categories of test slowness.** Common causes in this codebase:
   - WFO integration tests that run full TimeSeriesSplit on real data (`tests/test_wfo_engine.py`, 34 tests)
   - Shadow evaluation tests that load the full feature matrix
   - Research script smoke tests with long compute paths
   
   Create `tests/conftest.py` (or update if it exists) with a `--fast` marker that skips tests marked `@pytest.mark.slow`.

5. **Mark the 5 slowest tests** with `@pytest.mark.slow` in their respective test files without changing test logic.

### Autoresearch invocation

```
/autoresearch
Goal: Reduce the total pytest runtime for the pgr-vesting-decision-support test
      suite while keeping all 1,683 tests passing. The primary lever is identifying
      and optimizing fixture setup, mock boundaries, and parametrize redundancy in
      the slowest tests. Do NOT delete tests or reduce assertion coverage.

Metric: bash scripts/measure_test_time.sh 2>/dev/null | grep elapsed_seconds= | cut -d= -f2

Scope: Any test file in tests/ (fixture refactoring, parametrize reduction,
       mock boundaries, conftest.py shared fixtures). Source files in src/ may
       be modified only to add/improve caching on pure functions.
       Do NOT modify: config/model.py, any results/research/ scripts,
       any scripts/ entry points, or any test assertion logic.
       Test count must not drop below 1,683.

Guard: python -m pytest --tb=short -q 2>&1 | tail -3 | grep -E "1[6-9][0-9]{2} passed|[2-9][0-9]{3} passed"

Direction: lower_is_better

Iterations: 20

Log: results/research/v_test_runtime_autoresearch_log.jsonl
```

### Guard design rationale

The guard checks that at least 1,683 tests pass (the `1[6-9][0-9]{2}` pattern matches 1600–1999). This prevents autoresearch from achieving a "speedup" by skipping tests. The pattern intentionally allows test count to increase (new tests added while optimizing fixtures are valid) but not to decrease.

### Known risks

- **Fixture refactoring may hide slow I/O:** If a shared fixture caches a database load, the first test in a session pays the cost but subsequent tests appear "fast." The aggregate runtime may not decrease if the bottleneck is I/O that was previously parallelized implicitly.
- **Mocking real computations changes test fidelity:** If autoresearch mocks out WFO calls in integration tests to speed them up, it may miss regressions. The guard only checks test count, not assertion quality. Review any new mocks manually before accepting the iteration.
- **Windows path sensitivity:** The codebase runs on Windows 11 Pro. Timing measurements via bash may be imprecise if bash is running under WSL or Git Bash. Verify `scripts/measure_test_time.sh` gives consistent results across three runs before trusting the metric.

### Success criteria

Total runtime reduced by ≥ 20% from the baseline established in setup task 3, with 0 test failures and test count ≥ 1,683.

### Sequencing note

Run independently — no dependencies on other loops. Can run **in parallel with any other loop** since it only modifies test infrastructure. Best run first overnight to reclaim iteration budget for other loops.

---

## Target 3 — Ridge Alpha Grid Exploration (High-Alpha Region)

### Prerequisites

The current Ridge alpha grid in `src/models/regularized_models.py:build_ridge_pipeline()` spans `np.logspace(-4, 2, 50)` (αmax = 100). The v37 research cycle showed that the single most effective intervention was post-hoc shrinkage (equivalent to high regularization), yet the Ridge tuning grid has never been extended into the high-alpha region (100 – 10,000). The loop tests whether the inner TimeSeriesSplit CV selects higher-alpha Ridge solutions when given the chance, improving pooled OOS R² beyond −0.1310.

Required before loop runs:
- **`results/research/v133_ridge_alpha_sweep.py`** — standalone WFO sweep that accepts `--alpha-min FLOAT --alpha-max FLOAT --n-alpha INT` and prints `pooled_oos_r2=X.XXXX` to stdout. Uses the same WFO configuration as production (`WFO_TRAIN_WINDOW_MONTHS=60`, `WFO_TEST_WINDOW_MONTHS=6`, `WFO_EMBARGO_MONTHS_6M=6`, `WFO_PURGE_BUFFER_6M=2`).
- **`config/model.py`** — add `RIDGE_ALPHA_MAX: float = 1e2` and `RIDGE_ALPHA_MIN: float = 1e-4` as named constants so the sweep respects config boundaries.

### Estimated iteration time

One full WFO pass across all production benchmarks takes approximately 4–8 minutes depending on benchmark count and training window. At 8 benchmarks × 60/6 = 10 folds each, this is ~80 Ridge CV fits per iteration. Overnight budget ≈ 60–120 iterations at 4 min/iter.

### Setup tasks

1. **Add constants to `config/model.py`** (after the existing shrinkage alpha block, ~line 46):
   ```python
   # v133 — Ridge inner CV alpha grid bounds (autoresearch-tunable)
   RIDGE_ALPHA_MIN: float = 1e-4   # lower bound of logspace grid
   RIDGE_ALPHA_MAX: float = 1e2    # upper bound — extend to 1e4 for high-alpha sweep
   RIDGE_ALPHA_N: int = 50         # grid resolution
   ```

2. **Create `results/research/v133_ridge_alpha_sweep.py`** with this interface:
   ```python
   def run_ridge_alpha_sweep(
       alpha_min: float,        # e.g. 1e-4
       alpha_max: float,        # e.g. 1e4
       n_alpha: int = 50,       # grid resolution
       benchmarks: list[str] | None = None,  # default: PRIMARY_FORECAST_UNIVERSE
   ) -> dict[str, float]:      # keys: pooled_oos_r2, pooled_ic, pooled_hit_rate, pooled_sigma_ratio
   ```
   
   The script must:
   - Pass `alphas=np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)` to `build_ridge_pipeline()`
   - Run the standard `run_multi_benchmark_wfo()` with `model_type="ridge"`
   - Pool results across benchmarks using inverse-variance weighting (same as production)
   - Print `pooled_oos_r2=X.XXXX` as the first stdout line (autoresearch reads this)

3. **Establish the baseline reading** (must match v38 production baseline):
   ```bash
   python results/research/v133_ridge_alpha_sweep.py --alpha-min 1e-4 --alpha-max 1e2 --n-alpha 50
   # Expected: pooled_oos_r2=-0.1310 (±0.005)
   ```
   If baseline does not match, do not run the loop — investigate the discrepancy first.

4. **Write `tests/test_research_v133_ridge_alpha_sweep.py`** with:
   - Smoke test: default grid produces pooled_oos_r2 in range [−0.30, 0.10]
   - Guard test: `alpha_max < alpha_min` raises ValueError
   - Guard test: `n_alpha < 10` raises ValueError

### Autoresearch invocation

```
/autoresearch
Goal: Determine whether extending the Ridge inner-CV alpha grid into the high-alpha
      region (100–10,000) yields a pooled OOS R² better than the current production
      baseline of −0.1310 (v38 shrinkage α=0.50). The v37–v60 research cycle showed
      that shrinkage was the single most effective intervention; this loop tests
      whether the Ridge regularization path can internalize the same effect via
      stronger alpha values rather than post-hoc prediction scaling.

Metric: python results/research/v133_ridge_alpha_sweep.py --alpha-min 1e-4 --alpha-max $(cat results/research/v133_alpha_max_candidate.txt) --n-alpha 60 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2

Scope: The file results/research/v133_alpha_max_candidate.txt (a single float
       on one line, e.g. "10000.0"). Autoresearch tunes this one number.
       Valid range: [1e2, 1e5]. Values outside this range must be rejected
       by the guard.
       Do NOT modify: config/model.py, regularized_models.py, any test files,
       any production source files.

Guard: python -c "v=float(open('results/research/v133_alpha_max_candidate.txt').read()); assert 1e2 <= v <= 1e5, f'alpha_max={v} out of range'" && python -m pytest tests/test_research_v133_ridge_alpha_sweep.py -q --tb=short 2>&1 | tail -1 | grep "passed"

Direction: higher_is_better

Iterations: 30

Log: results/research/v133_ridge_alpha_autoresearch_log.jsonl
```

### Guard design rationale

The guard has two parts:
1. **Range check on `v133_alpha_max_candidate.txt`**: prevents autoresearch from setting an alpha so large that the inner CV collapses to a constant predictor (all predictions = grand mean), which would produce a deceptively good (low-variance) OOS R² on a small dataset.
2. **Pytest smoke on the v133 script**: ensures the harness remains runnable and the alpha_min/max validation logic has not been broken.

### Known risks

- **Overfitting the grid:** If the inner TimeSeriesSplit selects a very high alpha (e.g., 10,000) on the training folds but the test folds have different variance structure, OOS R² may improve in the backtest while degrading on prospective data. Flag any iteration where the selected alpha consistently hits the grid upper bound — this is a boundary condition that warrants extending the grid further rather than treating the current best as stable.
- **Confound with shrinkage:** The production v38 baseline includes post-hoc prediction shrinkage (α=0.50 applied to predictions after the fold). If this loop's Ridge CV selects high alpha, it effectively double-regularizes. Disable `ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA` in the sweep script to test the pure effect of Ridge alpha.
- **Benchmark heterogeneity:** Some benchmarks (GLD, DBC) have materially different variance profiles. A pooled metric may mask that high alpha helps some benchmarks and hurts others. Check per-benchmark R² in the sweep artifacts before promoting.

### Success criteria

Pooled OOS R² ≥ −0.10 (improvement of +0.03 over v38 baseline −0.1310) with pooled IC ≥ 0.1579 and hit rate ≥ 0.70. Both conditions must be met; IC/hit-rate must not regress.

### Sequencing note

Run **before** Target 7 (GBT hyperparameter search) — if high Ridge alpha already closes the performance gap, the GBT sweep may be deprioritized. Independent of all classifier targets (1, 5, 8).

---

## Target 4 — FRED Publication Lag Optimization

### Prerequisites

The current `FRED_SERIES_LAGS` dict in `config/model.py` assigns 1 month to most series and 2 months to NFCI and TRFVOLUSM227NFWA. Real-world publication lags vary: CPI sub-series (CUSR0000*) are typically released 2–3 weeks after month-end but sometimes slip to 5 weeks; T10Y2Y and GS-series are daily, effectively lag-free; PCE-related series (PPIACO, WPU45110101) can lag 4–6 weeks. Using lag=1 month universally may be overly conservative for daily-frequency series and insufficiently conservative for quarterly-adjusted releases.

Required before loop runs:
- **`results/research/v134_fred_lag_sweep.py`** — standalone harness accepting a JSON lag-override dict `--lag-overrides '{"GS10": 0, "T10Y2Y": 0, "VIXCLS": 0}'` and printing `pooled_oos_r2=X.XXXX` to stdout
- **`docs/data/fred_publication_lag_reference.md`** — reference table of actual FRED publication lags (one-time research task, not part of the loop; create before running)

Known FRED publication constraints to document before running:
- `GS10`, `GS5`, `GS2`, `T10Y2Y`, `T10YIE`: daily H.15 release, can use lag=0
- `VIXCLS`: daily, lag=0 acceptable
- `NFCI`: weekly (Friday), typical lag 1 week; 2-month lag is overly conservative
- `BAA10Y`, `BAMLH0A0HYM2`: daily/weekly, lag=0 or 1 acceptable
- `PPIACO`, `WPU45110101`: monthly, released ~3 weeks after month-end; lag=1 is correct
- `MORTGAGE30US`: weekly (Thursday), lag=0 or 1 acceptable
- `SP500_*_MULTPL`: monthly; lag=1 is appropriate
- `DCOILWTICO`: daily EIA release, lag=0 acceptable

### Estimated iteration time

One WFO sweep with modified lags takes approximately 5–10 minutes (requires re-building the lagged feature matrix from the database for each lag configuration, then running WFO). Overnight budget ≈ 48–96 iterations.

### Setup tasks

1. **Create `docs/data/fred_publication_lag_reference.md`** documenting the real-world publication timeline for each series in `FRED_SERIES_LAGS`. One table row per series with columns: `series_id | release_name | typical_lag_days | safe_lag_months | current_config_months | recommended_min`.

2. **Create `results/research/v134_fred_lag_sweep.py`** with this interface:
   ```python
   def run_lag_sweep(
       lag_overrides: dict[str, int],   # series_id -> lag_months (0, 1, or 2 only)
       benchmarks: list[str] | None = None,
   ) -> dict[str, float]:              # keys: pooled_oos_r2, pooled_ic, pooled_hit_rate
   ```
   The script must:
   - Merge `lag_overrides` into a copy of `config.FRED_SERIES_LAGS` (not mutate the original)
   - Re-build the lagged feature matrix using the overridden lags (call `build_feature_matrix()` with the modified lag dict)
   - Run WFO with the same settings as production
   - Reject any lag value < 0 or > 3 with ValueError

3. **Write `tests/test_research_v134_fred_lag_sweep.py`** with:
   - Default (no overrides) produces pooled_oos_r2 ≈ −0.1310 (baseline recovery test)
   - `lag_overrides={"GS10": -1}` raises ValueError
   - `lag_overrides={"NONEXISTENT_SERIES": 1}` raises KeyError (unknown series rejected)

4. **Create `results/research/v134_lag_candidate.json`** as the file autoresearch will edit:
   ```json
   {"GS10": 1, "GS5": 1, "GS2": 1, "T10Y2Y": 1, "T10YIE": 1, "VIXCLS": 1, "BAA10Y": 1, "BAMLH0A0HYM2": 1, "MORTGAGE30US": 1}
   ```
   (Only the 9 series with real-world lag < 2 weeks are eligible candidates for reduction.)

### Autoresearch invocation

```
/autoresearch
Goal: Determine whether reducing publication lag assumptions for daily/weekly FRED
      series (GS10, GS5, GS2, T10Y2Y, T10YIE, VIXCLS, BAA10Y, BAMLH0A0HYM2,
      MORTGAGE30US) from the current 1-month default to 0 months improves pooled
      OOS R² beyond the v38 baseline of −0.1310. These series are released daily
      or weekly, so a 1-month lag assumption discards up to 4 weeks of valid data
      from each training fold.

Metric: python results/research/v134_fred_lag_sweep.py --lag-overrides "$(cat results/research/v134_lag_candidate.json)" 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2

Scope: The file results/research/v134_lag_candidate.json only. Autoresearch
       may set any of the 9 eligible series to 0 or 1. Values of 2 or higher
       are not permitted for these series. Series NOT in the candidate file
       (NFCI, TRFVOLUSM227NFWA, CUSR0000*, PPIACO, WPU45110101, PCU*, MRTSSM447USN,
       DTWEXBGS, DCOILWTICO, THREEFYTP10, SP500_*) must not appear in the JSON.
       Do NOT modify: config/model.py, any source files, any test files.

Guard: python -c "import json; d=json.load(open('results/research/v134_lag_candidate.json')); eligible={'GS10','GS5','GS2','T10Y2Y','T10YIE','VIXCLS','BAA10Y','BAMLH0A0HYM2','MORTGAGE30US'}; assert set(d)==eligible, f'unexpected keys: {set(d)-eligible}'; assert all(v in (0,1) for v in d.values()), f'invalid lag values: {d}'" && python -m pytest tests/test_research_v134_fred_lag_sweep.py -q --tb=short 2>&1 | tail -1 | grep "passed"

Direction: higher_is_better

Iterations: 25

Log: results/research/v134_fred_lag_autoresearch_log.jsonl
```

### Guard design rationale

The guard enforces two invariants:
1. **Eligible-series-only constraint:** Autoresearch cannot accidentally reduce lags for CPI, PPI, or other monthly surveys that genuinely have multi-week publication delays. Reducing those lags would introduce look-ahead bias into the training folds, producing a spuriously good backtest.
2. **Binary constraint (0 or 1 only):** Fractional lags or negative lags are structurally impossible in the monthly feature builder. Constraining to `{0, 1}` keeps the search space interpretable and prevents degenerate configurations.

### Known risks

- **Leakage from lag=0:** Even for daily-frequency series, using lag=0 means the month-end value is assumed available on the first day of the following month. If the feature builder uses month-end prices and the target is computed over the next 6 months, lag=0 is defensible. Verify the exact calendar logic in `build_feature_matrix()` before trusting results.
- **Interaction with purge/embargo:** The 6-month purge buffer already removes most overlap between training and test targets. Reducing lags from 1 to 0 for daily series is unlikely to introduce new leakage given the existing embargo design, but this should be confirmed.

### Success criteria

Pooled OOS R² ≥ −0.10 with pooled IC ≥ 0.1579 and hit rate ≥ 0.70. Even a modest improvement (−0.12 to −0.11) with stable IC is worth promoting to config given zero model complexity cost.

### Sequencing note

Run **before** Target 3 (Ridge alpha) in the overnight sequence — lag optimization is a data transformation that improves signal quality for all downstream models. If lags are optimized first, the Ridge alpha sweep operates on cleaner features.

---

## Target 5 — Path B Temperature Scaling Parameter Tuning

### Prerequisites

The v130 adoption verdict established that temperature-scaled Path B beats Path A on all three adoption criteria. The current temperature grid in `results/research/v131_threshold_sweep_eval.py` spans `[0.50–0.95] ∪ [1.0–3.0]` using a prequential warmup of 24 months. The v131 sweep then found that asymmetric thresholds (0.15, 0.70) improve covered BA by +6pp. Two tunable axes remain unexplored:

1. **The warmup window size** (currently fixed at 24 months): smaller warmup = more predictions evaluated; larger warmup = more stable temperature estimates
2. **The temperature grid upper bound** (currently 3.0): some calibration literature suggests T > 3 produces better reliability curves for overconfident classifiers

Required before loop runs:
- `results/research/v131_threshold_sweep_eval.py` already exists and accepts `--low FLOAT --high FLOAT`
- Need to extend it (or create `results/research/v135_temp_param_search.py`) to accept `--temp-min FLOAT --temp-max FLOAT --warmup INT`

### Estimated iteration time

~10–20 seconds per eval (operates on the precomputed `v125_portfolio_target_fold_detail.csv`, no WFO re-training). Overnight budget ≈ 1,440–2,880 iterations — more than sufficient for a thorough grid.

### Setup tasks

1. **Create `results/research/v135_temp_param_search.py`** extending the v131 evaluator:
   ```python
   def evaluate_temperature_config(
       temp_min: float,         # lower bound of temperature search grid
       temp_max: float,         # upper bound of temperature search grid
       n_temps: int = 51,       # grid resolution
       warmup: int = 24,        # prequential warmup months
       low_thresh: float = 0.15,  # abstention lower (v131 winner)
       high_thresh: float = 0.70, # abstention upper (v131 winner)
       fold_detail_path: str | None = None,
   ) -> dict[str, float]:      # keys: covered_ba, coverage, brier, log_loss, ece
   ```
   The script must:
   - Load `v125_portfolio_target_fold_detail.csv`
   - Build the temperature grid as `np.linspace(temp_min, temp_max, n_temps)` (log scale if `temp_max / temp_min > 10`)
   - Apply prequential temperature scaling with the specified warmup
   - Apply the fixed (0.15, 0.70) thresholds (v131 winner — not re-tuned in this loop)
   - Print `covered_ba=X.XXXX` and `coverage=X.XXXX` to stdout
   - Exit 1 if coverage < 0.20

2. **Write `tests/test_research_v135_temp_param_search.py`** with:
   - Default config produces covered_ba ≈ 0.5725 (v130 baseline ± 0.01)
   - `temp_min > temp_max` raises ValueError
   - `warmup < 6` raises ValueError (minimum 6 months required for stable estimates)
   - `warmup > 48` raises ValueError (would consume more than half the evaluation window)

3. **Create parameter candidate files:**
   ```bash
   echo "5.0" > results/research/v135_temp_max_candidate.txt
   echo "24" > results/research/v135_warmup_candidate.txt
   ```

### Autoresearch invocation

```
/autoresearch
Goal: Tune the temperature scaling upper bound and prequential warmup window for
      the Path B classifier to maximize covered balanced accuracy at the v131
      asymmetric thresholds (low=0.15, high=0.70). Current temp-scaled BA = 0.5725;
      the promotion gate requires ≥ 0.60. Temperature range and warmup duration
      are the two degrees of freedom most likely to move this metric without
      changing any model architecture.

Metric: python results/research/v135_temp_param_search.py --temp-min 0.5 --temp-max $(cat results/research/v135_temp_max_candidate.txt) --warmup $(cat results/research/v135_warmup_candidate.txt) --low 0.15 --high 0.70 2>/dev/null | grep covered_ba= | cut -d= -f2

Scope: Two files only:
       results/research/v135_temp_max_candidate.txt (a single float, range [1.5, 10.0])
       results/research/v135_warmup_candidate.txt (a single integer, range [12, 42])
       Do NOT modify: v131_threshold_sweep_eval.py, v125_portfolio_target_fold_detail.csv,
       config/model.py, any source or test files.

Guard: python -c "t=float(open('results/research/v135_temp_max_candidate.txt').read()); w=int(open('results/research/v135_warmup_candidate.txt').read()); assert 1.5<=t<=10.0, f'temp_max={t} out of range'; assert 12<=w<=42, f'warmup={w} out of range'" && python -m pytest tests/test_research_v135_temp_param_search.py -q --tb=short 2>&1 | tail -1 | grep "passed"

Direction: higher_is_better

Iterations: 80

Log: results/research/v135_temp_param_autoresearch_log.jsonl
```

### Guard design rationale

- **`temp_max` range [1.5, 10.0]:** Values below 1.5 would overlap with the fixed lower portion of the temperature grid (0.50–0.95), producing a degenerate one-sided grid. Values above 10.0 correspond to extreme probability compression (logit/10 → near-zero probabilities) that collapse coverage.
- **`warmup` range [12, 42]:** Fewer than 12 months of warmup means temperature is estimated on very few data points and will be numerically unstable. More than 42 months (≈ 70% of the ~60-row evaluation window) leaves too few test rows to measure covered BA reliably.

### Known risks

- **Overfitting the temperature to the v125 fold detail:** The v125 fold detail CSV is the only data being optimized against. It contains ~60 rows. With 80 iterations, autoresearch is effectively doing grid search over a tiny dataset. Any discovered optimum should be validated on the v132 temporal hold-out before being promoted to config.
- **Coverage–BA tradeoff:** Higher temperature (T > 3) compresses probabilities toward 0.5, causing more predictions to fall inside the abstention band (0.15, 0.70) and reducing coverage. If coverage drops below 0.25, the BA estimate becomes unreliable. Monitor both metrics.

### Success criteria

Covered BA ≥ 0.60 at coverage ≥ 0.25, with Brier ≤ 0.20. This would represent progress toward the promotion gate criterion #3. The discovered (temp_max, warmup) values must be validated on the v132 temporal hold-out before config update.

### Sequencing note

Run **first** in the overnight sequence — it is the fastest loop (10–20s/iter) and directly addresses the promotion gate bottleneck (BA < 0.60). Results from this loop may change the temperature configuration used by Target 1 (feature subsetting). Run Target 5 → confirm temp parameters → then run Target 1.

---

## Target 6 — Research Backlog Prioritization via /autoresearch:predict

### Prerequisites

This is a meta-loop: autoresearch:predict is applied to the project codebase to generate multi-persona forecasts of which research directions will yield the highest expected improvement given the current model state. It requires:

- **`docs/research/backlog.md`** — structured backlog document listing all open research candidates with a standardized schema (see setup tasks). Create this before running.
- No new scripts or fixtures needed. autoresearch:predict reads the codebase and backlog directly.

### Estimated iteration time

autoresearch:predict performs static analysis with no model training. Expect 2–5 minutes per full predict run depending on codebase size. Overnight budget is not the constraint here — typically run once per quarter or after a major research cycle completes.

### Setup tasks

1. **Create `docs/research/backlog.md`** with one entry per open research candidate. Use this schema for each entry:
   ```markdown
   ### [Candidate ID] — [Short Name]
   **Status:** open | blocked | deferred
   **Priority:** high | medium | low (human-assigned)
   **Rationale:** One sentence describing the hypothesis.
   **Estimated effort:** S | M | L (S=<1 day, M=1–3 days, L=>3 days)
   **Depends on:** [list of candidate IDs that must complete first, or "none"]
   **Expected metric impact:** [which metric, direction, magnitude estimate]
   **Last touched:** [version or date]
   ```

   Seed the backlog with at least the following candidates (drawn from the v37–v60 and v123–v128 plans):
   - `BL-01` — Black-Litterman tau/view-confidence tuning (Target 8 of this plan)
   - `CLS-01` — SCHD per-benchmark classifier addition (deferred, ~v135+)
   - `CLS-02` — Firth's penalized regression for short-history benchmarks
   - `CLS-03` — Path A vs Path B production decision after 24 matured months
   - `REG-01` — v50 clip+shrink blend at ensemble level
   - `REG-02` — v57 rank-normalized GBT re-evaluation with high Ridge alpha
   - `FEAT-01` — DTWEXBGS (USD momentum) per-benchmark feature addition post v128
   - `FEAT-02` — WTI oil momentum 3m for DBC/VDE (requires EIA series verification)
   - `DATA-01` — EDGAR filing lag review (current `EDGAR_FILING_LAG_MONTHS=6`; verify against actual SEC EDGAR release history)
   - `INFRA-01` — Conformal prediction coverage calibration (`CONFORMAL_COVERAGE` currently hardcoded)

2. **Verify autoresearch:predict prerequisites** — the skill requires the codebase to be in a clean, readable state with no uncommitted merge conflicts.

3. **Write `docs/research/backlog_scoring_rubric.md`** — the rubric autoresearch:predict will use to score candidates. Include these axes:
   - **Signal quality ROI**: expected metric improvement per effort unit
   - **Promotion gate proximity**: does this item directly move a promotion criterion closer?
   - **Data risk**: does this require new external data dependencies or schema changes?
   - **Model complexity cost**: does this add a new hyperparameter or architectural branch?
   - **Reversibility**: can this be reverted cleanly if it degrades metrics?

### Autoresearch invocation

```
/autoresearch:predict
Goal: Prioritize the open research backlog in docs/research/backlog.md from the
      perspective of four expert personas:
      (1) Quantitative researcher focused on OOS R² improvement
      (2) Risk manager focused on minimizing promotion-gate violations
      (3) Data engineer focused on implementation effort and data reliability
      (4) Portfolio manager focused on the actual vesting-decision quality
      Each persona scores each backlog item on the five axes in
      docs/research/backlog_scoring_rubric.md. The output is a consensus ranking
      with per-persona dissents noted for items where opinions diverge > 2 points.

Metric: python -c "import json; data=json.load(open('results/research/v136_predict_output.json')); consensus=[x for x in data['rankings'] if x['consensus_score']>=7]; print(len(consensus))"

Scope: Read-only analysis of: docs/research/backlog.md, docs/research/backlog_scoring_rubric.md, all files in src/, config/, results/research/*.md, docs/superpowers/plans/*.md.
       Output must be written to results/research/v136_predict_output.json (schema: {"rankings": [{"id": str, "name": str, "consensus_score": float, "persona_scores": dict, "dissents": list}]}).
       Do NOT modify any source files, configs, or test files.

Guard: python -c "import json; d=json.load(open('results/research/v136_predict_output.json')); assert len(d['rankings'])>=10, 'must rank all 10+ backlog items'; assert all('consensus_score' in r for r in d['rankings']), 'missing consensus_score'"

Direction: higher_is_better

Iterations: 3

Log: results/research/v136_predict_autoresearch_log.jsonl
```

### Guard design rationale

The guard verifies structural completeness of the output JSON: all backlog items must appear in the rankings, and every item must have a `consensus_score`. This prevents a degenerate output that ranks only the "easy" items and omits contested ones. Three iterations are sufficient for this target — predict converges quickly on qualitative rankings.

### Known risks

- **Hallucinated metrics:** autoresearch:predict generates assessments based on code and documentation analysis, not empirical test runs. Any predicted metric impact should be treated as directional guidance, not a quantitative guarantee. Cross-check top-ranked items against the existing v37–v60 empirical results before accepting the prioritization.
- **Stale backlog:** The backlog is a snapshot. Items may have been partially addressed by in-progress research (e.g., Target 3 in this plan partially covers `REG-01`). Update `backlog.md` immediately after any loop completes to keep the predict input current.

### Success criteria

A ranked backlog with ≥ 5 items achieving consensus_score ≥ 7, with the top-3 items assigned clear "next quarter" priority labels. The ranking should be used as the input to the next research cycle planning session.

### Sequencing note

Run **last** in the overnight sequence, after all empirical loops have completed. The predict analysis is most valuable when it can incorporate the results of the other 7 loops. Update `backlog.md` with the outcomes of Targets 1–5, 7, 8 before running Target 6.

---

## Target 7 — GBT Hyperparameter Space Exploration

### Prerequisites

The current GBT configuration in `src/models/regularized_models.py:build_gbt_pipeline()` is: `max_depth=2, n_estimators=50, learning_rate=0.1, subsample=0.8`. This has been the default since v5.0. The v57 rank-normalized GBT result (pooled OOS R² = −0.2440) was worse than v38, but that used a target transformation rather than a hyperparameter change. The standard GBT at alternative hyperparameters (e.g., `max_depth=1`, which is a decision stump ensemble) has never been tested in the current ensemble context.

Required before loop runs:
- **`results/research/v137_gbt_param_sweep.py`** — harness accepting `--max-depth INT --n-estimators INT --learning-rate FLOAT --subsample FLOAT` and printing `pooled_oos_r2=X.XXXX` to stdout
- **`results/research/v137_gbt_params_candidate.json`** — parameter candidate file autoresearch edits

### Estimated iteration time

GBT WFO is slower than Ridge due to tree fitting: approximately 8–15 minutes per full sweep. Overnight budget ≈ 32–60 iterations.

### Setup tasks

1. **Create `results/research/v137_gbt_param_sweep.py`** with this interface:
   ```python
   def run_gbt_sweep(
       max_depth: int,           # tree depth, 1–4
       n_estimators: int,        # boosting stages, 10–200
       learning_rate: float,     # shrinkage, 0.01–0.5
       subsample: float,         # row fraction, 0.5–1.0
       benchmarks: list[str] | None = None,
   ) -> dict[str, float]:       # keys: pooled_oos_r2, pooled_ic, pooled_hit_rate
   ```
   The script must:
   - Accept both explicit flags (`--max-depth`, `--n-estimators`, `--learning-rate`, `--subsample`) **and** `--params-file <json_path>` (reads the same four keys from the JSON file; `--params-file` takes precedence)
   - Call `build_gbt_pipeline(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample)`
   - Run WFO with the GBT model in isolation (not the full ensemble) to isolate the GBT contribution
   - Pool results using inverse-variance weighting
   - Print `pooled_oos_r2=X.XXXX` as the first stdout line

2. **Create `results/research/v137_gbt_params_candidate.json`** with the baseline config:
   ```json
   {"max_depth": 2, "n_estimators": 50, "learning_rate": 0.1, "subsample": 0.8}
   ```

3. **Write `tests/test_research_v137_gbt_param_sweep.py`** with:
   - Default params produce pooled_oos_r2 in [−0.40, 0.10]
   - `max_depth > 4` raises ValueError
   - `n_estimators > 200` raises ValueError
   - `subsample < 0.5` raises ValueError

4. **Establish GBT-only baseline** (not the v38 ensemble baseline — GBT alone will differ):
   ```bash
   python results/research/v137_gbt_param_sweep.py --max-depth 2 --n-estimators 50 --learning-rate 0.1 --subsample 0.8
   # Record the standalone GBT baseline for comparison.
   ```

### Autoresearch invocation

```
/autoresearch
Goal: Find GBT hyperparameter combinations that improve the standalone GBT
      pooled OOS R² beyond its current default-config level. The v37–v60 cycle
      showed that complexity generally hurts, but depth-1 stumps (max_depth=1)
      have never been tested and may produce better variance control than depth-2
      trees in this small-sample regime (N ≈ 60–80 per benchmark per fold).

Metric: python results/research/v137_gbt_param_sweep.py --params-file results/research/v137_gbt_params_candidate.json 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2

Scope: results/research/v137_gbt_params_candidate.json only.
       Valid ranges: max_depth in {1, 2, 3, 4}, n_estimators in [10, 200] (integer),
       learning_rate in [0.01, 0.50], subsample in [0.50, 1.00].
       Do NOT modify: regularized_models.py, config/model.py, any test files,
       any production source files.

Guard: python -c "import json; p=json.load(open('results/research/v137_gbt_params_candidate.json')); assert p['max_depth'] in {1,2,3,4}; assert 10<=p['n_estimators']<=200; assert 0.01<=p['learning_rate']<=0.50; assert 0.50<=p['subsample']<=1.00" && python -m pytest tests/test_research_v137_gbt_param_sweep.py -q --tb=short 2>&1 | tail -1 | grep "passed"

Direction: higher_is_better

Iterations: 35

Log: results/research/v137_gbt_param_autoresearch_log.jsonl
```

### Guard design rationale

- **`max_depth` discrete constraint:** Tree depth must be an integer in {1, 2, 3, 4}. Depth > 4 in a 50-tree ensemble with N ≈ 70 observations would have more leaves than training samples — structural overfitting regardless of `subsample`.
- **`n_estimators` upper bound 200:** At `learning_rate=0.01`, 200 trees is already near the effective ESS floor given the dataset size. More trees would only add runtime without generalization benefit.
- **`subsample` lower bound 0.50:** Subsample < 0.5 means each tree sees fewer than 30 observations in the smallest folds — below the minimum defensible sample for 12+ features.

### Known risks

- **Interaction with Ridge in the ensemble:** The GBT-only sweep evaluates the GBT member in isolation. The optimal standalone GBT hyperparameters may not be optimal when combined with Ridge in the ensemble blend (inverse-variance weighting means a lower-variance GBT gets more weight, potentially displacing Ridge signal). Follow up with an ensemble-level evaluation of any discovered optimum before promoting to config.
- **v57 precedent:** The v57 research already tested rank-normalized GBT at the default hyperparameters and found pooled OOS R² = −0.2440 (worse than v38). This loop tests standard (non-rank-normalized) GBT, but the v57 result should temper expectations. Success here would contradict the v37–v60 trend — treat any large improvement with skepticism and run additional validation.

### Success criteria

GBT-only pooled OOS R² ≥ −0.15 (vs the standalone GBT baseline to be established in setup task 4). The improvement must come from a hyperparameter configuration that does not hit any guard boundary (i.e., not a degenerate config like `n_estimators=200, learning_rate=0.01`).

### Sequencing note

Run **after** Target 3 (Ridge alpha) — if Ridge already achieves the target OOS R² improvement, the GBT sweep is lower priority. Run concurrently with Targets 1, 4, 5 if overnight compute budget allows parallel loops.

---

## Target 8 — Black-Litterman Tau and View-Confidence Optimization

### Prerequisites

The BL model parameters in `config/model.py` are: `BL_TAU=0.05`, `BL_VIEW_CONFIDENCE_SCALAR=1.0`, `BL_RISK_AVERSION=2.5`, `BL_USE_BAYESIAN_VARIANCE=True`. These values were set during initial BL implementation and have not been systematically tuned. The BL model provides the recommended portfolio weight used in the monthly decision layer.

The optimization target is the quality of the BL-derived recommendation against the ex-post correct hold/sell decision. This requires a BL-specific eval script that maps BL output → recommendation → comparison with actual 6-month PGR return vs portfolio return.

Required before loop runs:
- **`results/research/v138_bl_param_eval.py`** — accepts `--tau FLOAT --view-confidence FLOAT` and prints `recommendation_accuracy=X.XXXX` to stdout (fraction of months where BL recommendation agreed with the ex-post correct decision)
- **`results/research/v138_bl_params_candidate.json`** — parameter candidate file

### Estimated iteration time

The BL model is computationally cheap (matrix algebra only, no WFO). Expect < 30 seconds per eval. Overnight budget ≈ 960+ iterations.

### Setup tasks

1. **Create `results/research/v138_bl_param_eval.py`** with this interface:
   ```python
   def evaluate_bl_params(
       tau: float,                     # prior uncertainty scalar, range [0.01, 0.50]
       view_confidence_scalar: float,  # scales Ω = RMSE² × scalar, range [0.25, 4.0]
       risk_aversion: float = 2.5,     # held fixed in this loop
       use_bayesian_variance: bool = True,  # held fixed
   ) -> dict[str, float]:             # keys: recommendation_accuracy, mean_kelly_fraction, policy_uplift
   ```
   
   The script must:
   - Load the production shadow consensus output (the CSV written by `consensus_shadow.py`)
   - For each evaluation month, recompute the BL posterior expected return using the specified tau and view_confidence_scalar
   - Compute the Kelly fraction under the BL posterior
   - Label the ex-post outcome: `correct` if BL recommended hold and PGR outperformed, or BL recommended sell/reduce and PGR underperformed
   - Print `recommendation_accuracy=X.XXXX` (fraction correct, higher = better)
   - Print `coverage=X.XXXX` (fraction of months with a non-neutral BL recommendation)

2. **Create `results/research/v138_bl_params_candidate.json`**:
   ```json
   {"tau": 0.05, "view_confidence_scalar": 1.0}
   ```

3. **Write `tests/test_research_v138_bl_param_eval.py`** with:
   - Default params produce recommendation_accuracy in [0.40, 0.80] (sanity bounds)
   - `tau < 0.01` raises ValueError
   - `view_confidence_scalar < 0.25` raises ValueError
   - Coverage ≥ 0.10 (BL must produce non-neutral recommendations for at least 10% of months)

4. **Establish the baseline recommendation accuracy:**
   ```bash
   python results/research/v138_bl_param_eval.py --tau 0.05 --view-confidence 1.0
   # Record recommendation_accuracy — this becomes the guard floor.
   ```

5. **Document the theoretical priors** in a comment block at the top of the eval script:
   - `tau` near 0 → BL prior dominates → recommendations move toward market equilibrium weights (lower vesting concentration)
   - `tau` near 0.5 → views dominate → BL tracks the regression ensemble prediction closely
   - `view_confidence_scalar` > 1.0 → reduce uncertainty in the view → stronger directional signals
   - `view_confidence_scalar` < 1.0 → increase uncertainty → more conservative recommendations

### Autoresearch invocation

```
/autoresearch
Goal: Find BL tau and view-confidence-scalar values that maximize the fraction of
      months where the BL recommendation (hold vs reduce/sell) agreed with the
      ex-post correct decision, given actual PGR 6-month returns vs the redeploy
      portfolio. Current config (tau=0.05, view_confidence=1.0) has not been tuned
      against realized outcomes. The v37–v60 diagnostics showed real predictive
      signal in the ensemble (Clark-West t=3.36); the BL layer should amplify this
      signal rather than dilute it toward the equilibrium prior.

Metric: python results/research/v138_bl_param_eval.py --params-file results/research/v138_bl_params_candidate.json 2>/dev/null | grep recommendation_accuracy= | cut -d= -f2

Scope: results/research/v138_bl_params_candidate.json only.
       Valid ranges: tau in [0.01, 0.50], view_confidence_scalar in [0.25, 4.0].
       risk_aversion and use_bayesian_variance are held fixed at their config values.
       Do NOT modify: config/model.py, any source files, any test files,
       the consensus_shadow output CSV.

Guard: python -c "import json; p=json.load(open('results/research/v138_bl_params_candidate.json')); assert 0.01<=p['tau']<=0.50, f'tau={p[\"tau\"]} out of range'; assert 0.25<=p['view_confidence_scalar']<=4.0, f'vc={p[\"view_confidence_scalar\"]} out of range'" && python -m pytest tests/test_research_v138_bl_param_eval.py -q --tb=short 2>&1 | tail -1 | grep "passed"

Direction: higher_is_better

Iterations: 60

Log: results/research/v138_bl_param_autoresearch_log.jsonl
```

### Guard design rationale

- **`tau` range [0.01, 0.50]:** `tau < 0.01` effectively sets the prior variance to zero, making BL recommendations independent of the regression ensemble views — equivalent to pure equilibrium weighting. `tau > 0.50` inverts the normal BL interpretation (views would dominate prior so heavily that the equilibrium component is noise). Both extremes would produce nonsensical recommendations.
- **`view_confidence_scalar` range [0.25, 4.0]:** Scalar < 0.25 doubles or more the view uncertainty, effectively nullifying the regression ensemble signal. Scalar > 4.0 makes the view precision (1/Ω) implausibly high relative to the observed RMSE of the regression ensemble.

### Known risks

- **Ex-post labeling look-ahead:** The `recommendation_accuracy` metric requires knowing the realized 6-month return, which is only available for historical months. The most recent 6 months of shadow history will not have matured outcomes. Ensure the eval script truncates correctly and does not include unmatured months in the accuracy calculation.
- **Low base rate of "sell" recommendations:** If the BL model rarely recommends selling (which is expected given the taxable account cost asymmetry), the recommendation accuracy metric will be dominated by "hold and PGR outperformed" outcomes, making it easy to achieve high accuracy by never recommending a sell. Monitor coverage and the precision of sell recommendations separately.
- **BL is a decision support layer, not a direct metric driver:** BL tau/view-confidence affects portfolio weight recommendations, not the regression ensemble OOS R² or classifier BA. An improvement here may not move the promotion gate criteria directly. This loop is most valuable for the decision-quality metrics (criteria #6, #7 in the promotion gate) rather than the model-quality metrics.

### Success criteria

Recommendation accuracy ≥ 0.65 (vs the baseline to be established in setup task 4), with coverage ≥ 0.25. Secondary constraint: mean Kelly fraction change from baseline ≤ +0.10 (prevent over-concentration recommendations).

### Sequencing note

Run **independently** — BL parameters do not affect any other loop's metric. Can run in parallel with Targets 3, 4, 7. Lower priority for overnight sequencing than Targets 5, 1 (which directly address the promotion gate).

---

## Recommended Overnight Sequence

Given an 8-hour overnight window with Claude Code or Codex running continuously, the following sequence maximizes expected research value.

### Current state summary

| Layer | Blocking factor | Distance to goal |
|---|---|---|
| Production regression | None — v38 is promoted and stable | Incremental improvement only |
| Shadow classifier promotion | BA covered = 0.5725; gate requires ≥ 0.60 | 2.75 pp gap |
| Promotion gate month count | 0 matured prospective months (shadow monitoring just started) | 24 months required; time-locked regardless of metric improvements |

**Critical insight:** The promotion gate requires 24 matured prospective months, which is a time-locked constraint that no autoresearch loop can accelerate. The overnight loops therefore serve two purposes: (a) push the BA metric toward 0.60 so it is ready when the 24-month gate opens, and (b) improve the regression baseline incrementally.

### Recommended sequence

**Phase 1 (Hours 0–2): Fast loops, high direct impact on promotion gate**

1. **Target 5 — Temperature scaling tuning** *(~20s/iter, 80 iterations, ~27 min total)*
   - Directly addresses the BA 0.5725 → 0.60 gap
   - Uses precomputed data, no WFO training
   - Run first — establishes optimal temp params for all subsequent classifier work
   - Expected outcome: likely to find BA ≥ 0.58 within 80 iterations; may reach 0.60 threshold

2. **Target 1 — Per-benchmark feature subsetting** *(~45s/iter, 40 iterations, ~30 min total)*
   - Second classifier-focused loop, builds on temp param result from Target 5
   - Run immediately after Target 5 completes; update `v135_temp_max_candidate.txt` with the winner before starting
   - Expected outcome: modest pooled BA improvement (+0.01 to +0.02); primary value is identifying stable per-benchmark configurations

**Phase 2 (Hours 2–5): Medium-speed loops, regression baseline improvement**

3. **Target 4 — FRED lag optimization** *(~7 min/iter, 25 iterations, ~3 hrs total)*
   - Data quality improvement with zero model complexity cost
   - If successful (OOS R² ≥ −0.10), the cleaner features benefit all downstream targets
   - Run before Target 3 so Ridge alpha search operates on lag-optimized features
   - Expected outcome: 0.01 to 0.03 pooled OOS R² improvement if daily-series lag=0 is defensible

4. **Target 3 — Ridge alpha grid** *(~6 min/iter, 30 iterations, ~3 hrs total)*
   - Tests whether stronger Ridge regularization can internalize the v38 shrinkage effect
   - Run after Target 4 (features should be cleaner)
   - Expected outcome: likely minimal improvement over v38 given that v37–v60 results show the Ridge path was already near-optimal; this loop confirms or falsifies that assumption

**Phase 3 (Hours 5–7): Slower loops, lower expected value but important to characterize**

5. **Target 7 — GBT hyperparameter search** *(~12 min/iter, 35 iterations, ~7 hrs if running alone)*
   - Lowest expected improvement based on v37–v60 precedent (GBT consistently underperformed v38)
   - If Phase 2 finishes early, start Target 7; otherwise defer to the next overnight session
   - Most valuable finding here would be `max_depth=1` (stumps) outperforming depth-2 trees
   - Skip if Phase 2 is still running at Hour 5

**Phase 4 (Final hour): Analysis and metadata**

6. **Target 6 — Backlog prioritization via /autoresearch:predict** *(~4 min/run, 3 iterations)*
   - Run last, after all empirical loops have completed and `backlog.md` has been updated with results
   - Produces the input to the next research planning session

### Loops deferred to a separate session

- **Target 2 — Test suite runtime reduction**: Run in a dedicated daytime session (not overnight) so the timing measurements are stable and results can be reviewed immediately. Overnight WFO loops running simultaneously would skew timing measurements.
- **Target 8 — BL tau/view-confidence**: BL optimization does not affect any promotion gate criterion directly. Defer to a daytime session where the shadow consensus CSV can be inspected alongside the results.

### Expected overnight outcomes

| Target | Expected metric after overnight | Confidence |
|---|---|---|
| 5 (temp scaling) | Covered BA ≥ 0.58, possibly ≥ 0.60 | High |
| 1 (feature subsetting) | Pooled covered BA +0.01 to +0.02 | Medium |
| 4 (FRED lags) | OOS R² −0.12 to −0.10 if lag=0 is defensible | Medium-low |
| 3 (Ridge alpha) | OOS R² at or near −0.1310 baseline | Low improvement expected |
| 7 (GBT) | Standalone GBT modest improvement; unlikely to beat v38 ensemble | Low |
| 6 (predict) | Ranked backlog with 5+ high-priority items | High (qualitative) |

### Go/no-go decision after overnight

After reviewing overnight results:
- If **Target 5** achieves covered BA ≥ 0.60: update `SHADOW_CLASSIFIER_HIGH_THRESH` and `SHADOW_CLASSIFIER_LOW_THRESH` in `config/model.py` (pending v132 temporal hold-out confirmation)
- If **Target 4** achieves OOS R² ≥ −0.10: update `FRED_SERIES_LAGS` for the winning lag=0 series and re-run the production WFO evaluation
- If **Target 3** shows OOS R² ≥ −0.10 with `alpha_max` hitting the grid boundary: extend the grid to 1e5 and run a follow-on session
- If **Target 7** finds `max_depth=1` wins: test the depth-1 GBT in the full ensemble (not just standalone) before promoting

All parameter changes require: (1) test suite passes, (2) the discovered value is not at a guard boundary, (3) a holdout evaluation using a portion of data not seen during the loop.
