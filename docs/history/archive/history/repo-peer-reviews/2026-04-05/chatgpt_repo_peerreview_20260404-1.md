# Peer Review of pgr-vesting-decision-support

## Executive summary

This repository has matured into a well-instrumented, production-grade “research + operations” system for recurring investment decision support around entity["company","The Progressive Corporation","insurance company"] RSUs, with strong governance patterns (clear production/research boundaries, a committed operational SQLite DB, CI + smoke tests, and extensive historical research artifacts). The core modeling approach—walk-forward optimization (WFO) on monthly features, with benchmark-relative targets and a small regularized/low-variance model family—is directionally well-chosen for Small-N finance forecasting. fileciteturn45file0L1-L1 fileciteturn49file0L1-L1

The highest-impact accuracy opportunities that **do not add model complexity** are overwhelmingly about: (a) **fixing subtle time-index alignment** between market-derived month-end business dates and “economic month-end” dates used by EDGAR/FRED tables; (b) **closing remaining schema/key mismatches** that silently drop features; and (c) **eliminating validation/diagnostic bugs** that can mislead you into thinking a model is stable (or unstable) when the diagnostic is wrong.

Key takeaways (prioritized for “accuracy without complexity” and “reducing silent error”):

- The single biggest likely accuracy drag is a **calendar month-end vs business month-end (ME vs BME) mismatch** between (i) feature/target indices (BME) and (ii) EDGAR (and probably macro) month-end indices (ME). This can introduce an *unintended extra month of lag* for many observations—especially when the calendar month-end lands on a weekend—quietly degrading signal strength. fileciteturn40file0L1-L1 fileciteturn62file0L1-L1
- There is a **confirmed silent-feature-drop bug** around ROE naming in the monthly EDGAR 8-K pipeline: live parsing produces `roe_net_income_trailing_12m`, but the DB upsert expects `roe_net_income_ttm`, so that column can remain null in production without a hard failure. fileciteturn35file0L1-L1 fileciteturn34file0L1-L1
- CPCV currently has a **diagnostic correctness bug**: recombined-path ICs may be computed from predictions that were overwritten by later splits (shared `all_y_hat`). That can distort the CPCV distribution and any downstream stability gates. fileciteturn68file0L1-L1
- The inner alpha-tuning CV in regularized models uses `TimeSeriesSplit` but **does not apply a purge/embargo gap**, which is a subtle leakage risk for overlapping forward-return labels. This is “small complexity” to fix (one parameter/constructor change), and it improves validity more than raw performance, but it can materially affect model selection stability. fileciteturn44file0L1-L1
- Several docs still carry drifted counts/claims (e.g., benchmark ETF count and “23 tickers” assumptions) that increase operator error risk. fileciteturn39file0L1-L1 fileciteturn42file0L1-L1 fileciteturn38file0L1-L1

## What I must learn to answer perfectly

To go from “rigorous peer review” to “I can sign off on this for ongoing investment use,” I would still need the following (currently unknown unless you provide them or they’re committed in artifacts):

- The **exact current production cross-check and recommendation-layer mode** you are using in practice (the repo describes multiple parallel layers and promotion gates). fileciteturn45file0L1-L1
- The **most recent run manifests / scoreboards** (e.g., the latest `results/monthly_decisions/*/run_manifest.json` plus any v21–v24 summary tables) to validate that the code paths we’re discussing are actually active and producing the expected features/signals. fileciteturn45file0L1-L1
- The **effective evaluable window length** per benchmark after proxy fills, ETF launch dates, and any benchmark-universe reductions (needed to size statistical tests and interpret IC). fileciteturn42file0L1-L1
- Whether your macro table (`fred_macro_monthly`) is indexed on **calendar month-end** or **business month-end**, and how frequently it is updated/revintage handled (critical for the alignment fix). fileciteturn40file0L1-L1
- The **distribution of missingness** in EDGAR-derived columns in the committed DB (to confirm which features are silently absent today and quantify expected gains from fixes). fileciteturn34file0L1-L1

## Repository state and architecture

The project is operationally structured as an end-to-end pipeline: ingestion → SQLite accumulation → feature/target construction → per-benchmark WFO modeling (with ensemble options) → calibrated probabilities and conformal intervals → decision/report artifacts → scheduled automation and commits back to entity["company","GitHub","code hosting platform"]. fileciteturn45file0L1-L1 fileciteturn48file0L1-L1 fileciteturn49file0L1-L1

A simplified architecture view (conceptual; names correspond to repo modules and workflows):

```mermaid
flowchart LR
  subgraph Ingestion
    AV[Prices & dividends API] --> DB[(SQLite operational DB)]
    SEC[EDGAR monthly + quarterly] --> DB
    MACRO[Macro series store] --> DB
  end

  subgraph FeaturesTargets
    DB --> FM[Feature matrix (monthly)]
    DB --> RR[Relative return targets]
  end

  subgraph Modeling
    FM --> WFO[Per-benchmark WFO models]
    RR --> WFO
    WFO --> ENS[Ensemble signals + diagnostics]
    ENS --> CAL[Calibration + conformal intervals]
  end

  subgraph Decisions
    CAL --> REC[Recommendation & tax/portfolio logic]
    REC --> ART[Monthly artifacts + run manifest]
  end

  ART --> CI[CI + smoke tests]
  ART --> OPS[Scheduled workflows commit DB + results]
```

Notable strengths that already align with “accuracy without unnecessary complexity”:

- WFO uses `TimeSeriesSplit` with a **gap equal to (horizon + purge buffer)**, explicitly designed to mitigate overlapping-return leakage. fileciteturn68file0L1-L1
- Feature engineering and DB-backed construction are well-annotated, with explicit lag helpers for EDGAR and macro data (conceptually correct direction). fileciteturn40file0L1-L1
- Calibration and conformal intervals are implemented as **thin wrappers** around existing predictions (i.e., they improve decision usefulness without complex new model classes). fileciteturn46file0L1-L1 fileciteturn47file0L1-L1
- CI runs ruff, a limited mypy pass, pytest, and production-script smoke tests—excellent for keeping the repo operable. fileciteturn49file0L1-L1

## Detailed findings

### Time-index alignment is likely undermining signal strength

Your core feature matrix is indexed on **business month-end** (`BME`) via resampling of daily prices for both features and the 6‑month forward DRIP target. fileciteturn40file0L1-L1 fileciteturn41file0L1-L1

But EDGAR monthly ingestion (8‑K exhibits) materializes `month_end` as **calendar month-end** (`ME`) using period conversion to timestamp “M”. fileciteturn62file0L1-L1

This calendar-vs-business mismatch matters because when a calendar month-end is on a weekend (common), a business-month-end index for that month will be **earlier** than the EDGAR/FRED calendar end. If you align with `reindex(..., method="ffill")`, the EDGAR value for “that month” is *not yet available on the earlier BME date*, so you effectively ffill from the **previous month**, introducing an additional unintended lag and reducing predictive power.

This is not theoretical: the EDGAR parser’s own tests use month-ended dates like `February 28, 2026`, which is a Saturday. fileciteturn61file0L1-L1 With BME indexing, “February 2026” would typically be Friday Feb 27, 2026; unless you normalize indices, February EDGAR metrics will not line up with February features/targets.

Why this is a priority: unintentionally shifting a large block of fundamental/macro features by an extra month can destroy weak-but-real signal, especially in a setting where IC is already expected to be small and noisy (a known result in return predictability research). citeturn1search0 citeturn1search1

### Silent schema/key mismatches remain in the advanced EDGAR 8‑K pipeline

The repo now has two EDGAR 8‑K processing paths:

- `src/ingestion/edgar_8k_fetcher.py` (focused on combined ratio + PIF, heavily tested) fileciteturn62file0L1-L1 fileciteturn61file0L1-L1  
- `scripts/edgar_8k_fetcher.py` (a richer feature extraction + CSV backfill path used for many P2.x fields) fileciteturn35file0L1-L1

In `scripts/edgar_8k_fetcher.py`, the live HTML parse returns the key `roe_net_income_trailing_12m`; comments note that CSV uses a different name and has a mapping. fileciteturn35file0L1-L1

However, the DB upsert normalization reads `roe_net_income_ttm` (not `roe_net_income_trailing_12m`). fileciteturn34file0L1-L1

Net effect: **live-run ROE net income can be silently dropped** (stored as NULL) unless an intermediate mapping step renames it. This is exactly the type of silent error that reduces accuracy while leaving the system “green.”

Minimal-complexity fix: accept both key names on ingest (DB client), or normalize in the fetcher before upsert (preferred in the fetcher, since it’s the source-of-truth naming translator).

### CPCV recombined-path IC computation is currently unreliable

CPCV is implemented in `run_cpcv()` and used as a stability/overfitting diagnostic. The code collects predictions from each split into a single vector `all_y_hat` and overwrites indices as splits progress. fileciteturn68file0L1-L1

Then recombined paths iterate through `cv.recombined_paths` and pull `all_y_hat[test_idx]`. But in CPCV, the same observation can appear in test sets across multiple splits; a single “all_y_hat” array cannot represent split-specific predictions. Overwrites mean that a recombined path can use predictions trained on the *wrong training fold combination*, biasing path ICs.

You have tests validating split counts and some structural properties, but none that would catch this “prediction provenance” bug. fileciteturn59file0L1-L1

Why this matters: CPCV is used to reason about generalization stability. A bug here increases the chance of either (a) falsely promoting an unstable candidate or (b) falsely rejecting a stable candidate.

External grounding: CPCV’s “recombined paths” concept is specifically designed to reconstruct test paths from split-specific predictions; tools like skfolio expose these semantics explicitly. citeturn2search0

### Inner alpha-tuning CV lacks embargo/purge, creating subtle leakage risk

Your WFO outer loop correctly uses a gap of `target_horizon + purge_buffer`. fileciteturn68file0L1-L1

But inside each fold, Lasso/ElasticNet/Ridge alpha selection uses `TimeSeriesSplit(n_splits=3)` with no `gap`. fileciteturn44file0L1-L1

Because the target is a **forward 6‑month overlapping return** series, adjacent monthly labels share most of their forward window. Without an inner gap, alpha selection can “see through” this autocorrelation structure in a way that inflates apparent validation quality and alters selected regularization strength.

This is a “validity-first” correction: it usually won’t make metrics magically higher, but it makes the tuning less biased and often makes the chosen regularization steadier across folds.

Related principle: for overlapping-horizon returns, HAC/Newey–West style adjustments are commonly used in inference to address serial correlation. citeturn2search1

### WFO dataset-size guard does not account for the (larger) gap

`run_wfo()` computes `total_gap = target_horizon_months + purge_buffer` and sets `TimeSeriesSplit(... gap=total_gap)`, but the “dataset too small” check only enforces `TRAIN + TEST`, not `TRAIN + GAP + TEST`. fileciteturn68file0L1-L1

Practically, you may still get folds (depending on scikit behavior and indices), but this increases the chance of edge-case failures or folds that do not behave as intended when data is short (e.g., for newer benchmarks or reduced-universe experiments). This is a trivial guard improvement that reduces fragility.

### Documentation drift persists in a few operational modules

Your README is unusually thorough and appears actively maintained. fileciteturn45file0L1-L1

However, several code docstrings still assume older counts:

- Scheduler asserts “23 tickers (PGR + 22 ETFs)” even though the configured universe count can differ (and the functions simply return `[PGR] + ETF_BENCHMARK_UNIVERSE`). fileciteturn39file0L1-L1 fileciteturn38file0L1-L1
- Multi-ticker return code and multi-benchmark WFO docstrings still reference “20 benchmarks” in places. fileciteturn42file0L1-L1 fileciteturn43file0L1-L1

These don’t directly change accuracy, but they increase the risk of running the wrong assumptions (e.g., API budgets and expected run time), and in production systems operator confusion is a major source of “silent errors.”

## Prioritized fixes that improve accuracy without adding complexity

The table below focuses on changes that (a) are low complexity, (b) reduce silent error, and (c) plausibly improve predictive alignment/accuracy or the correctness of diagnostics.

| Priority | Fix | Why it matters | Files | Effort |
|---|---|---|---|---|
| P0 | Normalize monthly indices consistently (choose one: **BME everywhere** or **ME everywhere**) and enforce at the DB boundary | Prevents unintended extra-month lag for EDGAR/macro features; likely highest raw accuracy gain | `src/processing/feature_engineering.py`, `src/ingestion/edgar_8k_fetcher.py`, DB getters in `src/database/db_client.py` | Medium |
| P0 | Fix CPCV recombined-path prediction provenance (store per-split predictions; do not reuse a single overwritten vector) | Makes CPCV stability/overfit diagnostics trustworthy | `src/models/wfo_engine.py` | Low |
| P0 | Fix ROE key mismatch (`roe_net_income_trailing_12m` → `roe_net_income_ttm`) in live EDGAR 8‑K path | Restores intended feature; removes silent nulls | `scripts/edgar_8k_fetcher.py`, `src/database/db_client.py` | Trivial |
| P0 | Add WFO minimum-length guard to include the gap: require `TRAIN + GAP + TEST` | Prevents fragile edge cases; improves correctness of fold geometry | `src/models/wfo_engine.py` | Trivial |
| P1 | Apply inner-CV `gap` when tuning alphas (use the same `total_gap` as outer loop, or at least horizon) | Reduces subtle leakage in model selection; improves stability | `src/models/regularized_models.py` (and hook from WFO) | Medium |
| P1 | Add weekend-month-end regression tests for index alignment | Prevents future regressions of the biggest silent accuracy bug | new tests in `tests/` (see plan below) | Low |
| P1 | Add a validation check that “expected” EDGAR/macro features are not **all-null** after load | Turns silent missingness into explicit warnings/errors | `src/processing/feature_engineering.py`, possibly monthly decision entrypoint | Low |
| P1 | Ensure all computed “monthly close” resamples use the same convention (avoid ME in one helper and BME elsewhere) | Removes subtle misalignment in derived features (e.g., relative spreads) | `src/processing/feature_engineering.py` | Low |
| P2 | Tighten/centralize schema column mapping for EDGAR (single authoritative mapping dict used by CSV + live parser) | Reduces future key drift (like the ROE bug) | `scripts/edgar_8k_fetcher.py`, `src/database/db_client.py` | Low |
| P2 | Add CPCV unit test that detects “overwritten prediction” bug | Locks correctness of diagnostic | `tests/test_cpcv.py` | Medium |
| P2 | Add a “data freshness + schema version” check to production scripts (fail closed when DB is stale beyond threshold) | Prevents running monthly decision on stale DB silently | `scripts/monthly_decision.py`, DB helpers | Low |
| P2 | Relax docstrings to reference `len(config.ETF_BENCHMARK_UNIVERSE)` not hardcoded counts | Reduces operator confusion and budgeting mistakes | `src/ingestion/fetch_scheduler.py`, `src/processing/multi_total_return.py`, `src/models/multi_benchmark_wfo.py` | Trivial |
| P2 | Add a lightweight “expected monotonicity” check for calibration/conformal outputs (bounds, non-NaN) | Prevents odd probability/interval outputs from silently shipping | `src/models/calibration.py`, `src/models/conformal.py` | Low |
| P3 | Optional: winsorize/clamp training targets per fold (e.g., 1–99% quantile) | Can reduce noise sensitivity with no new model class | WFO fold loop in `src/models/wfo_engine.py` | Low |
| P3 | Optional: log benchmark-level effective sample size and overlap-adjusted uncertainty | Makes reports less overconfident | reporting layer | Low |

Notes on “effort”:
- **Trivial**: single-line key rename, doc fixes, or adding a guard.
- **Low**: small refactor localized to one module plus tests.
- **Medium**: cross-cutting change that touches DB load paths + feature engineering + tests (the month-index normalization and inner-CV gap are here).

## Suggested tests, CI checks, and minimal validation experiments

### Tests to add (targeting silent error and leakage)

These are intentionally “small surface area” and should slot into your existing pytest suite. fileciteturn56file0L1-L1 fileciteturn49file0L1-L1

- **Index alignment regression test (weekend month-end)**
  - Build a tiny synthetic EDGAR-like monthly frame indexed at calendar ME including a month where ME is weekend (e.g., 2026‑02‑28) and assert that, after your normalization, it aligns to the same index convention as feature/target dates (BME if you keep BME).
  - Fail the test if February’s value is missing at the February feature date.
  - This directly protects the most likely accuracy issue. fileciteturn40file0L1-L1 fileciteturn62file0L1-L1

- **ROE key mapping test (live parse → upsert)**
  - Create a minimal dict record with `roe_net_income_trailing_12m` and pass through the normalization layer (either in the script or db_client). Assert the stored row includes `roe_net_income_ttm`.
  - This prevents silent null regressions. fileciteturn35file0L1-L1 fileciteturn34file0L1-L1

- **CPCV recombined prediction provenance test**
  - Construct a toy dataset where each split returns a distinct constant prediction (e.g., by monkeypatching the model builder to output a pipeline with deterministic output keyed to split id).
  - Assert that each recombined path uses the correct per-split predictions rather than overwritten values.
  - Your existing CPCV tests validate counts and NaNs, but not correctness of recombination. fileciteturn59file0L1-L1 fileciteturn68file0L1-L1

- **Inner-CV gap test**
  - Assert that the CV object passed into `LassoCV`/`ElasticNetCV`/`RidgeCV` is a `TimeSeriesSplit` with a configured `gap > 0` when using overlapping-horizon targets.
  - This is mostly a “policy enforcement” test. fileciteturn44file0L1-L1

### CI checks to consider adding (lightweight)

Your CI is already strong. fileciteturn49file0L1-L1 The following additions are small but materially reduce risk:

- A CI job that runs a “**feature-matrix smoke validation**”:
  - Build the feature matrix from the committed DB in dry-run mode and assert:
    - index is monotone increasing,
    - no duplicate timestamps,
    - key feature groups are not all-null (macro, EDGAR, price).
- Add a “**schema drift detector**” for expected DB columns:
  - Simple query against SQLite pragma table_info; fail if required columns missing.
- If you already commit run manifests, add a CI check that the manifest includes:
  - git SHA, schema version, and warnings count (these are already referenced in workflow summary). fileciteturn48file0L1-L1

### Minimal experiments to validate accuracy gains

These experiments are designed to validate *incremental* gains with **minimal runtime cost** and without introducing new model families.

Common evaluation metrics (use the ones already in the repo to stay consistent):
- Spearman IC (rank correlation) across OOS folds (primary).
- Directional hit rate (sign agreement).
- MAE of relative return predictions.
- (Optional) OOS R² relative to a historical-mean baseline (common in return prediction evaluation; often negative, so treat as diagnostic not a hard gate). citeturn1search0 citeturn1search1

Because labels are overlapping 6‑month returns, use overlap-respecting uncertainty:
- Use block bootstrap with block length = 6 months for confidence intervals (your calibration module already uses block bootstrap ideas; the same logic applies). fileciteturn46file0L1-L1
- Or use HAC/Newey–West style adjustments when doing formal inference. citeturn2search1

Recommended minimal experiments (sample sizes given in “effective months” terms):

- **Experiment A: month-index normalization (P0)**
  - Design: Run the full monthly pipeline twice over the same DB snapshot:
    - Control: current indexing behavior.
    - Treatment: normalized indexing (BME-consistent or ME-consistent).
  - Sample size: all available OOS months in the common window for your primary reduced universe (expect on the order of ~70–120 OOS month predictions per benchmark depending on start date and fold geometry), multiplied by the number of active benchmarks.
  - Success criteria:
    - Mean IC increases materially (even +0.02 can be meaningful in this regime).
    - CPCV positive-path fraction improves or becomes less volatile.
    - Feature importances become more stable (qualitative).
  - Why it’s minimal: it’s strictly a data-alignment change, not a model change.

- **Experiment B: ROE key mismatch fix (P0)**
  - Design: Re-run only the EDGAR ingestion → feature build → WFO on one benchmark (or your reduced benchmark set), with and without the ROE fix.
  - Sample size: all months where ROE would be non-null historically (likely large since you have long EDGAR history).
  - Success criteria:
    - ROE feature is no longer all-null.
    - Either IC/hit improves, or (equally valuable) the feature is shown to be non-helpful and can be removed from overrides.

- **Experiment C: inner-CV gap for alpha tuning (P1)**
  - Design: Keep everything else fixed; add inner `gap=total_gap` (or `gap=horizon`) for `TimeSeriesSplit` used by CV estimators.
  - Sample size: all OOS folds for a subset of benchmarks (start with 5 benchmarks to reduce runtime).
  - Success criteria:
    - Similar or slightly lower raw IC is acceptable if stability improves (less variation in chosen alpha, fewer regime-dependent swings).
    - Reduced dispersion of CPCV path ICs (diagnostic of robustness).

- **Experiment D: CPCV bug fix verification (P0)**
  - Design: Compare CPCV IC distributions before/after fix on the same dataset.
  - Sample size: same CPCV path count (currently C(8,2)=28 splits and n_test_paths per skfolio settings). citeturn2search0
  - Success criteria:
    - Distribution changes are explainable and stable under repeated runs (should be deterministic given fixed data).

## Recommended documentation edits and repo-structure changes

This repo is already unusually well documented. The changes below are about **keeping docs aligned with operational truth** and reducing confusion.

Documentation edits (high value, low effort):

- Replace hardcoded benchmark counts in docstrings with `len(config.ETF_BENCHMARK_UNIVERSE)` in:
  - `src/ingestion/fetch_scheduler.py` (currently describes 23 tickers / 22 ETFs). fileciteturn39file0L1-L1
  - `src/processing/multi_total_return.py` (mentions 20 benchmark ETFs). fileciteturn42file0L1-L1
  - `src/models/multi_benchmark_wfo.py` (mentions 20 models). fileciteturn43file0L1-L1

- Add a short “Index convention” section to `docs/architecture.md` (or `docs/data-sources.md`) documenting:
  - canonical monthly index (BME or ME),
  - how EDGAR month_end and macro month_end are normalized,
  - how forward returns are keyed.
  - This is the single most important way to prevent reintroducing the alignment bug. fileciteturn40file0L1-L1 fileciteturn41file0L1-L1

Repo structure suggestions (minimal change footprint):

- Consider creating a small `src/utils/time_index.py` (or similar) containing one canonical helper:
  - `normalize_monthly_index_to_bme(df_or_series)` (or to ME).
  - Use it in DB getters and feature engineering.
  - This avoids “fix in one place, drift in another” and is a low-complexity refactor.

- Add a `docs/known-issues.md` (or extend `docs/troubleshooting.md`) with “silent error” hazards:
  - index convention mismatch,
  - missing EDGAR keys / schema diffs,
  - stale DB runs,
  - API budget partial-ingestion risks.

## Risks and disclaimers

This system is clearly designed as decision support, but it is still a financial forecasting stack operating in a domain where true predictability is weak, unstable, and vulnerable to overfitting. That’s not a critique of your implementation; it’s an empirical regularity widely documented in the return predictability literature. citeturn1search1 citeturn1search0

Accordingly:

- Treat any single-month signal as **high-variance** and prioritize the repo’s governance gates (WFO stability, CPCV stability, baseline agreement, and interpretability) over incremental IC improvements.
- The most dangerous failure mode for ongoing use is **silent misalignment** (wrong months) or **silent missingness** (features null but pipeline “runs”). The P0 fixes above directly reduce this risk.
- Nothing in this report is investment advice; it’s a software and ML engineering quality review of the repository’s current state, focusing on correctness, leakage, and accuracy under minimal added complexity.