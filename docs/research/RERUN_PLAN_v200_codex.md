# Research rerun prompts — v200 onward (Codex)

> **Status (updated 2026-09-29): adopted.** This is the canonical v200 research plan (owner decision D1). The owner's decisions of 2026-09-27, D7 of 2026-09-28 and D8 of 2026-09-29, in the next section, amend it and take precedence over any conflicting text below. [`RERUN_PLAN_v200_claude.md`](RERUN_PLAN_v200_claude.md) is reference only, and its version numbers are void. Background: [the comparison of the two step-13V runs](../reviews/2026-09-26_step13v_comparison.md).

Written 2026-09-26 from [independent verification](../reviews/VERIFICATION_2026-09-26.md). This replaces separate 8a–8c prompts. It is a plan, not a performance result or authorization to fetch data. Copy the shared preamble and **one** numbered prompt into each new session. Use one PR per session. Start at v200; never reuse v1–v199. The gap separates the repaired data foundation from historical experiments and fix releases.

## Owner decisions and amendments (2026-09-27)

These decisions amend the plan below. Where the original text conflicts with them, they win. Amended places are marked with the decision (D1–D4) or addition (A1–A6) they come from.

**D1: this plan is canonical, with six additions from the Claude plan.**

- **A1 (v200): production-equivalence check.** With production settings (6M: 60-month train, 6-month test, gap 8; 12M: gap 15), the research harness must reproduce `pgr_vds.decision.signal_generation.generate_signals` and `pgr_vds.decision.health.compute_aggregate_health` on the same inputs and as-of date to 1e-9. The compared figures are aggregate R², equal-weight and pooled IC, hit rate and PT p. Report the result in the protocol bridge table, next to the stricter research gaps.
- **A2 (v200): R² fixtures.**
  - An oracle forecaster on overlapping 6M targets must score R² > 0 against the honest prevailing-mean comparator.
  - Pooling two benchmarks must not inflate R² (the concatenation case).
  - `src/research/v37_utils.compute_metrics` fails both (verification N2) and must not be used.
- **A3 (v202): the 2024 fiscal-month question.** `npw_growth_yoy` has a standard deviation of 0.198 in 2024 against 0.076 in 2016–2023. Before any block uses it, test whether Progressive's monthly reporting calendar changed in 2024, from the NPE/NPW first-month-of-quarter pattern in 2016–2026. If the change is confirmed, use trailing-12-month or quarterly NPW growth instead.
- **A4 (v202): per-era `pb_vs_pe` re-analysis.** Cover 2004–2014 and 2015 to the development end, with date-block inference and at most 20 pre-registered descriptive tests under their own Holm correction. The work is descriptive and outside the 38-candidate budget.
- **A5 (every step): owner summary.** Every study README starts with a plain-language summary for the owner: what was tested, the result in one sentence, what it means for the vest decision, and a definition of every metric used.
- **A6 (inventory): v10/v10.1 corrected.** It is operational hardening, not a model study. See the corrected v10 rows below.

**D2: no 24-month forward-data wait.** The owner does not want to wait for 24 months of new data. The retrospective quarantine defined below is therefore the final test for promotion.

- **The quarantine itself is unchanged:**
  - it is defined and sealed in v200;
  - v200–v206 never touch it;
  - v207 opens it once, in one registered batch, with all finalists frozen first.
- **The v207 rule.** At v207 a finalist may be recommended for promotion only if both of these hold:
  1. it met its own step's development threshold;
  2. on the quarantine, against the v200 control:
     - its primary metric is not worse: ΔR² ≥ 0 for regression, Brier not higher for classifiers, MAE not higher for the x-series, and after-tax uplift ≥ 0 for policies;
     - for regression finalists, equal-weight IC is not lower by more than 0.02, and no benchmark's R² is lower by more than 5 pp;
     - the paired date-block bootstrap test of its primary-metric improvement (the contract's settings), Holm-adjusted across the frozen finalists, gives a one-sided p below 0.10.

  A finalist that meets the development threshold but not the quarantine test is "consistent, not confirmed" and stays shadow.
- **Disclosure.** Earlier studies (v75, v129, v132) and both step-13V verification replays already saw the quarantine. v207 and any promotion PR must say so plainly. The owner accepts this weaker evidence in exchange for speed.
- **After any promotion:**
  - the monthly run keeps logging the v200 incumbent's forecasts next to the promoted model's;
  - a review after 12 more matured 6M outcomes applies the same rule;
  - a promoted model that fails the review is reverted through a governance PR.

  This review does not delay the promotion.
- **Superseded text:** the "Promotion evidence" paragraph below and v207 step 4.

**D3: fix sessions before research.** All five fix sessions, R1–R5 in [`PRE_V200_FIX_PROMPTS_codex.md`](../reviews/PRE_V200_FIX_PROMPTS_codex.md), run now, before v200. So do the dividend refresh and its offline check (R2-lite). The execution plan and model split are in that file.
- v200 starts only after all of these have merged and the required dividends pass freshness. Its `baseline_lock.json` pins that `master` commit and DB.
- This replaces the earlier order (R1–R3 before v200, R4 before v204, R5 before v206).

**D4: the monthly e-mail stays as it is.** Research sessions do not change its content or format; a later, separate session will revisit them.

**D7 (2026-09-28, after R3b merged): the v200 start condition, pinned inputs and two carried-forward conditions.** The fix phase is complete. This updates the plan to match it.

- **Start condition.** v200 starts after all of the following are on `master`:
  - R1–R5 (v186, v188, v189, v190);
  - the step-0 dividend refresh (`ed7997f`) and its check, R2-lite (v187, PR #144);
  - the tax-loss label fix, v192 (PR #143, decision D6 in the fix-prompts file);
  - R3b, v191 (decision D5, PR #145), which records the current baseline in [`docs/model-governance.md`](../model-governance.md).

  This replaces D3's list.
- **Pinned inputs.** The clean inputs for v200's `baseline_lock.json` are R3b's baseline:
  - code: `c3b4798b90a43f9dbd616981861e9ff01fa2de72`, R3b's code pin. The lock also records the `master` commit v200 actually runs on, which contains R3b. Any production-code change between the two must be listed. From `c3b4798` to `b446433`, R3b's merge, only docs changed.
  - DB: SHA256 `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`, the refreshed DB committed at `ed7997f` (R2-lite's "after" DB).

  The weekly workflows keep committing a new `data/pgr_financials.db` to `master`, so the DB on `master` is not a pin. Extract the pinned DB with `git show ed7997f:data/pgr_financials.db` to a path outside the repository, and verify its SHA256 before any read. If it does not match, stop.

  The seed pins in the preamble and in every step (`aae0be8…`, DB `7c68ef…`) stay as the parent seed hashes for provenance only. The dividend-staleness preflight in v200 step 1 is satisfied by R2-lite's record, not by a new repair PR. The v200 lock cites that record and R3b's closeout.
- **VWO March 2026 dividend (disclosure).** VWO has no March 2026 ex-date in the provider's full history. It has one in every year from 2013 to 2025. On 2026-09-28 the owner accepted the gap as provider data ([R2-lite record](../reviews/2026-09-28_R2_dividend_refresh_check.md), caveat 1). The 12 VWO targets whose windows span 2026-03-20 stay as stored: 6M anchors 2025-09-30 to 2026-02-27, and 12M anchors 2025-03-31 to 2025-08-29.
  - v200 lists them in its availability ledger and states the gap in its README.
  - At R3b's as-of date, 2026-09-21, VWO passes the freshness check. Back-dated checks inside the gap fail on VWO dividends; for example, R3b's 2026-05-22 replay fails `data_ready`. That failure is expected and is labelled, not repaired. No session re-fetches or edits VWO rows.
  - Any VWO-specific result that depends on those rows says so.
- **v206 prerequisite from R5.** [R5's closeout](../reviews/R5_event_tax_closeout.md) requires that the always-50% control and every candidate policy compute event outcomes with the same `compute_event_outcome` (`src/backtest/backtest_engine.py`), over one common mask of available events. Historical policy utilities were not recalculated after R5, so no earlier event-utility figure is comparable evidence.

**D8 (2026-09-29, after v200, v205 and the blocked v201 merged): lessons from the first three studies.** v200 (PR #147) produced the accepted comparator, lock SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`. v205 (PR #149) ran and sent no finalist. v201 (PR #148) stopped before fitting on a registry-pin check. These amendments stop the same problems recurring.

- **Registry pin: add-only check.** v200's lock pins the bytes of `research/registry.yaml`, and every later study must add its own entry. A plain hash check therefore fails, which is what blocked v201.
  - Verify the lock with the add-only rule v205 used (`verify_lock_with_registry_append` in `research/studies/v205_dividend_bvps/run.py`). The registry is checked against its bytes at v200's execution commit, and all 114 pinned entries must be unchanged. Every other consumed file must match its pin in the working tree.
  - Use the shared, tested version in `src/pgr_vds/research_lib/provenance.py` once one exists. If none exists yet, the next study moves it there with tests; it does not write its own copy.
  - This is not a successor lock: v200's lock is unchanged.
- **v201 rerun (authorized).** v201's blocked session fitted nothing and saw no results. The rerun keeps its six declared ideas (P1–P3, M1–M3 in v201's `candidate_ledger.json`) and slots 1–6. It is not a retry after results.
- **Runtime and typing.** Use Python 3.12.14 with the exact versions in v200's `runtime_lock.json`, in an isolated environment; never upgrade. CI runs mypy: run it on new `research_lib` code with the locked Python before pushing (v200's first CI run failed on six mypy errors).
- **Multiplicity slots.** The 38-test campaign family is numbered by study: v201 1–6, v202 7–14, v203 15–22, v204 23–28, v205 29–34 (used, no finalist), v206 35–38. Each study fills only its own slots; the others stay pending at p = 1 until v207 completes the Holm adjustment.
- **Consume v200's outputs.** Take the mature naive (prevailing-mean) forecasts, support and control forecasts from v200's pinned outputs; do not recompute them. v205's first attempt differed because it left out v200's pre-output history label (the 1999-11 origin).
- **Development dates in tests.** Tests, fixtures and CI smoke runs use an as-of date no later than 2022-08-31, v200's development bridge date, so no label they read reaches the quarantine boundary of 2023-09-29.
- **Carried to later steps:** v202 (A3 still open; PIF vintage), v204 (minimum support for the classifier lane) and v207 (VWO targets; finalists so far), in the notes inside those prompts.

**Execution notes (research phase).**
- **Order:** v200 comes first. After it, v205 (the x-series lane) can run in parallel with the chain v201 → v202 → v203 → v204 → v206. v207 comes last.
- **Recommended model split:**
  - v200, and v201–v204 and v206: built by the Codex model and reviewed by the Claude model. This keeps one builder on the main chain, the plan's author implementing its own validation contract, and a second model independently reproducing v200's baseline numbers.
  - v205: built by Claude, in parallel, and reviewed by Codex.
  - v207: run by Claude, which built the fewest candidates, and reviewed by Codex. The model that opens the quarantine once and applies the D2 rule should not be the one that built most of the candidates.
- **Reviews** use the cross-model review prompt in the fix-prompts file.

## What the evidence supports

The repairs materially change targets, features and evaluation. Rejected momentum, macro, EDGAR, calibration and validation studies deserve bounded reconsideration. Archived winners are not independent confirmations: many reuse the same predictions or selection windows. v127/v130/v135 are one calibration-selection lineage; v151 forwards baseline forecasts with candidate metadata rather than measuring the joint candidate; v159 Firth adoption is claimed in the backlog but not supported by the current implementation tree. Historical conclusions in the inventory are attributed records, not freshly replicated conclusions.

Eight sessions cover the useful questions. Each has a restricted candidate list and fixed control; none is a license to rerun hundreds of exploratory variants.

| Version | Area / study folder | Candidate budget, excluding fixed control | Dependency |
|---|---|---|---|
| v200 | Clean data/target/validation baseline; `v200_clean_baseline` | 0: incumbent + fixed endpoint controls | Authorized data repair + integrity preflight |
| v201 | Price, technical and macro availability; `v201_price_macro` | 6 | v200 lock |
| v202 | Insurance fundamentals, Gainshare and valuation; `v202_insurance_valuation` | 8 | v200; fold-causal v201 secondary control |
| v203 | Regularisation, shallow ensembles and benchmark consensus; `v203_model_consensus` | 8 | v200–v202 development results |
| v204 | Classification and prequential calibration; `v204_classification_calibration` | 6 | v200 and fold-causal v203 predictions |
| v205 | Dividend/BVPS x-series; `v205_dividend_bvps` | 6 across both lanes | v200 clean inputs + horizon12 lock |
| v206 | Decision policy and tax mapping; `v206_policy_tax` | 4 | Fold-causal v200–v204 forecasts; tax fixtures |
| v207 | One holdout synthesis; `v207_synthesis` | 0 new trials; ≤6 winners + incumbent | Prior sessions closed, manifest frozen |

Maximum **38 candidate blueprints** across v201–v206. Inner hyperparameter grids are declared below and logged separately; all attempted, failed, abandoned and revised candidates consume the budget. No discretionary extra variants or retry searches based on promising metrics. A harness correction invalidates affected comparisons and requires an amended preregistration, not a claim of independent replication.

## Prerequisites and two kinds of holdout

**Data gate:** the audited snapshot has20 stale dividend tickers (19 ETFs + ALL). *(Amended 2026-09-27, D3: the gate is met by the existing dividend-refresh workflow run plus the offline R2-lite check, or by the full R2 if the refresh fails.)* Before certifying v200, a separate explicitly authorized data-repair PR must backfill required histories, include a target rebuild/migration, and pass split/DRIP, freshness, coverage and accounting tests. These research prompts authorize no provider call or DB repair. v200 may reproduce the seed to diagnose the block, but must stop candidate comparison and label it PROVISIONAL if required dividends remain stale. Isolate CB entity-splice data until a separate identity repair passes. Stale research series cannot silently forward-fill. Latest-vintage macro data must be labeled; availability lags do not restore historical vintages.

**Retrospective quarantine:** reserve the most recent24 fully matured monthly target origins for each used horizon. On the audited snapshot the6M anchors are March2024–February2026; for12M they are September2023–August2025. Resolve exact BME dates from the target-end calendar in v200; fail if a claimed mature label ends after the pinned as-of date. Quarantine the **union** across horizons, and restrict development origins to before the earliest quarantined origin (September2023 here). Purge every development label whose outcome window reaches that boundary, including x-series targets. This prevents6M research from training on the12M holdout. Seal partition hashes and an access ledger before fitting. All v200–v206 metrics use development only. Score quarantined rows **once, in one batch at v207**, with all candidates frozen first.

Those rows were already touched by old research and the mandatory verification replays. They are not a virgin holdout. *(Amended 2026-09-27, D2:)* passing that batch under the v207 rule can support a promotion recommendation, and v207 and any promotion PR must disclose the prior exposure. Do not shift dates after seeing outcomes or conceal this history.

**Promotion evidence (superseded 2026-09-27, D2):** the original rule required 24 unused forward monthly origins (for example October 2026–September 2028, with 6M labels maturing by March 2029) before any promotion. The owner declined that wait. The D2 rule at the top of this plan replaces it.

## Shared preamble — paste with every step

> Run one bounded research session. Read AGENTS.md, the independent verification report, relevant original findings, registry entries and prior-study summaries before implementing. Branch from latest master and open one PR. Preserve unrelated work. Use at most 2 subagents. Use DB copies outside the tracked path or immutable reads. Never run fetchers or email. Do not call Alpha Vantage, FRED or EDGAR unless the session's user prompt explicitly authorizes it. If explicitly authorized, EDGAR calls must send `User-Agent: Jeff Hester jeffrey.r.hester@gmail.com`, stay below10 requests/second and cache responses. Any data write needs a separate explicit migration/rebuild PR. No yfinance fundamentals; use unadjusted prices with manual splits and fractional-share dividend reinvestment for returns. Compute rolling monthly features. Every mathematical module gets an independent expected-output pytest fixture, observed red before implementation and green after. Run `python -m pytest -o addopts="--tb=short" -q`, report pytest's own summary and exit, and compare tracked DB SHA256 before/after. Report inherited failures honestly.
>
> Seed inputs: Git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. The seed fails dividend freshness. *(Amended 2026-09-28, D7: the approved repair is R3b's baseline, code `c3b4798b90a43f9dbd616981861e9ff01fa2de72` and DB SHA256 `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6` from `git show ed7997f:data/pgr_financials.db`. The VWO March 2026 gap is accepted provider data; disclose it.)* v200 must bind any approved repair to its exact full commit/DB SHA256 in `outputs/baseline_lock.json`, with parent seed hashes and migration/rebuild IDs. Later sessions copy and verify those exact inputs, and record their own code commit separately. “Latest DB”, a branch name, parquet alone or a mutable path is not a pin. If the clean lock is absent or preflight fails, write a blocked README and stop fitting. No v201+ comparison against a provisional v200.
>
> Apply the validation contract below to all forecasts, classifiers, calibration and policies. Reusable new code belongs in `src/pgr_vds/research_lib/` (create this installed namespace in v200; it does not yet exist). Study runners go in `research/studies/v2NN_<slug>/run.py`, with `outputs/`, `README.md` and `provenance.json` in that folder. Register in `research/registry.yaml` and regenerate research/README.md via `python research/tools/registry.py --write`. Tests belong in `tests/research/`; no sys.path edits. Old helpers need tested causal adapters before reuse. Pin all consumed CSV/parquet/model/output files by SHA256. *(Amended 2026-09-29, D8: verify v200's lock, SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`, with the add-only registry check, not a plain hash of `research/registry.yaml`. Use Python 3.12.14 and the exact versions in v200's `runtime_lock.json`, and run mypy on new `research_lib` code before pushing. Fill only this study's multiplicity slots: v201 1–6, v202 7–14, v203 15–22, v204 23–28, v205 29–34, v206 35–38. Take mature naive forecasts, support and controls from v200's pinned outputs. Tests and smoke runs use an as-of date no later than 2022-08-31.)*
>
> No live config, live features, recommendation policy or model changes in research PRs. A winner needs a separate promotion PR under docs/model-governance.md after passing the v207 quarantine rule (D2). Do not change the monthly e-mail's content or format (D4). Start every study README with a plain-language summary for the owner (A5). Record blockers, attempts, negative results and discarded comparisons. End with what changed / what is left.

## Validation and inference contract — mandatory in every prompt

1. **Chronology:** monthly origins and explicit target end/availability dates. Outer `sklearn.model_selection.TimeSeriesSplit`: sliding60-month training window for h6 and120-month window for h12,6-month test window, gap=`2*h` months for horizon h. First h months purge overlapping targets, additional h months provide embargo, both ≥horizon. Add label-end checks against every test origin; calendar availability can require longer gaps. Earlier training only. No K-fold, CPCV, LOO, shuffle or full-sample scaling. Insufficient history means insufficient evidence, not reducing gaps. h6 gap12; h12 gap24.
2. **Nested selection:** three inner chronological TimeSeriesSplit folds with explicit test_size=6 and gap=2*h inside each outer training fold. Outer max_train_size=60 for h6 and120 for h12; inner max_train_size equals that horizon’s outer window. Minimum usable inner training support is24 unique months for h6 and60 for h12 (first inner folds start with30 and78 rows before availability exclusions). Split unique monthly dates before expanding benchmark-panel rows. Check support and skip unsupported folds; no sklearn default test sizing. Features, pruning, imputation medians, scaling, priors, hyperparameters, weights and thresholds learn only there. Fit transforms on training and apply unchanged to test. Monthly samples of annual targets remain overlapping, not independent annual events. Mark unsupported folds unscorable. Upstream recipes are learned choices too: downstream v203/v204/v206 must choose/reconstruct them inside each downstream inner training history. A globally chosen upstream winner may be shown only as an exploratory retrospective sensitivity, excluded from honest WFO/promotion thresholds. Freeze catalog/procedure, not a future-selected winner for earlier dates. Never use outer test outcomes to select their own parameters.
3. **Honest metrics:** OOS R²=`1−Σ(y−prediction)²/Σ(y−prevailing_mean)²`, where prevailing mean uses only labels whose outcomes arrived by that origin, including matured training history. Never include the current target. Report equal-weight benchmark IC and panel IC with date-clustered/HAC inference or moving-date-block bootstrap of length≥h, keeping all benchmarks for each date together. Compare hit rate with the constant majority-direction rule learned from mature past labels; report base rate and directional-skill test. Prequential calibration uses only matured past predictions/outcomes: Brier/log loss/ECE and nominal80% interval coverage. Mark warmup unevaluated; never fill it from future residuals. Report sample support, availability windows, costs and uncertainty; no row-pooled p-value as independent confirmation.
4. **Multiplicity:** freeze candidate register/ordered grids before fitting. One preregistered primary improvement test per candidate; Holm adjustment across38 campaign candidates at familywise.05, date blocks, seed20260926,2,000 bootstrap replicates. Report raw/adjusted p-values and all attempts. Pending/unused slots are assigned p=1 for conservative interim Holm reporting; v207 completes the campaign adjustment. A new feature subset, threshold, lag or recipe consumes a candidate. Secondary metrics are safeguards, not another route around a failed primary test. Keep different endpoints separate; adjust registered primary p-values together rather than pooling labels.
5. **Control/success:** v200 evaluates current production classes/features/settings under this stricter research protocol; it does not change live gaps. Isolate protocol changes from data repairs. Later candidates share v200 support, targets, folds and naive forecast. For primary6M regression: development ΔR²≥+.010 absolute vs v200, adjusted paired date-block p<.05, equal-weight IC loss≤.01, no directional/calibration deterioration. An active-policy proposal also needs hit-rate skill vs past-learned base rate with adjusted p<.05, ECE≤.10 and80% coverage.75–.85. Otherwise retain research/shadow status. Classifier/x/policy endpoints have additional thresholds below, not substitutes for regression skill.
6. **Provenance/holdout:** `provenance.json` records full input DB/git hashes, code commit/dirty status, baseline-lock hash, as-of/max-available dates, extraction SQL, target definitions/ends, splits/dividends/FRED/EDGAR provenance, runtime/dependency lock, seed, folds/purge/embargo, blueprint/inner-grid budgets, attempts, metric definitions, output hashes and quarantine/access-ledger hashes. No unversioned feature cache. v200–v206 never compute holdout metrics; v207 opens the retrospective quarantine once after freezing finalists. Promotion follows the v207 quarantine rule (D2).

## v200 — Clean data, targets and honest baseline: one new-session prompt

> Execute only v200_clean_baseline, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** The incumbent has a reproducible, modest or absent edge once data and evaluation are repaired; retain its model specification.
>
> **Revisits:** v9/v18/v20–v25, v37/v38, v66–v73, v128/v129, v134/v142; F01–F18,F22/F23,F33–F35.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 0 alternatives: one incumbent forecast blueprint plus non-searched endpoint controls. Preserve production Ridge’s50 penalties `np.logspace(-4,4,50)` inside nested folds, distinct from the10 shrinkage values (.05,.10,.15,.20,.25,.30,.40,.50,.75,1.00) selected from matured past labels. These are declared tuning choices, not50 outer-test candidates. No metric/window search. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. Preflight the seed copy. Required dividend staleness blocks a clean baseline; report the20 stale tickers and require the separate authorized repair lock. *(Amended 2026-09-28, D7: the repair is R2-lite's verified refresh, pinned by R3b's baseline. Preflight that pinned DB, cite the R2-lite record and R3b's closeout in the lock, and list the 12 VWO targets affected by the accepted March 2026 gap in the availability ledger.)* Check split/DRIP targets, month-end windows, filing/release availability, quarter identities, duplicate macro months and missing live features. Exclude unresolved CB and optional stale research series.
2. After repair preflight passes, create the minimal installed research_lib provenance/temporal/metric package with independent red/green tests. Do not wrap old cv=None, CPCV, origin-only holdout filtering, future-residual warmup or full-frame pruning.
3. Hash development/quarantine partitions before fitting. Score fixed production Ridge+shallow-GBT features/settings on development folds. Cite the existing13V provisional table only; its full-history September replay includes quarantined origins and must not run again before v207. A reproduction belongs in the registered v207 batch, not v200.
4. Lock non-searched matched-endpoint controls too: current same-label PathB classifier/probability stream; past12M cash dividend amount; past-only mean annual excess-dividend/current-BVPS ratio; past-only prevailing BVPS growth; past-only mean absolute6M PGR DRIP return. Include exact target/unit definitions and support for v204/v205 in baseline_lock.json. Save all development control forecasts, per-date/benchmark residuals, honest naive forecasts, support and warmup flags. Research gap12/24 is stricter than live6M gap8 /12M gap15: isolate the protocol effect in a bridge table, without tuning. Freeze dependencies and forbid silent later upgrades.
5. (A1) Production-equivalence check. With production settings (6M gap 8, 12M gap 15), reproduce `generate_signals` and `compute_aggregate_health` on the same inputs and as-of date to 1e-9 (aggregate R², equal-weight and pooled IC, hit rate, PT p), at an as-of date inside the development period. Report it in the bridge table.
6. (A2) R² fixtures in `tests/research/`: an oracle forecaster on overlapping 6M targets scores R² > 0 with the honest prevailing-mean comparator, and pooling two benchmarks does not inflate R². Show that `src/research/v37_utils.compute_metrics` fails both (red) and the new `research_lib` metric passes (green).

> **Outputs/provenance:** `research/studies/v200_clean_baseline/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** Acceptance requires all data/availability/temporal fixtures passing, matching hashes, sufficient chronological support, complete reporting and forecasts reproducible within1e-10 on a second read-only run. No promotion in v200. Freeze its metrics as comparator; do not require positive R² to publish a baseline.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v201 — Price/technical and macro availability: one new-session prompt

> Execute only v201_price_macro, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Calendar windows and single publication lags may rescue a small block previously rejected because its inputs were broken.
>
> **Revisits:** v9,v15/v18–v20,v43–v45/v54,v133/v134,v162–v165; F01/F03/F05/F06/F07/F13/F24/F27.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 6 one-factor blueprints:3 price/TA and3 macro, no winner combinations. Use the identical production50-penalty Ridge grid for candidate and control, with identical10-value past-only shrinkage and fixed incumbent GBT settings. Only the feature block changes; no extra tuning recipes. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. Freeze six candidates: P1 calendar3/6/12M momentum block; P2 adjusted13-week volatility plus52-week-high distance; P3 two lean trend/reversal indicators chosen from the v162 inventory before metrics. M1 slope/real-yield-change at observed availability; M2 VIX/NFCI/credit at observed availability; M3 insurance-PPI minus cost-index gap with one publication lag. Extra indicators/lags consume the same total6 budget.
2. Reuse v200 returns/folds; explicitly remove replaced columns. Raw macro observations get one calendar-period availability rule; document vintage limits and reject stale/double-lagged values. Store all blueprints before execution.
3. Fit nested fold-local transforms and outer gap12 for6M. Emit corrected/archived feature examples (mom12, April2020 VIX, rate gap), future-perturbation fixtures and all six development results. No provider backfill.

> **Outputs/provenance:** `research/studies/v201_price_macro/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** ΔR²≥.010 vs v200, adjusted paired p<.05, IC loss≤.01 and no directional/calibration deterioration, per contract. One winner may advance to v207 as a finalist (D2).
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v202 — Insurance fundamentals, Gainshare and valuation: one new-session prompt

> Execute only v202_insurance_valuation, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Clean filing dates, share basis and YoY may change the value of underwriting, PIF, Gainshare and P/B-versus-P/E features.
>
> **Revisits:** v15–v21,v42/v49/v55/v60,v128/v129,v136/v137,v142/v143,pb_vs_pe,x12/x17/x19; F09–F12/F15–F18/F23/F33–F35.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 8 one-factor blocks, no pairwise combinations. Identical production50-penalty Ridge/10-value causal shrinkage grids in candidate/control; incumbent GBT unchanged. Only the feature block changes. At mostone winner. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. Freeze eight addition/replacement blocks: trailingCR/change; calendarPIF growth excluding property-definition drift; Gainshare proxy from trailingCR/PIF; investment-income growth/percent book yield; latest-share-basis BVPS growth; trailingROE with correctQ4; split-consistent P/B-versus-P/E spread; calendarNPW/rate-adequacy gap. No scanning every historical feature.
2. Align observations to actual filing/release dates. Missing dates use an explicitly conservative fallback and sensitivity table, not optimized lags. v142's old lag winner is not a fact. A later amendment cannot enter an earlier fold; flag source-vintage limitations.
3. Test units, leap-Feb keys, negativeNI, missing old months, common-equity identities and split continuity. Prune inside inner training only. Rebuild pb_vs_pe inference using date blocks and matched support. A v201 secondary comparator is reconstructed by selecting from its preregistered catalog inside each inner training history; the globally selected development winner is exploratory only, not honest evidence.
4. *(Amended 2026-09-29, D8: v205 used trailing-12M NPW growth without running this test, so it is still open here. `pif_total` differs from first-reported values in 50 months from 2019-07 (step 3b's single-definition rebuild, F11); PIF features are not a historical vintage, so disclose it as v205 did in its `source_vintage_audit.json`. Report against frozen v201 only after the v201 rerun has merged.)* (A3) Before any block uses `npw_growth_yoy`, test the 2024 fiscal-month hypothesis from the NPE/NPW first-month-of-quarter pattern, 2016–2026 (2024 standard deviation 0.198 against 0.076 in 2016–2023). If it is confirmed, use trailing-12-month or quarterly NPW growth in the affected blocks, and record the finding.
5. (A4) Per-era `pb_vs_pe` re-analysis (2004–2014; 2015 to the development end) with date-block inference: at most 20 pre-registered descriptive tests under their own Holm correction, outside the 38-candidate budget. It informs the valuation blocks but cannot promote anything on its own.

> **Outputs/provenance:** `research/studies/v202_insurance_valuation/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** Same ΔR²≥.010, adjusted p<.05 and safeguards vs v200; report relative to frozen v201 too. Plausibility, repeated annual samples or in-sample fit cannot justify promotion.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v203 — Regularisation, shallow ensembles and benchmark consensus: one new-session prompt

> Execute only v203_model_consensus, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Simple regularisation and causal weighting may beat the incumbent after repairing targets and its naive comparator.
>
> **Revisits:** v37–v41,v46–v59,v70/v72/v75,v110/v112/v114,v125/v126,v138/v140/v141/v144/v149/v150,BL01; F02/F04/F13/F25/F31.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 8 complete blueprints, no cartesian products. Ridge/Lasso inner penalty {1,10,100}; tree depths {1,2}, fixed100 estimators and learning rate.05. Max3 choices per blueprint/fold. No RidgeCV(cv=None). Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. Freeze the upstream candidate catalog and a fold-local selector before metrics. At mostone upstream block may enter each fold, selected only from that fold’s inner training evaluations, never by the globally best development OOS result. Candidates: Ridge-only; Lasso-only; depth1/2 GBT-only; fixed50:50 Ridge/GBT; equal benchmark aggregation; causal matured-error quality weighting; investable-only fixed equal weighting; VTI-for-VOO on matched actual-data support. Each is a full blueprint vs v200; no holdout-dependent asset choice.
2. Distinguish forecast and redeploy universes; exclude unresolvedCB. Weights/alpha/shrinkage must be nested or prequential. Synthetic alpha perturbation must change predictions, resolving v140's old flat-metric ambiguity. BL01 remains attributed synthetic evidence, not an additional candidate or empirical portfolio proof.
3. Retain date/benchmark panels. Replace v138/v149/v150 full-sample residual-variance warmup with a declared past-only prior. Test label-end purge and future-outcome invariance. Use the same realised prevailing-mean comparator.

> **Outputs/provenance:** `research/studies/v203_model_consensus/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** ΔR²≥.010, adjusted p<.05 and all regression safeguards. Choose the simplest qualifying complete blueprint. At mostone forecast candidate advances; synthesis cannot assemble untested winner combinations.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v204 — Classification and prequential calibration: one new-session prompt

> Execute only v204_classification_calibration, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Classifier/calibration conclusions may change with current features, mature labels and honest base rates instead of reused selected probabilities.
>
> **Revisits:** v46–v59,v87–v96,v111/v113/v115–v121,v125/v127/v130–v135,v138/v145–v150,v154–v158,claimedv159,v162–v165; F02/F04/F13/F24/F25.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 6 blueprints; logistic inner C {0.1,1,10}, or temperature {0.8,1,1.2}, max3 choices. No joint threshold/feature/temperature sweep. Decision bands fixed. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. *(Amended 2026-09-29, D8: v200's same-label PathB control has only 72 raw and 35 calibrated scored dates. Before fitting, preregister a minimum-support rule that closes the lane when support is too thin, as v205 did for its annual lane, and report the support each gate is judged on.)* Freeze six alternatives: matched PathA Platt mapping of fold-causal regression scores; matched PathB unweighted logistic; matched PathB balanced logistic; matched PathB Firth logistic; temperature calibration of fold-causal PathB; past-only base-rate/score shrinkage of PathB with declared prior. PathA/PathB use the same benchmarks, dates,3% actionable-sell labels and training availability, differing only in score-versus-feature architecture. v200 classifier and uncalibrated fold-causal sources are fixed controls. Reconstruct upstream streams using only each downstream training history, never a global winner selected on later outcomes.
2. Keep actionable-sell and relative-outperformance endpoints separate. Preregister actionable-sell under the current3% threshold as primary. Score actual current features; imputation/scaling stays inside inner training pipelines. Warmup cannot use future means.
3. Calibrate only after past labels mature. Keep live PathB bands.30/.70 fixed; archived alternative bands are descriptive. Direction-aware overlays and later maturity attachment require fixtures. Verify Firth actually fits distinct predictions: metadata is not implementation. Emit Brier/log loss/ECE, balanced accuracy, hit/base rate and date-block uncertainty on common dates.

> **Outputs/provenance:** `research/studies/v204_classification_calibration/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** Brier loss at least5% lower than v200 same-label classifier control, adjusted paired p<.05; balanced accuracy +.02, ECE≤.10 and no worse log loss; directional skill beats past-learned base rule. At mostone shadow lane advances; no immediate regression-policy override.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v205 — Dividend/BVPS x-series lanes: one new-session prompt

> Execute only v205_dividend_bvps, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Clean DRIP, December-special, calendar, share-basis and provenance handling may change x-series conclusions more than algorithm complexity.
>
> **Revisits:** x1–x24, especiallyx1/x2/x9/x12/x15/x17/x19/x22–x24; v128/v143; F03/F05/F08/F09–F12/F15–F18/F23/F25/F30/F33–F35.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 6 total:3 dividend and3 BVPS. Ridge alpha {1,10,100} nested, max3 choices each. Occurrence/amount tasks separate; no algorithm tournament. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. Create hand-calculated targets: raw price with splits/fractional DRIP; calendar matching; December/Q1 specials; trailing12M BVPS on common share basis; mature end dates. No raw price.shift(-h), unversioned parquet or future target mean. Freeze joint6M/12M partition locks before fitting.
2. Dividend12M lane: D1 past dividend+CR; D2 add calendarPIF/NPW; D3 Gainshare/percent book yield on the annual excess-dividend/current-BVPS survivor endpoint. Primary endpoint next12M per-share cash dividend amount for D1/D2, fixed past12M dividend control. D3 instead explicitly revisits the x-series annual excess-dividend/current-BVPS survivor: define excess using the archived x23 contract before metrics, and compare only with its v200 past-only same-label control; do not mix these endpoint errors. Ordinary/special components are descriptive decompositions, not one-class occurrence “success”.
3. BVPS12M lane: B1 lagged split-consistent BVPS/ROE; B2 add underwriting/PIF; B3 the x16/x17 persistent-BVPS/no-change-PB6M structural survivor with its archived mapping frozen before fitting and split/DRIP-correct absolute6M PGR target. B1/B2 use next12M BVPS growth and fixed past-only growth control; B3 compares to the v200 absolute6M DRIP mean control. Keep units/endpoints separate. Rolling monthly labels remain overlapping:12-month blocks and independent annual-event counts govern uncertainty.
4. Horizon12 outer120/6/24; horizon6 outer60/6/12. Three inner folds with test_size6 and minimum support60/24 respectively; all filing-date aligned. Report ΔR²/IC where meaningful, MAE, proper direction/base comparisons, prequential calibration and80% coverage. Insufficient annual support closes a lane; never reduce exclusion. These winners do not inherit6M stock-return policy claims.

> **Outputs/provenance:** `research/studies/v205_dividend_bvps/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** Each lane: MAE≥10% lower than its v200 matched-label control, adjusted block p<.05, ΔR²≥.01, no directional/calibration deterioration and sufficient independent events. Nominate at most one winner across the two lanes to keep v207 total≤6; decide using development evidence before holdout.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v206 — Decision policy and tax mapping: one new-session prompt

> Execute only v206_policy_tax, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Correct tax boundaries and honest probabilities may improve utility; old50% agreement may merely reflect forced defaults.
>
> **Revisits:** v9/v11–v14/v17/v20–v28,v35/v37/v53/v57,v70/v75/v116/v117/v120/v121; F02/F13/F14/F19–F24.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 4 policies: current mapping; fixed25/50/75 monotone bands; calibrated direction-supported abstention; continuous capped25–75 mapping. Always50% is fixed utility control. No inner tax-rate or band search. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. Freeze exact definitions and constants before scoring. Use v200 and one fold-causal v203/v204 source, selected inside the current outer training history; globally development-selected streams are exploratory only; do not optimize policy by benchmark. Policies have no extra estimator; source forecasts retain nested TimeSeriesSplit gap12 and utility uses the same outer dates.
2. Backtest actual vest/decision timestamps and horizon, quantify F22's suspected offset rather than copying nearest-target shortcuts. *(Amended 2026-09-28, D7: R5 (v190) fixed the offset. The always-50% control and every candidate policy use the same `compute_event_outcome` over one common available-event mask; do not reuse pre-R5 event utilities.)* Freeze documented owner tax rates, gain/lot scenarios and costs. Show sensitivity without choosing favorable cases. Anniversary+one-day LTCG, per-share gain ranking, unvested exclusion, signed breakeven, wash windows and expected proceeds need red/green fixtures. No silently invented owner tax assumptions.
3. Report after-tax utility, tail loss, turnover and abstention with date-clustered uncertainty. Also report source R²/IC/hit/base/calibration; constant asset drift is not model skill. Missing diagnostics fail closed. Research uses no CPCV; do not modify live policy here.

> **Outputs/provenance:** `research/studies/v206_policy_tax/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition:** Primary mean after-tax uplift≥0.25 percentage points/decision vs always50%, adjusted block p<.05, lower95% paired improvement bound>0. No worse95th-percentile loss, no unsupported bullish selling>50%, and all source skill/calibration safeguards. Otherwise keep default; correct tax fixtures alone do not justify promotion.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## v207 — Frozen comparison and promotion recommendation: one new-session prompt

> Execute only v207_synthesis, with the shared preamble and full validation contract. Create research artifacts, not a live promotion.
>
> **Hypothesis:** Development winners must survive a common sealed check; overlap, selection and prior holdout use may erase apparent improvement.
>
> **Revisits:** Allv200–v206; promotion lineagesv13/v21/v22/v36/v38/v72/v75/v128/v142; remainingF01–F35 limits.
>
> **Pinned inputs:** seed git `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`, DB SHA256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`, as-of2026-09-26. Clean comparisons require the approved-repair v200 baseline_lock.json with its exact replacement git/DB hashes; verify before running. Pin consumed earlier forecasts/output files by SHA256. Do not use a mutable latest snapshot.
>
> **Method:** rolling monthly features; outer TimeSeriesSplit60-month train for h6 /120 for h12,6-month test, purge h plus embargo h (`gap=2*h`), three chronological inner folds (test_size=6; minimum usable training24 months for h6 /60 for h12) for all learned choices; label-end and filing/publication-date availability checks. No K-fold/CPCV/LOO, shuffle, full-sample scaling or holdout access before v207. Honest realised prevailing-mean OOS R², date-clustered IC significance, hit rate versus past-learned base rate and prequential calibration, as in the contract.
>
> **Budget:** 0 search. Freeze≤6 total winners plus incumbent: one per earlier candidate session, including at most one x lane. No recipe/feature mix/threshold invented here. Register before execution; campaign-Holm-adjusted primary tests; no extra search after inspecting results.

1. *(Amended 2026-09-29, D8: v205 nominated no finalist, so at most five remain, one each from v201–v204 and v206. Complete the Holm adjustment from each study's slot ledger. The 12 VWO targets affected by the accepted March 2026 dividend gap, listed in v200's `outputs/vwo_accepted_gap.csv`, are all in the quarantine: score them as stored, and preregister a sensitivity run that leaves them out, reported next to the primary result.)* Audit lock/input hashes, candidate count≤38, nested choices/gaps, date-clustered inference, quarantine ledger and abandoned attempts. Disqualify affected candidates if preflight/chronology fails. Freeze outputs/finalists_manifest.json and nominate champion(s) from development evidence before unsealing. Use matched controls for annual/classification endpoints, not incomparable metrics.
2. Open all retrospective quarantines once in one registered batch for finalists/controls. Report every outcome. Keep horizon-sized date blocks, prevailing-mean R², IC, hit/base skill, prequential calibration, coverage and matched utility. A repeated execution may reproduce identical files only; it cannot change candidates after label access. Log accesses.
3. *(Amended 2026-09-27, D2.)* Check each finalist's development evidence against its step's unchanged threshold, then apply the D2 quarantine rule to the batch. Failures remain research-only. A finalist that passes both may be recommended for promotion. One that passes the development threshold only is "consistent, not confirmed" and stays shadow. Write outputs/promotion_recommendation.md with uncertainty, the quarantine's prior exposure (v75, v129, v132, both step-13V replays), source limitations, candidates rejected and the owner decision.
4. *(Superseded 2026-09-27, D2: no forward-origin wait.)* Pre-register D2's post-promotion review instead. The v200 incumbent is logged next to any promoted model. After 12 more matured 6M outcomes, the same rule decides whether the promoted model stays or is reverted through a governance PR. No live config changes here.

> **Outputs/provenance:** `research/studies/v207_synthesis/run.py`, `README.md`, `provenance.json`, `outputs/` containing candidate/fold/availability ledgers, predictions, metrics, comparison and closeout. Record full input/code hashes, baseline lock, runtime/dependencies, grids, targets and holdout access. Register in `research/registry.yaml`, regenerate research/README.md. Shared new code in `src/pgr_vds/research_lib/`; math tests in `tests/research/`. No unhashed cache inputs.
>
> **Preregistered threshold/disposition (amended 2026-09-27, D2):** Recommend promotion only for a finalist that met its development threshold and passes the D2 quarantine rule, with winners chosen before the quarantine is opened. If none qualifies, recommend no promotion. Low effective sample size/wide intervals are legitimate negative results.
>
> Run required red/green math tests, full suite and DB-hash check; report inherited failures and negative results. Open one research-only PR. End with what changed / what is left. Promotion requires a separate governance PR after the v207 quarantine rule (D2).

## Historical inventory and prioritisation

This inventory covers all157 integer IDs v9–v165, all24 x-series IDs, BL01 and two auxiliary folders. The registry has113 current study folders (86 v,24 x,BL01,2 auxiliary); a release is not automatically a study. Missing/proposed/integration IDs are accounted for without inventing results. HIGH/MEDIUM/LOW is a qualitative conclusion-change likelihood, not a probability or freshly measured result. Historical promotion means attributed integration; current use is qualified where evidence differs.

## Evidence map

- R: research/registry.yaml and research/README.md (113 entries: promoted 3, retained 7, shadow 13, closed 90; 86 numbered v studies, 24 x studies, bl01, pb_vs_pe, test_runtime).
- D: docs/reviews/REPO_REVIEW_2026-09-25.md, original findings F01-F35. These findings explain historical contamination. CHANGELOG v171-v185 describes subsequent repair work; it does not establish that archived study outputs were regenerated.
- L: docs/history/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md; V11 through V29 closeouts, docs/history/results/V13_RESULTS_SUMMARY.md and V19_RESULTS_SUMMARY.md; research/legacy/v*.
- A: docs/history/superpowers/plans/2026-04-10-v37-v60-results-summary.md.
- B: docs/history/superpowers/plans/2026-04-10-v66-v73-calibration-and-decision-layer.md and 2026-04-10-v74-v78-quality-weighted-promotion.md.
- C: docs/history/superpowers/plans/2026-04-11-v87-v96-classification-hybrid-research.md; individual outputs under research/studies/v87_* through v96_*.
- E: docs/history/superpowers/plans/2026-04-11-v102-v117-post-review-enhancement-plan.md and 2026-04-11-v118-v121-prospective-shadow-monitoring.md; individual v110-v121 summary outputs.
- P: research/studies/v125_portfolio_target_classifier/outputs/v125_portfolio_target_summary.md (actually headed v126), v127 summary, v128 feature-search summary, v129 VGT robustness audit, v130 temperature summary, v132 threshold-validation summary, v133-v138 search summaries.
- F: docs/history/closeouts/V142_CLOSEOUT_AND_V143_NEXT.md, V145_CLOSEOUT_AND_V146_NEXT.md, V150_CLOSEOUT_AND_V151_NEXT.md, V152_CLOSEOUT_AND_HANDOFF.md and individual v140-v150 summaries.
- T: docs/history/closeouts/V158_CLOSEOUT_AND_HANDOFF.md, V164_CLOSEOUT_AND_HANDOFF.md, V165_CLOSEOUT_AND_HANDOFF.md; docs/research/backlog.md and CHANGELOG.
- X: docs/research/x_series_resume_2026-04-24.md, research/studies/x24_indicator_contract/outputs/x24_research_memo.md; registry-linked per-x plans/memos.
- BL: docs/history/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md and research/studies/bl01_tau_sweep_eval/bl01_tau_sweep_eval.py.

## Defect families used below

I = price/features: F01 weekly unadjusted prices used as trading-day windows; F15 split/share-basis mistakes and synthetic spreads; F24 TA on weekly raw OHLCV; F27 spliced CB and macro latest-vintage issues.
T = total-return targets: F03 VOO split; F05 VGT split; F08 stale dividends; F22 partial-week duplicates, inconsistent horizons and as-of truncation. The VGT 2026 defect affects full-history/current fits; it should not be asserted to contaminate a study actually restricted to pre-2024 observations.
M = macro timing: F06 double lags/row shifts over duplicate month labels; F07 missing live FRED updates; F27 latest-vintage and aggregation issues. Availability lags do not make revised observations point-in-time.
E = EDGAR inputs: F09 quarterly annual/Q4/ROE errors; F10 book-yield units; F11 PIF leap keys/definition break; F12 sign loss; F15 per-share basis; F16 row-based YoY gaps; F17 omitted months; F18 equity/debt misparse; F23 filing-date alignment; F33 no historical provenance; F34 combined ratio; F35 missing ROE key. Not every field is used by every model: assign exposure by requested family, not a claim of universal direct feature use.
V = validation/reporting: F04 current-label naive mean; F13 fixed-alpha, quality-weight, calibration/conformal selection and pooled inference; F25 LOO RidgeCV in v39-v59 and 48 target rows crossing the old holdout boundary. The latter requires purging labels by end date, not merely filtering origin dates.
S = shadow/policy: F02 CPCV always fails and forbidden K-fold method; F19 tax/MC errors; F20 action mapping not validated/fails open; F21 constant confidence; F24 stale Path B row, monitoring never matures, direction-insensitive veto. Stale current-row scoring is separate from the archived WFO result.
X = x-specific: F25 raw price shift(-h), no dividend DRIP, ME/BME mismatch, December specials missed; F15 raw BVPS split discontinuity; F30 unversioned parquet cache changes x winners.

Additional code findings, separate from original review:

- v143 calls prune_feature_overrides on baseline[feature_df] before rerunning WFO; src/research/v139_utils.py computes its correlation matrix over that whole supplied frame. This is future-feature-dependent selection. Rebuild pruning inside each inner training fold.
- v138 _build_proxy_frame fills initial prequential residual MSE with residual_sq.mean() over all rows and uses realized.var(ddof=0) over all rows for prior variance. Its v149/v150 descendants inherit a future-outcome-dependent proxy. Replace warmup with a declared prior or matured training data only.
- v148 is odds rescaling of preserved Path B probabilities plus temperature calibration, not a classifier refit with class_weight. Its no-benefit result addresses that proxy only.
- v140 historically reported identical metrics for all tested alphas. Current code overrides the fixed-alpha config and passes it to reconstructed predictions. Before a new experiment, verify a synthetic alpha perturbation changes predictions, and distinguish this fixed research parameter from the new live prequential alpha grid. Flat historical metrics are not proof shrinkage is irrelevant.
- src/research/v37_utils.py still has fixed HOLDOUT_START='2024-04-01' and load_relative_series filters only target origin. It also pools flat arrays, losing date/benchmark structure before inference. The new harness must avoid these old loaders/metric shortcuts or explicitly repair them.

## Legacy studies absent from registry

All conclusions below are historical, not certified after September repairs. `promoted` here describes historical integration, not current recommended settings.

| ID | Question and evidence-based old conclusion | Historical promotion | Relevant defects | Rerun conclusion-change likelihood / reason |
|---|---|---|---|---|
| v9 / v9.1 | Feature, target, universe, pooling, weekly-snapshot and classifier comparisons. Smaller balanced_core7, lean Ridge/GBT and simple sign policies looked best; no broad replacement promoted. Five experiment scripts are registered, but broader v9 legacy bakeoffs/policy artifacts remain outside it (L). | Research; promotion deferred | I,T,M,E,V,S | HIGH: several whole-universe and policy comparisons use the broken baseline; reproduce representative lean controls only. |
| v11 | Diversification-first reduced-universe candidate vs simple policy. Ridge+GBT best model; historical_mean + neutral_band_3pct best policy; modest edge did not justify stack promotion (L). | No model promotion | I,T,M,E,V,S | MEDIUM: corrected model edge could move; simple diversification framing is a preference rather than model finding. |
| v12 | Simpler historical-mean shadow vs live monthly behavior. Both averaged 50% sell; shadow signal changes 0 vs live 5; steadier recommendation framing (L). | Supported later v13.1 recommendation promotion | V,S,T | HIGH for comparative action evidence: CPCV forced defaults can make identical action results uninformative. |
| v13 / v13.1 | Package simpler diversification-first baseline and holdings/redeploy guidance. shadow_promoted became default while 4-model output remained cross-check (V13 results). | Recommendation-layer promotion | S,V | LOW for packaging; HIGH if treating its 50% agreement as utility evidence. No standalone numeric rerun. |
| v14 | Reduced-universe prediction replacement. Continue shadowing ensemble_ridge_gbt, retain v13.1; narrow fixed-budget feature work next (L). | Research/shadow candidate only | I,T,M,E,V | MEDIUM: conservative non-promotion may survive, rankings uncertain. |
| v15 | Fixed-budget feature swaps. rate_adequacy_gap_yoy for GBT and BVPS growth for linear models led; strongest GBT candidate still negative mean OOS R2, ensemble not yet tested (L). | Research features, deferred ensemble gate | I,M,E,V (notably book yield, BVPS, PIF) | HIGH: winning feature families directly repaired. |
| v16 | Modified Ridge+GBT vs reduced live and historical mean. Improved vs reduced live, insufficient edge for direct promotion; shadow_for_v17 (L). | Shadow research | I,T,M,E,V,S | HIGH: directly uses repaired v15 features. |
| v17 | Candidate as visible cross-check over recent snapshots. Keep current; steadier candidate disagreed directionally with simple baseline in every reviewed month (L). | No replacement | I,T,M,E,V,S | HIGH: alignment and forecast sign could change. |
| v18 | Narrow benchmark/peer-relative swaps to reduce sign bias. Keep v16 research-only; swaps did not remove bias enough (L). | None | I,T,M,E,V, CB splice | HIGH: raw/synthetic peer-relative input defects directly bear on question. |
| v19 | Complete original 46-feature inventory. 44 tested, peer-CR and FCF yield source-blocked; swaps improved utility but no replacement proof (L). | Several useful features fed later assembly; no direct replacement | I,M,E,V; latest-vintage public backfills | HIGH for feature winners; LOW for two explicit source blocks unless sources change. |
| v20 | Assemble strongest v16-v19 swaps. Best stack better metrics but divergence from simple baseline; continue_research_keep_current_cross_check (L). | None | I,T,M,E,V,S | HIGH: composite ranking depends on repaired inputs/actions. |
| v21 | Full historical point-in-time comparison replaces recent-window gate. Promote ensemble_ridge_gbt_v18 visible cross-check (L). | Cross-check winner adopted v22 | I,T,M,E,V,S | HIGH: promotion-sensitive and PIT claim limited by F33/F27 provenance. |
| v22 | Integrate v21 cross-check, keep v13.1 recommendation layer (L). | Cross-check integration | Same inherited evidence | LOW standalone: integration, no independent model experiment. Re-evaluate winner with new control. |
| v23 | Extend history via research-only pre-inception proxies. extended_history_confirms_candidate v18 (L). | Confirmed cross-check, proxies research-only | T,I,M,E,V; proxy lineage | HIGH: target/synthetic series and longer pooled inference need correction. |
| v24 | VTI vs VOO forecast universe. keep_voo/current_voo_actual (L). | Incumbent universe retained | T specifically missing VOO split, V | HIGH: directly corrupted comparison asset. Include one bounded clean universe test. |
| v25 | Repair earlier alignment/validation and rerun v20-v24. v21 promotion, v23 confirmation and keep_voo survived that repair (L). | Supported v22 | Remaining September I,T,M,E,V,S | MEDIUM as remediation history; not an independent model hypothesis. New repair supersedes it. |
| v26 | Package cross-check, warning cleanup and output validation (L). | Operational integration | S, inherited metrics | LOW: do not rerun as a separate research version. |
| v27 | Separate sell decision from investable redeploy universe. balanced_pref_95_5, modest 0.25 tilt, VOO/VGT/SCHD/VXUS/VWO/BND (L). | Redeploy presentation/heuristic integration | T,S; preference-conditioned utility | LOW for role separation; MEDIUM if deriving tilt/default weights empirically. |
| v28 | Prune forecast universe to investable list? No; retain wider forecast context vs narrower buy list (L). | No forecast pruning | T,V,S | MEDIUM: comparative forecast evidence changes; retain conceptual distinction without numerical rerun. |
| v29 | Clarify forecast/redeploy labels in monthly outputs (L). | Presentation | No independent statistical hypothesis | LOW; no model rerun. |

## Version IDs without study folders: explanation and gaps

The registry's contract is one entry per research/studies folder, not one per release. It expressly excludes research/legacy/v9-v28. Thus absent numeric IDs are not automatically missing experiments or registry integrity failures. There are 71 absent IDs in v9-v165: 10-36, 61-69, 74, 76-86, 97-109, 123, 124, 126, 139, 151-153, 159-161.

- v10/v10.1 *(corrected 2026-09-27, A6)*: operational hardening, not a model study. [`docs/history/results/V10_1_RESULTS_SUMMARY.md`](../history/results/V10_1_RESULTS_SUMMARY.md) records post-v9 baseline reconciliation, workflow hardening, schema discipline, CI and documentation, with the recommendation "Promote with caveats". It explicitly did not address model accuracy. Do not rerun it, and do not interpret v9.1 as integer v91.
- v11-v24 and v27-v28: legacy result folders; captured above. v13/v22 are integration records, v25/v26 repair/packaging and v29 presentation in closeouts/results.
- v30: docs/history/plans/codex-v30-plan.md is operational hardening/retry/logging/freshness, not a single quantitative study.
- v31: conformal coverage/drift/performance-log integration; old health inference affected by F13/F31, now should be reconstructed as causal diagnostics rather than rerunning an operational version.
- v32: feature stability/VIF/policy/heuristic diagnostic wiring; archived metrics need refresh, not an independent candidate search.
- v33: config modularization and mypy; no model rerun.
- v34: BL diagnostic-only shadow wiring and hygiene; no independent empirical winner.
- v35: git history identifies 35.0 Monte Carlo tax, 35.1 retrain trigger, 35.2 utility relocation, 35.3 Streamlit dashboard. MC mathematics affected F19; drift F31. Those are validation/fix questions, not model-family sweeps.
- v36: git subjects contain both feat(v36) property tests and v36.0 promotion of ensemble_ridge_gbt_v18 to primary production. Version-label collision; promotion belongs to historical v18/v21 evidence, property tests were criticized F28. Do not assign one study conclusion to all v36 work.
- v61-v65: no dedicated named study/plan/subject was found in scoped searches. Treat as unresolved numbering gap, not presumed results. Decimal v6.1-v6.5 are different releases.
- v66: documentation framing. v67: ensemble-aligned monthly diagnostics. v68: shared Clark-West. v69: benchmark-quality export. All recorded in B.
- v74: shadow consensus integration; v75 registered holdout replay. v76: live quality-weighted integration. The plan proposed v77/v78 tracking/promotion stages, but execution snapshot describes v76 promotion. Do not invent separate v77/v78 numerical trials.
- v79-v80: post-promotion wiring stabilization in 2026-04-11-v79-v80-post-promotion-stabilization.md; missing artifacts restored and equal-weight cross-check kept temporarily. No separate study.
- v81-v86: docs/workflow/email/dashboard/structured-summary adoption and cross-check retirement (2026-04-11-v81-v88 plan). v87/v88 were re-scoped into classifier target/feature studies, so the earlier plan's version descriptions are not their final experiment questions.
- v97: deep research review prompt. v98-v101: no dedicated named study found in scoped searches; do not extrapolate an outcome.
- v102-v109: archive intake, artifact/surface contracts, dashboard, orchestration, as-of guard, tests, classifier-history/maturity logging. Operational prerequisites for E, not separate empirical studies.
- v123-v124: portfolio-weight aggregation and VGT/VIG investable classifier additions (2026-04-12-v123-v125 alignment plan), shadow integrations.
- v126: methodology repair folded into v125 scripts/artifacts; matched Path A comparator, rolling max_train=60, feasible split counts. Its study output is headed v126; lack of separate folder is explained.
- v139: autoresearch scaffold. v151: reporting-only follow-on lane integration. v152: final closeout. v153: archive/backlog/scaffold for v154-v158. No independent rerun.
- v159: problematic gap. Backlog says complete; historical plan header says completed and references V159 closeout and CHANGELOG entry, but current checkout has no src/reporting/firth_shadow.py, no V159 closeout, and no CHANGELOG v159 section. The plan itself forwards baseline probabilities with Firth result metadata rather than fitting Firth. Do not claim Firth predictions are currently integrated from these records.
- v160-v161: archived reports/preregistered TA direction and src/research/v160_ta_features.py feature factory under v160-v164 plan; registered empirical stages start v162.

Other registry gaps: many closeout=null entries lack a direct conclusion pointer (v122/v129-v131/v133-v138 plus auxiliary analyses); several registry promoted_to paths refer to old src layouts. There is no src/pgr_vds/research_lib at this checkout. Helpers are in src/research. The migration path requested by the parent needs checking against the actual tree before any plan names canonical package files.

## Registered v studies: per-study assessment

Historical claims and figures below come from named summary/closeout artifacts, not current reruns. Status is registry status unless the evidence requires a caveat. `closed` does not mean disproved under corrected data.

| ID | Question | Recorded conclusion / evidence | Promotion | Defects most relevant | Change likelihood and reason |
|---|---|---|---|---|---|
| v37 | Baseline lean Ridge+GBT | R2 -0.2269, IC .1579, hit .7002, sigma .6851 (A) | closed control | I,T,M,E,V | HIGH: baseline magnitude/health affected; must replace rather than use stale CSV delta. |
| v38 | Global post-hoc shrinkage | alpha .50 best; R2 -0.1310 with unchanged IC/hit, sigma .3426 (A) | promoted rule; now prequential alpha grid, fixed .50 research-only | F13 same-OOS selection, F04/F25, I/T/M/E | HIGH: review causal alpha+weights produced much worse reported R2; retain shrinkage hypothesis, reselect causally. |
| v39 | Ridge alpha range | Underperformed v38 (A) | closed | F25 LOO alpha, I/T/M/E/V | HIGH: alpha choice 3.7 vs temporal 110 directly refutes tuning protocol. |
| v40 | Ridge-only / constrained GBT / 80-20 blend | Best constrained GBT -0.2355, worse v38 (A) | closed | F25 LOO, I/T/M/E/V | MEDIUM: simple blend worth one joined control; historical ranking invalid, no evidence of reversal. |
| v41 | Fold-local target winsorization | Underperformed v38 (A) | closed | F25 LOO, T (split-driven outliers), V | HIGH: repairing target outliers changes the exact value proposition of winsorization. |
| v42 | Expanding / decayed / capped windows | Underperformed v38 (A) | closed | F25 LOO, I/T/M/E/V | MEDIUM: window sensitivity changes, but no reason for broad expansion sweep. |
| v43 | Seven-feature subsets | Underperformed v38 (A) | closed | F25 LOO, I/M/E/V | HIGH: corrected feature meanings/timing could reorder small subsets. |
| v44 | Fold-local block PCA | Worse than v38 (A) | closed | F25 LOO, I/M/E/V | LOW: expensive added complexity had no edge; small-N rationale unchanged. |
| v45 | BayesianRidge priors/replacement/+GBT | Materially worse v38 (A) | closed | I/T/M/E/V; F25 covers cycle methods | LOW: no surviving edge; reusing as mandatory replacement not justified. |
| v46 | Per-benchmark outperformance logistic | Accuracy .6533, BA .5292, Brier .2502; promising sidecar (A) | closed, classifier direction fed later research | I/T/M/E/V | MEDIUM: modest BA, corrected labels may change edge; rerun as baseline in newer classifier group. |
| v47 | Composite benchmark formulations | Worse v38 (A) | closed | F25 LOO, T,V | MEDIUM: composite returns affected by VOO and universe definition; cover as matched portfolio target rather than original sweep. |
| v48 | Regression panel pooling/fixed effects | Worse v38 (A) | closed | F25 LOO, I/T/M/E/V; grouped dates | MEDIUM: corrected labels/inference matter; one regularized pooled control only if needed. |
| v49 | Hard-market/vol/curve features | Worse v38 (A) | closed | F25 LOO, M/V | MEDIUM: macro timing changed; low priority because hard splits reduce sample. |
| v50 | Clip predictions to train percentiles | clip+shrink -0.2300 best later-stage result, still worse v37/v38 (A) | closed | F25 LOO, T,V | MEDIUM: one robust calibration control after target repair; no broad grid. |
| v51 | Peer pooling/two-stage sector signal | Strongly negative (A) | closed | F27 CB splice, F15 peer spreads, F25/V | HIGH for interpretation, LOW promotion priority: data lineage undermines peer test but no need to reopen large family. |
| v52 | 1M/3M test windows | Strongly negative R2 despite higher IC in one variant (A) | closed | F25/V, T overlap | MEDIUM: common scoring dates and horizon purge essential; merge into small window check. |
| v53 | ARD replacements | Strongly negative (A) | closed | I/T/M/E/V | LOW: high-complexity replacement did not survive. |
| v54 | Gaussian-process replacements | Best RBF R2 -.2828, worse v38 (A) | closed | I/T/M/E/V | LOW: unfavorable old result, scarce sample; not worth separate rerun. |
| v55 | Rank-transformed targets | Strongly negative (A) | closed | F25 LOO, T,V | MEDIUM: split-corrupted target ranks matter, but no prior edge to justify broad transform search. |
| v56 | 12M horizon Ridge | Strongly negative (A) | closed | F25 holdout crossing, T/V; current outer gap 15 (adequate), historical inner-CV path needs inspection | HIGH: requires horizon-aware validation and label-maturity gate; low priority absent need for 12M decision horizon. |
| v57 | Logs/ranks/lags | Rank-normalized GBT -.2440, still worse v38 (A) | closed | F25 LOO, I/M/E/V | MEDIUM: only shallow ranked GBT control if new ensemble group warrants it. |
| v58 | Domain FRED feature additions | Strongly negative (A) | closed | F06/F07/F27 M, F25/V | HIGH: feature lags/history repaired; a bounded family test is sensible, not repeat broad feature expansion. |
| v59 | 18-feature imputation strategies | Strongly negative (A) | closed | F25 LOO, M/E missingness, V | HIGH for old missingness claim: many NaNs were defects; avoid treating imputation as solution to corrupt ingestion. |
| v60 | CW / CE / MSE decomposition | CW t3.3567 p.0004, CE+.0330, variance share38.4%, bias1.4%; interpreted signal/calibration problem (A) | closed diagnostics; helpers later integrated | F04/F13/F25 overlap/panel inference, I/T/M/E | HIGH: recompute per-period/date-clustered evidence; old significance not promotion proof. |
| v70 | Benchmark-specific prequential alpha | A_prior12 R2 -.1041 vs v38 -.1310, lower hit rate (B) | closed | F13 fixed baseline/full-OOS selection, T/V | MEDIUM: concept already causal, rerun alongside pooled shrinkage, not independently. |
| v71 | Conservative affine recalibration | R2 -.1530 to -.1592, degraded v38 (B) | closed | T/V baseline, I/M/E | LOW: negative conservative result; retain as optional falsification control only. |
| v72 | Quality-weighted consensus | R2 -.0445, NW IC .362, preserved policy uplift (B) | promoted quality mode v76 | F13 weights same-OOS/gated IC, V/T/I/M/E | HIGH: gate inflation confirmed; evaluate lagged weights vs equal-weight with grouped dates. |
| v73 | Regression magnitude + logistic gate | No forecast-accuracy gain; design pattern only (B) | closed | T/V/S | MEDIUM: new causal policy comparison may differ; avoid optimizing forecast metric for action rule. |
| v75 | Quality shadow holdout replay | Apr2024-Mar2026, IC .1265 vs .1359, signal changes6 vs4, 100% mode/sell agreement, top weight18.36% (B) | supported quality promotion v76 | F13 full-OOS quality; F02 forced defaults; F25 boundary | HIGH: old holdout already seen/consumed, identical defaults weak action evidence; do not call it fresh confirmation. |
| v87 | Classifier target taxonomy | actionable_sell_3pct slightly edged simple underperformance (C) | closed research target | T,V/S policy alignment | HIGH: threshold-sensitive labels move under correct DRIP/windows. |
| v88 | Stepwise feature-family expansion | Lean baseline best, broader families worse BA/calibration (C) | closed | I/M/E,V | HIGH: corrupted lean and candidate features change family comparison. |
| v89 | Separate linear classifier family | Balanced logistic best decision baseline (C) | closed, family used by shadow | I/T/M/E,V | MEDIUM: high-bias family prior remains sound; confirm on matched dates. |
| v90 | Pooled vs separate classifiers | Pooled shared logistic better BA; strongest pure-model result (C) | closed, shadow uses separate-calibrated branch | T/V; pooled same-date dependence | MEDIUM: corrected labels/grouped fold geometry may change pooled-vs-separate result. |
| v91 | Shallow nonlinear classifier sweep | Did not improve pooled linear reference (C) | closed | I/T/M/E,V | LOW: no surviving edge; bounded shallow control enough. |
| v92 | Prequential calibration + abstention | Improved ECE; calibrated separate logistic best abstention branch (C) | closed, supported shadow | F13 evaluation vs tuning and F24 maturity, T/V | HIGH: thresholds/calibration selected on limited reused probabilities; use nested selection and matured labels. |
| v93 | Basket/breadth vs panel labels | Benchmark panel best monthly-policy usefulness (C) | closed | T/V,S | MEDIUM: align to real investable portfolio and fresh targets, not broad label proliferation. |
| v94 | Hybrid gate architectures | Classifier-only panel gating best policy, not hybrid (C) | closed | T/V,S | HIGH: reference action mapping/quality selection need clean policy frame. |
| v95 | Promotion-style policy replay | Same classifier-only result; stability vs production insufficient (C) | closed | T/V,S | HIGH: weak candidate edge and action/default defects affect outcome. |
| v96 | Classification program candidacy | continue_research_no_promotion, shadow interpretation layer (C) | shadow classification_shadow.py | Inherits C, F24 live scoring/maturity | MEDIUM: non-promotion prudent; rewrite synthesis once corrected evidence exists. |
| v110 | Gemini veto regression-sell gate | veto .50 mean policy .0681, agreement .9383,17 hold changes (E outputs) | closed; winner used by v113 | F24 wrong-direction veto, T/V/S | HIGH: directly flawed gate interpretation. |
| v111 | Permission-to-deviate overlay | .50 mean policy .0410, agreement .4877,1 hold change, action rate0 (E outputs) | closed | T/V/S | MEDIUM: no action/poor agreement provides little independent support. |
| v112 | Decision-aligned target alternatives | deviate_from_default_50pct_sell mean policy .0665, agreement .9012 (E outputs) | closed | T/V/S | HIGH: alternate labels tied to flawed default/action mapping. |
| v113 | Uplift/agreement/stability constrained selection | veto .50 eligible, uplift only .0005, agreement .9383 (E outputs) | shadow gate overlay | F24 veto, T/V/S | HIGH: tiny edge easily affected; nested selection required. |
| v114 | Selected gate shadow summary | veto .50, source v110, promotion_eligible True (E outputs) | closed summary; actual shadow integration | Inherited v113 | LOW independent; regenerate from clean candidate, no standalone search. |
| v115 | Horizon-aware monitoring summary | matured0, no Brier/logloss/ECE (E outputs) | closed summary | F24 maturity flag never updated | HIGH: monitoring readiness conclusion changes once maturity actually recomputed; no backtest rerun. |
| v116 | Limited gate candidacy | mature_ready=True and limited_ready=True, but keep shadow (E outputs), contradictory v115 matured0 | closed | F24/V/S, readiness-contract discrepancy | HIGH: readiness needs explicit count gate; no evidence actual matured future history. |
| v117 | Classifier selects recommendation mode? | Defer promotion pending longer shadow and matured evidence (output md) | closed | Inherited gate, F24 | LOW for defer conclusion until real 24 matured months; regenerate governance assessment later. |
| v118 | Simulated prospective replay | 162 review months, agreement .9383,10 disagreement months,cumulative+.0812 (E) | closed replay | T/V/S; retrospective selected candidate | HIGH: replay is not prospective evidence, corrected veto/targets can remove gain. |
| v119 | Disagreement-focused scorecard | All gain+.0812 in10 disagreement months,max streak6 (E outputs) | closed | T/V/S, clustered/overlap inference | HIGH: concentrated gain and churn fragile; report dates and CIs. |
| v120 | Promotion assessment | advance_to_real_time_shadow_monitoring; churn fail,matured0 (E outputs) | closed | F24,S/V | MEDIUM: real-time shadow reasonable, readiness numerical inputs unreliable. |
| v121 | Prospective phase synthesis | Same candidate/results, next real-time monitoring (E outputs) | closed | Inherited v118-v120 | LOW separate rerun; regenerate synthesis. |
| v122 | Current classifier coefficient/audit snapshot | Balanced separate logistic, lean12,calibration; pooled BA .5827; calibrated BA .5132/ECE .0813 (audit output) | closed snapshot | I/T/M/E,V; all-history fit vs rolling WFO mismatch | HIGH: coefficients/snapshot based on changed features; audit after clean comparison not stand-alone winner. |
| v125 | Path B portfolio-target logistic vs Path A | v126-remediated matched84 rows: covered BA .645 vs .500, calibration worse; keep secondary (P) | shadow composite classifier | T (VOO), I/M/E,V; F24 stale current-row separate issue | HIGH: modest sample and corrupted composite labels; matched clean Path A/B essential. |
| v127 | Path B calibration sweep | Platt best reliability but BA loss .145 vs raw; no adoption (P) | shadow (program lineage; no v127 winner adopted then) | T/V calibration selection | HIGH: baseline comparison changed in v130 without new data; score candidate policies on one predeclared criterion. |
| v128 | 72-feature benchmark search | 4 switched; pooled covered BA .5016 vs .5000, little total gain; VGT 2-feature .9474 on21 covered looked striking (P) | shadow feature-map CSV | I/M/E,T,V multiple selection; F05 only later/current VGT labels | HIGH: feature map selected after OOS searches, unstable VGT, directly repaired inputs. |
| v129 | Feature-map temporal robustness | VGT advantage only5/9/21 covered across2022/23/24; UNSTABLE, do not adopt, lean VGT (P) | shadow lean override | T/I/M/E,V | MEDIUM: good rejection may survive, but confirm covered sample/selection stability. |
| v130 | Revised temp adoption vs matched Path A | BA .5725 vs .500, Brier .1917 vs .2058; adopt temp shadow by relaxed comparison criterion (P) | shadow temp scaling | T/V (same84 rows reused) | HIGH: candidate criterion moved after results; nested causal calibration decisive. |
| v131 | Asymmetric abstention threshold search | Registry says .15/.70; log also contains kept .10/.55 late iteration; v132 rejects .15/.70 candidate | retained incumbent .30/.70, no adoption | T/V threshold search/reuse | HIGH: threshold records disagree and tiny subset selection fragile. |
| v132 | Temporal validation of thresholds | Selection63/holdout21; winner .10/.60; baseline BA .5 vs candidate .5, winner coverage0; DO NOT ADOPT (P) | retained .30/.70 | T/V; old holdout already inspected | MEDIUM: prudent rejection credible, labels may change; don't reconsume its holdout as fresh. |
| v133 | Ridge alpha upper-bound sweep | 1000 best R2-.4548, IC.1181,hit.6992;10k near tie (P) | closed research candidate | I/T/M/E,V | HIGH: proper temporal tuning and corrected target scale needed; merge with v39 controls. |
| v134 | FRED publication-lag sweep | All1 baseline-.1578; T10YIE0 best-.1573; registry notes lag0 reallylag1 on pre-lagged DB (P/R) | retained lag config, rerun pending | F06 double lag/duplicates, F27 vintage | HIGH: interpretation of tested lag is directly wrong. Test documented releases, never optimize unavailable data. |
| v135 | Temperature ceiling/warmup search | max2.5,warmup42 best covered BA.6987/coverage.5476/Brier.1589;80 grid on same84 rows (P) | closed research settings; downstream shadow candidate lineage | T/V selection on reused probabilities | HIGH: selection/threshold circularity, nested mature calibration required. |
| v136 | Persona backlog ranking | DATA-01 and REG-01 ranked first in stored JSON | closed prioritization | Repaired findings now supersede rationale | LOW: no quantitative rerun; rewrite queue from actual defects/dependencies. |
| v137 | Standalone shallow GBT parameters | depth1,trees25,lr.05,subsample.8 best R2-.2675; did not clear old success bar (P/backlog) | closed/deferred standalone | I/T/M/E,V | MEDIUM: direct GBT features changed; only bounded ensemble control worth revisit. |
| v138 | BL tau/view uncertainty replay proxy | tau.05,scalar.75 selected; coverage.2531,accuracy.8293,uplift.0010,sell_precision0 (P) | closed proxy | Full-sample warmup residual/prior var; T/V/S | HIGH: direct leakage; single-asset proxy cannot prove multi-asset BL policy. |
| v140 | Bounded fixed shrinkage revisit | .35-.65 identical metrics; keep .50 (F) | retained research setting; live now prequential | F13/V; effect-of-parameter check | HIGH: flat result and now-changed implementation require causal prediction-level verification. |
| v141 | Fixed Ridge/GBT blend | .60 best balanced R2-.1624 vs-.1634 baseline,IC.1263,hit.6935 (F) | shadow follow-on lane | I/T/M/E,V selection; stale fixed shrink | HIGH: tiny .001 R2 edge could easily reorder; fit blend using inner folds only. |
| v142 | EDGAR fixed lag0-3 | Keep2, lag1 improves IC but worse R2/hit (F); registry flags same-frame selection | retained old lag; filing-date repair supersedes live | F23/E,V | HIGH: real availability alignment changes question; release-date mapping should not be optimized by score. |
| v143 | Correlation pruning threshold | rho.80 R2-.1569,IC.1411,hit.6944 (F) | shadow follow-on | Full-frame corr pruning plus I/M/E,V | HIGH: selection explicitly sees future features and edge tiny. |
| v144 | Conformal nominal coverage/ACI gamma | .75/.03 empirical .749 vs target.75 (F) | shadow follow-on | F13 same-sample coverage/mature residual handling, T/V | HIGH: match-to-target alone optimizes its own target; causal coverage/width/tail risk required. |
| v145 | Train/test window sweep | Keep60/6;48/6 raises IC.1856 but hit drops.6535 (F) | retained60/6 | I/T/M/E,V; common sample | MEDIUM: competing windows must share dates and horizon-safe gaps; no obvious winner. |
| v146 | Threshold follow-through on temp winner | Keep research .15/.70,BA.6987/coverage.5476 (F) | closed, not live threshold adoption | v135 selection,T/V | HIGH: tuned reused probabilities and rejected v132 pair; merge threshold/calibration inside one experiment. |
| v147 | Coverage-weight Path A/B aggregate proxy | Multiplier1,BA.5000,coverage.4405; no bounded gain (F) | closed | T/V preserved probability proxy | MEDIUM: basic no-gain likely, avoid repeating proxy; real matched predictions instead. |
| v148 | Positive-weight proxy | Odds rescale .75-2,keep1;no BA gain (F/code) | closed | Proxy not actual refit;v135,T/V | MEDIUM: cannot infer actual class_weight ineffective; no need separate proxy rerun. |
| v149 | Kelly fraction/cap on v138 | .50/.25 utility.0021 vs .0010;coverage.4506,success.7671 (F) | shadow follow-on | v138 full-sample variance/warmup,T/V/S tax | HIGH: inherited leakage and one-asset proxy sizing; replace with actual portfolio/lot utility only after signal validation. |
| v150 | Neutral band after Kelly | Keep.015; utility flat.0021,selectivity trades coverage (F) | shadow follow-on incumbent | Same v138/v149 proxy defects | HIGH for policy evidence; LOW priority separate band sweep. |
| v154 | Firth stabilization for thin benchmarks | VMBS+.0412,BND+.0704 BA_covered, sole winner (T) | registry closed; research winner, actual v159 integration unverified | I/T/M/E,V; short-history class imbalance | MEDIUM: plausible low-variance correction, but thin covered samples/labels need confirmation. |
| v155 | WTI3M addition DBC/VDE | DBC+.0051,VDE+.0206 below.04 gate, no_benefit (T) | closed | M/E/I/T/V; source vintage | MEDIUM: recent lag repair might change localized edge; only bounded two-benchmark check after source audit. |
| v156 | USD momentum BND/VXUS/VWO | -.0767/0/+.0087 below.03 gate, no_benefit (T) | closed | M/I/T/V | MEDIUM: macro timing corrected; weak evidence for reopen but cheap representative check. |
| v157 | Term-premium3M diff | Best VDE+.0169 below.02, BND-.0875, no_benefit (T) | closed | M/T/V latest-vintage | MEDIUM: small threshold gap could change; provenance gate precedes optional bounded test. |
| v158 | Classifier/feature synthesis | Firth sole research winner; three feature additions no benefit (T) | closed synthesis | Inherited154-157 | LOW independent: regenerate after survivors, no new search. |
| v162 | Broad pruned TA screen | Replacement-style survivors supported v163/v164, no all-TA expansion (T) | closed screen | F01/F24 raw weekly TA,I/T/V | HIGH: TA meanings/windows/splits wrong; narrower replacement survivors sufficient before broad reopen. |
| v163 | Capped TA survivor confirmation | OBV replaces mom12,NATR replacesvol63,one ratio Bollinger VWO strongest (T) | closed confirmation | F01/F24,I/T/V; survivors selected on old screen | HIGH: directly compromised inputs and selection. |
| v164 | TA synthesis | replacement_candidate, justify later shadow only (T) | closed synthesis | Inherited162-163 | HIGH if repeating recommendation; regenerate after corrected narrow survivor test. |
| v165 | Prediction-level TA replacements | PlusVWO%B BA.6247,Brier.2348,+.0584 BA/-.0656 Brier,8/8 positive; minimal+.0460/-.0517;shadow_monitor (T) | shadow reporting-only | F01/F24,I/T/V; unfinished April anchor | HIGH: exact features broken; unusually universal gain requires honest corrected matched comparison. |

## X1-X24: all research-only; per-study assessment

None is production or shadow promoted at this checkout (R/X). A packaged indicator is not an integrated lane. X-series resume conclusions below are historical; x24 explicitly says it writes no production/monthly/shadow artifacts.

| ID | Question | Recorded conclusion | Exposure | Change likelihood / rationale |
|---|---|---|---|---|
| x1 | Feature inventory, target sufficiency | Separate absolute lane; 22 annual dividend snapshots | X, I/M/E, F30 unpinned cache | HIGH: targets/inventory eligibility and source dates require rebuild; sample size limitation may remain. |
| x2 | Absolute direction baselines1/3/6/12M | Did not clear base-rate gate | X,I/M/E,V | HIGH: corrected split/DRIP labels change classification; rerun one high-bias baseline only. |
| x3 | Direct return baselines | Mostly baseline-heavy; only12M drift clears no-change | X,I/M/E,V | HIGH: raw target definition directly wrong; naive drift must share clean maturity and dates. |
| x4 | BVPS forecast leg | Beat no-change all four horizons; strongest early lane | X specifically split basis, E,V | HIGH: gross2006 BVPS discontinuity could create/reverse edge. |
| x5 | BVPS × P/B decomposition | Stable anchor no_change_pb; structurally useful | X,I/E,V | HIGH: split affects both legs and recombination; retain simple anchor as control. |
| x6 | Annual two-stage special-dividend model | Low-confidence small-sample sidecar | X missed December specials, E,V; annual scarcity | HIGH for occurrence/size labels, LOW for confidence upgrade without more annual variation. |
| x7 | Targeted TA replacements | ta_minimal_plus_vwo_pct_b cleared2/4 horizons, no broad TA justification | X,F01/F24 TA,I/V | HIGH: exact features and labels miscomputed. |
| x8 | Cross-lane synthesis | Shadow readiness not_ready | Inheritsx1-x7 | LOW separate rerun; regenerate after corrected controls. |
| x9 | BVPS bridge features/interactions/baselines | Improved1/3M, not6/12M | X split basis/cache,E,V | HIGH: bridge target/flows/share basis materially change. |
| x10 | Capital-enhanced dividend sidecar | Better x6 EV MAE, still low confidence | X labels, E capital/sign/equity/PIF, V | HIGH: corrected payout years/current BVPS can reorder. |
| x11 | Capital-lane synthesis | continue_research | Inheritsx9/x10 | LOW independent; regenerate. |
| x12 | Raw-vs-dividend-adjusted BVPS audit | Helped3/6M, not1/12M; 2006 split treated capital event | X split basis, E,F30 cache | HIGH: audit premise directly refuted by known split; first x rerun prerequisite. |
| x13 | Adjusted decomposition comparisons | Only6M structural path clearly survived | X,E/I,V | HIGH: winning leg depends on x12 flawed audit. |
| x14 | Indicator synthesis | Narrow one6M structural candidate | Inheritsx12/x13 | LOW independent; regenerate. |
| x15 | Bounded P/B regime overlay | No overlay beats no-change P/B | X split/price ratio,E,V | MEDIUM: corrected ratios may reorder, but no evidence complex overlay now useful. |
| x16 | Structural indicator package | adjusted_structural_bvps_pb_6m research candidate | Inheritsx13-x15 | LOW packaging; HIGH if interpreting packaged name as validated edge. |
| x17 | Persistent-BVPS bridge | Helped3/6M; separate capital creation/payout policy | X split basis/cache,E,V | HIGH: economically useful framing survives, numerical edge may not. |
| x18 | Dividend-policy labels / regime audit | Dec2018 policy break; December-February payout window | X December omission,E date availability | HIGH for rebuilt earlier labels; audit payout definitions from frozen records before fitting. |
| x19 | Post-policy dividend-size model | Betterx10 overlapping years, only3OOS years | X/E,V small-N/multiple choices | HIGH: targets/scales/inputs change; tiny sample cannot support strong promotion. |
| x20 | Policy synthesis | Occurrence one-class overlap; size only identifiable | Inheritsx18/x19 | LOW for one-class limitation, which must be recounted but cannot be cured by ML. |
| x21 | Dividend-size target scales | Excess dividend/current BVPS best | X split basis/December payouts,E,V | HIGH: both numerator and denominator definitions require clean units. |
| x22 | Stronger dividend-size baseline challenge | to_current_bvps survives challengers | X/E,V only3OOS postpolicy years | HIGH: small-sample winner readily changes; use error-by-year and no significance overclaim. |
| x23 | Dividend package | research_size_indicator_candidate; occurrence underidentified | Inheritsx18-x22 | LOW independent; regenerate package with uncertainty after clean size comparison. |
| x24 | Unified structural/dividend contract | Research bundle6M structural + annual size watch; no wiring | Inheritsx16/x23 | LOW packaging: update only after surviving evidence, no direct reporting promotion. |

## BL-01 and auxiliary studies

| Study | Question / recorded conclusion | Promotion/exposure | Change likelihood |
|---|---|---|---|
| bl01 | 5×5 tau/risk-aversion grid ×50 synthetic scenarios; incumbenttau.05/RA2.5 rank_corr.8643, bestalternative+.009 <.05 gate; keep_incumbent (BL) | retained defaults; optional risk_free_rate arg added. Synthetic y_hat=mean_ic*.12 intentionally aligns views with IC; metric is weight-IC rank, not actual policy utility. BL is shadow diagnostic. | LOW for numerical sanity under unchanged synthetic design; HIGH if presented as real empirical optimality after corrected view sign/horizon. Do not rerun same synthetic sweep as portfolio proof. |
| pb_vs_pe | P/B vs P/E predictors; roe_gap apparent edge concentrated2023+, CWp.069, fullperiod pooling concealed within-era PB IC; original inference ~494 comparisons and overlapping horizons (D/F25) | closed; no live use; current summary artifact reproducibility did not make inference valid | HIGH: per-era/nonoverlap/date-aware inference and multiple-testing can change conclusion; optional structural group only, no exhaustive493+ redo. |
| test_runtime | Faster tests without weakened assertions; runtime-summary task | closed tooling; no financial target | LOW: no numerical research rerun. Existing passing tests are not independent proof of old model findings (F28). |


## Explicit per-ID coverage for all 71 versions absent from the registry

Every absent ID below is classified from evidence found; unresolved or proposed IDs have no fabricated study conclusion. For legacy empirical studies, the conclusion/defects/likelihood are in the legacy table above. Other rows identify non-study work or an evidence gap.

| ID | Classification | Evidence and disposition |
|---|---|---|
| v10 | Non-study process/hardening *(corrected 2026-09-27, A6)* | `docs/history/results/V10_1_RESULTS_SUMMARY.md`: v10.1 workflow, schema, CI and documentation hardening; "Promote with caveats"; no model claim. Do not rerun. |
| v11 | Legacy empirical study | research/legacy/v11 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v12 | Legacy empirical study | research/legacy/v12 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v13 | Non-study integration/documentation | V13_RESULTS_SUMMARY: simpler recommendation-layer promotion v13.1; integration, assessed above. |
| v14 | Legacy empirical study | research/legacy/v14 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v15 | Legacy empirical study | research/legacy/v15 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v16 | Legacy empirical study | research/legacy/v16 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v17 | Legacy empirical study | research/legacy/v17 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v18 | Legacy empirical study | research/legacy/v18 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v19 | Legacy empirical study | research/legacy/v19 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v20 | Legacy empirical study | research/legacy/v20 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v21 | Legacy empirical study | research/legacy/v21 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v22 | Non-study integration/documentation | V22 closeout: integrated v21 winner as visible cross-check; no separate experiment. |
| v23 | Legacy empirical study | research/legacy/v23 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v24 | Legacy empirical study | research/legacy/v24 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v25 | Remediation/packaging | V25 closeout/results: earlier alignment/validation repair or packaging, superseded by new corrected baseline. |
| v26 | Remediation/packaging | V26 closeout/results: earlier alignment/validation repair or packaging, superseded by new corrected baseline. |
| v27 | Legacy empirical study | research/legacy/v27 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v28 | Legacy empirical study | research/legacy/v28 plus history closeout/results; assessed in legacy table above; registry deliberately excludes legacy folders. |
| v29 | Non-study integration/documentation | V29 closeout: role/label clarification, no model-stack change. |
| v30 | Operational/diagnostic/refactor sequence | docs/history/plans/codex-v30-plan.md; not a single quantitative candidate study. Relevant inherited diagnostics assessed above. |
| v31 | Operational/diagnostic/refactor sequence | docs/history/plans/codex-v31-plan.md; not a single quantitative candidate study. Relevant inherited diagnostics assessed above. |
| v32 | Operational/diagnostic/refactor sequence | docs/history/plans/codex-v32-plan.md; not a single quantitative candidate study. Relevant inherited diagnostics assessed above. |
| v33 | Operational/diagnostic/refactor sequence | docs/history/plans/codex-v33-plan.md; not a single quantitative candidate study. Relevant inherited diagnostics assessed above. |
| v34 | Operational/diagnostic/refactor sequence | docs/history/plans/codex-v34-plan.md; not a single quantitative candidate study. Relevant inherited diagnostics assessed above. |
| v35 | Operational/tax/dashboard sequence | git subjects v35.0-v35.3: MC tax, retrain trigger, research utility relocation, Streamlit. Mathematical verification/fixes rather than model rerun. |
| v36 | Version-label collision | git subjects distinguish v36 property tests and v36.0 primary-model promotion; no invented unified conclusion. |
| v61 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v62 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v63 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v64 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v65 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v66 | Non-study integration/documentation | v66-v73 plan: documentation and research framing. |
| v67 | Non-study integration/documentation | v66-v73 plan: align monthly diagnostics to ensemble OOS path. |
| v68 | Non-study integration/documentation | v66-v73 plan: Clark-West reusable/production diagnostics. |
| v69 | Non-study integration/documentation | v66-v73 plan: benchmark-quality export; refresh causally from new controls. |
| v74 | Non-study integration/documentation | v74-v78 plan: quality-weighted shadow integration; empirical replay is registered v75. |
| v76 | Non-study integration/documentation | v74-v78 execution snapshot: live quality-weighted integration backed by v72/v75; re-evaluate inherited evidence. |
| v77 | Planned stage / execution gap | v74-v78 plan proposed tracking/promotion stages but records actual promotion at v76; no separate numerical study established. |
| v78 | Planned stage / execution gap | v74-v78 plan proposed tracking/promotion stages but records actual promotion at v76; no separate numerical study established. |
| v79 | Post-promotion stabilization | v79-v80 historical note restores ensemble reporting/artifact wiring; no independent candidate trial. |
| v80 | Post-promotion stabilization | v79-v80 historical note restores ensemble reporting/artifact wiring; no independent candidate trial. |
| v81 | Non-study integration/documentation | v81-v88 plan: docs and workflow contract synchronization. |
| v82 | Non-study integration/documentation | v81-v88 plan: email/report parity refresh. |
| v83 | Non-study integration/documentation | v81-v88 plan: dashboard data-model adoption. |
| v84 | Non-study integration/documentation | v81-v88 plan: dashboard distribution/static snapshot. |
| v85 | Non-study integration/documentation | v81-v88 plan: structured monthly_summary.json contract. |
| v86 | Non-study integration/documentation | v81-v88 plan: visible cross-check retirement, retain diagnostic CSV. |
| v97 | Review prompt | 2026-04-11-v97-deep-research-review-prompt.md; no empirical conclusion. |
| v98 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v99 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v100 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v101 | Unresolved numbering/evidence gap | No dedicated folder or named study record found in scoped file/commit-subject search; no conclusion asserted. |
| v102 | Non-study integration/documentation | v102-v117 plan: archive intake/review provenance. |
| v103 | Non-study integration/documentation | v102-v117 plan: structured summary/classifier artifacts. |
| v104 | Non-study integration/documentation | v102-v117 plan: decision-surface clarity/disagreement display. |
| v105 | Non-study integration/documentation | v102-v117 plan: dashboard/surface architecture. |
| v106 | Non-study integration/documentation | v102-v117 plan: orchestration/shared intermediates refactor. |
| v107 | As-of target remediation | v102-v117 plan: backdated leakage hardening; original F22 found residual leak; later target repair supersedes. |
| v108 | Non-study integration/documentation | v102-v117 plan: decision-layer/artifact contract tests; passing tests not research replication. |
| v109 | Non-study integration/documentation | v102-v117 plan: classifier history/maturity logging; F24 later found never-maturing flag. |
| v123 | Shadow integration | v123-v125 alignment plan: portfolio-weighted investable pool aggregate. |
| v124 | Shadow integration | v123-v125 alignment plan: VGT/VIG classifier additions; no separate folder. |
| v126 | Methodology remediation within v125 | v126 methodology-hardening note; matched Path A and rolling WFO; regenerated v125 output headed v126. |
| v139 | Scaffold | CHANGELOG v139: shared follow-on evaluation scaffold, no independently selected candidate. |
| v151 | Reporting integration | V152/CHANGELOG and shadow_followon source: forward baseline probabilities with candidate metadata; not joint candidate performance evidence. |
| v152 | Closeout | V152_CLOSEOUT_AND_HANDOFF: synthesizes registered v140-v150 results. |
| v153 | Research intake/scaffold | v153-v158 plan: archive/backlog update before Firth/feature experiments. |
| v159 | Claimed integration, unsupported here | Backlog/plan says complete but implementation module, closeout and version changelog section absent; no Firth prediction adoption established. |
| v160 | TA archive/feature factory | v160-v164 plan and src/research/v160_ta_features.py; pure feature construction, empirical stages v162-v165; raw weekly TA defect F24. |
| v161 | TA preregistration/factory stage | v160-v164 grouped plan covers archive, inventory and factory; no separately measured v161 outcome established. |

## Explicit exclusions and stop rules

- Do not rerun packaging/report/plan/dashboard/workflow/refactor releases as experiments (v10/v13/v22/v25/v26/v29–v36; v66–v69/v74/v76–v86; v97/v102–v109/v123/v124/v126/v139/v151–v153/v160/v161). Verify inherited contracts or baseline artifacts. v159 integration is unsupported here; v204 tests Firth without assuming adoption.
- Do not repeat every v39–v59 alpha/PCA/HMM/panel/GP/neural/architecture search. LOO evidence is invalid; small samples favor regularised linear models/shallow trees. v203 uses eight blueprints and small grids. A complex model needs a new hypothesis before more research.
- Do not repeat all v87–v96/v110–v150 thresholds/calibration recipes. Reused84-row probabilities and changed criteria are not replications. v204 tests six bounded alternatives on matched endpoints with fixed bands.
- Do not repeat BL01 synthetic IC/tau grids as empirical portfolio proof, full-history fracdiff tuning, hard regime classifiers or near-one-class annual dividend occurrence tournaments. They lack adequate independent outcomes or repaired-data questions; v205 tests amounts/BVPS directly.
- Defer CB peers until identity repair, optional macro/valuation feeds until availability/freshness/vintage audit, x-series ranking until targets/provenance pass. Do not guess missing values to keep fitting.
- Preserve LOW-priority negative results as history. Revisit HIGH-priority conclusions via representative blocks, not all combinations. If no candidate survives chronology/multiplicity, close with no promotion; do not expand budgets.

Dividend repair, Windows guard fixes, production CPCV removal, old sys.path migration and stale governance docs require separate remediation PRs. Keep them visible in research provenance and promotion review.
