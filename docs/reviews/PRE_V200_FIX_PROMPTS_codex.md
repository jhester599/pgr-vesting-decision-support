# Pre-v200 Remediation Prompts and Implementation Plan

> **Status (2026-09-27): adopted, with amendments.** The owner's decisions and the execution plan in the next section take precedence over any conflicting text below. The original text was committed unchanged in PR #136; amended places are marked. Background: [the comparison of the two step-13V runs](2026-09-26_step13v_comparison.md).

> **For implementing agents:** use `superpowers:executing-plans` to execute one session at a time with its verification checkpoints. If the user chooses delegation, use `superpowers:subagent-driven-development`, with at most two subagents. The current task writes this prompt document; it does not execute the fixes.

**Goal:** remove the verified safety, data and production-validation blockers before establishing the v200 research baseline.

**Architecture:** separate offline test/workflow fixes, provider-backed data repair, and production validation into independently reviewable PRs. Repair data on copies through a reproducible script, then replay the decision under chronological validation and update the current governance baseline. Later shadow and policy fixes use restricted development data once the research holdout is sealed.

**Tech stack:** Python 3.11+ as currently required by packaging, pytest, SQLite, pandas/numpy, scikit-learn TimeSeriesSplit, requests, GitHub Actions. Follow AGENTS.md for types, PEP 8 and quantitative constraints.

Prepared 2026-09-26 from [the independent verification](VERIFICATION_2026-09-26.md) and [the v200 research plan](../research/RERUN_PLAN_v200_codex.md). Read the relevant original sections in [the repository review](REPO_REVIEW_2026-09-25.md) before changing code. Location references below use the restructured repository; confirm them on latest master rather than copying the review's old line numbers.

## Owner decisions and execution plan (2026-09-27)

These decisions and this plan amend the prompts below. Where the original text conflicts with them, they win. The research-plan decisions are in [the canonical research plan](../research/RERUN_PLAN_v200_codex.md#owner-decisions-and-amendments-2026-09-27): D1 makes it canonical, with six additions, and D2 drops the 24-month wait.

- **D3: run all fix sessions now.**
  - R1–R5 all run before v200, in parallel where their files do not overlap.
  - The dividend data is repaired through the existing refresh workflow (step 0) and checked offline in the new R2-lite session.
  - The full R2 prompt below is the fallback.
- **D4: keep the monthly e-mail as it is.** No session changes its content or format, except the gate and CPCV lines that R3's new contract requires. Content and format will be revisited in a later session.
- **D5 (added 2026-09-27, after R3 merged): the refreshed-DB replay and the current baseline move to a new session, R3b.** R3 merged (PR #140) before the step-0 refresh, with its code complete and its replay recorded only as a pre-refresh smoke comparison. R2-lite stays a docs-only data check. R3b runs after R2-lite merges and does only the replay and the governance baseline, so the data change and the baseline change stay separately attributable.

### Step 0: dividend refresh (owner, no session)

- **How:** in GitHub, open Actions → "Weekly Data Accumulation" → Run workflow, and set `dividend_refresh` to `true`.
- **When:** Monday 28 September (any time after 00:00 UTC), or any other day with no scheduled Alpha Vantage use.
  - One run can make up to 23 calls: the daily limit of 25 minus a reserve of 2. That covers all 20 stale tickers.
  - On 27 September the Sunday peer fetch had already used 8 calls, so a run that day would have covered only 15.
  - The Friday price run and the Sunday peer run also use calls.
- **Check:** the run's "Check price, split and dividend integrity" step must pass.
  - If it reports stale tickers, dispatch the run again the next day.
  - If a second run still leaves required tickers stale, use the full R2 prompt.
- **Backstop:** the Wednesday 30 September cron runs the same refresh anyway.

### Sessions, builders and order

| Session | Builder | Reviewer | Start | Merge after | CHANGELOG | Main files |
|---|---|---|---|---|---|---|
| R1, offline safety (with additions) | Codex, on Windows | Claude | now | anytime | v186 | `tests/repo_guard.py`, repo tests, `.github/workflows/ci.yml`, `.github/workflows/peer_bootstrap.yml`, `scripts/initial_fetch.py`, a chart test |
| R2-lite, dividend check | Claude | Codex | after step 0 has committed | step 0 | v187 | one `docs/reviews/` record; no code |
| R3, validation, gates, baseline | Claude | Codex | now (code); replay after R2-lite | R2-lite | v188 | `src/pgr_vds/decision/*`, `src/models/wfo_engine.py`, `src/reporting/decision_rendering.py`, `src/reporting/run_manifest.py`, governance docs |
| R3b, refreshed-DB replay and baseline *(D5)* | Claude | Codex | after R2-lite merges | R2-lite, R3 | v191 | `docs/model-governance.md`, one `docs/reviews/` closeout and rows CSV; no code |
| R4, TA maturity, Firth metadata | Codex | Claude | now | R3 | v189 | `src/pgr_vds/decision/artifacts.py` (shared with R3), `src/reporting/classification_artifacts.py`, `src/models/classification_monitoring.py`, backlog and registry docs |
| R5, vest windows, rebalancer | Codex | Claude | now | anytime | v190 | `src/backtest/backtest_engine.py`, `src/portfolio/rebalancer.py`, tax tests |

**Why this split:**
- R1 needs Windows to reproduce and fix its three failures. Codex sessions run on the owner's Windows machine; Claude sessions run on Linux in the cloud.
- R3 is the largest production change. Claude builds it and Codex reviews it, so the model that wrote the R3 specification checks the implementation against it.
- R4 and R5 go to the Codex builder after R1. They are smaller, and apart from one shared file in R4 they are independent of R3.
- Every PR is reviewed by the other model before it merges, using the prompt below.

**Rules for running in parallel:**
- At most two subagents inside a session, as before.
- Sessions that touch the same file do not merge in parallel. R4 merges after R3, then merges `master` and resolves the overlap in `artifacts.py`.
- Data and baseline changes stay attributable. Only the step-0 refresh changes the DB. The replay and new baseline run on the refreshed DB, after R2-lite merges: in R3b since D5 (R3 merged first and recorded only a pre-refresh smoke replay).
- **CHANGELOG:** each session uses its pre-assigned version and inserts its section in version order. On a conflict, keep both sections.
- **Branches:** `codex/R<n>-<slug>` for Codex sessions; Claude sessions use the branch the session names.
- v200 starts after all five PRs, R2-lite and R3b *(D5)* have merged and the required dividends are fresh.

### Cross-model review prompt

Paste this into the reviewing model's session, with N replaced by the PR number:

> Review PR #N in `jhester599/pgr-vesting-decision-support` against its session prompt: the session the PR names (one of R1–R5, R2-lite or R3b) in `docs/reviews/PRE_V200_FIX_PROMPTS_codex.md`, including the 2026-09-27 amendments.
>
> Work read-only:
> - use scratch clones and DB copies outside the repository;
> - make no fetcher runs, provider calls or e-mail;
> - do not push to the PR branch.
>
> Check the following, and report each with evidence:
> 1. Every checklist item and the exit gate are met.
> 2. Each claimed red/green counterfactual reproduces in a scratch clone: the named test fails with the fix reverted and passes with it restored.
> 3. The full suite (`python -m pytest -o addopts="--tb=short" -q`) passes. Quote pytest's summary line and exit code, and note your platform.
> 4. The committed DB's sha256 is unchanged by the tests.
> 5. The PR introduces no forbidden method (K-fold, CPCV, leave-one-out, shuffling, full-sample scaling), no provider call and no change outside the session's scope, and it uses the pre-assigned CHANGELOG version.
>
> Post one review. List blocking findings first, each with file:line, what fails and how to reproduce it; mark optional suggestions as optional. Approve only if nothing blocks.

## Recommendation: multiple sessions, with related small fixes combined

*(Superseded 2026-09-27 by D3 and the execution plan above: all five sessions run now, in parallel lanes, before v200.)* **Use three required sessions before v200, in order R1 → R2 → R3.** Merge each PR before starting the next. Do not combine all three into one session: they have different inputs, failure modes and verification. R2 can need a continuation if the provider quota or access prevents completion; resume its existing work rather than committing partial data as a clean baseline.

| Session | Combine in this session | Separate boundary / reason | When |
|---|---|---|---|
| R1 — Offline safety and workflow checks | Windows DB guard, path-dependent tests, a targeted Windows CI check, peer-bootstrap schema query | No provider calls or data changes; establishes trustworthy verification before repair | Before R2 and v200 |
| R2 — Dividend history and target rebuild | Cached dividend backfill, repair script, both-horizon target rebuild, row diffs, recurring refresh regression checks | Data source, quota and provenance need their own review; no validation/policy changes | Before R3 and clean v200 |
| R3 — Chronological production validation and governance | Retire CPCV, genuine WFO completion, fail-closed required-data checks, decision replay and current baseline documentation | Behavior and gate changes should be assessed on the repaired DB, separate from data changes | Before v200 |
| R4 — Shadow outcomes and truthful integration metadata | TA ledger maturation, candidate-versus-baseline provenance, Firth adoption documentation | Separate shadow-only surface; can wait while v200–v203 run | Before v204 |
| R5 — Vest-event timing and remaining tax boundary | Quantify suspected event/target offset; fix if confirmed; anniversary-based rebalancer warning | Event/lot semantics affect policy utility rather than the regression baseline | Before v206 |

R4 and R5 are included so the remaining dependent work has paste-ready prompts. They are not prerequisites for starting v200 if R1–R3 pass and their affected paths are excluded from baseline evidence. If R4/R5 would change v200's data, target definitions or model forecasts, stop and regenerate the baseline lock before downstream comparisons.

Broad removal of old sys.path exceptions, strict-lint adoption, dependency cleanup and abandoned utilities can follow separately. The new v200 harness must exclude uncorrected research helpers. Exclude CB entity-splice history until a separate identity repair passes; exclude stale optional macro/valuation feeds until a separately authorized refresh/vintage audit. No historical holdout becomes untouched because these fixes land.

## Starting point and readiness gate

The verified code snapshot is `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d`; DB SHA256 is `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`. These are historical evidence pins, not instructions to overwrite a later repaired DB. Fetch latest master, record its full commit and actual DB hash, and inspect differences from those pins.

The verification/research documents were added in [PR #134](https://github.com/jhester599/pgr-vesting-decision-support/pull/134). Prefer starting R1 after that documentation PR merges. If it has not merged, read its documents as context while keeping the implementation branch based on latest master; do not copy unmerged production changes.

The source full-suite result at verification was:

```text
3 failed, 2491 passed, 1 skipped, 109 warnings in 305.29s (0:05:05)
```

R1 must reproduce and repair those Windows failures. Later sessions must preserve a passing suite. An unavailable Windows runner or provider credential is a stated verification/data blocker, not a passing check.

Before v200, require:

- [ ] R1–R5, R2-lite and R3b merged *(amended 2026-09-27, D3 and D5)*, and the full suite passing, with pytest's own summary and exit code recorded.
- [ ] Required dividend histories pass freshness checks; both 6M and 12M returns rebuilt from raw prices, splits and fractional DRIP.
- [ ] No production CPCV/K-fold execution, including fallback paths and active tests invoking it as validation.
- [ ] Required WFO results, finite live inputs and required feed freshness fail closed when absent.
- [ ] Current governance baseline identifies the repaired DB SHA256, code commit, runtime and metric definitions; old snapshots remain labeled historical.
- [ ] v200 can copy and hash the repaired input and create its clean lock. This document does not itself create that lock or certify a model edge.

## Shared preamble — paste before any one session prompt

Copy this shared preamble, then the entire selected R1–R5 section, including its file list, checkboxes, commands and exit gate. Each section is one session's complete scope.

> Work on one remediation session only. Read AGENTS.md, this document, the independent verification and the cited original findings. Use latest master and one PR for the session; use a `codex/` branch unless the user specifies another name. Preserve unrelated work. Use at most two subagents. Do not infer that a historical CHANGELOG claim is verified.
>
> Python functions need type hints and PEP 8. No K-fold, CPCV, leave-one-out validation, shuffling or full-sample scaling in active validation. Use chronological TimeSeriesSplit with explicit purge/embargo and maturity/availability checks. Research keeps the v200 contract's horizon-sized purge and embargo; do not silently retune production gaps or feature/model parameters while fixing unrelated code. Use unadjusted prices and manual split/fractional-share DRIP total returns. No yfinance fundamentals or historical ratios.
>
> Never run fetchers or the monthly decision, even dry runs, against the tracked DB. Run all DB work on copies outside the repository. Original DB inspection is immutable read-only: `sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)`. Commit DB changes only as the documented output of an explicit migration/rebuild script included in the PR. No real email. Provider calls are forbidden unless the selected prompt below expressly authorizes them; only R2 does. If another session is separately authorized to use EDGAR, send `User-Agent: Jeff Hester jeffrey.r.hester@gmail.com`, remain below 10 requests/second and cache responses.
>
> Every fix needs a named regression test, with its original defect observed red and its repair observed green. Expected mathematical values must be hand-computed or independently derived, not copied from the function under test. Save commands, pytest summaries and exit codes in `docs/reviews/<session>_closeout.md`. Run `python -m pytest -o addopts="--tb=short" -q`; report its exact summary and exit. Compare tracked DB SHA256 before/after tests. Use cached/synthetic inputs for tests, with no live API calls or email. Perform a counterfactual reversal in a scratch clone where the fix is to a fixture, workflow or safety guard, so the revised test cannot pass against the original defect.
>
> Update CHANGELOG with the version pre-assigned in the execution plan: R1 v186, R2-lite v187, R3 v188, R4 v189, R5 v190, R3b v191 *(amended 2026-09-27; R3b added by D5)*. Never use v200 or later. Do not change the monthly e-mail's content or format beyond what the session's prompt requires (D4). Update active docs when production behavior changes. Do not edit the historical verification report to erase its observations. Open one PR, attach it to the chat, and end with what changed / what is left. Passing tests do not establish an investment-performance improvement.
>
> After v200 seals its quarantine, remediation tests/replays use synthetic fixtures or the pinned development partition only. Do not reopen recent holdout outcomes to debug a shadow/policy change. Pre-v200 replays of already-inspected historical decisions are smoke verification, must be recorded as such, and cannot become new promotion evidence.

## R1 — Windows safety guard and offline workflow repair

**Session recommendation:** combine these small offline fixes in one session and one PR. Run before R2. No provider calls, fetchers, email or tracked-data changes.

**Paste this prompt after the shared preamble:**

> Execute R1 only. Repair the verified Windows test DB-access escape and two path-dependent failures, then fix the peer-bootstrap summary's invalid column. Relevant evidence: verification V01/V02/V09; original F26/F28. Keep existing protection against swallowed write errors, unmarked DB reads and read-write artifact opens.
>
> Start by reproducing these exact failing nodes on Windows:
>
> - `tests/integration/repo/test_test_suite_hygiene.py::test_classify_access[sqlite3.connect-args11-False-True]`
> - `tests/integration/repo/test_test_suite_hygiene.py::test_guard_fails_exactly_the_probes_that_touch_the_repo`
> - `tests/integration/repo/test_restructure_phase1.py::test_email_reads_report_and_charts_from_artifacts`
>
> Work in an external scratch clone with its own DB copy while intentionally probing the old guard. Record red output before changing implementation or assertions.

**Current files and responsibilities:**

- `tests/repo_guard.py`: `_as_path`, `_is_committed_db`, `classify_access`; normalize SQLite file URIs and native paths consistently.
- `tests/integration/repo/test_test_suite_hygiene.py`: native/URI classification and nested probe outcomes.
- `tests/guard_probe.py`: end-to-end blocked/allowed access assertions.
- `tests/integration/repo/test_restructure_phase1.py`: email reader's missing-file path assertion; never send email.
- `.github/workflows/ci.yml`: targeted Windows regression job; preserve the existing Linux full suite and artifact separation.
- `.github/workflows/peer_bootstrap.yml`: execute the actual summary query against `daily_prices.date`.
- `tests/integration/repo/test_workflow_contracts.py`: executable schema-backed workflow regression.

- [ ] Add URI cases for `Path.as_uri()`, percent-encoded spaces, `file:...?...mode=ro&immutable=1`, native case variations on Windows and relative paths. For an unmarked test, every committed-DB spelling is refused; artifact-marked read-only is allowed, read-write is refused. Memory DB and genuine external paths stay allowed. Preserve URI authority/drive semantics; do not fix Windows by making POSIX paths incorrect.
- [ ] Normalize the nested pytest node-ID path separator before comparison, or use structured JUnit outcomes from the child. A passed call followed by a guard teardown error still counts as failed. Keep the entire `_EXPECTED_PROBES` comparison, sidecar checks and swallowed-error detection.
- [ ] Make the email missing-path test compare a normalized path extracted from the exception. Do not replace it with a generic `raises(FileNotFoundError)` assertion: it must catch a reader pointed at the old `results/` directory.
- [ ] Add `test_peer_bootstrap_summary_uses_existing_date_column` to the workflow-contract file. Extract/execute the workflow's actual inline summary code against a temporary DB with the real schema and known dates. Old code must fail with `no such column: price_date`; repaired code must print the known first date. Do not trigger the GitHub workflow or provider bootstrap.
- [ ] Add a Windows CI job running these regression files on Python 3.12. Retain Linux checks. If Linux execution is unavailable locally, use existing CI to obtain that result; do not claim it was run locally.
- [ ] Run the targeted files, then the exact full-suite command. Require exit 0 on Windows and preserve Linux passing status. Do not fix this by skipping Windows tests or weakening the audit hook.
- [ ] In a scratch clone, restore the old URI normalization, old slash-only parser/assertion and old SQL separately. Show their revised tests fail, restore the fixes, and show green.
- [ ] Add `docs/reviews/R1_safety_closeout.md` with results, DB hashes, platform/runtime and remaining collection/C-level guard limits. Update CHANGELOG; open one PR. Do not call the guard a complete filesystem sandbox.

**Additions (approved 2026-09-27):**

- [ ] Add a test for step 4b's market-cap consistency guard (verification N6). It should feed `verify_monthly_frame` in `src/reporting/capital_return_data.py` a frame whose market cap is not price × shares, and expect the `ValueError`. Show that it fails with the `raise` removed and passes with it restored.
- [ ] `scripts/initial_fetch.py` ignores `--force` (`del force`), although `initial_fetch_prices.yml` passes it. Either honour the flag or remove it from the script and the workflow input, and add a test either way.
- [ ] Add a top-level `permissions: contents: read` block to `.github/workflows/ci.yml`. CI only reads, so it needs no more.

**Focused commands:**

```powershell
python -m pytest -o addopts="--tb=short" -q tests/integration/repo/test_test_suite_hygiene.py tests/integration/repo/test_restructure_phase1.py tests/integration/repo/test_workflow_contracts.py
python -m pytest -o addopts="--tb=short" -q
python scripts/checks/check_doc_links.py
Get-FileHash -Algorithm SHA256 -LiteralPath data/pgr_financials.db
```

**Exit gate:** the three observed failures are repaired without skips; DB/sidecars unchanged; schema query executes; exact pytest summary/exit and old-code counterfactual failures recorded. R2 starts only after merge.

## R2 — Cached dividend repair and both-horizon rebuild

*(Amended 2026-09-27, D3: the dividend data is repaired by the step-0 refresh and checked by R2-lite. Use this full R2 prompt only if the refresh fails or leaves required tickers stale.)*

**Session recommendation:** one separate data-repair session/PR. Keep provider access, reproducible repair, target rebuild and continuity checks together. Provider quota may require a continuation; no partially refreshed DB may be called a clean v200 input.

**Paste this prompt after the shared preamble:**

> Execute R2 only after R1 merges. Relevant evidence: verification V03; original F03/F05/F08/F22 and the feed portion of F26. This prompt explicitly authorizes Alpha Vantage **DIVIDENDS** REST requests only, for missing/stale histories in the current PGR, ETF and existing peer universe. Use an already configured credential, obey the configured quota/reserve/pacing, cache responses and count every request/retry. No FRED, EDGAR, price, fundamentals or other provider calls. This authorization applies when this prompt is run as a new session; creating this document did not authorize calls today.
>
> The verified stale list is VTI, VOO, VWO, VIG, VGT, VHT, VFH, VIS, VDE, VPU, VNQ, KIE, VXUS, VEA, SCHD, BND, BNDX, VCIT, VMBS and ALL. Recompute the due list on latest master's immutable snapshot; these names are evidence rather than permission to ignore newly stale required tickers. Explicitly distinguish known non-payers such as GLD from missing history. Do not alter PGR forecasts, validation, tax mapping or model parameters.

**Current and new files:**

- `src/ingestion/multi_dividend_loader.py`: existing parsing, retry/pacing and request accounting.
- `src/database/db_client.py`: freshness and dividend upsert; preserve idempotent keys and unrelated tables.
- `scripts/weekly_fetch.py`, `.github/workflows/weekly_data_fetch.yml`, `.github/workflows/peer_data_fetch.yml`: recurring dividend refresh and observable stale/deferred outcomes; no real workflow dispatch.
- `scripts/rebuild_relative_returns.py`: existing offline split/bar/6M/12M rebuild and diff tooling.
- Create `scripts/repair_dividend_histories.py`: explicit copy-only cached repair/rebuild entry point; invoke with `python -m scripts.repair_dividend_histories` so no new sys.path edit is needed.
- Extend `tests/unit/scripts/test_weekly_fetch_dividend_feed.py`; create `tests/unit/scripts/test_repair_dividend_histories.py` for script behavior and independent math.
- Keep `tests/integration/data/test_db_price_integrity.py` as the stored-data consistency check after repair.

- [ ] Write the missing-dividend fixture before implementation: a 6M window starts at $100, includes a 4-for-1 split, ends at $25, and reinvests a post-split $0.25 dividend at $25. Independent share accounting gives 4.04 shares, $101 value, **+1%** return. A flat non-dividend PGR comparator gives **−1%** PGR-relative return. The stale-history version gives 0%; show the regression distinguishing them.
- [ ] Name new tests `test_cached_dividend_repair_restores_split_crossing_return`, `test_repair_is_idempotent`, `test_repair_refuses_tracked_db_paths`, `test_failed_provider_payload_is_not_marked_fresh`, `test_scheduled_future_dividend_is_not_a_realised_return` and `test_budget_counts_advisories_and_retries`. Use synthetic/cached JSON; never call the provider from pytest.
- [ ] Require explicit `--input-db`, `--output-db`, `--cache-dir`, `--as-of`, `--report`, `--rows-csv`, `--manifest` arguments. Input/output must be different resolved paths outside the tracked DB location; refuse aliases to it. Default execution is cache-only. A separate explicit `--fetch-av` mode may fill the cache under this prompt's narrow authorization; it operates only on the output copy.
- [ ] Cache unmodified response payloads with ticker, request time, content hash and sanitized request metadata. Cache no credentials or key-bearing URLs. Do not print the key or exception URLs containing it. Persist an external quota journal so making a new DB copy cannot reset the apparent account request count; include known account usage and configured reserves. Stop on hard quota/advisories as defined by the loader; resume without repeated successful downloads.
- [ ] Validate payload status, ex-date, per-share amount and duplicate keys before applying. Preserve scheduled future entries as such, excluding them from realised target windows. Do not infer “no dividend” from a failed/empty/advisory response. Apply successful cached histories idempotently to the output copy.
- [ ] Rebuild both 6M and 12M targets from the repaired output copy using raw prices, canonical splits and fractional DRIP, with actual BME window-end maturity. Keep before/after tables and publish per-ticker/per-horizon row changes, largest differences, sign flips and input/output hashes.
- [ ] Compare the entire target key set plus `pgr_return`, `benchmark_return` and `relative_return`, not only relative differences. Independently recompute representative windows containing newly restored dividends, VOO October 2013 and VGT April 2026 splits. Verify income/ratio/equity identities, monthly coverage and FRED duplication are unchanged.
- [ ] Replay the cached repair from the same input into a second output DB with network blocked. Logical tables and targets must match; a second application must make no economic-row changes. Exclude timestamps/request journal fields when comparing logical idempotence and state that exclusion. Finalize SQLite cleanly with no WAL/SHM sidecars.
- [ ] Audit recurring refresh using mocked provider responses and limited budgets. Due tickers must eventually be selected, partial success must remain visible, retries must be counted, and targets must be rebuilt after dividend changes. Fix only demonstrated continuity bugs; do not run the broad weekly pipeline, which also fetches EDGAR/FRED/prices.
- [ ] Commit the repaired tracked DB only as the explicitly documented output of this included repair/rebuild script, after successful preflight and reviewable diffs. Include safe cached inputs or a durable content-addressed reference sufficient for an offline replay; never commit credentials. Record any provider redistribution constraint and missing reproducibility input instead of hiding it.
- [ ] Add `docs/reviews/R2_dividend_repair_closeout.md`, safe provenance/row-diff artifacts and CHANGELOG. Required histories must pass freshness at the pinned as-of date; optional unavailable histories must be excluded/documented. If credentials/quota/data are missing, publish the incomplete status and stop clean-baseline certification rather than manufacturing a passing refresh.

**Focused commands, all using copied data:**

```powershell
python -m scripts.repair_dividend_histories --help
python -m pytest -o addopts="--tb=short" -q tests/unit/scripts/test_weekly_fetch_dividend_feed.py tests/unit/scripts/test_repair_dividend_histories.py tests/integration/data/test_db_price_integrity.py
python -m pytest -o addopts="--tb=short" -q
Get-FileHash -Algorithm SHA256 -LiteralPath data/pgr_financials.db
```

The closeout must include the actual repair invocation with resolved copy paths and pins. Tests leave the tracked DB unchanged; the later intentional script-produced DB replacement has its own before/after hashes and is identified separately.

**Exit gate:** successful offline replay, required dividend freshness, independently correct targets, complete row/provenance diffs and full-suite exit 0. R3 starts only after merge. No claim that a repaired target automatically improves investment skill.

## R2-lite — verify the dividend refresh (offline)

*(Added 2026-09-27, D3.)* **Session recommendation:** one small docs-only session, run after the step-0 refresh has committed to `master`. No provider calls, fetchers or e-mail.

**Paste this prompt after the shared preamble:**

> Execute R2-lite only. The owner ran the existing dividend refresh (`scripts/weekly_fetch.py --dividend-refresh`, through the "Weekly Data Accumulation" workflow) instead of the full R2 repair script. Verify and record its effect. Relevant evidence: verification V03 / N1; original F03/F05/F08/F22.
>
> 1. **Commits and hashes.** Identify the refresh commit or commits on `master` ("chore: weekly data update …" after 2026-09-27), the DB before (the first parent) and the DB after. Record both sha256 values and the result of the run's integrity step.
> 2. **Diffs.** On immutable copies outside the repository, diff:
>    - `daily_dividends`: new rows per ticker, ex-date range, amounts, and any removed or changed rows;
>    - `monthly_relative_returns` for both horizons: rows changed per benchmark, the largest absolute change in `benchmark_return` and `relative_return`, and sign flips of `relative_return`.
> 3. **Freshness.** Run `db_client.check_dividend_freshness`. No required ticker (PGR and the 8 primary benchmarks) may be STALE. List any other STALE ticker, and treat GLD as a known non-payer.
> 4. **Hand check.** Recompute by hand at least three stored targets whose windows contain a newly added dividend, from raw prices, canonical splits and fractional DRIP, as in Appendix C of `VERIFICATION_2026-09-26.md`. They must match the stored values to 1e-12.
> 5. **Nothing else moved.** Confirm there are no unexplained price jumps and no duplicate FRED months, the EDGAR identities are unchanged, and every changed target row is explained by a new dividend.
> 6. **Record.** Write `docs/reviews/<date>_R2_dividend_refresh_check.md` with the commits, hashes, diff tables and checks. Add CHANGELOG v187 ("dividend backfill verified; no code change"). Open one docs-only PR.

**Exit gate:**
- the required dividends are fresh;
- the hand-computed targets match;
- every changed row is explained.

If a required ticker is still stale, stop and report it: the owner re-dispatches the refresh, or the full R2 prompt is used.

## R3b — refreshed-DB replay and current baseline

*(Added 2026-09-27, D5.)* **Session recommendation:** one small session after R2-lite merges, with no code change. Builder: Claude. Reviewer: Codex. CHANGELOG v191. It finishes the two R3 items that needed the refreshed DB: the February–September 2026 replay and the pinned current baseline. If R2-lite's exit gate did not hold (a required dividend still stale), R3b waits.

**Paste this prompt after the shared preamble:**

> Execute R3b only, after R2-lite has merged. Relevant evidence: verification V08; [`R3_validation_closeout.md`](R3_validation_closeout.md) ("What is left", item 1) and its pre-refresh replay rows [`R3_smoke_replay_rows.csv`](R3_smoke_replay_rows.csv); the R2-lite record in `docs/reviews/`. No provider calls, fetchers or e-mail. Change no code, feature, threshold, parameter, target, consensus weighting or gate. If the replay seems to need such a change, stop and report it instead.
>
> 1. **Pins.**
>    - The code commit: latest `master`, which must contain R3 (v188) and R2-lite (v187).
>    - The DB: R2-lite's "after" DB. If a later weekly update has changed `data/pgr_financials.db` on `master`, extract R2-lite's version with `git show <R2-lite after commit>:data/pgr_financials.db` to a copy outside the repository, and use that.
>    - Record both sha256 values and the runtime (Python and package versions).
> 2. **Replay.**
>    - Use an external scratch checkout at the pinned commit, with its own copy of the pinned DB.
>    - Run it with `PYTHONPATH=<clone>:<clone>/src`, because the editable install points at the source checkout, and with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (three separate assignments), because BLAS thread oversubscription made R3's replay hang.
>    - Command: `python scripts/replay_monthly_decisions.py --committed-dates --out <CSV outside the repository>`, with cwd set to the clone.
>    - Record the actual paths. Record the sha256 before and after of the clone's DB, the source DB and the source ledgers. Confirm `git status` is clean in the clone afterwards.
>    - Run the latest as-of date a second time and confirm the `monthly_summary.json` metrics are identical.
> 3. **Attribution against R3's pre-refresh replay.**
>    - Compare row by row with the `R3 1b94bf0` rows of `R3_smoke_replay_rows.csv`: same gates and model code, DB `f453ab98…` before the refresh. Every difference is therefore a data change.
>    - For each month, explain every change in mode, sell percentage, gate status and metric (OOS R², equal-weight and pooled IC, hit rate, base rate, Pesaran–Timmermann p, ECE, coverage) by the dividends that R2-lite shows were added. Name the benchmarks and the target windows involved.
>    - `data_ready` should now pass where the dividends are fresh. A month that still fails must name the feed and say why, for example a dividend genuinely due by that as-of date that is still missing.
>    - Also compare with the verification's replay tables.
>    - State that back-dated readiness (`readiness_basis = backdated_reconstruction`) reflects repaired values, not what the original decisions had available.
> 4. **Governance baseline.** In `docs/model-governance.md`:
>    - Replace "Current state after R3 (pre-refresh, not the new baseline)" with a pinned **current baseline**: the DB sha256, code commit, runtime, `MODEL_HEALTH_METRICS_VERSION` and `DECISION_GATE_CONTRACT_VERSION`.
>    - For the latest as-of date, include: OOS R² against the prevailing mean of realised targets; equal-weight IC; pooled IC with its Driscoll–Kraay p; hit rate against the constant-rule base rate with the Pesaran–Timmermann p; the readiness gates; prequential ECE; trailing conformal coverage against 80 %.
>    - State the current recommendation, and the calibration and coverage limitations.
>    - Keep the pre-refresh R3 table and the step-5 table as labelled, dated history.
>    - Update the Consequences of [ADR 0008](../decisions/0008-chronological-validation-and-readiness-gates.md) with a pointer to the baseline. Do not edit committed monthly artifacts or the performance log.
> 5. **Record.**
>    - Write `docs/reviews/R3b_baseline_closeout.md` with the commands, paths, pins, hashes, the before/after decision table and the attribution.
>    - Commit the replay rows as `docs/reviews/R3b_refreshed_replay_rows.csv`, and add CHANGELOG v191.
>    - Run the full suite with the tracked DB sha256 recorded before and after, plus `check_doc_links.py`. Open one PR.
>
> R3b changes no code, so the preamble's red/green regression rule has no fix to apply to; the replay determinism check (step 2) and the attribution (step 3) are its verification. This replay is a pre-v200 smoke comparison of already-inspected history, not promotion evidence, and it promises no better return.

**Focused commands:**

```text
python -m pytest -o addopts="--tb=short" -q
python scripts/checks/check_doc_links.py
```

Run the replay only with cwd set to the external scratch checkout and an output CSV outside the source repository.

**Exit gate:**
- the replay completed (exit 0) on the pinned refreshed DB, with every DB and ledger hash unchanged;
- every difference from R3's pre-refresh replay is attributed to the data change;
- the current governance baseline is pinned (DB, code, runtime, metric and gate versions), with the older baselines kept as dated history;
- full-suite exit 0.

v200 starts only after R3b merges; its clean lock copies and hashes the same refreshed DB.

## R3 — Pure chronological validation, readiness gates and current baseline

*(Amended 2026-09-27, D5: R3 merged in PR #140 with the code and gate items done and a pre-refresh smoke replay recorded in [`R3_validation_closeout.md`](R3_validation_closeout.md). Its refreshed-DB replay and new current baseline are done by [R3b](#r3b--refreshed-db-replay-and-current-baseline) instead.)*

**Session recommendation:** one production-behavior PR after R2. Combine CPCV retirement, its gate replacement and the associated report/manifest/governance changes; splitting them would leave callers and policy inconsistent. Do not combine this with provider-backed repair.

**Paste this prompt after the shared preamble:**

> Execute R3. *(Amended 2026-09-27, D3: the code work may start in parallel with R1, R4 and R5. The replay and the new baseline must use the refreshed DB, after R2-lite merges.)* Relevant original findings F02/F04/F07/F13/F20/F21/F26, verification V08 and the observed warning-only live-input path. No provider calls, fetchers or email. Use the repaired DB copy and keep model features, thresholds, parameters, targets and consensus weighting fixed.
>
> Remove combinatorial K-fold validation from active execution. A retired CPCV diagnostic must not prevent a valid WFO result from being used, and removing it must not make missing WFO results or stale/missing required data pass. Use the existing honest chronological metrics. Add an explicit readiness contract rather than fabricating a constant PASS to replace CPCV.

**Current files and responsibilities:**

- `src/pgr_vds/decision/signal_generation.py`: remove the representative `run_cpcv` call/import; derive actual WFO completion and live-input status.
- `src/models/wfo_engine.py`: retire the active combinatorial implementation; retain historical result deserialization only if required. A compatibility `run_cpcv` entry point may raise a clear unsupported-method error, never execute K-fold.
- `src/reporting/decision_rendering.py`: replace `cpcv_completed` with genuine `wfo_completed` and `data_ready` gates; keep existing R²/IC/directional thresholds and sell mapping unchanged.
- `src/pgr_vds/decision/health.py`, `pipeline.py`, `rendering.py`, `artifacts.py`: carry and report the new contract consistently, removing live CPCV-dependent text and completeness warnings.
- `src/database/db_client.py`: use existing per-feed and dividend freshness; add only required readiness aggregation.
- `src/reporting/run_manifest.py` and `scripts/replay_monthly_decisions.py`: propagate gate/policy version and readiness; old payloads remain clearly historical, not synthetic current validation.
- `tests/integration/pipeline/test_validation_gating.py`, `test_monthly_pipeline_e2e.py`, `test_dry_run_read_only.py`: replace obsolete CPCV assertions with chronological contracts.
- Create `tests/integration/pipeline/test_production_validation_contract.py`: end-to-end no-CPCV and fail-closed readiness checks.
- `docs/model-governance.md`, `docs/decisions/0006-validation-gates-and-cpcv-diagnostic.md`, active runbook/report docs and CHANGELOG: document resulting behavior and new current baseline while preserving the labeled step-5 historical baseline.

- [ ] Add `test_live_decision_does_not_invoke_cpcv`: spy on the old call/constructor while running the synthetic production path. Assert zero calls after the fix. A raised sentinel alone is insufficient because the current caller catches exceptions; the spy must prove the attempted call count is zero.
- [ ] Add `test_healthy_wfo_is_independent_of_retired_cpcv`, `test_incomplete_wfo_blocks_actionable`, `test_missing_live_feature_blocks_actionable`, `test_stale_required_dividend_blocks_actionable`, `test_unknown_data_readiness_blocks_actionable` and `test_missing_or_nonfinite_quality_metric_blocks_actionable`. Observe red for newly corrected defects; preserve existing already-green missing-metric safety tests.
- [ ] Define `aggregate_health["wfo_completed"]` from the configured required model/benchmark pairs, actual nonempty fold/results, usable finite OOS predictions, chronological split audit and available outcome/feature dates. Missing/failed required results are false; optional excluded assets are explicitly listed. Do not infer completion merely from one successful model, a CPCV object or a truthy unknown value.
- [ ] Define `aggregate_health["data_ready"]` from finite required live features **before imputation**, required price/FRED/EDGAR freshness and required dividend history. Store `missing_live_features` and `stale_required_feeds` in the report/manifest. Treat unknown readiness as false. Explicitly allow audited non-dividend payers; never convert an unsuccessful dividend request into that classification.
- [ ] Evaluate readiness against the requested decision's as-of date, with no later feature, price, filing or realised outcome entering a backdated run. Add `test_backdated_readiness_uses_decision_date` and `test_later_data_cannot_make_backdated_inputs_ready`. Distinguish repaired historical values from evidence that the original decision actually had those values available.
- [ ] Use the following hand-controlled healthy gate fixture, extending it with the readiness fields above. Old code fails because its missing CPCV gate is FAIL; new code must return only actual chronological/readiness/quality gates, all PASS. Each negative fixture changes one field and must give DEFER-TO-TAX-DEFAULT/50%, with the reason named in all output surfaces.

```python
health = {
    "oos_r2": 0.03,
    "pt_p_value": 0.01,
    "agg_hit": 0.70,
    "constant_rule_hit_rate": 0.50,
    "wfo_completed": True,
    "data_ready": True,
    "missing_live_features": [],
    "stale_required_feeds": [],
}
# mean_ic = 0.09; representative_cpcv = None.
# Expected gate names: oos_r2, mean_ic, directional_skill,
#                      wfo_completed, data_ready.
# Expected healthy statuses: all PASS.
# Unknown/False readiness or completion: FAIL, never ACTIONABLE.
```

- [ ] Audit actual outer/inner TimeSeriesSplit boundaries, purge/embargo values, label-end checks and fold-local scaling. Preserve the verified production protocol unless a concrete leakage test fails; any necessary protocol change requires its own documented metric attribution. Do not adopt the stricter research settings by silently changing the production baseline in this cleanup.
- [ ] Remove `CombinatorialPurgedCV` imports/construction from active production validation. Replace active CPCV test executions with WFO coverage, no-future-training and fail-closed tests; historical artifacts/docs may still describe retired diagnostics. Preserve test coverage of real missing validation, rather than deleting the old safety assertion without a replacement.
- [ ] Give the new gate contract a documented stable version in manifests. Change the metric version only if a metric definition changes; gate changes and input repairs are separately attributed. Do not silently overwrite the old performance log or committed monthly artifacts.
- [ ] Replay February–September 2026 on an external scratch checkout containing the repaired DB copy using `scripts/replay_monthly_decisions.py --committed-dates`. This is a pre-v200 smoke comparison of already-inspected history. Compare to R2's input/code and the verification tables; explain each mode, sell percentage and metric difference. Record data changes separately from removal of CPCV and new readiness rules.
- [ ] Verify dry-run hashes for copied DB, source DB and source artifacts/ledgers, and run import smoke plus the full suite. No active pipeline may reach K-fold; no failed readiness/completion can enable action.
- [ ] Update current governance with full DB/code/runtime pins, honest R² comparator, IC inference, hit versus base rate, calibration/coverage limitations and current recommendation. Preserve the historical +5.08% step-5 table as dated history; replace claims about current CPCV requirements. No promise of a better return.
- [ ] Add `docs/reviews/R3_validation_closeout.md`, CHANGELOG and one PR. Include a before/after decision table and the actual red/green outputs. R3's output is the code/data starting point for v200, which creates its own research-protocol lock.

**Additions (approved 2026-09-27):**

- [ ] **D4.** Keep the monthly e-mail's content and format. Change only the lines the new gate contract requires: the removed CPCV text and the named reasons for a deferral.
- [ ] **Decision log (verification N8).** In a separate commit, update the `artifacts/monthly_decisions/decision_log.md` header template to the current rules, and annotate the 6 historical `[DRY RUN]` rows as dry runs. Do not delete history.

**Focused commands:**

```powershell
python -m pytest -o addopts="--tb=short" -q tests/integration/pipeline/test_production_validation_contract.py tests/integration/pipeline/test_validation_gating.py tests/integration/pipeline/test_monthly_pipeline_e2e.py tests/integration/pipeline/test_dry_run_read_only.py
python -m pytest -o addopts="--tb=short" -q
python scripts/checks/check_doc_links.py
python scripts/checks/check_sys_path_edits.py
```

Run the replay command only with cwd set to the external scratch checkout and an output CSV outside the source repository. Record the actual paths and pinned hashes in the closeout rather than supplying a command that defaults to the tracked DB.

**Exit gate:** active production uses chronological validation only, no manufactured completeness, stale/missing required data fails closed, all output surfaces agree, full-suite exit 0, and repaired current baseline is documented. Start v200 only after this PR merges and R1/R2 remain green.

## R4 — TA shadow maturation and honest candidate metadata

**Session recommendation:** *(amended 2026-09-27, D3: run now, before v200. It shares `src/pgr_vds/decision/artifacts.py` with R3, so merge R3 first and then merge `master` into R4.)* Separate shadow-only session before v204. It may run after v200 starts if it obeys the sealed-development restriction. Do not implement or promote Firth in this maintenance session.

**Paste this prompt after the shared preamble:**

> Execute R4 only. Read F24/F30 and verification V07. Fix the TA shadow ledger's creation-time-only maturity behavior. Make current-use documentation and metadata accurately distinguish a baseline forecast with candidate metadata from a separately fitted candidate. No provider calls, data backfills or live recommendation changes.
>
> Inspect current master for the Firth implementation and distinct candidate prediction artifacts before asserting adoption. The audited tree did not substantiate the backlog's v159 adoption claim; correct the current status with a dated evidence note if that remains true. Historical planned/claimed adoption is not proof of current use.

**Current files:**

- `src/pgr_vds/decision/artifacts.py`: `update_shadow_histories` currently matures classifier history, but only appends TA entries.
- `src/models/classification_monitoring.py`: existing maturity/end-date behavior to extend or share.
- `src/reporting/classification_artifacts.py`: ledger schema/append keys; preserve issuance-time forecasts.
- `src/reporting/shadow_followon.py`: reporting-only baseline payload with `candidate_sources`.
- `tests/unit/reporting/test_classification_artifacts.py`, `test_shadow_followon.py`; create `tests/unit/models/test_ta_shadow_monitoring.py`.
- `docs/research/backlog.md`, relevant current-use rows in `research/registry.yaml` and generated `research/README.md`; preserve historical records with clear dated annotations.

- [ ] Add `test_ta_history_matures_on_actual_horizon_end`, `test_ta_history_does_not_mature_early`, `test_ta_maturity_is_idempotent`, `test_backdated_ta_run_ignores_later_outcomes` and `test_dry_run_does_not_rewrite_ta_ledger` using copied/synthetic ledgers and DBs. Old `update_shadow_histories` must fail the maturity regression.
- [ ] Use a synthetic January 2020 origin with six-month BME endpoint July 31: July 30 evaluation is immature; July 31 may mature only when the complete realised basket outcome exists. A later data row cannot change the earlier backdated status. Missing outcomes remain unknown even if the calendar horizon passed.
- [ ] Recompute TA maturity on later runs from each row's anchor/horizon and as-of date. Fill realised outcomes without changing issued probability, feature date, recipe, recommendation or candidate identity. Use stable row keys and prevent duplicate rows. Extend shared maturity code without introducing new live-data loading or model fits.
- [ ] Dry runs can compute matured monitoring in memory but must not append/rewrite either ledger. For an existing committed ledger correction, include an explicit copy-only deterministic repair script and row diff; apply it only as documented, never by running a production decision on the tracked DB.
- [ ] Inspect follow-on outputs: baseline probability/stance copied with candidate metadata must retain `reporting_only` and be described as baseline-derived. Candidate application/performance is asserted only with distinct model/prediction/code/data provenance. Keep already-correct reporting-only behavior; do not manufacture a candidate fit to support an old claim.
- [ ] Update unsupported current-use claims with a dated audit note; regenerate the registry README if entries change. Add regression checks for any corrected code/metadata contract. Do not rewrite an archived study's original conclusion as if it were newly rerun.
- [ ] Run focused/full tests and a synthetic or sealed-development shadow replay. Verify identical regression predictions and sell recommendations; only monitoring/provenance may change. If core forecasts change, investigate and invalidate the affected baseline before further research.
- [ ] Add `docs/reviews/R4_shadow_closeout.md`, CHANGELOG and one PR. Compare ledger before/after, show red/green tests, and label Firth as research-only pending v204 evidence when current implementation is unsupported.

**Focused commands:**

```powershell
python -m pytest -o addopts="--tb=short" -q tests/unit/models/test_ta_shadow_monitoring.py tests/unit/reporting/test_classification_artifacts.py tests/unit/reporting/test_shadow_followon.py tests/integration/pipeline/test_dry_run_read_only.py
python -m pytest -o addopts="--tb=short" -q
python research/tools/registry.py
python scripts/checks/check_doc_links.py
```

**Exit gate:** TA maturity is as-of correct and idempotent, dry-run ledgers unchanged, current-use claims supported, and no core recommendation or model promotion introduced. v204 may consume shadow evidence only after this merges or explicitly exclude the affected ledger.

## R5 — Verify vest-event windows and align tax-boundary warnings

**Session recommendation:** *(amended 2026-09-27, D3: run now, before v200. Its files do not overlap the other sessions'.)* Separate event/tax session before v206. The event offset remains a suspicion, so quantify it before altering behavior. Combine with the remaining rebalancer anniversary warning because both concern the same vest-date decision contract.

**Paste this prompt after the shared preamble:**

> Execute R5 only. Read F19/F22 and their PARTIAL/adjacent limitations in verification. No provider calls or model/policy tuning. Use synthetic or pinned development-only data if the holdout is sealed. Determine whether the first monthly target on/after a vest event measures the intended event-to-horizon return; do not assume the suspected bug is confirmed. Fix only demonstrated timing errors. Separately reuse the already-correct calendar eligibility helper for the rebalancer's remaining 365-day warning logic.

**Current files:**

- `src/backtest/backtest_engine.py`: `run_historical_backtest` currently takes the first precomputed target on/after `event_date`.
- `src/processing/total_return.py`, `multi_total_return.py`: correct raw-price/split/DRIP and calendar window helpers; monthly targets and event outcomes are distinct contracts.
- `src/portfolio/rebalancer.py`: `_check_stcg_boundary` derives days to eligibility from a fixed 365-day zone.
- `src/tax/capital_gains.py`: existing `ltcg_eligible_date`; reuse its verified anniversary-plus-one-day behavior.
- `tests/unit/backtest/test_backtest_engine.py`, `tests/unit/tax/test_stcg_boundary.py`, `test_tax_hand_computed.py`; add narrowly scoped helper tests if extracting an event-return function.
- `docs/model-governance.md` and policy/tax documentation affected by a confirmed correction; CHANGELOG.

- [ ] Add `test_vest_event_return_uses_event_window_not_next_month_target`, `test_event_outcome_requires_complete_horizon`, `test_event_return_uses_only_observable_start_price` and `test_event_window_split_dividend_matches_hand_calculation`. Use a mid-month event, materially different next-month target, a split/dividend inside the true window and an outcome ending after evaluation as-of. Assert independent raw-price/DRIP amounts, not the same DB row-selection rule.
- [ ] Before changing code, record exact event date, feature/forecast anchor, last observable starting bar, stated outcome endpoint and currently selected target's own start/end. Quantify mismatch and return difference. Weekly bars only approximate execution prices; document that limitation and do not use a later bar as an allegedly available event-date close.
- [ ] If confirmed, calculate an explicit event-to-end outcome using the documented price convention, canonical splits and fractional DRIP. Require outcome-end availability. Do not alter the repaired monthly target table to make event joins appear correct. Keep model target horizon and realised policy holding period clearly labeled.
- [ ] If the suspicion is disproved under the intended documented contract, add an independent regression protecting that alignment and close with no timing change. Report the evidence; do not create a red test for an invented bug.
- [ ] Add `test_stcg_warning_uses_calendar_eligibility_across_leap_year` and `test_stcg_warning_stops_on_first_ltcg_day`. Under the repository's verified eligibility contract: vest March 1, 2023 has first eligible date March 2, 2024; the February 29, 2024 warning says 2 days remaining, March 1 says 1, and March 2 has no STCG warning. Also cover February 29 acquisition and ordinary anniversaries using independently asserted dates.
- [ ] Replace fixed 365-day qualification/day-count calculations with `ltcg_eligible_date`. Clarify any retained zone-window override as an alert-window setting, not the tax eligibility rule. Preserve unvested/empty-lot filtering, actual lot data, tax rates and sell mapping. No new owner tax assumptions or policy thresholds.
- [ ] Run focused/full tests and compare event/lot outputs on the development-only fixture. Show which historical utility numbers change from outcome timing versus warning text; no evidence from untouched/quarantined rows. Core regression targets/predictions must remain unchanged.
- [ ] Add `docs/reviews/R5_event_tax_closeout.md`, CHANGELOG and one PR. Include confirmation/disconfirmation of the suspected offset, exact date/return fixtures, red/green for confirmed repairs and remaining weekly execution-price approximation. Before v206, ensure its always-50% and candidate policies share the same corrected event/outcome convention.

**Focused commands:**

```powershell
python -m pytest -o addopts="--tb=short" -q tests/unit/backtest/test_backtest_engine.py tests/unit/tax/test_stcg_boundary.py tests/unit/tax/test_tax_hand_computed.py tests/unit/processing/test_total_return.py
python -m pytest -o addopts="--tb=short" -q
python scripts/checks/check_doc_links.py
```

**Exit gate:** confirmed timing defects repaired or the suspicion closed with evidence; tax-boundary warnings share the existing calendar eligibility contract; policy utility can use matched event outcomes; no production model promotion.

## What is intentionally left to the research plan

- v200 creates `src/pgr_vds/research_lib/`, provenance locks, unique-date chronological splits, label-end checks and independent metrics fixtures. It must not reuse old full-frame correlation pruning, future-residual warmup, origin-only holdout filtering, raw x-targets or unversioned parquet unchanged.
- v201–v203 revisit representative repaired feature/model/consensus hypotheses. These remediation prompts do not rerun old parameter searches or select features.
- v204 tests genuinely fitted, matched classifiers and calibration. Correcting metadata is not Firth adoption.
- v205 repairs/tests research target definitions before interpreting x-series results; dividend source completeness from R2 is only its data prerequisite.
- v206 tests economic policy usefulness; correct tax/event arithmetic alone does not demonstrate utility.
- v207 opens the retrospective quarantine once. Prior inspection cannot be undone. Promotion still needs the separately reserved unused evidence and a governance PR.

*(Amended 2026-09-27, D3 and D5.)* After R1–R5, R2-lite and R3b merge, proceed to v200 rather than waiting for every legacy cleanup. Sessions may run in parallel on disjoint files, as the execution plan at the top sets out. The data change (step 0 and R2-lite) and the baseline change (R3b, on R3's code) stay sequenced, so each remains attributable. At most two subagents inside a session.
