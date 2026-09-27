# Step 13V: comparison of the two verification runs (2026-09-26)

- **Purpose.** The step-13V task was run twice, on the same inputs, by two different coding agents: verify the fixes for the [2026-09-25 review](REPO_REVIEW_2026-09-25.md), then plan the research re-run. This document compares the two runs and checks every disagreement against the code and data. It also records the recommended next steps. It is the reference for what was done before v200, and why.
- **What both runs audited:** `master` at `aae0be8` (PR #133), and the committed DB with sha256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`.
- **Written:** 2026-09-26, at `master` `db64f8b`, after both PRs had merged.
- **Author and bias.** This comparison was written by the same agent session that produced the Claude run. To limit bias, I settled each disagreement by re-checking the code or data rather than by preference. The evidence is listed in each row, and the commands are in the appendix.
- **Safety.** Every check here ran offline, in scratch clones on DB copies. No fetcher, e-mail or provider call was made, and nothing in production changed.

Terms used below:

- **OOS R²:** how much smaller the forecast errors are than those of a naive guess (the average of the outcomes already known at the forecast date). 0 % means no better than that guess.
- **IC:** the rank correlation between forecasts and outcomes.
- **Base rate:** the hit rate of always saying "PGR beats the fund", which is 68–70 % over this history.
- **PT p:** the p-value of the Pesaran–Timmermann test, which asks whether the up/down calls beat chance.
- **K-fold and CPCV:** validation schemes that train on data from after the test period. `AGENTS.md` forbids them. CPCV (combinatorial purged cross-validation) is a K-fold variant.
- **Walk-forward:** train only on the past, test on the next period, then step forward.
- **Holdout:** data kept unseen until one final test.
- **Holm:** a correction that keeps the chance of any false "win" at 5 % when many candidates are tested.

## 1. Summary for the owner

**Bottom line.** The two runs agree on everything that affects your vest decision. They disagree on bookkeeping and on how strict the research should be.

- **Where they agree:**
  - The recommendation would have been the same in every month from February to September 2026: defer to the tax default and sell 50 %.
  - The two runs replayed those eight months on different operating systems and library versions. The results match to every printed digit, so the numbers can be trusted.
  - The model beats the naive guess slightly: R² is +1 % to +6 %.
  - It does not beat the simple rule "PGR usually wins": its up/down calls are right 60–64 % of the time, against 68–70 % for the rule.
  - ETF dividends are still missing for 20 tickers.
- **Finding statuses.** The runs agree on 28 of the 35 review findings. For the other seven, the Codex run scored the whole finding, including leftover research work; the Claude run scored only the production defect. I checked all seven, and the Codex rating (PARTIAL) holds for each:
  - two are plain errors in the Claude report: CPCV still runs every month and can still block a recommendation, and the technical-analysis shadow ledger never marks its forecasts as matured;
  - one depends on the platform: a test safety guard has a hole on Windows;
  - four are partly fixed because the review itself listed a research re-run or an unconfirmed suspicion.

  **Reconciled count: 18 fixed, 15 partly fixed, 2 not addressed, none worse.**
- **Tests.**
  - On Linux (Claude run) every test passes.
  - On Windows (Codex run) 3 tests fail. One of them is a real hole in the guard that keeps tests away from the real database. I confirmed the mechanism. It matters because you work on Windows.
- **Each run found things the other missed** (combined list in section 7):
  - **Codex run:** the Windows guard, the CPCV gate, the TA ledger, look-ahead in two old research studies (v138, v143), an unsupported claim that Firth was integrated (v159), the rebalancer's 365-day count, and the suspected offset in the backtest's vest windows.
  - **Claude run:** a look-ahead in the research code's R² helper, an untested market-cap guard, the stale decision-log header, and a fresh mutation study.
- **Research plans.** Both plans start at v200, but the version numbers mean different things (for example, v203 is EDGAR in one plan and model regularisation in the other). **Pick one plan before any research session starts.**
  - **The main difference is the final test.** The Claude plan uses the most recent 24 months as the one-time final test. The Codex plan points out that those months have already been looked at, by earlier studies and by both verification runs. So it uses them only for a retrospective check, and it requires 24 months of new data, collected after the plan is frozen, before anything is promoted. The earliest promotion would then be around 2029.
  - **My recommendation:** use the Codex plan as the backbone, with six additions from the Claude plan (section 8). The wait costs little today, because the current model cannot beat the simple rule and the 50 % default stays in place.
- **Before the research starts.** The Codex run's three fix sessions, R1–R3, are sound. I recommend them with two adjustments (section 9):
  - **R2:** let the scheduled Wednesday dividend refresh do the backfill instead of writing a new repair script. The existing workflow can fetch all 20 tickers in one run and rebuilds the targets. Then check the result offline.
  - **R1 and R3:** fold in the small items from the Claude list.

**What needs a decision from you:**

1. **Which research plan is canonical.** I recommend the Codex plan with the six additions. Then mark the other plan "reference only", so each v2NN number means one thing.
2. **How strict the final test is.** Either:
   - require 24 months of new, forward data before any promotion, and keep winners as shadow signals until then (recommended); or
   - accept the already-seen last 24 months as enough (faster, weaker evidence).
3. **Approve the fix sessions:**
   - R1, the dividend refresh plus an offline check, and R3, all before v200;
   - R4 and R5 later, before the classification and decision-policy research steps.
4. **Monthly e-mail content.** Whether it keeps showing the Path B sell probability, the "80 %" ranges and the "chance PGR outperforms" figure (from the Claude report).

## 2. What each run produced

| | Codex run | Claude run |
|---|---|---|
| Verification report | [`VERIFICATION_2026-09-26.md`](VERIFICATION_2026-09-26.md) | [`VERIFICATION_2026-09-26_claude.md`](VERIFICATION_2026-09-26_claude.md) |
| Research plan | [`RERUN_PLAN_v200_codex.md`](../research/RERUN_PLAN_v200_codex.md) (v200–v207) | [`RERUN_PLAN_v200_claude.md`](../research/RERUN_PLAN_v200_claude.md) (v200–v210) |
| Pre-v200 fix prompts | [`PRE_V200_FIX_PROMPTS_codex.md`](PRE_V200_FIX_PROMPTS_codex.md) (R1–R5). The owner supplied it in chat; it is committed unchanged with this comparison. | None: the fixes are listed as new issues N1–N10 |
| PR, and when it merged | [#134](https://github.com/jhester599/pgr-vesting-decision-support/pull/134), commit `afd1d5f` (22:00 UTC), merged as `d7d92a4` (23:36 UTC) | [#135](https://github.com/jhester599/pgr-vesting-decision-support/pull/135), commits `6e6dc49` (23:23 UTC) and `840ab43` (the rename), merged as `db64f8b` (23:48 UTC) |
| Platform | Windows PowerShell; Python 3.12.14; numpy 2.5.3; xgboost 3.4.1 | Linux; Python 3.11.15; numpy 2.4.6; xgboost 3.2.0 |
| Subagents | 2, read-only (history map, study inventory) | None: they hit the usage limit, so the inventory was done directly |
| Finding statuses | 18 FIXED, 15 PARTIAL, 2 NOT ADDRESSED | 25 FIXED, 8 PARTIAL, 2 NOT ADDRESSED |
| New issues | V01–V10 | N1–N10 |

## 3. Where the runs agree

Both runs independently found the following:

- **The decisions.** The replayed decision for every month from February to September 2026 is DEFER-TO-TAX-DEFAULT with a 50 % sale. Mode, sell %, consensus label, R², IC, pooled IC, hit rate, base rate and PT p are identical to the printed precision. For example, September is:
  - R² +2.86 %;
  - equal-weight IC 0.0762;
  - pooled IC 0.102 (date-clustered p 0.125);
  - hit rate 62.8 % against a 68.1 % base rate;
  - PT p 0.39.
- **The change since step 5** comes from step 6's filing-date placement of EDGAR rows (F23), not from the restructure. Both runs replayed the step-5 code (`69889ad`) on the current DB.
- **The restructure preserved output.** A September replay at `b609af6` (before the restructure) and at `aae0be8` gives 10 of 11 files byte-identical. The manifest differs only in `git_sha`, timestamp and `script_name`.
- **Data integrity:**
  - no unexplained price jumps;
  - no duplicate FRED months;
  - 265 monthly EDGAR rows with none missing, and no identity violations;
  - monthly net income matches the quarterly XBRL figure in all 73 quarters;
  - no missing live features.
- **Dividends:** 20 tickers are stale (19 ETFs plus ALL). PGR is current; GLD pays no dividend.
- **Repository checks:**
  - the registry lists 113 studies with 0 problems;
  - the link checker passes (340 → 342 files);
  - 197 files outside `tests/conftest.py` still edit `sys.path`;
  - there are no `.py` files under `results/`;
  - there are 17 leave-one-out `RidgeCV(cv=None)` calls in 16 old research scripts.
- **Model health:**
  - calibration error (ECE) is 14–19 %;
  - "80 %" intervals cover 41–49 % of outcomes;
  - the governance health baseline still shows step-5 numbers (Claude N10, Codex V08).
- **Open findings:**
  - F25 (research methods) and F31 (unused utilities) are not addressed;
  - the suspected 2024 fiscal-month change in Progressive's monthly filings is uninvestigated (Claude N5, Codex F16);
  - `peer_bootstrap.yml` queries a nonexistent `price_date` column.

## 4. Finding statuses: the seven disagreements

In every case the Claude run said FIXED and the Codex run said PARTIAL. The "review asked" column quotes the [original review](REPO_REVIEW_2026-09-25.md).

| Finding | What the review asked | What I checked here | Reconciled |
|---|---|---|---|
| F01, weekly unadjusted prices | The fix list ends "Then re-run the feature-selection research" (v18/v20). | Both runs verified the production features against hand calculations (`mom_12m` −0.115). The feature selection has not been re-run. | **PARTIAL.** Production is fixed; the feature re-selection is pending (price step of either plan). |
| F02, CPCV | Preferred fix: "demote CPCV to diagnostic-only and gate on walk-forward metrics". The title notes CPCV is a K-fold, which `AGENTS.md` forbids. | `src/pgr_vds/decision/signal_generation.py:160` still runs `run_cpcv` every month. `src/reporting/decision_rendering.py:164` defines a `cpcv_completed` gate that fails closed when CPCV does not run. The verdict no longer gates, but completion does. | **PARTIAL.** Codex is right. The Claude report's "CPCV no longer gates" is wrong. |
| F15, per-share basis | The evidence includes the research x1/x9/x12/x17 targets built on raw values. | The production per-share series are fixed (P/B has no 2006 cliff). The x-series research code is unchanged. | **PARTIAL.** Production is fixed; the research targets are pending (x-series step). |
| F22, target windows | Three CONFIRMED defects with fixes, plus one SUSPECTED item: the backtest's realised window is about one month off the model target (`backtest_engine.py:229-240`). No fix was listed for it. | Both runs verified the three fixes. `src/backtest/backtest_engine.py` has had no commits since the review, so the suspicion is neither confirmed nor cleared. | **PARTIAL** until the suspicion is tested (Codex R5). |
| F23, fixed EDGAR lag | Fix: place rows at the first decision date on or after `filing_date`. The evidence notes that v142 chose the lag by in-sample R². | Both runs verified the production placement. v142 is not revisited, and a fallback lag remains for missing filing dates. | **PARTIAL** (borderline): the listed fix is done; the research conclusion is pending. |
| F24, shadow layer | "Monitoring never matures … recompute maturity at attach time." | The classifier ledger is re-matured (`src/pgr_vds/decision/artifacts.py:313,341`, `attach_matured_classifier_outcomes`). TA ledger rows set `is_horizon_mature` once, at creation (`src/reporting/classification_artifacts.py:259`), and are only appended afterwards (`artifacts.py:346-357`). | **PARTIAL.** Codex is right. The Claude report missed the TA ledger. |
| F28, test gaps | Weekly/split/gap/leap fixtures; vacuous tests. | On Linux every test passes and 24 of 24 mutations are caught. On Windows, the new DB guard misreads `file:///C:/…` URIs (`tests/repo_guard.py:64-70`), so a test can open the committed DB by URI without being flagged. Confirmed here with Python's Windows path rules. | **PARTIAL** on Windows, the owner's platform. |

**Reconciled totals: 18 FIXED, 15 PARTIAL, 2 NOT ADDRESSED, 0 REGRESSED.** This is the Codex count under its "whole finding" rule. Of these seven, only F02 touches the live decision, and only by blocking it when CPCV fails to run. That block defers the decision, which is also what happens today.

## 5. Tests and counterfactual checks

| | Codex run | Claude run |
|---|---|---|
| Full suite (`python -m pytest -o addopts="--tb=short" -q`) | `3 failed, 2491 passed, 1 skipped, 109 warnings in 305.29s (0:05:05)`, exit 1 (Windows) | `2494 passed, 1 skipped, 136 warnings in 542.04s (0:09:02)`, exit 0 (Linux) |
| The 3 Windows failures | the guard URI escape (V01), and two tests that assume `/` path separators (V02) | not reproducible on Linux; the guard mechanism is confirmed here |
| Revert one fix per step, run its test | 16 reverts, every step covered | 17 reverts, every step covered |
| Mutation study (F28) | not re-run; cites step 9's 0 of 18 | re-run in 3 clones: 0 of 18 F28 mutations survive (the review had 16 of 18), and 0 of 6 extras |
| DB hash before and after | unchanged | unchanged |

The test totals are consistent: 3 + 2491 + 1 = 2494 + 1 = 2495.

## 6. Corrections to each report

These are facts found wrong in this comparison. They supersede the statements in the reports, which are left as written.

**Claude report** ([`VERIFICATION_2026-09-26_claude.md`](VERIFICATION_2026-09-26_claude.md)):

1. **F02.** "CPCV no longer gates" (section 3) and "a labelled diagnostic that cannot gate" (section 7) are wrong: completion is still a fail-closed gate, and CPCV runs every month. The status becomes PARTIAL.
2. **F24** FIXED is wrong: the TA shadow ledger never re-matures. The status becomes PARTIAL.
3. **F28** FIXED holds on Linux only: the Windows guard has a hole. The status becomes PARTIAL.
4. **F01, F15, F22, F23** were scored on the production defect only. The review's own fix list or evidence includes a research re-run or an open suspicion, so the status becomes PARTIAL.
5. **Step-5 replay (section 9).** "Replaying `69889ad` on today's DB reproduces the step-5 CSV exactly" is exact for February and March only.
   - The published CSV was generated at 01:14 UTC on 2026-09-26 (`c8e6f66`). That was before the step-5 branch merged the weekly data update of the same day (`a31ca5f`, merged at `5618113`, 01:24 UTC).
   - Replaying step-5 code for 2026-08-20 gives:
     - on the DB before the update (sha `f53d0bd7…`): R² 0.049603136, the published value;
     - on today's DB: 0.049614215, the value both verification runs got.
   - The attribution to step 6 stands. The Codex report's "not causally isolated" difference is explained by this data update.
6. **Totals.** The 25 / 8 / 2 count becomes 18 / 15 / 2.

**Codex report and plan:**

1. **V06** cites `src/research/v138_utils.py`. The code is in `research/studies/v138_bl_param_eval/v138_bl_param_eval.py` (lines 95 and 103). The finding itself is confirmed.
2. **v10.** The plan says no study or closeout was found for v10. [`docs/history/results/V10_1_RESULTS_SUMMARY.md`](../history/results/V10_1_RESULTS_SUMMARY.md) exists.
3. **Step-5 CSV.** The report's small difference from the published step-5 CSV, left unexplained there, is explained above.

**Interpretation difference, not an error:** the September report shows a 65.8 % "chance PGR outperforms" next to an UNDERPERFORM consensus.
- The Codex run treats this as honest: a probability calibrated to a 68 % base rate can exceed 50 % while the average forecast is negative.
- The Claude run (N3) treats it as confusing for the reader.

Both are right. The calculation is sound; the presentation needs a label or an explanation.

## 7. Combined list of open issues

"Where" names the session that should handle each issue:
- R1–R5 are the pre-v200 fix sessions in [`PRE_V200_FIX_PROMPTS_codex.md`](PRE_V200_FIX_PROMPTS_codex.md);
- v200 is the research baseline.

Severity is the reconciled one.

| # | Issue | Found by | Severity | Checked here | Where |
|---|---|---|---|---|---|
| 1 | ETF dividends are stale for 20 tickers. The monthly decision neither shows nor gates on it. | both (N1, V03) | High | counts agree | refresh (section 9); R3 `data_ready` gate |
| 2 | On Windows the test DB guard misreads `file:///C:/…` URIs. | Codex (V01) | High on Windows | confirmed by simulation | R1 |
| 3 | On Windows two tests assume `/` path separators. | Codex (V02) | Medium | observed by Codex | R1 |
| 4 | CPCV, a K-fold, runs every month, and its completion gates the action. | Codex (F02) | Medium (`AGENTS.md` rule) | confirmed | R3 |
| 5 | The governance health baseline still shows step-5 numbers. | both (N10, V08) | Medium | agree | R3 |
| 6 | The research helper `v37_utils.compute_metrics` scores 6M targets against a 1-month naive, and its pooling concatenates benchmarks. | Claude (N2); Codex flags the pooling and the fixed holdout date | Medium (research) | Claude's oracle test: honest R² +4.9 %, helper −2.4 % | v200 (both plans build a new `research_lib`) |
| 7 | v143 correlation pruning sees the whole feature history. | Codex (V05) | Medium (research) | consistent with `src/research/v139_utils.py:57-73` | v200 harness rule |
| 8 | The v138 proxy warm-up (inherited by v149/v150) uses full-sample statistics. | Codex (V06) | Medium (research) | confirmed | v200 harness rule |
| 9 | The backlog calls v159 (Firth in the shadow lane) "complete", but there is no implementation. | Codex (V07) | Medium (docs) | confirmed: no `src/reporting/firth_shadow.py`, no CHANGELOG v159 section | R4 |
| 10 | "Chance PGR outperforms" is shown next to a forecast of the opposite sign. | Claude (N3) | Medium (reporting) | same September numbers in both | owner decision; R3 |
| 11 | The "80 %" intervals cover 41–49 %. | both (N4) | Medium (reporting) | agree | owner decision; interval research |
| 12 | The 2024 fiscal-month change in the monthly filings is uninvestigated. | both (N5, F16) | Medium (research) | agree | EDGAR research step |
| 13 | The backtest's vest-event window is suspected to be about 1 month off. | Codex (F22); the review marked it SUSPECTED | unknown until tested | `backtest_engine.py` unchanged since the review | R5 |
| 14 | The TA shadow ledger never re-matures. | Codex (F24) | Low–Medium (shadow only) | confirmed | R4 |
| 15 | The rebalancer counts a fixed 365 days to long-term status; the tax module correctly uses the anniversary + 1 day. | Codex (F19, adjacent) | Low | confirmed: `src/portfolio/rebalancer.py:268`, `config/tax.py:40` | R5 |
| 16 | `peer_bootstrap.yml` queries the nonexistent `price_date` column. | both (N9, V09) | Medium (the workflow fails after doing its work) | Codex reproduced the SQL error | R1 |
| 17 | `initial_fetch --force` is ignored; `ci.yml` has no `permissions:` block; Actions are pinned by tag. | both (N9, F26) | Low | agree | R1 or a hygiene PR |
| 18 | Step 4b's market-cap consistency guard has no test. | Claude (N6) | Low | removing the `raise` fails no test | R1 |
| 19 | `decision_log.md` has a stale header and 6 `[DRY RUN]` rows. | Claude (N8) | Low | — | R3 or a hygiene PR |
| 20 | The `src/research` package shadows the top-level `research/` namespace when `src/` is on the path. | both (N7; Codex runtime note) | Low | agree | v200 (move to `src/pgr_vds/research_lib`) |
| 21 | Live code reads research-study output files (v113, v128, v141–v150). | Codex (F30); Claude noted the v128 map | Low–Medium | confirmed: `src/reporting/shadow_followon.py:14`, `src/models/classification_gate_overlay.py:46`, `src/models/v129_feature_map.py:31`, `config/features.py:348` | later provenance work |
| 22 | PEP 8 is not enforced (ruff selects only E9, F63, F7, F82). Docs say Python 3.10; packaging requires 3.11. | both | Low | `pyproject.toml:99` | hygiene PR |
| 23 | 197 files still edit `sys.path`. | both | Low | agree | staged hygiene |
| 24 | CB entity splice, TRV dividend overlap, FZROX proxy rows, latest-vintage FRED. | both (F27) | Low–Medium (research) | agree | exclude from research until repaired |
| 25 | Ineffective utilities: fracdiff, a BLP fit that ignores outcomes, full-history drift IC. | both (F31) | Low (not used by the live decision) | Codex toy check | unassigned |

## 8. Research plans compared

| Topic | Codex plan (v200–v207) | Claude plan (v200–v210) |
|---|---|---|
| Sessions | 8: baseline; price/technical + macro; insurance fundamentals + valuation; model class + consensus; classification + calibration; dividend/BVPS x-series; policy/tax; synthesis | 11: baseline; price/technical; macro; insurance/EDGAR; valuation; model class; universe/consensus; classification; policy/tax; x-series; synthesis |
| Candidate budget | 38 across v201–v206, with inner grids declared | 74, plus ≤ 20 descriptive valuation tests and ≤ 9 finalists |
| Multiple testing | Holm across all 38 campaign candidates; unused slots count as p = 1 | Holm within each step |
| Outer validation | 6M: 60-month train, 6-month test, gap 12 (purge 6 + embargo 6). 12M: 120 / 6 / gap 24. A bridge table shows the effect against production's gap 8 | production settings: 6M gap 8 (horizon 6 + 2); 12M gap 15 |
| Inner selection | 3 inner walk-forward folds (test 6, gap 2h, minimum 24 / 60 months of training) | nested inner walk-forward with the same gap, or prequential |
| Earlier winners in later steps | re-selected inside each fold; winners chosen on the whole development period are exploratory only | later steps start from finalists chosen on the whole development period (for example, the v205 base) |
| Final test | The latest 24 matured origins are quarantined (6M: March 2024–February 2026; 12M: September 2023–August 2025) and opened once, at v207. Passing supports only shadow status. **Promotion needs 24 new forward origins** (for example October 2026–September 2028; 6M labels mature by March 2029). | The latest 24 realised 6M dates are the holdout, opened once at v210. The plan notes their earlier exposure, and a promotion recommendation is allowed from them. |
| Regression success | ΔR² ≥ +1.0 pp and adjusted p < 0.05; IC loss ≤ 0.01; no directional or calibration loss. An active policy also needs directional skill (p < 0.05), ECE ≤ 0.10 and 80 % coverage between 75 and 85 % | ΔR² ≥ +1.0 pp and Holm p < 0.05; IC loss ≤ 0.01; hit − base not down > 1 pp; no benchmark down > 3 pp; policy 90 % CI lower bound ≥ −0.25 pp |
| Policy success | uplift ≥ +0.25 pp per decision against always-50 %, adjusted p < 0.05, 95 % lower bound > 0 | v208: uplift ≥ +0.25 pp per decision against always-50 %, 90 % lower bound > 0. The per-candidate rule above is a non-inferiority bound of −0.25 pp |
| Harness | new `research_lib` (provenance, temporal splits, metrics); forecasts reproducible to 1e-10 on a second run | new `research_lib` (frames, holdout, evaluation, splits, ledger, provenance); a production-equivalence check at `DEV_END` to 1e-9 |
| Data precondition | a separate, authorised dividend-repair PR (R2) before v200 | dividend-freshness preflight in v200; the owner dispatches the refresh |
| Study inventory | all 157 IDs v9–v165, including the 71 without study folders; flags v138/v143 leakage, the v148 proxy, v151 metadata forwarding and the unsupported v159 claim; misses v10 | one row per ID or block v9–v170; includes v10/v10.1; gaps v61–v65 and v98–v101; a skip list with reasons |
| Readability | dense; each prompt repeats long boilerplate | plainer prompts and a plain-language preamble |

**Assessment.** The Codex plan is stricter in the four places that decide whether a research "win" can be believed:

1. **The final test.** The recent 24 months have already been examined:
   - v75 replayed April 2024–March 2026;
   - v129 and v132 used parts of them;
   - both verification runs replayed February–September 2026.

   Only data collected after the plan is frozen is truly unseen.
2. **Carrying winners forward.** Carrying a winner chosen on the whole development period into later steps leaks that choice into their development results. The Codex plan re-selects inside each fold.
3. **Multiple testing.** One Holm correction across 38 candidates controls false wins across the whole programme. Per-step Holm across 74 candidates does not.
4. **Validation gaps.** Research gaps of twice the horizon are the conservative choice, and the bridge table keeps them comparable with production.

The Claude plan is better on readability, on the concrete harness design and its production-equivalence check, and on its explicit valuation and universe steps.

**Recommendation.** Adopt the Codex plan's sequence (v200–v207) and its final-test rule as canonical, with these six additions from the Claude plan:

1. **In v200**, run the production-equivalence check: the harness at `DEV_END`, with production settings, must match `generate_signals` / `compute_aggregate_health` to 1e-9. Report it in the bridge table.
2. **In v200**, add the fixtures for issue 6:
   - an oracle forecaster on overlapping 6M targets must score R² > 0;
   - pooling two benchmarks must not inflate R².
3. **In the insurance/EDGAR step** (Codex v202), test the 2024 fiscal-month hypothesis (issue 12) before using `npw_growth_yoy`.
4. **In the same step**, run the per-era `pb_vs_pe` re-analysis as descriptive, Holm-corrected work. It does not count toward the candidate budget.
5. **In every study README**, start with a plain-language summary for the owner, as in the Claude plan's preamble.
6. **In the inventory**, add the v10/v10.1 entry.

Then mark the Claude plan "reference only; its version numbers are void".

**The cost of this choice.** No model change can be promoted before the forward data matures. If the plan is frozen by September 2026, that is about March 2029 for 6M targets. Until then, winners run as shadow signals, and the 50 % tax default continues. The model cannot beat the base rate today, so little is lost by waiting.

## 9. Pre-v200 fix sessions (Codex R1–R5)

The Codex run's attached prompts propose five sessions. R1–R3 come before v200; R4 and R5 come before the classification and policy research steps.

| Session | What it does | Assessment | Adjustments |
|---|---|---|---|
| **R1**, offline safety | fix the Windows guard URI escape and the two path-separator tests; fix the `peer_bootstrap.yml` summary query; add a targeted Windows CI job | **Agree.** The guard defect is confirmed (issue 2), and both runs found the query bug. A small Windows job is worth it because the owner and at least one agent work on Windows. | also add: a test for the step-4b market-cap guard (issue 18); honour `initial_fetch --force` and add a CI `permissions:` block (issue 17) |
| **R2**, dividend repair | a new repair script with cached Alpha Vantage payloads, a quota journal, a rebuild of both horizons, row diffs and an idempotence replay | **Agree the data must be fixed before v200, but use the existing path first.** `scripts/weekly_fetch.py --dividend-refresh` already fetches due tickers, rebuilds the 6M and 12M targets and runs the integrity check. It runs every Wednesday (next: 30 September) or by dispatch with `dividend_refresh: true`. One run can use up to 23 calls (`AV_DAILY_LIMIT` 25 − `DIVIDEND_REFRESH_AV_RESERVE` 2), enough for the 20 stale tickers. | after the refresh, run an offline "R2-lite" session: diff dividends and both-horizon targets between the DB before and after (from git history), confirm freshness, and record it in `docs/reviews/`. Use the full R2 only if the workflow run fails or leaves required tickers stale. |
| **R3**, validation, gates and baseline | remove CPCV from the live path; add `wfo_completed` and `data_ready` gates that fail closed (including stale required dividends); replay February–September; refresh the governance baseline | **Agree.** It completes the review's preferred F02 fix, turns issue 1 into a fail-closed gate and fixes issue 5. The monthly action is unlikely to change, because it already defers. | also: the decision-log header (issue 19); and, if the owner decides so, label or hide "chance PGR outperforms" and the "80 %" ranges (issues 10–11). It is a production change, so it needs a CHANGELOG entry (v186 or later) and a governance note. |
| **R4**, TA maturity and Firth metadata | re-mature the TA shadow ledger; correct the v159 "complete" claim | **Agree.** Shadow only. | before the classification step |
| **R5**, vest windows and the rebalancer | test the suspected backtest offset and fix it only if confirmed; use the anniversary rule in the rebalancer | **Agree.** | before the policy step |

## 10. Recommended sequence

1. **Now:** R1. It is offline and independent of the data.
2. **Wednesday 30 September:** the scheduled dividend refresh (or dispatch it). Check that the run's "Check price, split and dividend integrity" step passes.
3. **R2-lite:** record and verify the refresh offline.
4. **R3:** production validation, gates and the current baseline. Merge it before v200.
5. **v200:** the clean baseline, from the plan the owner chooses (section 11).
6. **v201 onwards**, with R4 before the classification step and R5 before the policy step.
7. **After the synthesis step:** collect the forward data, if the owner chooses the strict final test. Then a separate governance PR for any promotion.

Production fixes take CHANGELOG versions v186 and up. Research keeps v200 and up. Both runs' plans agree on this split.

## 11. Decision record

Fill this in when the owner decides, in a separate commit, with the date.

| Decision | Options | Recommendation | Owner's choice |
|---|---|---|---|
| Canonical research plan | Codex (v200–v207) + six additions / Claude (v200–v210) | Codex + additions | *pending* |
| Final-test rule | forward 24 months before promotion / already-seen last 24 months | forward data | *pending* |
| Dividend repair route | scheduled refresh + R2-lite / full R2 | refresh + R2-lite | *pending* |
| Fix sessions before v200 | R1 → refresh + R2-lite → R3 | approve | *pending* |
| E-mail: Path B, "80 %" ranges, "chance PGR outperforms" | keep / label as experimental / hide | label as experimental or hide until re-validated | *pending* |

## Appendix: how the checks in this document were run

All commands ran on `master` `db64f8b` or in scratch clones outside the repository.

```bash
# Finding statuses: parse the status column of both verification tables.
grep -n '^| F' docs/reviews/VERIFICATION_2026-09-26.md docs/reviews/VERIFICATION_2026-09-26_claude.md

# F02: CPCV still runs and its completion gates the decision.
grep -n 'run_cpcv(' src/pgr_vds/decision/signal_generation.py        # line 160
grep -n 'name="cpcv_completed"' src/reporting/decision_rendering.py   # line 164

# F24: TA ledger rows set maturity once, at creation; only classifier rows are re-matured.
grep -n 'is_horizon_mature = run_date' src/reporting/classification_artifacts.py  # line 259
sed -n 300,360p src/pgr_vds/decision/artifacts.py

# F22: no change to the backtest since the review.
git log --oneline 9887288..db64f8b -- src/backtest/backtest_engine.py   # (no output)

# F19 (adjacent): rebalancer day count.
sed -n 260,270p src/portfolio/rebalancer.py; grep -n STCG_ZONE_MAX_DAYS config/tax.py
```

```python
# F28 / V01: the guard's URI handling under Windows path rules (run on Linux).
import ntpath, nturl2path
from urllib.parse import urlparse, unquote
committed = r"C:\Users\Jeff\repo\data\pgr_financials.db"
uri = "file:///C:/Users/Jeff/repo/data/pgr_financials.db?mode=ro"
guard_view = ntpath.normpath(unquote(urlparse(uri).path))  # what tests/repo_guard.py:64-70 does
print(guard_view)  # \C:\Users\Jeff\repo\data\pgr_financials.db -> no match, access not flagged
print(nturl2path.url2pathname(urlparse(uri).path) == committed)  # True: the correct conversion
```

```bash
# Step-5 CSV vs replay: step-5 code (69889ad) on the DB before and after the 2026-09-26 weekly update.
git show c8e6f66:data/pgr_financials.db > <scratch>/at_c8e6f66.db     # sha256 f53d0bd7...
# In a scratch clone at 69889ad with that DB copy, sockets blocked, keys unset:
python scripts/monthly_decision.py --dry-run --as-of 2026-08-20 --skip-fred
#   R² 0.049603136 (published CSV: 0.0496031). The same run on today's DB (7c68efbd...): 0.049614215.
git log --format='%h %ad %s' --date=iso -- data/pgr_financials.db | head -4  # c8e6f66 01:14, 5618113 01:24

# Research leakage and the Firth claim.
sed -n 88,104p research/studies/v138_bl_param_eval/v138_bl_param_eval.py   # residual_sq.mean(), var(ddof=0)
sed -n 57,76p src/research/v139_utils.py                                   # corr over the supplied frame
ls src/reporting/firth_shadow.py; grep -n v159 docs/research/backlog.md CHANGELOG.md

# Dividend refresh budget and target rebuild.
grep -n -E 'AV_DAILY_LIMIT|DIVIDEND_REFRESH_AV_RESERVE' config/api.py
grep -n -E 'def run_dividend_refresh|def _rebuild_targets|build_relative_return_targets' scripts/weekly_fetch.py
```
