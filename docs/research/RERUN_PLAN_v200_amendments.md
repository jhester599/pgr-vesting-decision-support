# Research rerun plan v200: amendments after v200

> **Status (2026-10-01): adopted.** These owner decisions amend the canonical plan, [`RERUN_PLAN_v200_codex.md`](RERUN_PLAN_v200_codex.md), and take precedence over any conflicting text in it. Read them together with that plan's D1–D7. D9 (2026-10-01) takes precedence over D8 (2026-09-29) where they conflict.

## Why this is a separate file

v200's accepted lock, [`baseline_lock.json`](../../research/studies/v200_clean_baseline/outputs/baseline_lock.json) (SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`), pins the exact bytes of 131 input files. They include the canonical plan (`4bea95d…`), `docs/model-governance.md` and the review records it cites. Every later study verifies those pins against its working tree. Only `research/registry.yaml` is exempt, through the add-only check below.

So the canonical plan and the other pinned documents must not change while later studies verify against v200's lock. An edit to any of them stops every later study before fitting. Amendments after v200 go here instead. This file is not pinned.

## How to use it in a session

v203 and v204 are cancelled (D9); do not run them. For v202, v206 and v207, paste, in this order:

1. the canonical plan's shared preamble;
2. the [preamble addendum](#preamble-addendum--paste-after-the-shared-preamble) below;
3. the canonical plan's validation contract;
4. the step's prompt from the canonical plan, except for v206: paste the [v206 replacement prompt](#v206-replacement-prompt-d9--paste-instead-of-the-canonical-v206-prompt) below instead;
5. for v202 and v207, that step's note from [step notes](#step-notes--paste-after-the-steps-prompt) below.

## D8 (2026-09-29, after v200, v205 and the blocked v201 merged): lessons from the first three studies

v200 (PR #147) produced the accepted comparator. v205 (PR #149) ran and sent no finalist. v201 (PR #148) stopped before fitting on a registry-pin check. These amendments stop the same problems recurring.

- **Registry pin: add-only check.** v200's lock pins the bytes of `research/registry.yaml`, and every later study must add its own entry. A plain hash check therefore fails, which is what blocked v201.
  - Verify the lock with the add-only rule v205 used (`verify_lock_with_registry_append` in `research/studies/v205_dividend_bvps/run.py`). The registry is checked against its bytes at v200's execution commit, and all 114 pinned entries must be unchanged. Every other consumed file must match its pin in the working tree.
  - Use the shared, tested version in `src/pgr_vds/research_lib/provenance.py` once one exists. If none exists yet, the next study moves it there with tests; it does not write its own copy.
  - This is not a successor lock: v200's lock is unchanged.
- **Pinned documents stay unchanged.** No study edits the canonical plan, `docs/model-governance.md` or any other file pinned in v200's lock. Later amendments go into this file.
- **v201 rerun (authorized).** v201's blocked session fitted nothing and saw no results. The rerun keeps its six declared ideas (P1–P3, M1–M3 in v201's `candidate_ledger.json`) and slots 1–6. It is not a retry after results.
- **Runtime and typing.** Use Python 3.12.14 with the exact versions in v200's `runtime_lock.json`, in an isolated environment; never upgrade. CI runs mypy: run it on new `research_lib` code with the locked Python before pushing (v200's first CI run failed on six mypy errors).
- **Multiplicity slots.** The 38-test campaign family is numbered by study: v201 1–6, v202 7–14, v203 15–22, v204 23–28, v205 29–34 (used, no finalist), v206 35–38. Each study fills only its own slots; the others stay pending at p = 1 until v207 completes the Holm adjustment.
- **Consume v200's outputs.** Take the mature naive (prevailing-mean) forecasts, support and control forecasts from v200's pinned outputs; do not recompute them. v205's first attempt differed because it left out v200's pre-output history label (the 1999-11 origin).
- **Development dates in tests.** Tests, fixtures and CI smoke runs use an as-of date no later than 2022-08-31, v200's development bridge date, so no label they read reaches the quarantine boundary of 2023-09-29.
- **Carried to later steps:** v202 (A3 still open; PIF vintage), v204 (minimum support for the classifier lane) and v207 (VWO targets; finalists so far), in the step notes below.

## D9 (2026-10-01, after v201 attempt 2 closed): end the forecast campaign early and reframe v206

v201 attempt 2 (PRs #150 and #151) closed with no finalist, as v205 (PR #149) had. With v200's baseline, the evidence so far is:

- **The incumbent:** 6M development R² +0.011, 95% date-block interval −0.132 to +0.123; equal-weight IC −0.042; 12M R² −0.191.
- **The campaign:** 12 of the 38 slots are used (v201 1–6, v205 29–34). None met its development threshold, and every Holm-adjusted p is 1.0. The largest 6M gain, v201's P1, is ΔR² +0.058 with raw p 0.087.
- **The live decision:** all 25 rows in `artifacts/monthly_decisions/decision_log.md`, dry runs included, say sell 50%. The forecast has never moved the vest decision.

The campaign is unlikely to detect a realistic edge:

- The 144 6M development origins are about 24 non-overlapping periods. Every relative return contains PGR's own move, so the eight benchmarks add little independent information.
- The first Holm step needs p < 0.05/38 ≈ 0.0013. The bootstrap p is (1 + k)/2,001, where k counts the centred draws at least as large as the observed gain, so at most 1 of the 2,000 draws may reach it.
- A normal approximation from v201's 6M results (P1, P3 and M1) puts the gain that would clear that step at about ΔR² 0.11–0.17. That is two to three times P1's gain. Published return predictors rarely reach more than a few points of out-of-sample R².

More forecast candidates on the same data are therefore expected to end with no finalist, whatever the truth. That result is absence of evidence, not proof that no edge exists. The vest decision is better served by the parts that need no forecast: concentration and tax.

Decisions:

- **v202 runs, and it is the last forecast study.** Its prompt, 8-block budget, slots 7–14 and thresholds are unchanged, and the D8 step note still applies. It is the most PGR-specific hypothesis left, and v205's nearest miss (D2) used the same fundamentals.
  - If v202 nominates no finalist, the forecast campaign under this plan ends.
  - A later forecast study needs a new owner decision and its own plan. It does not reuse this plan's slots. If v207 leaves the quarantine sealed, that plan may open it once.
- **v203 and v204 are cancelled.** They apply other estimators, weightings and a classifier to inputs that carry no demonstrated signal. v204 also had only 35 calibrated PathB dates (D8).
  - Their version numbers are retired, never reused.
  - Their slots, 15–22 and 23–28, stay in the family at p = 1. The family stays at 38 tests: shrinking it after seeing v201's and v205's results would be a results-dependent change, and keeping it is the conservative choice.
  - D8's v204 note is void.
- **v206 is reframed as a forecast-free study.** As written, it tests forecast-driven policies against always-50%. With no forecast skill, that outcome is already known.
  - The [replacement prompt](#v206-replacement-prompt-d9--paste-instead-of-the-canonical-v206-prompt) compares policies that need no forecast and reports their trade-offs for the owner. It nominates no winner by p-value.
  - It uses no campaign slots, so 35–38 stay at p = 1, and it never opens the quarantine.
  - It no longer depends on v203 or v204, so it can run in parallel with v202.
- **v207 depends on v202.**
  - With a v202 finalist, v207 runs as planned for that one finalist, with D8's VWO note.
  - Without one, v207 is a short close-out. It completes the Holm table from the slot ledgers, records no promotion and leaves the quarantine sealed. Its partition hashes and access ledger stay unchanged, so a future plan can still use it once.
- **No live change.** The live model keeps deferring to the 50% default, and D4 stands. Changing the default sell rule, for example after v206, needs its own governance PR once the owner has decided.
- **Order and model split.** v202 and v206 run in parallel, built by Codex and reviewed by Claude. v207 runs last, by Claude, reviewed by Codex.

## Preamble addendum — paste after the shared preamble

> *(D8, 2026-09-29; slots amended by D9, 2026-10-01.)* Verify v200's lock, SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`, with the add-only registry check (`verify_lock_with_registry_append`; use the shared `research_lib` version if one exists), not a plain hash of `research/registry.yaml`. Do not edit the canonical plan, `docs/model-governance.md` or any other file pinned in that lock. Use Python 3.12.14 and the exact versions in v200's `runtime_lock.json`, and run mypy on new `research_lib` code before pushing. Multiplicity slots: v202 fills 7–14; v206 fills none. v201 (1–6) and v205 (29–34) are used. v203 (15–22), v204 (23–28) and v206's former slots (35–38) stay at p = 1, and the family stays at 38. v203 and v204 are cancelled. Take mature naive forecasts, support and controls from v200's pinned outputs. Tests and smoke runs use an as-of date no later than 2022-08-31.

## Step notes — paste after the step's prompt

**v202:**

> *(D8.)* v205 used trailing-12M NPW growth without running the A3 test, so A3 (step 4) is still open here. `pif_total` differs from first-reported values in 50 months from 2019-07 (step 3b's single-definition rebuild, F11): PIF features are not a historical vintage, so disclose it as v205 did in its `source_vintage_audit.json`. Report against frozen v201 only after the v201 rerun has merged.
>
> *(D9.)* The v201 rerun has merged (PRs #150 and #151), so report against frozen v201 attempt 2. v202 is the last forecast study under this plan. Next to each candidate's primary 6M result, also report the approximate ΔR² that would clear the first Holm step at its own spread: 3.01 times the standard deviation of its bootstrap ΔR² draws. This descriptive figure shows the owner whether a "no finalist" result is absence of evidence. It adds no test, slot or route to promotion.

**v204:** cancelled by D9. D8's note no longer applies.

**v207:**

> *(D8.)* v205 nominated no finalist, so at most five remain, one each from v201–v204 and v206. Complete the Holm adjustment from each study's slot ledger. The 12 VWO targets affected by the accepted March 2026 dividend gap, listed in v200's `outputs/vwo_accepted_gap.csv`, are all in the quarantine. Score them as stored, and preregister a sensitivity run that leaves them out, reported next to the primary result.
>
> *(D9.)* v201, v205 and the reframed v206 send no finalist, and v203 and v204 are cancelled, so at most one finalist remains, from v202.
>
> - **With no v202 finalist:** run v207 as a close-out only. Complete the Holm table: slots 1–14 and 29–34 from the ledgers, slots 15–28 and 35–38 at p = 1. Record no promotion. Do not open the quarantine; verify that its partition hashes and access ledger are unchanged. The D8 VWO note does not apply. Cite v206's trade-off table as the owner's decision input.
> - **With a v202 finalist:** run v207 as planned for that one finalist, with the D8 VWO note.

## v206 replacement prompt (D9) — paste instead of the canonical v206 prompt

> Execute only `v206_policy_tax`, with the shared preamble, the preamble addendum and the validation contract's chronology, availability and provenance rules (items 1 and 6). Items 2–5 cover learned forecasts; nothing is learned here. Create research artifacts, not a live promotion.
>
> **Question:** With no demonstrated forecast skill, what does each forecast-free vest policy trade between PGR concentration, after-tax outcome and tax cost? The study informs an owner choice. It does not pick a winner by historical return. PGR beat its benchmark in about 71% of v200's 942 scored 6M rows (origins 2011–2023), so a return ranking rewards whichever policy held more PGR in hindsight.
>
> **Pinned inputs:** v200's lock, verified with the add-only registry check (D8). Owner lot data stays outside git, like the gitignored `data/processed/position_lots.csv`. Committed outputs use a synthetic lot schedule and report percentages or per-share values, never the owner's dollar amounts.
>
> **Policies, frozen before scoring, with no search:**
>
> - **C0, control:** sell 50% of each vest, the current default.
> - **S1:** sell 100% at vest.
> - **S2, concentration cap:** at each vest, sell vest shares until PGR is at most X% of the investable portfolio, and keep the rest.
> - **S3:** S2 plus tax-aware trimming of earlier lots toward the cap, with the existing lot rules: LTCG lots before STCG, highest per-share basis first, loss harvesting and wash-sale windows.
> - **Reference only, not a candidate:** sell 0% (hold everything), to show the concentration end of the trade-off.
>
> **Owner inputs, frozen in `outputs/preregistration.json` before any event outcome is read:**
>
> - the cap X; if the owner gives none, run X = 10%, 20% and 30% side by side;
> - the fund that receives sale proceeds: VTI (history from 2001) unless the owner names another;
> - federal LTCG and STCG rates, NIIT and state rate; the defaults are those in `config/tax.py`;
> - PGR's starting share of the investable portfolio; if the owner gives none, run 25%, 50% and 75% side by side;
> - shares per vest: a fixed number per vest unless the owner supplies a schedule.
>
> List every default used in the README. Invent no other owner assumption.
>
> **Method:**
>
> 1. **Events:** the vest calendar in `config/tax.py` (January 19 and July 17), applied to every year with full PGR and fund price, split and dividend coverage. Score only events whose outcome windows end, and are available, before the quarantine boundary of 2023-09-29. Every policy uses `compute_event_outcome` over one common mask of available events (D7, R5). Unadjusted prices with manual splits and fractional-share dividend reinvestment, as in the preamble.
> 2. **Tax:** apply the frozen rates to realised gains lot by lot, by character, with the existing tested helpers. Write red/green fixtures for anniversary-plus-one-day LTCG, per-share gain ranking, unvested-share exclusion, wash-sale windows and after-tax proceeds. One fixture checks that a sale on the vest date realises no gain, because the RSU basis is the vest-date price.
> 3. **Report** for each policy and scenario:
>    - PGR's share of the portfolio along the path (median and maximum);
>    - after-tax value relative to C0, 6 and 12 months after each event and at the end of the path;
>    - the worst event, the 5th percentile and the largest month-end peak-to-trough fall of the combined position;
>    - realised tax by character, and tax as a share of proceeds;
>    - shares sold per event;
>    - the per-event gap to hold-all and to S1, so the owner sees what diversification cost in PGR's strong periods and saved in its weak ones.
>
>    Uncertainty: a moving-block bootstrap over event dates, with blocks of two events (12 months), seed 20260926 and 2,000 replicates. Intervals are descriptive.
>
> **Disposition:** no p-value gate, no campaign slot, no Holm entry and no quarantine access. Write `outputs/policy_tradeoffs.md` with one plain-language table per scenario for the owner. The README's owner summary (A5) says which trade-offs the evidence supports and that it promises no better return. Any owner choice goes into a separate governance PR that changes the live default; this PR changes nothing live.
>
> **Outputs/provenance:** as in the canonical v206 prompt.
>
> Run the red/green math tests, the full suite and the DB-hash check, and report inherited failures. Open one research-only PR. End with what changed / what is left.
