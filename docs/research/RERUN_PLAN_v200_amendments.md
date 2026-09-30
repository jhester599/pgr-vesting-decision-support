# Research rerun plan v200: amendments after v200

> **Status (2026-09-29): adopted; updated 2026-09-30 after v201's rerun.** These owner decisions amend the canonical plan, [`RERUN_PLAN_v200_codex.md`](RERUN_PLAN_v200_codex.md), and take precedence over any conflicting text in it. Read them together with that plan's D1–D7.

## Why this is a separate file

v200's accepted lock, [`baseline_lock.json`](../../research/studies/v200_clean_baseline/outputs/baseline_lock.json) (SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`), pins the exact bytes of 131 input files. They include the canonical plan (`4bea95d…`), `docs/model-governance.md` and the review records it cites. Every later study verifies those pins against its working tree. Only `research/registry.yaml` is exempt, through the add-only check below.

So the canonical plan and the other pinned documents must not change while v201–v207 verify against v200's lock. An edit to any of them stops every later study before fitting. Amendments after v200 go here instead. This file is not pinned.

## How to use it in a session

Paste, in this order:

1. the canonical plan's shared preamble;
2. the [preamble addendum](#preamble-addendum--paste-after-the-shared-preamble) below;
3. the canonical plan's validation contract;
4. the step's prompt from the canonical plan;
5. for v202, v204 and v207, that step's note from [step notes](#step-notes--paste-after-the-steps-prompt) below.

## D8 (2026-09-29, after v200, v205 and the blocked v201 merged): lessons from the first three studies

v200 (PR #147) produced the accepted comparator. v205 (PR #149) ran and sent no finalist. v201 (PR #148) stopped before fitting on a registry-pin check. These amendments stop the same problems recurring.

- **Registry pin: add-only check.** v200's lock pins the bytes of `research/registry.yaml`, and every later study must add its own entry. A plain hash check therefore fails, which is what blocked v201.
  - Verify the lock with the add-only rule v205 used (`verify_lock_with_registry_append` in `research/studies/v205_dividend_bvps/run.py`). The registry is checked against its bytes at v200's execution commit, and all 114 pinned entries must be unchanged. Every other consumed file must match its pin in the working tree.
  - Use the shared, tested version: `verify_registry_growth` in `src/pgr_vds/research_lib/snapshot.py`, added by v201's rerun (PR #151) with tests in `tests/research/test_v201_snapshot.py`. Do not write another copy.
  - This is not a successor lock: v200's lock is unchanged.
- **Pinned documents stay unchanged.** No study edits the canonical plan, `docs/model-governance.md` or any other file pinned in v200's lock. Later amendments go into this file.
- **v201 rerun (authorized).** v201's blocked session fitted nothing and saw no results. The rerun keeps its six declared ideas (P1–P3, M1–M3 in v201's `candidate_ledger.json`) and slots 1–6. It is not a retry after results. Done in PR #151: all six ran at 6M and 12M, none passed its gates, and v201 sent no finalist.
- **Runtime and typing.** Use Python 3.12.14 with the exact versions in v200's `runtime_lock.json`, in an isolated environment; never upgrade. CI runs mypy: run it on new `research_lib` code with the locked Python before pushing (v200's first CI run failed on six mypy errors).
- **Multiplicity slots.** The 38-test campaign family is numbered by study: v201 1–6 (used, no finalist), v202 7–14, v203 15–22, v204 23–28, v205 29–34 (used, no finalist), v206 35–38. Each study fills only its own slots; the others stay pending at p = 1 until v207 completes the Holm adjustment.
- **Consume v200's outputs.** Take the mature naive (prevailing-mean) forecasts, support and control forecasts from v200's pinned outputs; do not recompute them. v205's first attempt differed because it left out v200's pre-output history label (the 1999-11 origin).
- **Development dates in tests.** Tests, fixtures and CI smoke runs use an as-of date no later than 2022-08-31, v200's development bridge date, so no label they read reaches the quarantine boundary of 2023-09-29.
- **Carried to later steps:** v202 (A3 still open; PIF vintage), v204 (minimum support for the classifier lane) and v207 (VWO targets; finalists so far), in the step notes below.

## Preamble addendum — paste after the shared preamble

> *(D8, 2026-09-29.)* Verify v200's lock, SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`, with the add-only registry check (`verify_registry_growth` in `pgr_vds.research_lib.snapshot`), not a plain hash of `research/registry.yaml`. Do not edit the canonical plan, `docs/model-governance.md` or any other file pinned in that lock. Use Python 3.12.14 and the exact versions in v200's `runtime_lock.json`, and run mypy on new `research_lib` code before pushing. Fill only this study's multiplicity slots: v201 1–6, v202 7–14, v203 15–22, v204 23–28, v205 29–34, v206 35–38. Take mature naive forecasts, support and controls from v200's pinned outputs. Tests and smoke runs use an as-of date no later than 2022-08-31.

## Step notes — paste after the step's prompt

**v202:**

> *(D8.)* v205 used trailing-12M NPW growth without running the A3 test, so A3 (step 4) is still open here. `pif_total` differs from first-reported values in 50 months from 2019-07 (step 3b's single-definition rebuild, F11): PIF features are not a historical vintage, so disclose it as v205 did in its `source_vintage_audit.json`. Report against frozen v201, the rerun in `research/studies/v201_price_macro/outputs/attempt2/` (PR #151), which sent no finalist.

**v204:**

> *(D8.)* v200's same-label PathB control has only 72 raw and 35 calibrated scored dates. Before fitting, preregister a minimum-support rule that closes the lane when support is too thin, as v205 did for its annual lane, and report the support each gate is judged on.

**v207:**

> *(D8.)* v201 and v205 nominated no finalist, so at most four remain, one each from v202–v204 and v206. Complete the Holm adjustment from each study's slot ledger. The 12 VWO targets affected by the accepted March 2026 dividend gap, listed in v200's `outputs/vwo_accepted_gap.csv`, are all in the quarantine. Score them as stored, and preregister a sensitivity run that leaves them out, reported next to the primary result.
