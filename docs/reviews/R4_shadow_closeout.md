# R4 shadow closeout — v189 (2026-09-27)

Scope: R4 only, per the [amended execution plan](PRE_V200_FIX_PROMPTS_codex.md).
Read [AGENTS.md](../../AGENTS.md), [F24/F30](REPO_REVIEW_2026-09-25.md),
[independent verification V07](VERIFICATION_2026-09-26.md) and the
[second verification](VERIFICATION_2026-09-26_claude.md).
Historical CHANGELOG claims were not treated as adoption or verification.

## Pins and isolation

- Latest master fetched before implementation:
  `dc12291b0717368c04d7c233bac88d4477f0307d` (includes merged R1).
- Branch: `codex/R4-shadow-maturity`.
- Windows; test runtime CPython 3.12.14, pytest 9.0.2. Dependencies reused
  from the existing verifier installation through a path-only `.pth`; R4's
  external venv has its own editable scratch installation. No package download.
- External scratch clone (independent Git objects, `--no-hardlinks`):
  `C:\Users\Jeff\AppData\Local\Temp\pgr-R4-20260927\scratch`.
- Commands/logs/exits/hash records: same external parent, `runs.jsonl` and
  per-command `.log`/`.exit` files. The driver captures the exact commands,
  source/scratch DB SHA256 before and after, pytest summaries and exits.
- Source and scratch DB SHA256 before tests:
  `f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d`.
- Unrelated `.codex/` preserved. No DB or committed ledger replacement.

All R4 monitoring/repair regressions and the parity replay use synthetic
inputs and external temporary DBs/ledgers. Full-suite artifact tests and the
existing copied-DB dry-run test are offline smoke verification of already
inspected historical decisions, not new promotion evidence. No recent holdout
outcome was opened to debug R4 or evaluate a candidate. The Python startup
audit hook rejects socket connect/DNS in subprocesses. No provider calls or
real email; fetchers and decisions never ran against the tracked DB.

## Behavior and provenance

TA monitoring shares cached-target maturity logic with classifier monitoring.
Every TA row uses its issued feature anchor and horizon, the actual BME
endpoint and the evaluation as-of date. All configured basket constituents
must have finite cached realised outcomes. Calendar passage alone does not
establish an outcome. Backdated runs clear later stored outcomes. Issued
probability, stance, anchor, recipe, recommendation and candidate identity
remain unchanged; `(as_of_date, variant)` is the first-issuance key. New batch
duplicates are discarded; extra issuance metadata is retained. Existing
classifier calendar-maturity behavior remains unchanged for ordinary 6M rows.
Dry runs may compute monitoring in memory but persist neither ledger.

`shadow_followon.py` retains `reporting_only=True` and explicitly records
`forecast_source=baseline_shadow`, `candidate_application=metadata_only`,
`candidate_fitted=False`. Probability/stance and copied decision overlay are
baseline-derived. Candidate settings do not establish candidate application
or performance. No email content/format, validation gaps, model parameters,
feature recipes, raw-price/split/fractional-DRIP target construction or live
recommendations were changed.

Current-master inspection found Firth fitting/prediction only in
`src/research/v154_utils.py` and the archived v154 study; no active model/config
import, Firth reporting module, v159 implementation/closeout or distinct
monthly Firth prediction artifact substantiates the backlog's adoption claim.
The v159 file is an archived plan. Current backlog/registry/README now say
research-only pending v204 evidence, with a dated evidence note. Archived
conclusions and the independent verification were not rewritten or rerun.

## Regression evidence and commands

Test commands use `python -m pytest -o addopts="--tb=short" -q` with the explicit
external interpreter in the recorded driver. The initial project environments
lacked hypothesis/PyYAML; these startup failures were not defect evidence.
An initial full-suite attempt had 26 collection errors (exit 2) because the
verification driver's `PYTHONPATH` exposed `src/research` as `research`.
This was corrected only in the external environment; repository collection
and assertions were unchanged.

Before implementation, new TA/artifact/follow-on tests against original code:
`7 failed, 18 passed in 1.63s`, exit **1**. Named red tests:

- `test_ta_history_matures_on_actual_horizon_end`
- `test_ta_maturity_is_idempotent`
- `test_backdated_ta_run_ignores_later_outcomes`
- `test_ta_maturity_uses_each_rows_horizon`
- `test_ta_shadow_variant_history_preserves_issued_forecast_on_duplicate_key`
- `test_ta_entry_uses_bme_and_asof_not_wall_clock`
- `test_followon_records_baseline_derived_provenance`

The early-maturity and dry-run tests already passed; their guards were retained.
After the first repair, adding existing `test_shadow_layer.py` gave
`40 passed in 1.41s`, exit **0**. Before the copy-only repair script existed,
`-k "missing or repair"` gave `2 failed, 3 passed, 7 deselected in 1.22s`,
exit **1**, for the named copy-only/determinism and repository-path tests.

Final command results (external scratch cwd):

```powershell
python -m pytest -o addopts="--tb=short" -q tests/unit/models/test_ta_shadow_monitoring.py tests/unit/reporting/test_classification_artifacts.py tests/unit/reporting/test_shadow_followon.py tests/integration/pipeline/test_dry_run_read_only.py
python -m pytest -o addopts="--tb=short" -q tests/unit/models/test_ta_shadow_monitoring.py tests/unit/reporting/test_classification_artifacts.py tests/unit/reporting/test_shadow_followon.py tests/unit/models/test_shadow_layer.py tests/unit/reporting/test_shadow_current_use_docs.py
python -m pytest -o addopts="--tb=short" -q
python research/tools/registry.py
python scripts/checks/check_doc_links.py
```

| Check | Exact summary | Exit |
|---|---|---|
| Requested focused tests | `33 passed, 1 warning in 54.37s` | 0 |
| Expanded synthetic/contract tests | `45 passed in 1.56s` | 0 |
| Full suite | `1 failed, 2531 passed, 1 skipped, 111 warnings in 302.55s (0:05:02)` | 1 |
| Registry (including README regeneration) | `[registry] 113 studies, 0 problems` | 0 |
| Active doc links | `[doc-links] 348 files, 0 broken links` | 0 |
| Strict new/shared monitoring Python style (`ruff check --isolated --select E,W --line-length 79`) | `All checks passed!` | 0 |
| Import ordering (new/shared monitoring modules) | `All checks passed!` | 0 |
| `git diff --check` | no errors | 0 |

The full suite's sole failure is unchanged master
`tests/unit/tax/test_property_tax_boundaries.py::test_optimize_sale_orders_losses_then_ltcg_then_stcg`.
It is the same counterexample already documented in the
[R1 closeout](R1_safety_closeout.md): price 20, basis
`20.000000000000004`, shares `409.8506658027625`; rounded total-dollar gain
loses a tiny per-share loss and labels the lot LTCG/STCG. Both tax code and
the property test have zero diff from pinned master. A separate clean external
clone, `master_tax_clone`, at `dc12291b0717368c04d7c233bac88d4477f0307d`
reproduced the unchanged node with **`1 failed in 0.50s`**, exit **1**.
No saved Hypothesis counterexample was deleted, seed selected, test skipped
or unrelated tax fix added. The Windows full-suite exit-zero gate remains
unmet; this R4 PR stays draft pending that separately scoped repair.

Scratch counterfactuals keep current tests and reverse only the indicated
implementation/docs, then restore them. Each command uses the base pytest
command above with the named node/module. Latest recorded summaries:

| Reversal | Named test/module | Red / exit | Restored green / exit |
|---|---|---|---|
| Restore original `artifacts.py` | `test_ta_history_matures_on_actual_horizon_end` | `1 failed in 1.15s` / 1 | `1 passed in 1.03s` / 0 |
| Restore original backlog/registry/README | `tests/unit/reporting/test_shadow_current_use_docs.py` | `2 failed in 0.44s` / 1 | `2 passed in 0.40s` / 0 |
| Remove TA dry-run write guard | `test_dry_run_does_not_rewrite_ta_ledger` | `1 failed in 1.16s` / 1 | `1 passed in 1.01s` / 0 |
| Remove complete-basket gate | `test_ta_missing_basket_outcome_remains_unknown` (row/null/infinite) | `3 failed in 1.19s` / 1 | `3 passed in 1.04s` / 0 |
| Remove copy-only repository-path guard | `test_ta_repair_rejects_repository_paths` | `1 failed in 1.13s` / 1 | `1 passed in 1.00s` / 0 |

The last guard reversal fails before a DB open: the required early rejection
is absent and the missing external ledger raises `FileNotFoundError`. No
tracked DB is touched while intentionally reversing the safety guard.

Synthetic shadow replay command: `python <external>/synthetic_replay.py`,
exit **0**. Original and repaired ledger integration use the same January
2020 issuance, externally created SQLite targets and supplied forecast
fixtures; no model is fitted or candidate manufactured. AAA = -0.08 and
BBB = -0.04 give a hand-computed equal-weight return of -0.06 and sell
label 1 at the -0.03 threshold. July 30 is immature; July 31 is eligible
only with both finite outcomes. July 2020 and January 2021 BME dates are
independently specified in tests. A later outcome row cannot alter a July 30
backdated status; missing outcomes remain unknown even after calendar maturity.

Issued regression prediction **-0.02**, sell recommendation **0.75** and
TA probability **0.31** are identical in the before/after fixture replay;
all issued TA fields and the baseline-derived overlay are byte/frame-identical.
Only three monitoring cells change, as recorded in the committed
[synthetic row diff](R4_synthetic_ta_row_diff.csv). This is fixture parity and
monitoring smoke verification, not a production fit replay or promotion evidence.
The repair script independently reproduces the same synthetic repaired ledger
and diff while leaving both input files byte-identical.

Source and scratch DB hashes after the full/focused/expanded tests, every
reversal and baseline reproduction match the before hash:
`f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d`.
`git diff --exit-code -- data/pgr_financials.db artifacts/monthly_decisions`
returns **0**. The committed source ledgers remain byte-identical:

- Classifier history SHA256:
  `5dc860f0a87bff909466874a36faf81681be6c98ba4b678f18a56f6926d1951b`.
- TA history SHA256:
  `9252f5f82a27f1104b952605472051098e6bc6c73f9da7041380f6bec139516e`.

One docs/provenance subagent and one read-only reviewer were used. The
reviewer found no blocking R4 code/spec issue; their review is not the required
cross-model PR review. No performance improvement is established.

## Copy-only repair procedure

No committed ledger correction is applied in this PR. If a ledger correction
is explicitly approved later, copy its DB and CSV into an external directory,
pin their hashes and evaluation date, then run from an installed checkout:

```powershell
python -m scripts.repair_ta_shadow_history --db-copy C:/temp/r4/db.db --ledger-copy C:/temp/r4/ledger.csv --output C:/temp/r4/repaired.csv --row-diff C:/temp/r4/diff.csv --as-of 2020-07-31
```

All four paths must be distinct and outside Git checkouts, including resolved
symlink targets. The copied DB opens with `?mode=ro&immutable=1`; inputs are
never rewritten. Duplicate issuance keys fail for manual reconciliation.
Review the cell diff and input/output hashes before any later commit. Never
use a production decision to repair committed data. Use only synthetic or
pinned development inputs after v200 quarantine, never recent holdout outcomes.

## Remaining work

R4 must merge after R3. Rebase/merge latest master and resolve the shared
`artifacts.py` change after R3 lands, preserving both sessions. Cross-model
review remains required by the execution plan. v204 may use corrected TA
evidence only after R4 merges or must explicitly exclude the affected ledger.
Firth requires distinct fitted-model/prediction/code/data provenance in v204;
this PR supplies no promotion or investment-performance evidence.
