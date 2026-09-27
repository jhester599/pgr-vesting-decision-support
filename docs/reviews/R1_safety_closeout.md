# R1 safety closeout — v186 (2026-09-27)

Scope: R1 only, including the approved N6, ignored-force and CI-permissions
additions in the [execution plan](PRE_V200_FIX_PROMPTS_codex.md).
Evidence read: [AGENTS.md](../../AGENTS.md),
[independent verification V01/V02/V09](VERIFICATION_2026-09-26.md),
[original F26/F28](REPO_REVIEW_2026-09-25.md), and the
[second verification's N6](VERIFICATION_2026-09-26_claude.md).
Historical CHANGELOG statements were not treated as verified results.

## Pins and isolation

- Latest master fetched before implementation:
  `449fcdecee9abfaa8114f9cdee63ad2c7ffb5cc4`.
- Branch: `codex/R1-offline-safety`.
- Source checkout:
  `C:\Users\Jeff\.codex\worktrees\189a\pgr-vesting-decision-support`.
- External clone (independent Git objects and DB bytes, no hardlinks):
  `C:\Users\Jeff\AppData\Local\Temp\pgr-R1-20260927\scratch with spaces`.
- Evidence directory:
  `C:\Users\Jeff\AppData\Local\Temp\pgr-R1-20260927`.
- Windows, CPython 3.12.14; pytest 9.0.2, pandas 3.0.6, numpy 2.5.3,
  scikit-learn 1.9.1, scipy 1.18.1, skfolio 1.4.0, xgboost 3.4.1,
  matplotlib 3.11.2, hypothesis 6.135.7, PyYAML 6.0.2.
  An isolated venv reuses the previous verifier's installed dependency
  directory via a path-only `.pth`; it has its own editable scratch install.
  Runtime dependency ranges remain unpinned.

Source and scratch DB SHA256 before reproduction:
`f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d`.
The DB is 5,808,128 bytes. The historical verifier hash is different; no
historical DB was substituted. No DB migration or replacement is in this PR.
The unrelated source `.codex/` directory was preserved.

Tests use synthetic inputs, external temporary DBs or the clone's private
DB copy. A subprocess-inherited `sitecustomize.py` audit hook refuses
`socket.connect` and `socket.getaddrinfo`; no live provider call or real
email is possible through those Python sockets. The missing-report email
test exits before delivery. No provider bootstrap workflow was dispatched.
Allowed committed-DB SQLite probes use `?mode=ro&immutable=1`.

`run_checks.py` records commands, cwd, platform, exit, pytest summary and
source/scratch DB/sidecar hashes, sizes and modification times in
`runs.jsonl`, with per-run `.log` and `.exit` files. These local logs are
supplementary; the commands and results below are the committed evidence.
Nested probe children intentionally return 1; their parent requires that
exit and compares the complete expected-outcome dictionary.

## Original red reproduction

Before implementation or assertion changes, in the external clone:

```powershell
python -m pytest -o addopts="--tb=short" -q 'tests/integration/repo/test_test_suite_hygiene.py::test_classify_access[sqlite3.connect-args11-False-True]' 'tests/integration/repo/test_test_suite_hygiene.py::test_guard_fails_exactly_the_probes_that_touch_the_repo' 'tests/integration/repo/test_restructure_phase1.py::test_email_reads_report_and_charts_from_artifacts'
```

`original-red.log`: **`3 failed in 1.60s`**, exit **1**.
The classifier returned `None` for an unmarked percent-encoded Windows
`Path.as_uri()` DB read. The child showed the unmarked URI probe passing.
The email reader raised the correct artifact path with backslashes, which
the slash-only regex rejected. This runtime printed slash node IDs in the
child; the added synthetic backslash parser test separately proves the
path-separator defect observed in V02.

New regressions against unrepaired implementation/workflows:

```powershell
python -m pytest -o addopts="--tb=short" -q tests/integration/repo/test_test_suite_hygiene.py tests/integration/repo/test_workflow_contracts.py tests/unit/scripts/test_capital_return_charts.py
```

`added-regressions-red.log`: **`11 failed, 42 passed in 7.77s`**, exit **1**.
Failures include the URI policy, authority handling, memory URI,
backslash parser, actual summary SQL, CI permissions/job and ignored force.
The market-cap validator already worked; its new test's red is the explicit
raise-removal counterfactual below.

## Changes and independently specified expectations

- Use platform `url2pathname` conversion and native absolute/case
  normalization. Preserve URI authorities as UNC/double-slash paths;
  `/C:/...` remains a POSIX absolute path on POSIX. Test native, case,
  relative, `file:` relative/native, localhost, `Path.as_uri()`, encoded
  spaces, `mode=ro&immutable=1`, memory and genuine external paths.
  Unmarked DB access is refused; artifact read-only is allowed; artifact
  read-write is refused. Existing swallowed-write detection is retained.
  Repeated SQLite `mode` parameters fail closed; SQLite's later mode can
  override an earlier `memory`/`ro`. URI/memory interpretation applies only
  to SQLite events, never native file operations. A synthetic DB containing
  integer 23 independently demonstrates the repeated-mode disk read.
- Normalize child node separators before parsing. A passed call followed
  by teardown error remains failed, even if another pass line follows.
  Retain the full probe comparison, residue and sidecar checks. Add actual
  native-case, relative, swallowed-read and artifact read-write URI probes.
- Extract and normalize the exception's missing path and compare it to the
  literal required artifact path. Changing the reader to old `results/`
  fails this assertion. No email content or format changes.
- Execute the workflow's actual `Bootstrap summary` Python heredoc against
  an external synthetic DB built from `src/database/schema.sql`. Two
  deliberately out-of-order dates, January 3 and February 7, 2020, require
  `2 prices (from 2020-01-03), 1 dividends` for each of the four peers.
  Repair only the SQL column from `price_date` to `date`.
- Remove the ignored initial-fetch `force` Python parameter, CLI option,
  dispatch input and flag plumbing. Test CLI rejection before any fetch
  runs and inspect both script and workflow. Loader freshness/quota logic
  stays unchanged; active workflow docs describe the removed no-op option.
- The N6 frame directly supplies $10 and 600 million same-basis shares,
  so market cap must be $6,000 million. A supplied $6,001 million must
  raise `ValueError`; no frame-builder output supplies the expectation.
- CI has top-level `contents: read` and Python 3.12 Windows regressions.
  Linux unit/integration, research and artifacts remain separate. Existing
  network-mocked entrypoint smoke commands now run after copying the
  checkout to `mktemp` and installing that external copy editable. Copy
  preparation has its own step; the four-command network-guard contract
  remains unchanged. The smoke working directory/PYTHONPATH select the copy.

## Counterfactual reversals

Each command below has the prefix
`python -m pytest -o addopts="--tb=short" -q` and runs in the external clone.
`counterfactuals.py` restores one defect, runs the named tests, restores the
fixed bytes in `finally`, then reruns the same tests. All red exits are 1;
all restored green exits are 0. A slash-only parser is separately reversed
from URI normalization, so one repair cannot conceal the other defect.

| Reintroduced defect / named test suffix | Red summary | Restored green summary |
|---|---|---|
| Original `tests/repo_guard.py`; hygiene `::test_committed_db_spellings_obey_read_policy`, `::test_file_uri_preserves_authority_and_platform_drive_semantics`, `::test_guard_fails_exactly_the_probes_that_touch_the_repo` | `4 failed in 1.39s` | `4 passed in 1.31s` |
| Remove separator normalization; hygiene `::test_probe_parser_preserves_teardown_failure` | `1 failed, 1 passed in 0.42s` | `2 passed in 0.33s` |
| Restore original slash-only assertion; phase1 `::test_email_reads_report_and_charts_from_artifacts` | `1 failed in 0.43s` | `1 passed in 0.35s` |
| Reader uses old `results/monthly_decisions`; same phase1 node | `1 failed in 0.42s` | `1 passed in 0.34s` |
| Restore `MIN(price_date)`; workflows `::test_peer_bootstrap_summary_uses_existing_date_column` | `1 failed in 0.56s` (`no such column: price_date`) | `1 passed in 0.48s` |
| Remove market-cap `raise`; charts `::test_verify_monthly_frame_rejects_inconsistent_market_cap` | `1 failed in 0.68s` (`DID NOT RAISE`) | `1 passed in 0.57s` |
| Restore old script/workflow force; workflows `::test_initial_fetch_removes_ignored_force_option` | `1 failed in 0.46s` | `1 passed in 0.94s` |
| Original CI; workflows `::test_ci_has_read_only_permissions`, `::test_ci_runs_windows_safety_regressions_on_python312` | `2 failed in 0.46s` | `2 passed in 0.39s` |
| First-mode-only interpretation; hygiene `::test_repeated_sqlite_modes_cannot_hide_committed_db_access` | `4 failed in 0.43s` | `4 passed in 0.35s` |
| Apply URI-memory interpretation to native files; hygiene `::test_native_file_writes_do_not_get_sqlite_memory_exemption` | `2 failed in 0.40s` | `2 passed in 0.33s` |
| Original CI smoke cwd; workflows `::test_ci_entrypoint_smokes_use_an_external_checkout_copy` | `1 failed in 0.47s` | `1 passed in 0.39s` |
| Restore only workflow's ignored force input/plumbing; workflows `::test_initial_fetch_removes_ignored_force_option` | `1 failed in 0.98s` | `1 passed in 0.92s` |

Abbreviations refer to these exact files:

- hygiene: `tests/integration/repo/test_test_suite_hygiene.py`
- phase1: `tests/integration/repo/test_restructure_phase1.py`
- workflows: `tests/integration/repo/test_workflow_contracts.py`
- charts: `tests/unit/scripts/test_capital_return_charts.py`

CI external-smoke-copy test before its change:
`python -m pytest -o addopts="--tb=short" -q tests/integration/repo/test_workflow_contracts.py::test_ci_entrypoint_smokes_use_an_external_checkout_copy`
gave **`1 failed in 0.48s`**, exit **1** (`ci-smoke-copy-red.log`).

## Verification results and limits

Final targeted Windows commands, cwd set to the external clone:

```powershell
python -m pytest -o addopts="--tb=short" -q tests/integration/repo/test_test_suite_hygiene.py tests/integration/repo/test_restructure_phase1.py tests/integration/repo/test_workflow_contracts.py
python -m pytest -o addopts="--tb=short" -q tests/unit/scripts/test_capital_return_charts.py tests/unit/scripts/test_initial_fetch_logging.py
```

Exact summaries: **`63 passed in 7.79s`**, exit **0**;
**`12 passed in 1.88s`**, exit **0**. No target was skipped.
The internal reviewer independently passed 40 pure Windows classification
assertions, exit 0, using never-opened synthetic paths. Both guard review
findings were repaired and reversed red/green before this final run.

An initial exact full-suite run gave **`1 failed, 2507 passed, 1 skipped,
141 warnings in 331.34s (0:05:31)`**, exit **1**. The failure is
`tests/unit/tax/test_property_tax_boundaries.py::test_optimize_sale_orders_losses_then_ltcg_then_stcg`.
It is outside R1: unchanged master tax code compares per-share loss while
sorting, but subtracts rounded total dollars while labeling sale results.
At price 20 and basis `20.000000000000004` with `409.8506658027625` shares,
the negative per-share difference rounds away in the total-dollar subtraction.
The lot is classified as LTCG/STCG rather than LOSS, violating the existing
property. The optimizer and property test have zero diff against pinned
master. Repeating the unchanged node with the retained Hypothesis examples
gave **`1 failed in 0.43s`**, exit **1** (`unchanged-master-tax-red.log`).
No saved counterexample was deleted and no seed was selected for a green run.
The owner was asked whether to extend scope; until authorized, this remains
an explicitly documented exit-gate blocker.

The next exact full-suite run gave **`2 failed, 2514 passed, 1 skipped,
127 warnings in 326.65s (0:05:26)`**, exit **1**. Alongside the unchanged
tax counterexample, `test_ci_smoke_tests_run_with_the_network_mocked` caught
the copy-preparation commands inserted into a step whose existing contract
requires exactly four guarded smoke commands. Copy preparation was moved
to a separate step, with the copy selected through `working-directory`
and `PYTHONPATH`. The existing test was retained unchanged.
That node plus `test_ci_entrypoint_smokes_use_an_external_checkout_copy`
gave **`2 failed in 0.53s`**, exit **1**, before the adjustment and
**`2 passed in 0.46s`**, exit **0**, afterward. Command:
`python -m pytest -o addopts="--tb=short" -q tests/integration/repo/test_workflow_contracts.py::test_ci_entrypoint_smokes_use_an_external_checkout_copy tests/integration/pipeline/test_ops_contracts.py::test_ci_smoke_tests_run_with_the_network_mocked`.

The final exact full-suite rerun and branch CI outcomes are recorded below
when those executions finish. PR [#138](https://github.com/jhester599/pgr-vesting-decision-support/pull/138)
is a draft because the full-suite exit gate is not met. R2 has not started.

Checks already run:

- `ruff check .`: `All checks passed!`, exit 0 (repository's existing
  syntax/fatal-lint selection; new code uses annotations and PEP 8 layout).
- `python scripts/checks/check_doc_links.py`: `[doc-links] 347 files,
  0 broken links`, exit 0.
- `git diff --check`: exit 0.
- `Get-FileHash -Algorithm SHA256 -LiteralPath data/pgr_financials.db`:
  unchanged `f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d`.
  Every recorded test/reversal compares source and scratch hash, size and
  modification time, including sidecar lists. No WAL/SHM/journal sidecar
  existed before or appeared afterward; all 51 runs then recorded matched.

Linux was unavailable locally (`wsl --list --quiet`: WSL not installed).
The fetched master's [existing CI run](https://github.com/jhester599/pgr-vesting-decision-support/actions/runs/36318626284)
had three successful Linux jobs, with summaries copied from its actual logs:
`1946 passed, 1 skipped, 548 deselected, 71 warnings in 247.62s (0:04:07)`
(unit/integration), `350 passed, 2145 deselected, 1 warning in 14.28s`
(research), and `198 passed, 2297 deselected, 28 warnings in 202.61s (0:03:22)`
(artifacts). Those are baseline remote results, not local or R1 branch runs.

An early targeted invocation mistakenly used source cwd: `67 passed in
9.64s`, exit 0. It is excluded from copy-only verification evidence. The
only source SQLite inspection was the allowed immutable read-only artifact
probe; read-write attempts were refused and source hash/sidecars remained
unchanged. The suite was rerun in the external clone. Its first attempt
gave `1 failed, 66 passed in 9.10s`, exit 1 because Git refused the
clone's ownership; a process-scoped exact `safe.directory` fixed that
environment error. The subsequent external run gave `67 passed in 8.82s`,
exit 0. No safety assertion was changed for this environment issue.

No new skip, fetcher replay, investment backtest, feature/model/validation
parameter change or historical report edit belongs to R1. The existing
unconditional classification-shadow integration skip is not a Windows
workaround. No historical/recent outcome is new promotion evidence.

The audit hook is **not a complete filesystem sandbox**. It activates
during test setup/call/teardown, not import or collection; C-extension and
subprocess filesystem operations are not generally intercepted. It cannot
prove every filesystem alias or SQLite URI-flag interpretation safe. Keep
external copies mandatory for deliberate probes, migrations and replays.
Passing tests do not establish an investment-performance improvement.
Cross-model Claude review and merge remain required before R2.
