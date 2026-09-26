# Review 2026-09-25, step 11 — restructure phase 4: docs and tests layout (WP13)

Phase 4 of section 5 ("Repository structure review and tidy-up plan") of
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md), finding F32.
Phases 0–2 were step 7 ([report](2026-09-25_step7_restructure_phases_0_2.md))
and phase 3 was step 10 ([report](2026-09-25_step10_restructure_phase3.md)).

No production code path changed. The workflows install the same
dependencies from `pyproject.toml` instead of `requirements.txt`. Nothing
here touches the DB, a fetcher or `monthly_decision.py`, so there is no
replay to compare (steps 7 and 10 showed the replay is identical when only
files move).

The branch starts with a move-only commit (contents as on `master`), so
git pairs all 322 moved files at 100 % similarity and `git log --follow`
works across the move; the later commits make every edit. The move-only
commit does not pass the tests on its own (moved tests still climb to the
wrong repository root); review the commits together.

## 1. What moved

| From | To | Files |
|---|---|---:|
| `docs/plans/` | `docs/history/plans/` | 28 |
| `docs/superpowers/` (`plans/`, `specs/`) | `docs/history/superpowers/` | 49 |
| `docs/closeouts/` | `docs/history/closeouts/` | 32 |
| `docs/results/` | `docs/history/results/` | 24 |
| `docs/archive/` (with `history/`) | `docs/history/archive/` | 44 |
| `docs/repo-hygiene-review-2026-04-19.md` | `docs/history/` | 1 |
| `docs/gemini-prompts.txt` | `docs/history/archive/history/prompts/` (the move that audit proposed) | 1 |
| `claude.md` | `CLAUDE.md` (now a pointer to `AGENTS.md`) | 1 |
| `tests/test_*.py` | `tests/unit/<package>/`, `tests/integration/<kind>/`, `tests/research/` | 141 |
| `tests/guard_probe_wp12.py` | `tests/guard_probe.py` | 1 |

Deleted (git history keeps them): root `archive/` (14 scripts, 1 test, 3
READMEs; listed in [`docs/history/retired-code/README.md`](../history/retired-code/README.md)
with a `git show 282a6b3:…` line), `requirements.txt`,
`requirements-dev.txt`, `requirements-dashboard.txt` and
`constraints-dev.txt`.

New: [`docs/history/README.md`](../history/README.md) (index),
[`docs/decisions/`](../decisions/README.md) (index and seven records),
`tests/integration/repo/test_restructure_phase4.py`, 17 `__init__.py` files.

## 2. Docs history

- **One index.** `docs/history/README.md` lists every sub-folder with its
  file count, era and contents, says where each kind of record goes now
  (decisions, reviews, studies, CHANGELOG, plans and closeouts), and how to
  find things. `docs/history/plans/README.md` and `results/README.md` point
  at their new neighbours.
- **Links.** A one-off script resolved every relative link in every tracked
  Markdown file from the file's *old* location against the `master` tree,
  mapped the target through the moves, and wrote it back relative to the
  file's *new* location (104 files). Prose mentions of the old
  paths were updated in live docs (README, ROADMAP, operator docs, research
  READMEs, `research/registry.yaml`, the x1 study script); dated records
  (`docs/history/`, `docs/reviews/`, CHANGELOG) keep their wording and only
  had link targets rewritten.
- **Link checker.** `scripts/checks/check_doc_links.py` now checks
  `docs/decisions/*.md` and `docs/history/**/*.md`: 339 files, 0 broken
  (151 before this step, when the history trees were exempt). Checking the
  history trees exposed 13 links that were already broken on `master`
  (nine absolute `/Users/Jeff/.codex/worktrees/…` paths, two peer-review
  links one level too shallow, and two bare `ROADMAP.md`/`CHANGELOG.md`
  links in a spec); all 13 now resolve.

## 3. Decision records

`docs/model-governance.md` had a "Recent Promotion Record" list plus two
long sections for the step 5 gates and the step 6 mapping. Each promotion
decision is now one file in `docs/decisions/`:

| # | Decision | Source of the content |
|---|---|---|
| 0001 | Post-ensemble shrinkage (v38) | v37–v60 results summary, registry, v140/V152 closeout |
| 0002 | Quality-weighted consensus (v72 → v76) | v66–v73 plan, v74–v78 promotion cycle |
| 0003 | Stabilisation, `monthly_summary.json`, cross-check retired (v79–v86) | v79–v80 and v81–v88 plans |
| 0004 | Classifier stays shadow-only (v102–v117) | v102–v117 plan, v117 study output |
| 0005 | TA variants reporting-only (v160–v169) | v160–v164 plan, CHANGELOG v164–v169, V165–V169 closeouts |
| 0006 | Gates, realised-only health, CPCV diagnostic (step 5) | governance section and step 5 report (moved) |
| 0007 | ACTIONABLE sell mapping (step 6) | governance section and step 6 report (moved) |

Each has a header table (status, date, where it lives, study or findings,
report) and Context / Decision / Evidence / Consequences / Sources. 0001–0005
are reconstructed from the linked plans and closeouts, and say so; numbers in
them are quoted from those records, not recomputed. `model-governance.md`
keeps a Decision Record summary table, the gates and mapping in force (one
paragraph), the 2026-09-21 health baseline (the table later months are
compared against), and a Promotion Rule line requiring a new decision file
in the same PR as any promotion.

## 4. Root-level cleanup

- **`CLAUDE.md`.** `claude.md` was a byte-for-byte copy of `AGENTS.md`, and
  Claude Code looks for `CLAUDE.md` (F29). It is now a nine-line pointer that imports
  `@AGENTS.md` and point to CONTRIBUTING, so the two cannot drift.
- **`pyproject.toml`.** Step 7 had already folded `pytest.ini`, `mypy.ini`
  and `ruff.toml`. This step folds the requirements files: runtime
  dependencies were already listed identically in `[project]`; the `dev`
  extra now carries the constraints file's exact pins (`pytest==9.0.2`,
  `ruff==0.13.2`, `mypy==1.18.2`, `hypothesis==6.135.7`, `pyyaml==6.0.2`);
  the `dashboard` extra has Streamlit. All nine operational workflows run
  `pip install -e .` (was `pip install -r requirements.txt`), CI runs
  `pip install -e ".[dev]"`, and every `setup-python` step keys its pip
  cache on `pyproject.toml` (`cache-dependency-path`), since there is no
  `requirements.txt` for its default key.
- **Root `archive/`** is deleted rather than moved into `docs/history/`:
  it held Python code that no workflow or test ran, and keeping `.py` files
  under `docs/` would put them back in reach of ruff and the `sys.path`
  checker. Its 14 `sys.path` allowlist entries are gone.

## 5. Tests layout

| Folder | Files | What |
|---|---:|---|
| `tests/unit/backtest/` | 2 | `src/backtest` |
| `tests/unit/config/` | 1 | `config/` |
| `tests/unit/dashboard/` | 1 | `dashboard/` |
| `tests/unit/database/` | 5 | `src/database` |
| `tests/unit/ingestion/` | 10 | `src/ingestion` |
| `tests/unit/models/` | 24 | `src/models` |
| `tests/unit/portfolio/` | 5 | `src/portfolio` |
| `tests/unit/processing/` | 20 | `src/processing` |
| `tests/unit/reporting/` | 12 | `src/reporting` |
| `tests/unit/tax/` | 7 | `src/tax` |
| `tests/unit/scripts/` | 35 | the CLI and utility scripts in `scripts/` (most are `monthly_decision`, `edgar_8k_fetcher` and `weekly_fetch`) |
| `tests/integration/pipeline/` | 9 | several layers end to end: monthly pipeline, dry runs, entry-point imports, ops and workflow contracts, validation gating, mutation kills |
| `tests/integration/data/` | 3 | row-level integrity of the committed DB |
| `tests/integration/repo/` | 9 | repository contracts: docs, links, layout phases 0–4, workflows, test-suite hygiene |
| `tests/research/` | 119 | the 118 study tests from step 10, plus `test_v15_setup.py` (`src/research/v15.py`) |

A test's folder is its primary target. A test that imports one `src`
package (plus `config` or a DB fixture) is a unit test of that package; a
test of a script is under `unit/scripts/`, since the review's target layout
turns `scripts/` into `cli/` (phase 5); a test that crosses layers, reads
the committed DB for its assertions or checks the repository is an
integration test.

**Renamed after the module they test** (32 files, one split, one helper):

| Old (`tests/`) | New (`tests/`) |
|---|---|
| `test_v62_schema_and_csv.py` | `unit/database/test_pgr_edgar_monthly_schema_and_csv.py` |
| `test_live_mapping_wp8.py` | `unit/models/test_live_policy_backtest.py` |
| `test_v50_ensemble.py` | `unit/models/test_multi_benchmark_wfo_ensemble.py` |
| `test_shadow_layer_wp8.py` | `unit/models/test_shadow_layer.py` |
| `test_v74_guards.py` | `unit/models/test_cpcv_and_obs_ratio_guards.py` |
| `test_v27_redeploy_portfolio.py` | `unit/portfolio/test_redeploy_portfolio.py` |
| `test_edgar_filing_timing_wp10.py` | `unit/processing/test_edgar_filing_timing.py` |
| `test_v45_features.py` | `unit/processing/test_feature_engineering_macro_predictors.py` |
| `test_v60_features.py` | `unit/processing/test_feature_engineering_high_52w_and_peers.py` |
| `test_v63_channel_mix_features.py` | `unit/processing/test_feature_engineering_channel_mix.py` |
| `test_v64_p2x_features.py` | `unit/processing/test_feature_engineering_pgr_fundamentals.py` |
| `test_price_features_wp2.py` | `unit/processing/test_price_features.py` |
| `test_reporting.py` | `unit/reporting/test_backtest_report.py` |
| `test_v12_shadow.py` | `unit/reporting/test_snapshot_summary.py` |
| `test_tax_wp8.py` | `unit/tax/test_tax_hand_computed.py` |
| `test_bl_fallback_monthly.py` | `unit/scripts/test_monthly_decision_bl_fallback.py` |
| `test_diagnostic_report.py` | `unit/scripts/test_monthly_decision_diagnostic_report.py` |
| `test_dividend_feed.py` | `unit/scripts/test_weekly_fetch_dividend_feed.py` |
| `test_fred_pgr_refresh_and_nan_live_features.py` | `unit/scripts/test_weekly_fetch_fred_pgr_refresh.py` |
| `test_monthly_ensemble_alignment.py` | `unit/scripts/test_monthly_decision_ensemble_alignment.py` |
| `test_monthly_logging.py` | `unit/scripts/test_monthly_decision_logging.py` |
| `test_monthly_report_tax.py` | `unit/scripts/test_monthly_decision_report_tax.py` |
| `test_policy_backtest_monthly.py` | `unit/scripts/test_monthly_decision_policy_backtest.py` |
| `test_v13_recommendation_layer.py` | `unit/scripts/test_monthly_decision_recommendation_layer.py` |
| `test_v813_recommendation_mode.py` | `unit/scripts/test_monthly_decision_recommendation_mode.py` |
| `test_integration.py` | `integration/pipeline/test_engine_smoke.py` |
| `test_fred_pipeline_wp3.py` | `integration/pipeline/test_fred_pipeline.py` |
| `test_mutation_kills_wp12.py` | `integration/pipeline/test_mutation_kills.py` |
| `test_ops_wp8.py` | `integration/pipeline/test_ops_contracts.py` |
| `test_v129_dual_track_integration.py` | `integration/pipeline/test_v129_feature_map_dual_track.py` |
| `test_validation_gating_wp7.py` | `integration/pipeline/test_validation_gating.py` |
| `test_wp12_test_hygiene.py` | `integration/repo/test_test_suite_hygiene.py` |
| `test_v65_p26_p27_p28.py` | split: `unit/scripts/test_edgar_8k_fetcher_html_exhibit.py`, `unit/scripts/test_monthly_decision_calibration_plot.py`, `unit/reporting/test_email_sender.py` |
| `guard_probe_wp12.py` (helper) | `guard_probe.py` |

`test_v129_feature_map.py` keeps its name: the module is
`src/models/v129_feature_map.py`. The study tests in `tests/research/`
keep theirs (see Judgement calls).

**Code the move needed.**

- **Repository root.** Moved files sit two folders deeper: every
  `Path(__file__).resolve().parents[1]` / `.parent.parent` became
  `parents[3]`, the two `os.path.dirname` chains gained two levels, and the
  live-mapping fixture path is `parents[2] / "fixtures"`.
- **`sys.path`.** 14 moved tests inserted the repository root (or
  `scripts/`) into `sys.path`; the package is installed and
  `tests/conftest.py` already adds the root, so the inserts were removed
  (and `import sys` where it was then unused). The split calibration-plot
  test imports `scripts.monthly_decision` instead of `monthly_decision`.
- **Cross-test imports.** Three helpers imported from other test files now
  use the full path (`tests.unit.scripts.test_edgar_8k_parser_breadth`,
  `tests.unit.scripts.test_monthly_decision_policy_backtest`,
  `tests.integration.pipeline.test_engine_smoke`).
- **Packages.** Every test folder has an `__init__.py`, so same-named files
  in different folders cannot clash under pytest's default import mode.
- **Markers.** `tests/conftest.py` adds `unit`, `integration` or `research`
  to each item from its first folder under `tests/`
  (`pytest_collection_modifyitems`, `tryfirst` so the markers exist before
  `-m` deselects). The markers are described in `pyproject.toml`.
- **CI.** `ci.yml` has three jobs: `test`
  (`-m "not artifact and not research"`, plus lint, checks, mypy and the
  offline smoke runs), a new `research` job
  (`-m "research and not artifact"`), and `artifacts` (`-m artifact`).
- **Mutation study.** `scripts/checks/mutation_study.py` lists its test
  files by path under `tests/`. It used to filter the list with
  `.exists()`, so a moved file was silently skipped: since step 10 moved
  `test_research_v72_quality_weighted_consensus.py` into `tests/research/`,
  a re-run of the three consensus mutations (M10–M12) would have silently left it out. Their step 9 kills came from `test_mutation_kills_wp12.py`, which was still found, so the recorded result stands. It now
  stops with an error if a listed file is missing, and
  `test_mutation_study_names_existing_test_files` checks all 45.
- **References.** Every live mention of a moved test (docs, `dashboard/data.py`,
  `src/processing/pgr_edgar_validation.py`, `scripts/generate_edgar_data_dictionary.py`
  and the dictionary it writes, `monthly_decision.yml` comments, the
  allowlist header) uses the new path.

**Nothing lost.** `pytest --collect-only` on `master` and on this branch
collect the same 2,511 test names, apart from the two phase-0/hygiene tests
rewritten for the new layout (`test_requirements_txt_matches_pyproject_dependencies`
→ `test_pyproject_is_the_only_dependency_list`,
`test_root_archive_is_labeled_as_code_archive` →
`test_retired_code_index_replaces_root_archive`); the branch adds the 26
phase-4 tests (2,537). The three CI selections partition the suite exactly:

| CI job | Selection | Tests |
|---|---|---:|
| `test` | `-m "not artifact and not research"` | 1,989 |
| `research` | `-m "research and not artifact"` | 350 |
| `artifacts` | `-m artifact` | 198 |
| | total | 2,537 |

By layer marker: `unit` 1,687, `integration` 334, `research` 516.

## 6. Tests: failing before, passing after

`tests/integration/repo/test_restructure_phase4.py` (26 tests), run on a
clean `master` worktree (282a6b3, with the file copied in) and on this
branch:

| Test | `master` | branch |
|---|---|---|
| `test_history_tree_moved_under_docs_history[plans / superpowers / closeouts / results / archive]` (5) | fail | pass |
| `test_history_keeps_sub_folders` | fail | pass |
| `test_history_index_links_every_sub_folder` | fail | pass |
| `test_link_checker_covers_history_and_decisions_and_passes` | fail | pass |
| `test_no_live_file_points_at_the_old_history_paths` | fail | pass |
| `test_one_decision_file_per_promotion` | fail | pass |
| `test_governance_keeps_a_summary_table_of_every_decision` | fail | pass |
| `test_claude_md_points_to_agents_md` | fail | pass |
| `test_pyproject_is_the_only_dependency_and_tool_config` | fail | pass |
| `test_every_workflow_installs_from_pyproject` | fail | pass |
| `test_root_archive_is_retired` | fail | pass |
| `test_no_test_files_left_at_the_top_of_tests` | fail | pass |
| `test_unit_tests_mirror_the_code` | fail | pass |
| `test_integration_and_research_folders` | fail | pass |
| `test_every_test_folder_is_a_package` | fail | pass |
| `test_no_version_numbered_test_files_outside_research` | fail | pass |
| `test_moved_tests_resolve_the_repo_root` | fail | pass |
| `test_layer_markers_are_applied_by_folder` | fail | pass |
| `test_this_file_carries_the_integration_marker` | fail | pass |
| `test_ci_runs_research_tests_in_their_own_job` | fail | pass |
| `test_mutation_study_names_existing_test_files` | fail | pass |
| `test_contributing_and_docs_map_describe_the_new_layout` | fail | pass |
| **Total** | **26 failed** | **26 passed** |

```text
master: 26 failed in 0.79s
branch: 26 passed in 0.95s
```

Two tests passed vacuously on `master` in a first draft (nothing to scan
there); they now also require the new folders, so all 26 fail before.

Updated for the new layout: `test_docs_hygiene.py` (docs map sections,
history READMEs, retired-code index replaces the root `archive/` README),
`test_restructure_phase0.py` (no `requirements.txt`; CI installs
`-e ".[dev]"`), `test_restructure_phase3.py` (skips the moved phase tests
by their new path) and `test_test_suite_hygiene.py` (guard probe path, CI
selection, seed helper import).

Repository checks on the branch: `ruff check .` all passed;
`check_doc_links.py` 339 files, 0 broken; `check_sys_path_edits.py`
0 new, 0 stale; `research/tools/registry.py` 113 studies, 0 problems.

## 7. Full suite

`python -m pytest -o addopts="--tb=short" -q` on this branch:

```text
2536 passed, 1 skipped, 105 warnings in 730.93s (0:12:10)
```

`master` (step 10's report): `2510 passed, 1 skipped`. The difference is
the 26 new phase-4 tests.

## Judgement calls

- **Unit folders mirror today's `src/`**, not the target `src/pgr_vds/`:
  the package directories have not been renamed yet (the review's target is
  `src/pgr_vds/<package>/`). When they are, `tests/unit/<package>/` keeps
  the same names.
- **Script tests are unit tests** (`tests/unit/scripts/`) when they test
  functions of one script; the review's phase 5 moves that logic into
  `src/pgr_vds/decision/` and `ingestion/edgar_monthly/`, and the tests can
  follow it.
- **Study tests keep their names.** `tests/research/test_research_v38_shrinkage.py`
  and the like are named after their study (`research/studies/v38_shrinkage/`),
  and the naming rule allows study ids there. Dropping the redundant
  `test_research_` prefix would touch 118 files and 109 study READMEs for
  no functional gain; it can be done with the study-registry tooling later.
- **`_wpN` names count as numbered.** Besides the `test_vNN_*` files, the
  review's work-package suffixes (`test_tax_wp8.py`, `test_ops_wp8.py`, …)
  were renamed too, because they say when a test was written, not what it
  tests. `test_v129_feature_map*.py` keep the version: it is the module's
  name.
- **The mixed v6.5 file was split**, not renamed: it tested the EDGAR HTML
  parser, a `monthly_decision` plot and the e-mail sender, so no one name
  fits. The 37 tests are unchanged (37 before, 37 after).
- **Root `archive/` is deleted**, not moved (section 4).
- **The hygiene audit and the prompts file** moved into `docs/history/`
  although the brief does not name them: the audit is a dated record whose
  recommendations this step completes, and it had proposed that exact move
  for the prompts file.
- **Decision records 0001–0005 are reconstructions.** They quote the plans,
  closeouts and study outputs of their cycle and link them; they add no new
  analysis. Their metrics predate the review's corrections (F04, F13) and
  say so where they are quoted.
- **Dated records keep their wording.** Only link targets were rewritten in
  `docs/history/`, `docs/reviews/` and CHANGELOG; prose that names an old
  path there still describes the repository as it was.

## Not done here (later steps)

- Phase 5: split `scripts/monthly_decision.py` into `src/pgr_vds/decision/`
  and merge the two `edgar_8k_fetcher.py` files; rename `src/` to
  `src/pgr_vds/` (the unit-test folders already use the package names).
- `ruff` still checks only fatal errors (F29, PEP 8).
- The WP11 re-runs flagged in the registry and in decision 0001.
- 199 allowlisted `sys.path` edits remain: 111 study scripts under
  `research/`, their 54 tests in `tests/research/`, and 34 scripts under
  `scripts/`. No test outside `tests/research/` edits `sys.path` any more.
