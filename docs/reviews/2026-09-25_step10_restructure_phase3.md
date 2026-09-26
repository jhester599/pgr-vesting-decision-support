# Review 2026-09-25, step 10 — restructure phase 3: one folder per study (WP13)

Phase 3 of section 5 ("Repository structure review and tidy-up plan") of
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md), finding F32 and
the research-output part of F30. Phases 0–2 were step 7
([report](2026-09-25_step7_restructure_phases_0_2.md)). Every move uses
`git mv`, so `git log --follow` shows each file's history from its old path.

The goal was no production behaviour change. The September 2026 replay is
identical before and after (section 6).

## 1. What moved

| From | To | Files |
|---|---|---:|
| `results/research/<id>_*.py`, `scripts/research/x*.py`, `scripts/research/pb_vs_pe_analysis.py`, the five root `scripts/*experiments*.py` | `research/studies/<id>_<slug>/` | 113 |
| `results/research/<id>_*` (outputs), `results/research/pb_vs_pe/` | `research/studies/<id>_<slug>/outputs/` | 259 |
| `results/research/summarize_all.py` (cross-study v37–v60 table) | `research/tools/summarize_v37_v60.py` | 1 |
| `tests/test_research_*.py`, `tests/test_v*_research.py`, `test_bl01_sweep.py`, `test_pb_vs_pe.py`, the five `test_*_experiments.py` | `tests/research/` (file names unchanged) | 118 |
| `results/v9/` … `results/v28/` (16 folders) | `research/legacy/v9/` … `research/legacy/v28/`, unchanged | 214 |

That is 705 renames. `git diff -M` pairs 704 of them at its default 50 %
similarity; the 705th, `tests/research/test_research_v58_fred.py`, is a
five-line file whose one path line changed, and pairs at `-M20%`.

**113 studies.** 110 numbered ids from `results/research/` and
`scripts/research/` (v37–v165, x1–x24, bl01), plus three:

- `v9_experiments/`: the five root `*experiments*` scripts (v9.2 features,
  v9.3 targets, v9.7 pooled benchmarks, confirmatory classifiers, weekly
  snapshots). Their committed outputs are in `research/legacy/v9/`; new runs
  write to `outputs/` in the study folder.
- `pb_vs_pe_analysis/`: the unnumbered P/B vs P/E study.
- `test_runtime_autoresearch/`: the two `v_test_runtime_*` outputs.

Studies were moved in the order the brief asked (the studies behind live
settings — v38, v72/v75 for v76, v128/v129, v131/v132, v134, v142 — then the
x-series, then the rest), in one commit, since the order does not change the
result.

**Folder names.** `<id>_<slug>`, where the slug is the rest of the (first)
script name, so `results/research/v38_shrinkage.py` becomes
`research/studies/v38_shrinkage/v38_shrinkage.py`. Studies with outputs but
no script get a slug from their outputs: `v46_classification` (the script is
`src/research/binary_classification.py` since step 7), `v136_backlog_ranking`,
`v158_synthesis`, `v164_ta_synthesis`.

**Removed.** `results/research/__init__.py` (the `results.research` package)
and `.gitkeep`. `results/` now holds only `backtests/` (10 CSVs written by
`scripts/feature_ablation.py`) and the gitignored `results/dry_run/`. It
contains no `.py` file.

## 2. Code changes the move needed

- **Imports.** `results.research.<m>`, `scripts.research.<m>` and
  `scripts.<name>_experiments` became `research.studies.<folder>.<m>` in
  study scripts and tests, including `patch("…")` targets. `research/` and
  `research/studies/` are namespace packages (no `__init__.py`), imported
  from the repository root like `tests.*`.
- **Repository root.** Study scripts sit one level deeper
  (`Path(__file__).resolve().parents[3]`; was `[2]`), and so do moved tests
  (`parents[2]`; was `[1]`). The v9 scripts used a chain of
  `os.path.dirname` and now use `parents[3]`.
- **Output paths.** `src/research/study_paths.py` (new) finds a study's
  folder from the id prefix of an output name:
  `study_output_path("v38_shrinkage_best_results.csv")` →
  `research/studies/v38_shrinkage/outputs/…`; `detail=True` returns the
  `outputs/detail/` path. `v37_utils.save_results`, `v102_utils` and every
  former `RESULTS_DIR / "<name>"` use it, so a study that reads another
  study's output (v118 → v113, v127/v130/v131 → v125, x24 → x8/x11/x16/x23, …)
  finds it in that study's folder. `v37_utils.RESULTS_DIR` is gone, so a
  missed reference fails loudly. Scripts that wrote to a local
  `OUTPUT_DIR = Path("results") / "research"` now write to their own
  `outputs/`.
- **`sys.path` allowlist.** 165 entries (study scripts and moved tests)
  renamed to the new paths; the count
  is unchanged and `check_sys_path_edits.py` reports 0 new, 0 stale.
- **Other:** `src/research/v15.py` reads the v14 summary from
  `research/legacy/v14/`; the x24 memo's four Windows-absolute prompt links
  are now relative; `research/tools/summarize_v37_v60.py` globs the study
  folders.

## 3. Production reads of study outputs

Production imports no study code. Three modules read committed study
outputs; each path is now a named constant, and
`tests/test_restructure_phase3.py` checks the files exist:

| Module | Constant | File |
|---|---|---|
| `config/features.py` (used by `src/models/v129_feature_map.py`) | `V128_BENCHMARK_FEATURE_MAP_PATH` | `research/studies/v128_benchmark_feature_search/outputs/v128_benchmark_feature_map.csv` |
| `src/models/classification_gate_overlay.py` | `DEFAULT_OVERLAY_RESULTS_PATH` (new) | `research/studies/v113_constrained_candidate_selection/outputs/v113_constrained_candidate_selection_results.csv` |
| `src/reporting/shadow_followon.py` | `FOLLOWON_CANDIDATE_PATHS` (new) | the v141, v143, v144, v149 and v150 candidate files |

`resolve_overlay_policy_variant` falls back to built-in defaults when its
file is missing, so a wrong path would not raise. It now resolves
`gemini_veto_0.50` from the moved file (the fallback is
`permission_overlay`), and the replay's `decision_overlays.csv` is
identical, which confirms the read.

## 4. Registry

- `research/registry.yaml`: one entry per study folder with `id`, `slug`,
  `date` (first commit of the study's files, from full history), `question`
  (the script's docstring summary, or written by hand for the seven studies
  without one), `status`, `promoted_to`, `closeout` and optional `notes`.
- Status is one of `promoted` (3: v38, v72, v75), `retained` (7: v131, v132,
  v134, v140, v142, v145, bl01 — tested a live setting and kept it),
  `shadow` (13: v96, v113, v125, v127, v128, v129, v130, v141, v143, v144,
  v149, v150, v165) or `closed` (90). `promoted_to` names the config
  constant or module each non-closed study feeds. v76 is the promotion of
  the v72 quality-weighted consensus (after the v74 shadow and the v75
  hold-out replay); it has no folder of its own, so v72 and v75 record it.
  Notes flag the WP11 re-runs the review asks for (v38, v128, v134, v142).
- `closeout` is the study's closeout note when one exists, else the plan or
  results summary that records its conclusion; two studies
  (`pb_vs_pe`, `test_runtime`) have none.
- `research/tools/registry.py` validates the registry (required fields, id
  and slug format, quoted dates, status vocabulary, `promoted_to` present
  exactly when not closed, closeout files exist, unique ids), checks that
  every folder under `research/studies/` has an entry and every entry a
  folder, and renders `research/README.md`. `--write` regenerates the
  README; with no flag it also fails if the README is stale. CI runs it as
  "Every research study folder is registered".
- It reads YAML, so PyYAML is a new dev dependency (`pyyaml>=6.0` in
  `pyproject.toml` and `requirements-dev.txt`; `pyyaml==6.0.2` in
  `constraints-dev.txt`). The runtime dependencies are unchanged.
- Each study folder has a README (question, date, status, notes, closeout,
  run command, outputs, tests), generated once from the registry and
  editable by hand from now on.

## 5. Legacy results and detail files

- `results/v9..v28/` → `research/legacy/` unchanged (214 files, 17 MB), with
  a README listing each folder, the script that wrote it and its closeout.
  The archived v11–v24 scripts, `v27_redeploy_portfolio_study.py`,
  `v28_forecast_universe_review.py` and the other v9 scripts still default
  to `results/vNN/`, so a re-run writes a new untracked folder there.
- `research/studies/*/outputs/detail/` is gitignored. The two committed
  `*_detail.csv` files over 1 MB outside the legacy folder leave the tree:
  `v128_regularized_selection_detail.csv` (2.7 MB) and
  `v162_ta_broad_screen_detail.csv` (2.2 MB). History is not rewritten.
  - v128 saves it with `save_results(..., detail=True)`; v162 writes it to
    `OUTPUT_DIR / "detail"`, and v163 (which reads it for its benchmark-family
    slices) takes it from there through a new `screen_detail_path`
    parameter, skipping the slices when it has not been regenerated, as it
    did before when the file was absent.
  - Each study README gives the command that regenerates the file and a
    `git show <sha>:results/research/<file>` line for the last committed
    copy; both lines were checked to return the 2,693,167- and
    2,189,974-byte files.
- The three legacy detail files over 1 MB
  (`v9/classifier_feature_selection_detail_20260403.csv`, 9.7 MB;
  `v9/regime_slice_detail_20260403.csv`, 1.7 MB;
  `v27/v27_redeploy_backtest_detail_20260405.csv`, 1.0 MB) stay, because the
  brief says to move the legacy folders unchanged. The 1 MB test exempts
  `research/legacy/` only.

## 6. Tests: failing before, passing after

`tests/test_restructure_phase3.py` (59 tests): layout (no `.py` under
`results/`, `scripts/research/` and `results/research/` gone, 14
live-setting/x-series/special studies have their script and README, every
study has a README and `outputs/` or a legacy link, study tests in
`tests/research/`), repository-root depth in study scripts and moved tests,
no code pointing at the old locations, the registry (every folder
registered; a stray folder and bad entries are rejected; README generated
from the registry; statuses of the live-setting studies; CI runs the
check), the 16 legacy folders and their README, the detail-file rules
(no committed `*_detail*.csv` over 1 MB outside legacy, `outputs/detail/`
ignored, v128/v162 write there), the resolver, and the production inputs.

| | Result |
|---|---|
| Unfixed `master` (f76cee2) | 59 failed |
| This branch | 59 passed |

Two depth checks passed vacuously on `master` at first (nothing to scan);
they now also require that 100+ files were scanned, so all 59 fail before.

Updated tests: `test_restructure_phase2.py` (no `scripts/research/`
exemption), `test_docs_hygiene.py` (`docs/results/README.md` points at
`research/studies/` and `research/legacy/`), the moved research tests
(imports, paths, depth), and `test_v129_feature_map.py` /
`test_v129_dual_track_integration.py` (feature-map path).

The link checker now also covers `research/README.md`,
`research/legacy/README.md` and the 113 study READMEs: 150 files, 0 broken
links (35 on `master`). It found the four `results/research/` links in
`docs/research/x_series_resume_2026-04-24.md`, now fixed.

Full suite, `python -m pytest -o addopts="--tb=short" -q`:

| | Result |
|---|---|
| `master` (f76cee2) | 2451 passed, 1 skipped, 150 warnings in 654.14s (0:10:54) |
| This branch | 2510 passed, 1 skipped, 111 warnings in 617.51s (0:10:17) |

## 7. September 2026 before and after

`scripts/replay_monthly_decisions.py --as-of 2026-09-21` (read-only dry run,
`--skip-fred`) in two clean worktrees, each with its own copy of the
committed DB (sha256 `7c68efbd…c35d`, unchanged after both runs): `master`
(f76cee2) and this branch (02e3fd4).

| File | `master` vs this branch |
|---|---|
| `recommendation.md`, `diagnostic.md`, `signals.csv`, `benchmark_quality.csv`, `consensus_shadow.csv`, `classification_shadow.csv`, `decision_overlays.csv`, `dashboard.html`, `monthly_summary.json`, `plots/calibration_curve.png` | identical (`cmp`) |
| `run_manifest.json` | only `git_sha` and `run_timestamp_utc` differ |
| Replay row (34 fields) | identical |

The recommendation is unchanged: DEFER-TO-TAX-DEFAULT / 50 %, consensus
UNDERPERFORM (LOW), mean forecast −1.43 %, OOS R² +2.86 %. The identical
`monthly_summary.json` includes the follow-on lane's `candidate_sources`,
read from the moved v141–v150 files.

## Judgement calls

- **Script names kept, not `run.py`.** The review's target layout shows
  `run.py`. Two folders hold more than one script (v129 has two, v9 has
  five), and scripts and tests import each other by module name, so each
  keeps its file name.
- **One v9 folder.** The five `*experiments*` scripts are v9.x experiments
  that all wrote to `results/v9/`; they share `v9_experiments/` rather than
  five folders with dotted ids that could not be imported.
- **Other root research scripts stay.** The brief names the
  `*experiments*` scripts. `benchmark_suite.py`, `benchmark_reduction.py`,
  `candidate_model_bakeoff.py`, `classifier_feature_selection.py`,
  `feature_cost_report.py`, `policy_evaluation.py`,
  `regime_slice_backtest.py`, `feature_ablation.py`,
  `v27_redeploy_portfolio_study.py` and `v28_forecast_universe_review.py`
  are still in `scripts/`, as is `results/backtests/`.
- **Tests keep their names** inside `tests/research/`; renaming them
  (`test_research_` prefix) is phase 4 with the unit/integration split.
- **Study outputs are records.** Committed summaries that mention
  `results/research/…` paths were not edited.
- **`sys.path` edits stay** in the moved scripts (renamed in the
  allowlist); removing them needs `research/` installed or run with `-m`,
  which is a separate change.

## Not done here (later steps)

- Phase 4 (docs history merge, `tests/unit|integration`, research tests in
  their own CI job, `CLAUDE.md`, config consolidation) and phase 5 (split
  `monthly_decision.py` and the two `edgar_8k_fetcher.py` files).
- Moving the remaining root research scripts and `results/backtests/`.
- Production still reads three study outputs (section 3). The review's
  naming rule ("production paths never contain a version or study ID")
  needs those promoted into `config/` or `artifacts/`.
- The WP11 re-runs flagged in the registry notes.
