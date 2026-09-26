# Review 2026-09-25, step 7 — restructure phases 0–2 (WP13)

Phases 0–2 of section 5 ("Repository structure review and tidy-up plan") of
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md), finding F32 and the
import part of F30. Phases 3–5 (one folder per study, docs/tests split,
monolith split) are later steps. Every move uses `git mv`, so history
follows the files (`git log --follow`).

The goal was no behaviour change. The September 2026 replay is identical
before and after (section 5).

## 1. Phase 0: safety nets

| Item | Where | What it checks |
|---|---|---|
| Package metadata | `pyproject.toml` | Package `pgr_vds` (`src`, `config`); `pip install -e .`; Python ≥ 3.11; runtime deps with `pandas>=3.0,<4`; `dev` / `dashboard` extras; pytest, mypy and ruff config |
| Link checker | `scripts/checks/check_doc_links.py` | Every relative link, image and reference definition in the active docs resolves; `#fragments` match a GitHub heading anchor |
| Import smoke test | `tests/test_entrypoint_imports.py` | Each of the 14 scripts and 5 modules a workflow runs imports in a fresh interpreter, from a temp working directory, without `PYTHONPATH`; the list covers every `python scripts/...` and inline `from src...` in `.github/workflows/` |
| `sys.path` rule | `scripts/checks/check_sys_path_edits.py` | No `sys.path` edit (or `site.addsitedir`) in a tracked `.py`/`.yml` file outside `tests/conftest.py` unless it is in `scripts/checks/sys_path_allowlist.txt`; a listed file that no longer edits `sys.path` also fails, so the list only shrinks |

- **Config consolidation.** `pytest.ini`, `mypy.ini` and `ruff.toml` are
  deleted; their settings are in `[tool.pytest.ini_options]`,
  `[tool.mypy]` (with the `scripts.monthly_decision` override) and
  `[tool.ruff]`, unchanged. `ruff check --show-settings` reports
  `pyproject.toml` as the settings path.
- **Dependency pin.** pandas ≥ 2.1 resolved to 3.0.6, which the suite and
  every replay since step 1 ran on; 2.x and 3.x differ (`pct_change`
  `fill_method`, F29). The pin is `>=3.0,<4` in both `pyproject.toml` and
  `requirements.txt` (the workflows install the latter); a test keeps the
  two lists equal. pandas 3 needs Python 3.11, so the package says ≥ 3.11
  and the README now does too. AGENTS.md still says 3.10+ (not edited here).
- **Active docs** are root `*.md`, `docs/*.md`, `docs/data/`,
  `docs/research/`, `docs/reviews/` and the `artifacts/` READMEs (34 files
  now; 28 files with 44 links on `master`). The history trees (`docs/plans`,
  `docs/superpowers`, `docs/closeouts`, `docs/results`, `docs/archive`,
  `archive/`, `results/`) are records and are not checked. The checker
  found 4 broken links on `master` (Windows-absolute `/C:/Users/...` paths in
  `docs/research/x_series_resume_2026-04-24.md`) and 2 that the phase 1
  move would have broken (`CHANGELOG.md`, `ROADMAP.md` → `decision_log.md`).
- **Allowlist.** 227 files edited `sys.path` after this change (81 in
  `results/`, 68 in `tests/`, 64 in `scripts/`, 14 in `archive/`), down from
  230 on `master`: the moved v46 module and the monthly-email heredoc no
  longer edit it (`python -` already puts the working directory on
  `sys.path`).
- **CI** runs `pip install --no-deps -e .` after the requirements, then the
  link checker and the `sys.path` rule as separate steps.
- **Not done:** the review's "record the repo tree in a manifest". The
  moves are verified by the phase 1 tests and by `git diff -M` (104
  renames, 100 % similarity except the v46 module at 84 %).

## 2. Phase 1: production artifacts under `artifacts/`

| From | To | Files |
|---|---|---|
| `results/monthly_decisions/` | `artifacts/monthly_decisions/` | 84 |
| `results/v14/shadow_reviews/` | `artifacts/shadow_reviews/` | 6 |
| `results/research/pgr_*.png` (the 12 `CHART_FILES`) | `artifacts/charts/` | 12 |
| `data/fetch_status.md` | `artifacts/ops/fetch_status.md` | 1 |

- `config/paths.py` (re-exported by `config`) holds `ARTIFACTS_DIR`,
  `MONTHLY_DECISIONS_DIR`, `DECISION_LOG_PATH`, `SHADOW_REVIEWS_DIR`,
  `CHARTS_DIR`, `OPS_DIR`, `FETCH_STATUS_PATH` and
  `DRY_RUN_MONTHLY_DECISIONS_DIR`. They are relative to the repo root, like
  `DB_PATH`.
- Readers and writers switched to the constants: `monthly_decision.py`
  (output folder, dry-run folder, `decision_log.md` read and append),
  `classification_artifacts` (both shadow ledgers), `email_sender` (report
  and the four attached charts), both chart scripts (`OUT_DIR`),
  `initial_fetch.py` (default `--status-file`),
  `verify_monthly_outputs.py` (`--base-dir`) and
  `replay_monthly_decisions.py`. `dashboard/data.py` cannot import `config`
  under `streamlit run` (only `dashboard/` is on `sys.path`), so it mirrors
  the path and a test compares the two.
- **Workflows.** Every `git add` now names one of
  `data/pgr_financials.db`, `artifacts/monthly_decisions/`,
  `"artifacts/charts/$chart"` or `artifacts/ops/fetch_status.md`, and none
  uses `|| true`. `post_initial_bootstrap.yml` used to run
  `git add results/ || true`. In `monthly_decision.yml` the chart list is
  `MONTHLY_CHARTS` and the step is "Regenerate monthly charts"; the freshness
  check and the commit use `artifacts/charts/`. The two initial-fetch
  workflows write and commit `artifacts/ops/fetch_status.md`.
- **Unchanged on purpose.** Dry runs still write to the gitignored
  `results/dry_run/monthly_decisions/` (not a production artifact). Earlier
  months' `run_manifest.json` files still list the paths they were written
  to; they are records. `results/v14/v14_shadow_review_20260404.csv` still
  names `results\v14\shadow_reviews\...` for the same reason.
- Each `artifacts/` folder has a README naming the script and workflow that
  write it. `docs/artifact-policy.md` and `docs/workflows.md` have a table of
  what each workflow commits; `docs/architecture.md`, `docs/operations-runbook.md`,
  `docs/decision-output-guide.md`, `docs/model-governance.md`, README and
  CONTRIBUTING name the new paths.

## 3. Phase 2: de-versioned library

- **Shims.** `src/research/v11.py` (no production importer; only
  `archive/tests/test_v11_research.py`) and the other eight two-line
  `from X import *` shims are deleted:

  | Shim | Real module |
  |---|---|
  | `benchmark_sets`, `diversification` | `src.portfolio.benchmark_sets`, `src.portfolio.diversification` |
  | `evaluation`, `policy_metrics` | `src.models.evaluation`, `src.models.policy_metrics` |
  | `v11`, `v27` | `src.portfolio.redeploy_buckets`, `src.portfolio.redeploy_portfolio` |
  | `v12`, `v22`, `v29` | `src.reporting.snapshot_summary`, `src.reporting.cross_check`, `src.reporting.confidence` |

  40 importing files were rewritten (13 in `archive/`, 13 in `scripts/`,
  12 in `tests/`, `src/research/v28.py`,
  `results/research/v162_ta_broad_screen.py`), plus a docstring in
  `monthly_decision.py` and the CI mypy list. The review counted 39
  importers; the extra one is the archived v11 test. A star import only
  re-exports public names, so every name the importers used exists in the
  real module; no test monkeypatched a shim.
- **`results/research/v46_classification.py` →
  `src/research/binary_classification.py`** (`git mv`, then edited).
  Production reached it through `scripts/monthly_decision.py` →
  `src.models.classification_shadow` → `src.research.v66_utils`, which
  imports `compute_binary_metrics`. At import it:
  - inserted the repo root into `sys.path`;
  - called `sys.stdout.reconfigure(encoding="utf-8")`;
  - installed three global "ignore" filters: `ConvergenceWarning`,
    `FutureWarning` from `sklearn.linear_model._logistic`, and
    "All-NaN slice encountered".

  All three are gone from import time. The study's `main()` sets the
  encoding and the filters (inside `warnings.catch_warnings()`) for its own
  run; `python -m src.research.binary_classification` re-runs it.
  `monthly_decision.py` now reconfigures stdout to UTF-8 in its `__main__`
  block, so its console output is unchanged, including on a Windows console.
- **Visible difference.** A monthly run no longer silences those warnings
  globally. The WFO block that counts ConvergenceWarnings already set
  `simplefilter("always")` inside `catch_warnings`, so its count is
  unchanged; warnings raised elsewhere now reach the log. The artifacts are
  identical (section 5).
- **Also:** `src/research/v37_utils.py` (in the same import chain) created
  `results/research/` at import; it now does so in `save_results`.

## 4. Tests: failing before, passing after

New tests (68): `tests/test_restructure_phase0.py` (19),
`tests/test_restructure_phase1.py` (12), `tests/test_restructure_phase2.py`
(15), `tests/test_entrypoint_imports.py` (22; one per entry point, plus
coverage and side-effect checks).

Run against unfixed `master` (b609af6, clean worktree, package not
installed): **56 failed, 12 passed**. The 12 that pass are the 12 script
import smokes that exist on `master`; they are positive controls, which pass
before and after. On this branch: all 68 pass.

The F30 check on `master`'s import chain (probe of
`import src.models.classification_shadow` run from the `master` worktree,
`PYTHONIOENCODING=ascii`):

| | `master` | This branch |
|---|---|---|
| `results.*` modules loaded | `results`, `results.research`, `results.research.v46_classification` | none |
| Blanket ignore filters added | ConvergenceWarning; sklearn `_logistic` FutureWarning; "All-NaN slice encountered" | none |
| `sys.path` changed | yes | no |
| `sys.stdout.encoding` | `utf-8` (reconfigured) | `ascii` (untouched) |

Existing tests updated for the new layout:

- `test_capital_return_charts.py`: `MONTHLY_CHARTS` and "Regenerate monthly
  charts".
- `test_dry_run_read_only.py`: the temp repo copies `artifacts/monthly_decisions`
  and checks the whole `artifacts/` tree is unchanged by a dry run.
- `test_repo_hygiene.py`: the no-`-q` check reads `pyproject.toml`.

Full suite, `python -m pytest -o addopts="--tb=short" -q`:

| | Result |
|---|---|
| `master` (b609af6) | 2329 passed, 2 skipped, 120 warnings in 530.35s (0:08:50) |
| This branch | 2397 passed, 2 skipped, 110 warnings in 551.26s (0:09:11) |

## 5. September 2026 before and after

`scripts/replay_monthly_decisions.py --as-of 2026-09-21` (a read-only dry run
with `--skip-fred`) in two clean worktrees, each with its own copy of the
committed DB (sha256 `7c68efbd…c35d`, unchanged after both runs): `master`
(b609af6) and this branch.

Every artifact of the dry run is byte-identical, except two fields of
`run_manifest.json` that record the run itself:

| File | `master` vs this branch |
|---|---|
| `recommendation.md`, `diagnostic.md`, `signals.csv`, `benchmark_quality.csv`, `consensus_shadow.csv`, `classification_shadow.csv`, `decision_overlays.csv`, `dashboard.html`, `monthly_summary.json`, `plots/calibration_curve.png` | identical (`cmp`) |
| `run_manifest.json` | only `git_sha` (b609af6 → this commit) and `run_timestamp_utc` differ |
| Replay row (`replay_monthly_decisions.py` CSV, 34 fields) | identical |

The recommendation is unchanged: DEFER-TO-TAX-DEFAULT / 50 %, consensus
UNDERPERFORM (LOW), mean forecast −1.43 %, OOS R² +2.86 %, equal-weight IC
0.0762, Pesaran–Timmermann p 0.391 (the step 6 numbers).

The dry run reads the previous decision from `decision_log.md` and the
classifier history from the shadow ledgers, both now under
`artifacts/monthly_decisions/`; an identical `recommendation.md` shows those
reads still find the moved files.

The first run on this branch took 18 minutes, because it shared the 4 CPUs
with the full test suite (load average 8, BLAS threads oversubscribed in the
logistic fits of the TA shadow variants). Re-run on its own, it took 110 s,
in line with `master`, and its output was again identical.

## Judgement calls

- **Package name.** `pgr_vds` is the distribution name; the importable
  packages stay `src` and `config`. Renaming them to `src/pgr_vds/...` would
  rewrite every import in the repo, which is phase 5 territory in the review's
  target layout.
- **requirements.txt kept.** Every workflow and the dashboard install it; the
  review's phase 4 folds the requirements files into `pyproject.toml`. Until
  then a test keeps the two dependency lists identical.
- **Only pandas is capped.** The brief asks for pandas; the other lower
  bounds are unchanged.
- **Dry-run output stays in `results/dry_run/`.** `artifacts/` is for
  committed production output; the dry-run path is now a config constant, so
  moving it later is a one-line change.
- **Shadow reviews.** Nothing writes `artifacts/shadow_reviews/` today (the
  v14 script is archived); the folder moved as the brief asks and its README
  says so.
- **Warnings.** The global filters were removed, as F30 asks, rather than
  re-installed in the entry point. Only log output changes.

## Not done here (later steps)

- Phase 3–5: one folder per study, `docs/history/`, `tests/unit|integration|research`,
  `CLAUDE.md`, splitting `monthly_decision.py` and the two
  `edgar_8k_fetcher.py` files.
- The other `results/research/*.py` modules that tests import are still
  under `results/` (phase 3). Production still *reads* committed research
  outputs: `results/research/v128_benchmark_feature_map.csv`
  (`src/models/v129_feature_map.py`), the v141–v149 candidate files
  (`src/reporting/shadow_followon.py`) and a path in
  `src/models/classification_gate_overlay.py` (F30).
- The 227 existing `sys.path` edits (the allowlist shrinks as files are
  migrated).
- AGENTS.md and ROADMAP still say Python 3.10+; the code needs 3.11.
