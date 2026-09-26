# Contributing

## Scope

This repository mixes production decision-support code with committed research
artifacts. Contributions should preserve the distinction between those two
layers.

## Branching

- Use a short branch off `master`.
- Prefer focused PRs that map to one logical workstream.
- If a change affects production workflows, documentation updates are required
  in the same PR.

## Before Opening a PR

Run the local CI-equivalent commands:

```bash
pip install -e ".[dev]"
ruff check .
python scripts/checks/check_doc_links.py
python scripts/checks/check_sys_path_edits.py
python research/tools/registry.py
python -m pytest -q -m "not artifact and not research"   # CI job: test
python -m pytest -q -m "research and not artifact"       # CI job: research
python -m pytest -q -m artifact                          # CI job: artifacts
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

The weekly and monthly dry runs are read-only (DB opened with `mode=ro`;
monthly artifacts go to the gitignored `results/dry_run/`). They leave the
committed DB and `artifacts/monthly_decisions/` untouched.

## Tests

The layout mirrors the code (review 2026-09-25, section 5, phase 4):

| Folder | What goes there | Marker |
|---|---|---|
| `tests/unit/<package>/` | Tests of one module. `<package>` mirrors the code: `backtest`, `database`, `ingestion`, `models`, `portfolio`, `processing`, `reporting`, `tax` (under `src/`), plus `config`, `dashboard` and `scripts` | `unit` |
| `tests/integration/pipeline/` | Several layers end to end: the monthly pipeline, dry runs, entry-point imports, workflow and ops contracts, mutation kills | `integration` |
| `tests/integration/data/` | Row-level integrity of the committed DB | `integration` |
| `tests/integration/repo/` | Repository contracts: docs, links, layout, workflows, test-suite hygiene | `integration` |
| `tests/research/` | Research studies (`research/studies/`) and `src/research/` | `research` |

- Name a test file after the module it tests: `test_<module>.py`, or
  `test_<module>_<aspect>.py` when a module has several (for example
  `unit/processing/test_feature_engineering_channel_mix.py`). No version or
  work-package numbers in new names; study ids appear only in
  `tests/research/`.
- `tests/conftest.py` applies the `unit`, `integration` and `research`
  markers by folder. Helpers shared across folders (`conftest.py`,
  `repo_guard.py`, `capital_return_fixture.py`, `guard_probe.py`) and
  `fixtures/` stay at the top of `tests/`. Every test folder has an
  `__init__.py`; import a helper from another test file by its full path
  (`from tests.unit.scripts.test_edgar_8k_parser_breadth import ...`).
- A test file that needs the repository root uses
  `Path(__file__).resolve().parents[3]` (two folders below `tests/`), or
  `parents[2]` in `tests/research/`.
- CI runs three jobs: `test` (`-m "not artifact and not research"`),
  `research` (`-m "research and not artifact"`) and `artifacts`
  (`-m artifact`). A plain `python -m pytest -q` runs all three.
- Mark a test `@pytest.mark.artifact` when its assertions are about
  committed data (`results/`, `artifacts/`, `data/`, the committed DB)
  rather than code. A test of production code that happens to load a
  committed input stays unmarked.
- `tests/conftest.py` installs a repository guard (`tests/repo_guard.py`).
  A test fails if it writes inside the repository tree or opens
  `data/pgr_financials.db`. `config.DB_PATH` and the feature-matrix cache
  point at `tmp_path`. An `artifact` test may open the committed DB
  read-only (`get_connection(path, read_only=True)` or a `mode=ro` URI).
  To run code that opens the default DB read-write on real data, use the
  `committed_db_copy` fixture.
- Seed random data with fixed integers, never `hash(...)`, which is salted
  per process.
- `scripts/checks/mutation_study.py` re-runs the F28 mutation study in a
  scratch clone. When you move one of its production sites, update the
  mutation; when you move or rename one of its test files, update its list
  (the script stops on a missing file).

## Generated Files

Do not edit these manually unless the change is specifically about generated
output shape or fixtures:

- `artifacts/*` (production outputs written by workflows; see
  `docs/artifact-policy.md`)
- `research/legacy/*` (the v9–v28 result folders, read-only)
- `research/studies/*/outputs/*` (written by each study's script)
- `research/README.md` (generated from `research/registry.yaml`)
- `data/pgr_financials.db`

If a code change intentionally alters a generated artifact, regenerate it and
include both the code and artifact update in the same PR.

## Packaging and Paths

- `pyproject.toml` defines the package (`pgr_vds`: `src` and `config`), its
  dependencies and the pytest, mypy and ruff config. It is the only
  dependency list: there are no requirements or constraints files. Runtime
  dependencies go in `[project] dependencies` (every workflow runs
  `pip install -e .`), test and lint tools in the `dev` extra, pinned
  exactly, and Streamlit in the `dashboard` extra. pandas is pinned to the
  tested major version (`>=3.0,<4`).
- New code imports `src` and `config` through `pip install -e .`, never by
  editing `sys.path`. `scripts/checks/check_sys_path_edits.py` fails on a new
  edit outside `tests/conftest.py`; remove a file from
  `scripts/checks/sys_path_allowlist.txt` when you drop its edit.
- Production output paths are constants in `config/paths.py`
  (`artifacts/...`); do not hard-code them. Nothing in `src/`, `config/`,
  `dashboard/` or a production script may import code from `results/`.
- A new workflow entry point must be added to
  `tests/integration/pipeline/test_entrypoint_imports.py` (the test fails until it is).

## Workflow Discipline

- Treat `.github/workflows/*.yml` as production code.
- Prefer explicit verification steps over silent success.
- Use concurrency groups for workflows that can mutate the committed DB or
  committed reports.
- Stage only intended files in workflow commit steps.

## Database and Migrations

- Add schema changes through ordered migration files under
  `src/database/migrations/`.
- Do not add more ad hoc `ALTER TABLE` logic unless it is part of the migration
  runner itself.
- Update migration tests when schema shape changes.

## Documentation Expectations

Update docs whenever you change:

- repo status or baseline
- operational workflows
- artifact policy
- contributor/operator process
- user-facing report or email behavior

The top-level docs map lives in `README.md`; `docs/README.md` says where a
new document goes. Finished plans, closeouts, result summaries and external
reviews live under `docs/history/` (index: `docs/history/README.md`); do not
edit them to match today's code. Retired code is deleted, not archived in the
tree: list it in `docs/history/retired-code/README.md` with the commit that
last had it. `scripts/checks/check_doc_links.py` checks every relative link
in the docs, including all of `docs/history/`.

## Research vs. Production

- Production code supports the scheduled workflows and the monthly decision
  output.
- Research code supports evaluation, experimentation, and future promotion
  decisions.
- Do not silently promote research code into production behavior.
- Record each promotion decision as the next numbered file in
  `docs/decisions/` and add a row to the Decision Record table in
  `docs/model-governance.md`, in the same PR (see `docs/decisions/README.md`).

## Research Studies

Each study is one folder, `research/studies/<id>_<slug>/`: its script(s), a
`README.md` (question, command, conclusion, closeout link) and `outputs/`.
Its tests go in `tests/research/`.

- Name outputs with the study id prefix (`v170_…`, `x25_…`) and write them
  with `src.research.study_paths.study_output_path(name)` (or
  `src.research.v37_utils.save_results`), which finds the folder from the id.
- Per-fold `*_detail.csv` files over 1 MB go to `outputs/detail/`
  (`study_output_path(name, detail=True)`), which is gitignored; say in the
  study README how to regenerate them. A test fails if one over 1 MB is
  committed outside `research/legacy/`.
- Register the study in `research/registry.yaml` and run
  `python research/tools/registry.py --write` to regenerate
  `research/README.md`. CI fails if a study folder is not registered or the
  README is stale.
- Run study scripts from the repository root. Production code must not
  import from `research/`; if production needs a study's output, name the
  path in `config/` or the reading module and record it in the registry's
  `promoted_to`.
