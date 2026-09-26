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
pip install -r requirements-dev.txt -c constraints-dev.txt
pip install --no-deps -e .
ruff check .
python scripts/checks/check_doc_links.py
python scripts/checks/check_sys_path_edits.py
python -m pytest -q
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

The weekly and monthly dry runs are read-only (DB opened with `mode=ro`;
monthly artifacts go to the gitignored `results/dry_run/`). They leave the
committed DB and `artifacts/monthly_decisions/` untouched.

## Tests

- CI runs `python -m pytest -q -m "not artifact"` in the `test` job and
  `python -m pytest -q -m artifact` in the `artifacts` job. A plain
  `python -m pytest -q` runs both.
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
  mutation.

## Generated Files

Do not edit these manually unless the change is specifically about generated
output shape or fixtures:

- `artifacts/*` (production outputs written by workflows; see
  `docs/artifact-policy.md`)
- `results/v9/*`
- `data/pgr_financials.db`

If a code change intentionally alters a generated artifact, regenerate it and
include both the code and artifact update in the same PR.

## Packaging and Paths

- `pyproject.toml` defines the package (`pgr_vds`: `src` and `config`) and
  the pytest, mypy and ruff config. Runtime dependencies are listed there and
  in `requirements.txt` (which the workflows install); keep the two equal.
  pandas is pinned to the tested major version (`>=3.0,<4`).
- New code imports `src` and `config` through `pip install -e .`, never by
  editing `sys.path`. `scripts/checks/check_sys_path_edits.py` fails on a new
  edit outside `tests/conftest.py`; remove a file from
  `scripts/checks/sys_path_allowlist.txt` when you drop its edit.
- Production output paths are constants in `config/paths.py`
  (`artifacts/...`); do not hard-code them. Nothing in `src/`, `config/`,
  `dashboard/` or a production script may import code from `results/`.
- A new workflow entry point must be added to
  `tests/test_entrypoint_imports.py` (the test fails until it is).

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

The top-level docs map lives in `README.md`.

## Research vs. Production

- Production code supports the scheduled workflows and the monthly decision
  output.
- Research code supports evaluation, experimentation, and future promotion
  decisions.
- Do not silently promote research code into production behavior.
- Promotion decisions should be documented in `docs/model-governance.md` and in
  a summary document for the relevant release.
