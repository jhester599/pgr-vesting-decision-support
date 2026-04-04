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
ruff check .
python -m pytest -q
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

## Generated Files

Do not edit these manually unless the change is specifically about generated
output shape or fixtures:

- `results/monthly_decisions/*`
- `results/v9/*`
- `data/pgr_financials.db`

If a code change intentionally alters a generated artifact, regenerate it and
include both the code and artifact update in the same PR.

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
