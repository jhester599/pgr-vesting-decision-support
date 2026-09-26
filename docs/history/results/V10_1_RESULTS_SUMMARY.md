# v10.1 Results Summary

## Recommendation

**Promote with caveats**

v10.1 materially improves the repository baseline, workflow safety, schema
discipline, CI coverage, documentation, and operational traceability. The repo
is substantially clearer and safer to operate than the pre-v10.1 baseline.

The main caveat is that v10.1 intentionally does not solve the underlying model
accuracy problem. It hardens the system around the current post-v9 conclusion
that production-model promotion should remain conservative.

## 1. Post-v9 Baseline Reconciliation

Completed:

- added `POST_V9_BASELINE.md`
- updated `README.md` to reflect the real post-v9 state
- classified the repo into production, research, provisional, and historical
  layers
- explicitly documented the status of `results/v9/`

Outcome:

- a reader can now tell what is live production behavior versus retained
  research evidence

## 2. Workflow Hardening

Completed:

- standardized workflow action versions to `actions/checkout@v5` and
  `actions/setup-python@v6`
- added concurrency controls to the core production workflows
- strengthened verification for:
  - weekly fetch
  - peer fetch
  - monthly 8-K fetch
  - monthly decision
- added GitHub job summaries for core production workflows

Outcome:

- workflow runs are easier to inspect
- overlapping operational runs are less likely to collide on the committed DB
  or monthly artifacts

## 3. CI / Lint / Smoke / Regression Gates

Completed:

- added `.github/workflows/ci.yml`
- added `requirements-dev.txt` and `constraints-dev.txt`
- added `ruff.toml`
- added `mypy.ini`
- CI now runs:
  - `ruff check .`
  - limited `mypy`
  - full `pytest`
  - production dry-run smoke checks

Outcome:

- repo health now has an automatic guardrail rather than relying only on manual
  local validation

## 4. Helper / Provider Cleanup

Completed:

- added `src/ingestion/provider_registry.py`
- centralized provider metadata for:
  - daily-limit behavior
  - DB logging semantics
  - cache semantics
- updated legacy AV and FMP client counters to use explicit provider metadata
- updated DB-side request logging to respect provider limits cleanly

Outcome:

- provider behavior is more explicit
- rate-limit semantics are less buried in ad hoc conditionals

## 5. Schema Versioning and Migrations

Completed:

- added `src/database/migration_runner.py`
- added `src/database/migrations/001_initial.sql`
- added `schema_migrations` to the schema snapshot
- updated `db_client.initialize_schema()` to:
  - apply ordered migrations
  - then reconcile legacy pre-migration DBs safely
- added migration tests

Outcome:

- schema evolution is now explicit and versioned
- fresh and legacy DB initialization paths are clearer and safer

## 6. Modularization

Completed:

- extracted recommendation-mode and decision-section rendering into
  `src/reporting/decision_rendering.py`
- kept `scripts/monthly_decision.py` as the orchestration entrypoint while
  reducing the amount of embedded report wording/business logic it owns

Outcome:

- the monthly decision script is still large, but the business-facing report
  logic now has a more testable home in `src/`

## 7. Logging, Run Metadata, and Observability

Completed:

- added `src/reporting/run_manifest.py`
- monthly decision runs now emit `run_manifest.json`
- manifests include:
  - run timestamp
  - git SHA
  - schema version
  - row counts
  - latest data dates
  - warnings
  - output file paths
- workflow summaries now expose key postconditions directly in Actions

Outcome:

- monthly artifacts are more traceable
- warnings are easier to find in both reports and workflow logs

## 8. Documentation Consolidation

Completed:

- replaced the stale top-level README with an overview + docs map
- added current docs:
  - `docs/architecture.md`
  - `docs/data-sources.md`
  - `docs/workflows.md`
  - `docs/model-governance.md`
  - `docs/operations-runbook.md`
  - `docs/decision-output-guide.md`
  - `docs/troubleshooting.md`
  - `docs/changelog.md`
  - `docs/artifact-policy.md`

Outcome:

- current-state docs are easier to find
- the project is less dependent on scattered historical plans for orientation

## 9. Contributor and Operator Guidance

Completed:

- added `CONTRIBUTING.md`
- documented:
  - local validation commands
  - generated-file discipline
  - workflow discipline
  - migration discipline
  - research vs. production expectations
- added operator/recovery guidance in the runbook and troubleshooting docs

Outcome:

- future contributors and future Codex sessions now have a clearer starting
  point

## 10. Dependency and Environment Management

Completed:

- separated runtime and developer tooling dependencies
- added a dev constraints file
- documented local CI-equivalent commands

Outcome:

- local and CI setup are more aligned
- developer-tooling drift is less ad hoc

## 11. Data and Artifact Strategy

Completed:

- documented artifact policy in `docs/artifact-policy.md`
- explicitly separated:
  - committed operational state
  - committed research evidence
  - local-only helper artifacts that should not be committed
- refreshed the April 2026 monthly artifacts to the v10.1 baseline and added a
  run manifest

Outcome:

- the repo is clearer about why it commits some generated state and how that
  state should be interpreted

## 12. Contract and Golden-File Style Testing

Completed:

- added workflow contract tests
- added migration tests
- added provider-registry tests
- added run-manifest tests
- preserved existing monthly recommendation-mode tests around the extracted
  rendering helpers

Outcome:

- workflow assumptions and new infrastructure boundaries now have automated
  protection

## Validation Performed

- `python -m pytest tests/test_migration_runner.py tests/test_provider_registry.py tests/test_run_manifest.py tests/test_workflow_contracts.py tests/test_v813_recommendation_mode.py -q`
- `python scripts/weekly_fetch.py --dry-run --skip-fred`
- `python scripts/peer_fetch.py --dry-run`
- `python scripts/edgar_8k_fetcher.py --dry-run`
- `python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred`
- `python -m ruff check .`
- `python -m mypy src/database/migration_runner.py src/ingestion/provider_registry.py src/reporting/run_manifest.py`
- `python -m pytest -q`

## Follow-on Recommendations

1. Keep the current production model stack in place until the reduced-universe
   v9.1 candidate bakeoff is ready.
2. If a new production candidate is promoted later, require it to update:
   - `POST_V9_BASELINE.md`
   - `docs/model-governance.md`
   - `docs/artifact-policy.md`
   - the monthly run manifest contents if new artifacts are added
3. Consider further modularization of `scripts/monthly_decision.py` and
   `scripts/edgar_8k_fetcher.py` in a future maintenance cycle, but avoid
   turning that into an unnecessary framework rewrite.

## Deferred on Purpose

- no new model classes
- no benchmark-universe promotion
- no new feature research
- no methodology expansion beyond the current v9 conclusions
- no aggressive artifact-storage redesign that would disrupt the current
  workflow model
