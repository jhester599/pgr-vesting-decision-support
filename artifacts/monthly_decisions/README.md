# Monthly decisions

Written by `scripts/monthly_decision.py` (workflow `monthly_decision.yml`;
the first report by `post_initial_bootstrap.yml`). One folder per month,
`YYYY-MM/`, plus the append-only `decision_log.md`,
`classification_shadow_history.csv` and `ta_shadow_variant_history.csv`.
`scripts/verify_monthly_outputs.py` checks each new month.

Moved here from `results/monthly_decisions/` in v180. The `run_manifest.json`
files of earlier months still list the paths they were written to.

What each file means: [`docs/decision-output-guide.md`](../../docs/decision-output-guide.md).
