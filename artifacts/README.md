# Production artifacts

Everything the scheduled GitHub workflows write and commit lives here
(review 2026-09-25, section 5, phase 1). Research output stays under
`results/`; nothing here is written by a research script.

| Folder | Written by | Committed by |
|---|---|---|
| [`monthly_decisions/`](monthly_decisions/README.md) | `scripts/monthly_decision.py` | `monthly_decision.yml`, `post_initial_bootstrap.yml` |
| [`charts/`](charts/README.md) | `scripts/repurchase_timeseries_charts.py`, `scripts/capital_return_charts.py` | `monthly_decision.yml` |
| [`ops/`](ops/README.md) | `scripts/initial_fetch.py --status-file` | `initial_fetch_prices.yml`, `initial_fetch_dividends.yml` |
| [`shadow_reviews/`](shadow_reviews/README.md) | nothing today (v14 study, archived) | — |

The paths are constants in `config/paths.py` (`config.MONTHLY_DECISIONS_DIR`,
`config.CHARTS_DIR`, `config.FETCH_STATUS_PATH`, `config.SHADOW_REVIEWS_DIR`).
Workflows stage these exact paths only. `--dry-run` output goes to the
gitignored `results/dry_run/`, never here. See
[`docs/artifact-policy.md`](../docs/artifact-policy.md).
