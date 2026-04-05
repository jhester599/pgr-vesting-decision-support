# PGR Vesting Decision Support - v26 corrected cross-check production baseline

PGR Vesting Decision Support is a tax-aware decision-support system for unwinding
a concentrated Progressive Corporation (`PGR`) RSU position in a taxable
account. The repository combines scheduled data ingestion, feature engineering,
walk-forward modeling, tax-lot analysis, and monthly reporting so the user can
make a more disciplined decision at each vest.

## Current Status

Status as of 2026-04-05:

- `v7.x` is fully implemented in runtime code: the ablation work, tax scenario
  framework, EDGAR parser hardening, monthly report cleanup, and
  observation-per-feature / CPCV governance are all live.
- `v8.x` is fully implemented in production operations: the repo baseline was
  reconciled, the checked-in database was backfilled from the committed CSV,
  the monthly workflow and email were refreshed, and the user-facing decision
  output now uses recommendation modes rather than overconfident prediction
  language.
- `v9.x` is a completed research layer, not a promoted production change. It
  added research harnesses, benchmark-reduction studies, target experiments,
  policy evaluation, a weekly snapshot experiment, and a tuned Ridge
  classifier-sidecar candidate. The v9 conclusion was to avoid promoting a
  production model change yet.
- `v10.1` hardens the repo around that post-v9 state: it clarifies research vs.
  production boundaries, improves workflow safety and CI, introduces explicit
  schema migrations, adds run manifests, documents artifact policy, and makes
  the codebase easier to operate and extend safely.
- `v11.x` is an implemented research loop focused on accuracy and usefulness
  under a diversification-first objective. It adds a canonical `results/v11/`
  scoreboard, diversification-aware universe selection, reduced-universe
  bakeoffs, policy redesign, sidecar classifier review, and production-like
  dry-run recommendation memos. The current v11 conclusion is still
  `do not promote a live model change yet`.
- `v12.x` is a shadow-promotion study layered on top of v11. It compares the
  live production monthly stack against the simpler diversification-first
  baseline over a rolling 12-month review window, writes side-by-side dry-run
  memos under `results/v12/`, and tests whether the recommendation layer
  should be simplified before any new model stack is promoted.
- `v13.x` is the production-facing follow-through from that result. It keeps
  the live model stack intact, promotes the steadier diversification-first
  recommendation layer, and adds explicit existing-holdings guidance,
  diversification-first redeploy guidance, and a live-stack cross-check to the
  monthly report and email.
- `v14.x` is a narrow post-v13 prediction-layer study. It keeps the promoted
  v13.1 recommendation layer fixed, retests reduced benchmark universes, and
  compares the live 4-model stack against lean Ridge/GBT-centered replacement
  candidates. The current v14 conclusion is `continue shadowing, do not
  promote yet`, with `ensemble_ridge_gbt` as the leading v15 candidate.
- `v15.x` is now executed through `v15.2`. It reviewed 4 external research
  reports, built a canonical candidate inventory, screened one-for-one
  replacements on the v14 Ridge/GBT survivors, confirmed the best winners
  across all deployed model types, and finished with a final cross-model
  bakeoff.
- `v16.x` is a completed narrow promotion study on the best v15 upgrades. It
  compares the modified Ridge+GBT pair against the reduced-universe live stack
  and the `historical_mean` baseline. The current v16 conclusion is
  `shadow_for_v17`: the modified pair is now the top reduced-universe row, but
  it still does not separate enough from the baseline to justify immediate
  production promotion.
- `v17.x` is a completed production-style shadow gate. It tests whether the
  modified Ridge+GBT pair should replace the current live stack as the visible
  cross-check under the promoted v13.1 recommendation layer. The v17
  conclusion is `keep_current_live_cross_check`: the candidate is steadier and
  much healthier on reduced-universe metrics, but it disagrees with the
  simpler baseline too consistently to improve the current user-facing
  experience yet.
- `v18.x` is a completed directional-bias reduction study. It tests only
  narrow benchmark-side and peer-relative one-for-one swaps on the modified
  Ridge+GBT pair. The v18 conclusion is `keep_v16_as_research_only`: the best
  swaps improved reduced-universe metrics again, but they did not reduce the
  candidate's directional disagreement with the promoted simpler baseline.
- `v19.x` is a completed remaining-feature closure pass. It backfilled public
  macro and valuation series, implemented the unresolved EDGAR-derived feature
  ideas, and produced a final traceability matrix for all 46 original v15
  feature candidates. The v19 conclusion is that `44 / 46` were fully tested
  and `2 / 46` are now explicitly blocked by missing source classes rather than
  left queued.
- `v20.x` is a completed synthesis and promotion-readiness gate. It assembled
  the strongest confirmed v16-v19 Ridge/GBT swap combinations into a small set
  of replacement stacks and compared them against the reduced live production
  cross-check, the `historical_mean` baseline, and the promoted simpler
  baseline over a 12-month monthly review window. The v20 conclusion is still
  `continue_research_keep_current_cross_check`.
- `v21.x` is a completed point-in-time historical comparison study. It fixes
  the narrow-window shadow-gate methodology by comparing the current live
  reduced cross-check and the leading v16-v20 assembled candidates over the
  full common evaluable history instead of a recent slice. The v21 conclusion
  is `promote_candidate_cross_check`, with `ensemble_ridge_gbt_v18` now
  clearing the historical agreement gate versus the promoted simpler baseline.
- `v22.x` is the narrow implementation step from that result. It keeps the
  promoted simpler diversification-first recommendation layer active and swaps
  the visible monthly cross-check to `ensemble_ridge_gbt_v18`.
- `v23.x` is a completed research-only extended-history proxy study. It uses
  stitched pre-inception benchmark proxies for `VOO`, `VXUS`, and `VMBS` to
  extend the common evaluable OOS window backward and confirm whether the
  `v21` cross-check promotion result still holds. It does.
- `v24.x` is a completed benchmark-definition study. It tests whether simply
  replacing `VOO` with `VTI` improves the reduced forecast universe. The
  answer is no: `VTI` adds raw history, but it does not improve the leading
  candidate enough to justify replacing `VOO`.
- `v25.x` is now complete. It is the correctness-first cycle prompted by
  external peer reviews: monthly index conventions were normalized, inner-CV /
  WFO / CPCV integrity issues were fixed, silent-failure guards were added,
  and the promotion-sensitive `v20-v24` studies were rerun on the corrected
  foundation. The key result survived: `ensemble_ridge_gbt_v18` remains the
  right visible cross-check candidate, and `v24` still concludes `keep_voo`.
- `v26.x` is now complete. It is a narrow productionization pass on top of
  `v25`: the remaining historical-comparison warning noise was cleaned up, the
  promoted visible cross-check was validated again through the monthly dry-run
  path, and the corrected repo state is now ready to package as the next PR.

## Production vs. Research

The repo now has explicit operating boundaries:

- Production:
  - `scripts/weekly_fetch.py`
  - `scripts/peer_fetch.py`
  - `scripts/edgar_8k_fetcher.py`
  - `scripts/monthly_decision.py`
  - scheduled GitHub workflows under `.github/workflows/`
  - committed database `data/pgr_financials.db`
  - monthly decision artifacts under `results/monthly_decisions/`
- Research / evaluation:
  - `src/research/`
  - v9 research scripts under `scripts/` such as
    `benchmark_suite.py`, `target_experiments.py`,
    `candidate_model_bakeoff.py`, and classifier experiments
  - research outputs under `results/v9/`
- Provisional:
  - v9 candidate production recommendations that have not yet been promoted
  - classifier-sidecar confidence work from v9
  - v11 diversification-first candidates and policy recommendations
  - v12 shadow-baseline recommendation-layer study
  - v14 reduced-universe replacement candidate shadowing
- Active recommendation-layer default:
  - v13.1 `shadow_promoted` recommendation-layer mode
- Active prediction-layer conclusion:
  - keep the live 4-model stack for now
- Active v15 status:
  - execution complete through `v15.2`
  - strongest confirmed replacements:
    - `rate_adequacy_gap_yoy` for GBT
    - `book_value_per_share_growth_yoy` for the linear models
  - best single v15 candidate:
    - `gbt_lean_plus_two__v15_best`
- Active v16 status:
  - promotion study complete
  - leading replacement stack:
    - `ensemble_ridge_gbt_v16`
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - keep the live production prediction stack unchanged for now
    - carry the modified Ridge+GBT pair into a narrower v17 shadow / promotion gate if we continue
- Active v17 status:
  - shadow gate complete
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - keep the current live production cross-check unchanged
    - keep `ensemble_ridge_gbt_v16` as a research candidate only
- Active v18 status:
  - bias-reduction study complete
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - keep the current live production cross-check unchanged
    - keep `ensemble_ridge_gbt_v16` as the leading research candidate
    - do not advance the v18 benchmark-side swaps to another promotion gate
- Active v19 status:
  - full original v15 feature inventory closed out
  - `44 / 46` original feature ideas tested through the swap framework
  - blocked source gaps:
    - `pgr_cr_vs_peer_cr`
    - `pgr_fcf_yield`
  - strongest newly resolved positive additions:
    - `pgr_pe_vs_market_pe`
    - `usd_broad_return_3m`
    - `auto_pricing_power_spread`
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - keep the current live production cross-check unchanged
    - move next to a narrow synthesis / promotion-readiness study, not another broad feature sweep
- Active v20 status:
  - synthesis / promotion-readiness study complete
  - best metric row:
    - `ensemble_ridge_gbt_v18`
  - best assembled best-of-confirmed stack:
    - `ensemble_ridge_gbt_v20_best`
  - key blocker:
    - the leading assembled stacks still show `0.0%` signal agreement with the promoted simpler baseline over the 12-month review window
    - `ensemble_ridge_gbt_v18` remained `UNDERPERFORM` in every reviewed month
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - keep the current live production cross-check unchanged
    - focus next on blocked-source work or narrower calibration diagnostics rather than another generic feature sweep
- Active v21 status:
  - point-in-time historical comparison complete
  - common evaluable window:
    - `2016-10-31` through `2025-09-30`
  - historical result:
    - `ensemble_ridge_gbt_v18` achieved `81.5%` signal agreement with the promoted simpler baseline
    - current live reduced cross-check achieved `64.8%`
  - current decision:
    - `ensemble_ridge_gbt_v18` is now the leading candidate to replace the current live production cross-check
    - keep the v13.1 recommendation layer unchanged
    - move next to a narrow production-promotion implementation / validation step rather than more generic feature hunting
- Active v22 status:
  - visible cross-check promotion complete
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - use `ensemble_ridge_gbt_v18` as the visible production cross-check
    - keep the underlying 4-model production signal path unchanged for now
- Active v23 status:
  - extended-history proxy validation complete
  - research-only stitched-history window:
    - `2013-04-30` through `2025-09-30`
  - historical result:
    - `ensemble_ridge_gbt_v18` achieved `78.7%` signal agreement with the promoted simpler baseline
    - current live reduced cross-check achieved `57.3%`
  - proxy caveat:
    - `VXUS <- VEA + VWO` and `VMBS <- BND` are strong research proxies
    - `VOO <- VTI` extends history usefully but is a looser proxy than ideal
  - current decision:
    - keep the v13.1 recommendation layer unchanged
    - keep `ensemble_ridge_gbt_v18` as the visible production cross-check
    - treat the promotion result as confirmed over the longer stitched-history window
- Active v24 status:
  - VTI-for-VOO replacement study complete
  - scenarios tested:
    - current `VOO`-based reduced universe
    - actual `VTI` replacement universe
    - stitched-history `VTI` replacement universe
  - result:
    - the current `VOO`-based universe remains best
    - `VTI` did not improve policy return, OOS R^2, or shadow agreement enough to justify replacement
  - current decision:
    - keep `VOO` in the reduced forecast universe
    - keep the v13.1 recommendation layer unchanged
    - keep `ensemble_ridge_gbt_v18` as the visible production cross-check
- Historical planning and review artifacts:
  - `docs/history/claude-v7-plan.md`
  - `docs/plans/codex-v8-plan.md`
  - `docs/plans/codex-v9-plan.md`
  - older roadmap / research review files in `docs/`

See [POST_V9_BASELINE.md](docs/baselines/POST_V9_BASELINE.md) and
[docs/model-governance.md](docs/model-governance.md) for the authoritative
boundary and promotion policy.

## Quickstart

### 1. Install runtime dependencies

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Optional developer tooling

```bash
pip install -r requirements-dev.txt -c constraints-dev.txt
```

### 3. Configure environment

Set the environment variables used by your workflow or local run:

- `AV_API_KEY`
- `FRED_API_KEY`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `EMAIL_FROM`
- `EMAIL_TO`

### 4. Run the main production entrypoints

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/edgar_8k_fetcher.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

### 5. Run the local CI-equivalent checks

```bash
ruff check .
python -m pytest -q
```

## Repository Layout

- `config.py`: central configuration and model / workflow constants
- `src/database/`: DB schema, migrations, and DB helpers
- `src/ingestion/`: data-provider clients and fetch helpers
- `src/models/`: walk-forward training, calibration, CPCV, and model pipelines
- `src/portfolio/`: vesting recommendation and portfolio-allocation logic
- `src/reporting/`: report and email rendering helpers
- `src/research/`: v9 research harnesses and policy evaluation utilities
- `results/v11/`: diversification-first research outputs from the v11 loop
- `results/v12/`: shadow-promotion comparisons between the live stack and the
  simpler diversification-first baseline
- `results/v14/`: reduced-universe prediction-layer bakeoffs, minimal feature
  surgery, and shadow review outputs
- `scripts/`: runnable CLI entrypoints for production and research tasks
- `results/monthly_decisions/`: committed production monthly decision outputs
- `results/v9/`: committed research outputs from the v9 program
- `docs/`: current operator, architecture, workflow, governance, and historical
  documentation

## Docs Map

- Baseline / current state:
  - [POST_V9_BASELINE.md](docs/baselines/POST_V9_BASELINE.md)
  - [V10_1_RESULTS_SUMMARY.md](docs/results/V10_1_RESULTS_SUMMARY.md)
  - [V11_RESULTS_SUMMARY.md](docs/results/V11_RESULTS_SUMMARY.md)
  - [V11_CLOSEOUT_AND_V12_NEXT.md](docs/closeouts/V11_CLOSEOUT_AND_V12_NEXT.md)
  - [V12_RESULTS_SUMMARY.md](docs/results/V12_RESULTS_SUMMARY.md)
  - [V12_CLOSEOUT_AND_V13_NEXT.md](docs/closeouts/V12_CLOSEOUT_AND_V13_NEXT.md)
  - [V13_RESULTS_SUMMARY.md](docs/results/V13_RESULTS_SUMMARY.md)
  - [V14_RESULTS_SUMMARY.md](docs/results/V14_RESULTS_SUMMARY.md)
  - [V14_CLOSEOUT_AND_V15_NEXT.md](docs/closeouts/V14_CLOSEOUT_AND_V15_NEXT.md)
  - [V15_RESULTS_SUMMARY.md](docs/results/V15_RESULTS_SUMMARY.md)
  - [V15_EXECUTION_SUMMARY.md](docs/results/V15_EXECUTION_SUMMARY.md)
  - [V15_CLOSEOUT_AND_V16_NEXT.md](docs/closeouts/V15_CLOSEOUT_AND_V16_NEXT.md)
  - [V16_RESULTS_SUMMARY.md](docs/results/V16_RESULTS_SUMMARY.md)
  - [V16_CLOSEOUT_AND_V17_NEXT.md](docs/closeouts/V16_CLOSEOUT_AND_V17_NEXT.md)
  - [V17_RESULTS_SUMMARY.md](docs/results/V17_RESULTS_SUMMARY.md)
  - [V17_CLOSEOUT_AND_V18_NEXT.md](docs/closeouts/V17_CLOSEOUT_AND_V18_NEXT.md)
  - [V18_RESULTS_SUMMARY.md](docs/results/V18_RESULTS_SUMMARY.md)
  - [V18_CLOSEOUT_AND_V19_NEXT.md](docs/closeouts/V18_CLOSEOUT_AND_V19_NEXT.md)
  - [codex-v19-plan.md](docs/plans/codex-v19-plan.md)
  - [V19_RESULTS_SUMMARY.md](docs/results/V19_RESULTS_SUMMARY.md)
  - [V19_CLOSEOUT_AND_V20_NEXT.md](docs/closeouts/V19_CLOSEOUT_AND_V20_NEXT.md)
- Architecture and operations:
  - [docs/architecture.md](docs/architecture.md)
  - [docs/workflows.md](docs/workflows.md)
  - [docs/operations-runbook.md](docs/operations-runbook.md)
  - [docs/troubleshooting.md](docs/troubleshooting.md)
- Data, artifacts, and governance:
  - [docs/data-sources.md](docs/data-sources.md)
  - [docs/model-governance.md](docs/model-governance.md)
  - [docs/decision-output-guide.md](docs/decision-output-guide.md)
  - [docs/artifact-policy.md](docs/artifact-policy.md)
- Process:
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [docs/changelog.md](docs/changelog.md)
- Historical context:
  - [ROADMAP.md](ROADMAP.md)
  - [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
  - older reviews and plans in [docs/](docs/)

## Key Operational Outputs

- `results/monthly_decisions/<YYYY-MM>/recommendation.md`
- `results/monthly_decisions/<YYYY-MM>/diagnostic.md`
- `results/monthly_decisions/<YYYY-MM>/signals.csv`
- `results/monthly_decisions/<YYYY-MM>/run_manifest.json`
- `results/monthly_decisions/decision_log.md`

## Notes on v9

v9 did not promote a new production model stack. It established the research
baseline, showed that benchmark reduction and leaner feature sets help, and
concluded that the next production promotion test should focus on a reduced
benchmark universe and leaner Ridge/GBT-centered candidates rather than model
class expansion.

The v9 closeout is documented in
[V9_CLOSEOUT_AND_V91_NEXT.md](docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md).

## Notes on v12

v12 did not promote a new production model stack either. It shadow-tested the
best v11 policy row, `baseline_historical_mean` with `neutral_band_3pct`,
against the live monthly stack over 12 recent monthly snapshots. The main
finding is that the live stack changed directional signals several times while
still landing on the same `DEFER-TO-TAX-DEFAULT` 50% sell action, whereas the
shadow baseline was steadier and easier to explain. That makes recommendation-
layer simplification the most plausible next promotion candidate.

## Notes on v13

v13 does not promote a new model stack either. Instead, it promotes the best
usefulness improvements from v11-v12 into the production-facing report and
email, and now also promotes the steadier recommendation layer:

- simpler-baseline cross-checks
- existing-holdings lot guidance
- diversification-first redeploy guidance
- simpler diversification-first recommendation layer as the active default

That makes the monthly output more useful immediately while still keeping the
live model stack unchanged.

## Notes on v14

v14 kept the promoted v13.1 recommendation layer fixed and retested the
underlying prediction layer on a reduced, diversification-aware benchmark
universe. The best replacement candidate was `ensemble_ridge_gbt`, which
improved on the reduced-universe live stack and stayed close to the
`historical_mean` baseline, but still did not earn immediate promotion. The
main v14 recommendation is to use fixed-budget feature replacement in v15
rather than reopening broader methodology expansion.

## Notes on v15

v15 completed the fixed-budget feature replacement cycle and produced real
feature wins without expanding feature count materially.

The two most important results were:

- `rate_adequacy_gap_yoy` replacing `vmt_yoy` in GBT
- `book_value_per_share_growth_yoy` replacing `roe_net_income_ttm` across the
  linear-model family

The best single v15 model in the final bakeoff was `gbt_lean_plus_two__v15_best`,
which improved mean sign-policy return versus both the baseline GBT and the
`historical_mean` sign-policy baseline. OOS R² is still negative, so v15 should
be treated as a successful feature-research milestone, not yet a final
production-promotion decision.
