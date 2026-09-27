# Research re-run plan v200–v210 (after the 2026-09-25 review fixes)

> **Status (added 2026-09-26):** two v200 research plans exist, and their version numbers mean different things: this one (v200–v210) and [`RERUN_PLAN_v200_codex.md`](RERUN_PLAN_v200_codex.md) (v200–v207). Neither is adopted yet. [The comparison](../reviews/2026-09-26_step13v_comparison.md) sets out the differences and a recommendation, and it records the owner's choice. Do not start a v2NN session until that choice is recorded there.

- **Written:** 2026-09-26, by the step-13V verification session. Read it with [`docs/reviews/VERIFICATION_2026-09-26_claude.md`](../reviews/VERIFICATION_2026-09-26_claude.md). "Verification N1" … "N10" below are that report's new issues.
- **Status:** proposal. Nothing here changes production. Each step is one new session: paste the [shared preamble](#4-shared-preamble-paste-first-in-every-session), then that step's prompt.
- **Numbering:**
  - The series starts at v200 and runs sequentially (v200 … v210).
  - v200 is the new clean baseline: the current production model re-evaluated on the repaired data with honest metrics. The gap after the fix work (CHANGELOG v171–v185) marks the new data foundation.
  - Numbers v1–v199 are never reused.
  - Production CHANGELOG versions after v185 continue at v186 and must skip the block this series uses.

## 1. Why re-run

Every study before v200 was run on data or metrics that the 2026-09-25 review found broken and that CHANGELOG v171–v185 fixed:

- price features from weekly, split-unadjusted bars;
- targets missing two ETF splits;
- FRED data lagged twice;
- mis-parsed EDGAR fields;
- an OOS-R² benchmark that contained the target;
- a CPCV gate that could never pass;
- health metrics chosen on the same data they reported on.

The verification confirmed the fixes. It also showed that the live model has positive OOS R² (+1 % to +6 %) but no directional skill over the 68–70 % base rate (Pesaran–Timmermann p 0.36–0.62 in every month from 2026-02 to 2026-09).

Most of the live design (the feature sets, the 8-benchmark universe, shrinkage, quality-weighted consensus, the shadow classifiers) was chosen by studies touched by those defects. This plan re-establishes that evidence with fewer, better-designed studies. It does not re-run the ~150 old ones one by one.

## 2. Inventory of prior research (v9–v165, x1–x24, bl01)

Sources:

- `research/registry.yaml` and `research/studies/*/outputs/*summary*.md`;
- `research/legacy/`, `docs/history/results/V*_RESULTS_SUMMARY.md`, `docs/history/closeouts/`, `docs/history/plans/`, `docs/history/superpowers/plans/`;
- `docs/research/backlog.md`, `docs/research/x_series_resume_2026-04-24.md`;
- `docs/decisions/`, the CHANGELOG, and the study code in `src/research/` and `research/studies/`.

`src/pgr_vds/research_lib/` does not exist yet; v200 creates it.

**Defect codes:**

- **D** (standard research frame): every study that built features with `build_feature_matrix_from_db` and targets from `monthly_relative_returns` before v172–v179 inherited all of these:
  - F01 price features;
  - F03 VOO split in targets;
  - F06 FRED lags and duplicates;
  - F10, F11, F12 EDGAR units, PIF and signs;
  - F15 per-share basis;
  - F16/F17 YoY gaps;
  - F18 equity;
  - F22 target windows;
  - F23 EDGAR lag.
- **M** (metrics):
  - F04: the OOS-R² naive included the target. This includes `src/research/v37_utils.compute_metrics`, which still uses a 1-month horizon on 6M targets (verification N2).
  - F13: Newey–West p-values over pooled rows; hit rate compared with 50 % instead of the ~68 % base rate; winners picked and reported on the same OOS period.
- **Specific codes:**
  - F05: VGT labels (the investable classifier pool from v124);
  - F02: CPCV, or recommendation modes locked to DEFER by F02 and F04;
  - F08: dividends (2026 windows, special dividends);
  - F09: quarterly ROE;
  - F24: the shadow layer;
  - F25: LOO `RidgeCV(cv=None)`, raw-price x-series targets, overlap inference;
  - F27: peer data (CB spliced, TRV duplicate dividends);
  - F30: unversioned `feature_matrix.parquet`;
  - F31: Black–Litterman views and BLP fit.

**Other columns:**

- **Change:** how likely the conclusion is to change on repaired data with honest metrics: H (high), M (medium), L (low).
- **Re-run:** the step that revisits the study, or "—" with the reason it is not worth re-running.

| ID | Question | Conclusion (key numbers) | Promoted? | Defects touched | Change | Re-run |
|---|---|---|---|---|---|---|
| v9 | Simplest target, universe and features for stable OOS behaviour (features, targets, pooled families, classifiers, weekly snapshots) | Weakness structural; a reduced universe and lean features help; pooled targets do not; sign rules beat the tiered map | No (fed v10–v14) | D, M | M: "reduce breadth, stay lean" likely holds; which features are lean rested on F01/F06 inputs | v200, v201–v203 |
| v10, v10.1 | Promotion discipline and production hardening | "Promote with caveats"; conservative promotion rules | Process | — | L | — (process, no model claim) |
| v11 | Diversification-aware universe and candidate models | Universe VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE; best candidate `ensemble_ridge_gbt`; best policy row historical mean + 3 % band | **Yes:** the 8-benchmark `PRIMARY_FORECAST_UNIVERSE` | D, M, F03 | M: diversification scores stand; accuracy ranking used corrupted VOO targets | v206 |
| v12 | Should a diversification-first baseline replace the live engine? | Same universe; ensemble best utility; historical mean + band best policy | Fed v13 | D, M | M | v206, v208 |
| v13 | Implement the simpler recommendation layer | `shadow_promoted` mode | Production release | — | L | — (implementation) |
| v14 | Can the prediction layer be simplified on the reduced universe? | `ensemble_ridge_gbt` best; keep | Kept | D, M | M | v205 |
| v15 | Which of 46 report-suggested features improve Ridge/GBT (one-for-one swaps)? | e.g. `rate_adequacy_gap_yoy` for `vmt_yoy` in GBT (R² Δ +0.117); produced the v15 lean sets | **Partly:** several live features | D (F01, F06, F10, F11, F15, F16), M | **H:** the screen ranked features on broken inputs with the leaky R² | v201–v203 |
| v16 | Promote the modified Ridge+GBT pair? | `shadow_for_v17` (edge too small) | Shadow | D, M | M | v205 (via v200) |
| v17 | Replace the live cross-check with it? | Keep the current cross-check | No | D, M | L | — (superseded by the v200 baseline) |
| v18 | Benchmark-side and peer-relative swaps to cut directional bias | Best swaps: `vwo_vxus_spread_6m` for `real_rate_10y` (GBT), `real_yield_change_6m` for `yield_curvature` (Ridge) | **Yes:** the live v18 lean feature sets | D (F01 momentum/vol; F15 relative spreads on unadjusted closes), M | **H:** the live features were chosen on the broken versions | v201–v203, v205 |
| v19 | Backfill macro/valuation series; finish the 46-feature swap cycle | Leader `ridge_lean_v1__v15_best` (IC 0.238, R² −0.636); peer CR and FCF yield blocked | No; `v19.py` wrote the duplicate FRED rows | D, M, F06/F27 | H | v202, v203 |
| v20 | Promotion readiness of the assembled stacks | Continue research; `ensemble_ridge_gbt_v18` | No | D, M | M | v200 |
| v21 | Promote the v18 stack as the cross-check? | `promote_candidate_cross_check` | **Yes:** later the live model | D, M | M–H: agreement with a baseline on broken features | v200, v205 |
| v22 | Implement the cross-check promotion | Implemented | Production release | — | L | — |
| v23 | Does v21 survive with pre-inception proxies? | Extended history confirms | No | D, M, F27 (proxy rows) | M | — (proxy history not recommended; universe in v206) |
| v24 | Replace VOO with VTI? | Keep VOO | Kept | D, M, **F03** (70.8 % of VOO's 6M target sum of squares was split error) | **H** | v206 |
| v25 | Peer-review synthesis; re-run v20–v24 with guards | Continue; promote cross-check | — | D, M | M | — (superseded by v200) |
| v26 | Diagnostic warning clean-up | — | Infra | — | L | — |
| v27 | Redeploy portfolio for sale proceeds | Bounded dynamic allocation; small bond sleeve | Reporting | F09 (`redeploy_buckets` roe), F21, F03 | M | v208 (sizing only) |
| v28 | Prune the forecast universe? | Keep the current universe | Kept | D, M, F03 | M | v206 |
| v29 | Interpretation layer (confidence snapshot, benchmark roles) | Reporting change | Reporting | — | L | — |
| v30–v36 | Peer-review follow-ups (code quality, tests, property tests) | Engineering | Infra | — | L | — |
| v37 | Baseline of the v11 lean Ridge+GBT ensemble | Pooled R² −0.227, IC 0.158, hit 0.700 | Measurement | D, M | n/a | v200 (replaces it) |
| v38 | Post-hoc shrinkage ŷ·α | α 0.50 best, R² −0.131 | **Yes:** the shrinkage rule (live, prequential since v178) | D, M (shrinking toward 0 scored against a leaky naive; chosen on the reporting window) | **H:** the honest naive is the prevailing mean, so shrinking toward it may beat shrinking toward 0 (step 5 left this open) | v205 |
| v39 | Ridge α grid extension | Worse than v38 | No | D, M, F25 (LOO chose α 3.7 vs ~110 under TS CV) | M | v205 |
| v40 | Ridge-only, constrained GBT, 80/20 blend | Constrained GBT −0.236 | No | D, M, F25 | M | v205 |
| v41 | Target winsorisation in folds | Worse | No | D, M, F25 | L | — (it tamed split-error outliers that no longer exist) |
| v42 | Expanding / decayed windows | Worse | No | D, M, F25 | M | v205 (window length) |
| v43 | 7-feature subsets | Worse | No | D, M | M | v201–v203 (feature families) |
| v44 | Blockwise PCA | Worse | No | D, M, F25 | L | — (small N; large margin; complexity) |
| v45 | BayesianRidge as primary | Materially worse | No (dropped in v11) | D, M | L | — (same) |
| v46 | Per-benchmark logistic classifier | Accuracy 0.653, BA 0.529, Brier 0.250 | Basis of the shadow classifier | D, M, F03 | M | v207 |
| v47 | Composite benchmark targets | Worse | No | D, M, F03, F25 | L | — (consensus weighting is tested in v206 instead) |
| v48 | Panel pooling with fixed effects | Worse | No | D, M, F25 | M: pooling reduces variance, may help on clean data | v205 (one candidate) |
| v49 | Regime features (hard market, high vol, inverted curve) | Worse | No | D (**F01** vol regime, **F06** curve), M, F25 | **H** | v201, v202 |
| v50 | Prediction winsorisation | Clip+shrink −0.230 | No | D, M, F25 | L | — (calibration handled in v205) |
| v51 | Peer pooling / two-stage sector signal | Strongly negative | No | D, M, F25, **F27** | L | — (until peer data is repaired) |
| v52 | 1M/3M test windows | Strongly negative R² | No | D, M, F25 | L | — (evaluation design, not a model) |
| v53 | ARDRegression | Strongly negative | No | D, M | L | — |
| v54 | Gaussian processes | Best −0.283 | No | D, M, F25 | L | — |
| v55 | Rank-target transforms | Strongly negative | No | D, M, F03, F25 | L | — (outlier taming no longer needed) |
| v56 | 12-month horizon | Strongly negative | No | D, M (the leaky naive is worst with 12M overlap), F03 | H for the number, L for the decision (the decision is 6-monthly) | v200 (descriptive only) |
| v57 | Logs, rank-normalisation, lags | Rank-norm GBT −0.244 | No | D, M, F25 | L | — |
| v58 | Domain-specific FRED features | Strongly negative | No | D (**F06**), M, F25 | **H** | v202 |
| v59 | Imputation, 18-feature panel | Strongly negative | No | D, M, F25 | L | — (F07/F17 removed most missing values) |
| v60 | Clark–West, MSE decomposition, CE gain | CW t 3.36 (p 0.0004); CE +0.033; variance share 38 % | Diagnostics | D, M | M | v200 (CW vs prevailing mean) |
| v61–v65 | — | No record found (numbering gap) | — | — | — | — |
| v66–v69 | Reconstructed OOS paths; Clark–West and benchmark-quality reporting | Reporting infrastructure | Infra | M | L | — |
| v70 | Per-benchmark prequential shrinkage | R² −0.104 vs −0.131; weaker hit rate | No ("worth tracking") | D, M | **H:** small margin on a leaky metric | v205 |
| v71 | Prequential affine recalibration | Degraded (−0.153 to −0.159) | No | D, M | M | v205 |
| v72 | Benchmark-quality-weighted consensus | R² −0.045, NW IC 0.362 | **Yes:** live direction (decision 0002) | D, M, **F13** (weights fitted on the graded record) | **H** | v206 |
| v73 | Hybrid gate (v38 regression + v46 probabilities) | No accuracy gain; kept as a design pattern | No | D, M | M | v208 |
| v74 | `consensus_shadow.csv` shadow tracking | Infra | Infra | — | L | — |
| v75 | "Holdout" replay 2024-04 … 2026-03 for v74 | Quality-weighted IC 0.136 vs 0.127; 100 % mode and sell agreement | **Yes:** supported v76 | D, M, F13, **F02** (modes always DEFER, so agreement was uninformative); it consumed the v37 holdout | **H** | v206 |
| v76–v86 | Promotion, stabilisation, e-mail/dashboard/summary contract | Production releases | Production | — | L | — |
| v87 | Target taxonomy for classification | `actionable_sell_3pct` chosen | No | D, M, F03 | M | v207 |
| v88 | Feature-family sweep (classifier) | Lean baseline best | No | D (F01, F06), M | **H** | v207 |
| v89 | Per-benchmark linear classifiers | Balanced logistic best | No | D, M | M | v207 |
| v90 | Pooled panel classifiers | Pooled shared logistic best BA | No | D, M | M | v207 |
| v91 | Nonlinear classifier sweep | No gain | No | D, M | L | — (small N) |
| v92 | Calibration and abstention | Prequential calibration improved ECE; separate logistic (0.30, 0.70) | No | D, M (F13) | M | v207 |
| v93 | Basket vs panel targets | Panel beats basket and breadth | No | D, M | L | — |
| v94 | Hybrid classifier + regression gate | Classifier-only panel gating best | No | D, M, F02 | M | v208 |
| v95 | Policy replay of hybrids | Classifier-only gating best policy | No | D, M, F02 | M | v208 |
| v96 | Classification programme decision | `continue_research_no_promotion`; shadow only (decision 0004) | **Shadow:** `classification_shadow.py` | D, M | M | v207 |
| v97 | Deep-research review prompt | Docs | — | — | L | — |
| v98–v101 | — | No record found | — | — | — | — |
| v102–v109 | Archive intake, summary contract, rendering, dashboard, orchestration, as-of leakage hardening (v107), tests, classifier history (v109) | Engineering | Infra | F22 (v107 kept a post-as-of target) | L | — |
| v110 | Gemini-style veto gate | Best `gemini_veto_0.50` | Fed v113 | D, M | M | v207 |
| v111 | Permission-to-deviate overlay | Best candidate (superseded by v113) | No | D, M | L | — |
| v112 | Narrow target reformulation | Best formulation noted | No | D, M | L | — (target choice is in v207) |
| v113 | Constrained selection of promotable candidates | `gemini_veto_0.50`, "promotion eligible" | **Shadow:** production reads its CSV (`classification_gate_overlay`) | D, M, **F13**, F02 | **H** | v207 |
| v114–v117 | Overlay summary, monitoring summary, limited gate, mode selector | Defer classifier-led modes | Derived summaries | D, M, F02, **F24** (monitoring never matured) | L | — (regenerated by v207) |
| v118–v121 | Prospective shadow replay, scorecard, assessment, phase summary | `advance_to_real_time_shadow_monitoring` | Shadow monitoring | D, M, **F22** (backdated as-of truncation leaked), F24 | M | v207 (one clean prospective replay) |
| v122 | Classifier audit (coefficients) | Audit note | — | D | L | — |
| v123–v124 | Portfolio-alignment plumbing; VGT and VIG added to the investable pool | Shadow release | Shadow | **F05** (VGT labels) | — | — (plumbing) |
| v125 | Path B composite portfolio-target classifier | Better covered BA than Path A, worse calibration | **Shadow:** `path_b_classifier.py` | D, M, **F05** (VGT weight 0.20 in the composite) | **H** | v207 |
| v126 | Methodology hardening (matched Path A/B folds) | Infra and tests | Infra | — | L | — |
| v127 | Path B calibration sweep | Best `path_b_platt`; `selected_next` none (BA Δ −0.145) | Shadow | D, M, F05, F13 | **H** | v207 |
| v128 | Benchmark-specific feature search (72 features) | BND, DBC, VIG, VGT switched; VGT BA 0.947 with 2 features | **Shadow:** production reads the map (`V128_BENCHMARK_FEATURE_MAP_PATH`) | D (F01, F06, F09 `roe`), M, **F05**, **F13** (72-feature search, same OOS) | **H** | v207 |
| v129 | Feature-map evaluation; VGT robustness audit | VGT not adopted (unstable); dual-track shadow | Shadow | D, M, F05 | **H** | v207 |
| v130 | Path B temperature-scaling adoption | BA Δ +0.0725 vs matched Path A → adopt (shadow) | Shadow | D, M, F05 | **H** | v207 |
| v131 | Asymmetric abstention thresholds | Found (0.15, 0.70) | Retained (thresholds stay 0.30/0.70) | D, M, F05 | M | v207 |
| v132 | Temporal hold-out of thresholds | DO NOT ADOPT (BA Δ 0) | Retained | D, M, F05 | L | — (folded into v207) |
| v133 | Ridge `alpha_max` sweep | Best 1000 (R² −0.455) | No | D, M | M | v205 |
| v134 | FRED publication-lag sweep | Keep all-ones (−0.1573 vs −0.1578); its "lag 0" was lag 1 | **Retained:** `FRED_SERIES_LAGS` | D (**F06**), M | **H:** the inputs were double-lagged, and lags are availability rules, not tunables | v202 (audit instead of sweep) |
| v135 | Path B temperature parameter search | Best (2.5, 42): BA 0.699, coverage 0.548 | Shadow parameter | D, M, F05 | M | v207 |
| v136 | Backlog ranking (persona-scored) | Process | — | — | L | — |
| v137 | GBT hyperparameter sweep | Depth 1, 25 trees, lr 0.05, subsample 0.8 (R² −0.268) | No | D, M | M | v205 |
| v138 | Black–Litterman parameter replay | τ 0.05, view confidence 0.75; sell precision 0 | No | D, M, **F31** | L | — (BL is diagnostic; fix F31 first) |
| v139 | Autoresearch follow-on scaffold | Infra | Infra | — | L | — |
| v140 | Shrinkage evaluation | Flat over 0.35–0.65; keep 0.50 | **Retained** (now research-only) | D, M | M | v205 (with v38) |
| v141 | Fixed Ridge/GBT blend | Ridge weight 0.60 (R² −0.1624 vs −0.1634) | Shadow (follow-on lane) | D, M | M | v205 |
| v142 | EDGAR filing-lag evaluation | Keep lag 2 (lag 1 IC 0.149 vs 0.126) | Retained, now only a fallback | D, M, **F23** (lag 0 is look-ahead) | L | — (superseded: rows are placed by filing date since v179; v203 only audits the placement) |
| v143 | Correlation-pruned features | ρ 0.80 (IC 0.141 vs 0.126) | Shadow | D, M | M | v205 |
| v144 | Conformal coverage backtest | (0.75, 0.03), coverage gap −0.001 | Shadow | D, M, **F13** (coverage in-sample; trailing is ~43 %) | **H** | v205 |
| v145 | WFO train/test window sweep | Keep (60, 6); (48, 6) IC 0.186 but lower hit rate | **Retained** | D, M (hit rate vs 50 %) | M | v205 |
| v146 | Path B threshold sweep | Keep | Shadow | D, M, F05 | L | — (in v207) |
| v147 | Coverage-weighted Path A/B aggregate | Keep multiplier 1.0 | No | D, M | L | — |
| v148 | Positive-class weight | Keep 1.0 | No | D, M | L | — |
| v149 | Kelly fraction/cap replay | {0.50, 0.25} (more aggressive) | Shadow | D, M, F21, F31 | M | v208 |
| v150 | Neutral band on Kelly | Keep 0.015 | Shadow | D, M | L | v208 (one candidate) |
| v151–v153 | Follow-on shadow lane; synthesis docs; next plan | Shadow release and docs | Shadow | — | L | — |
| v154 | Firth logistic for thin benchmarks | Adopt for VMBS (+0.041) and BND (+0.070) | Shadow (v159) | D, M | M | v207 |
| v155 | WTI 3M momentum (DBC/VDE) | No benefit (+0.005, +0.021) | No | D (**F06**), M | **H** | v202 |
| v156 | USD momentum (BND/VXUS/VWO) | No benefit (BND −0.077) | No | D (**F06**), M | **H** | v202 |
| v157 | Term-premium 3M differential | No benefit (best VDE +0.017) | No | D (**F06**), M | **H** | v202 |
| v158 | Synthesis of v153–v157 | Firth is the sole winner | — | — | — | — (summary) |
| v159–v161 | Firth shadow integration; TA scaffold and feature factory | Shadow/infra | Shadow | F01/F24 (TA factory) | L | — |
| v162 | TA broad screen | Survivor list | No | D, **F01/F24** (weekly unadjusted OHLCV with daily windows), M, F13 (broad screen) | **H** | v201 |
| v163 | TA survivor confirmation | Survivors confirmed | No | same | **H** | v201 |
| v164 | TA synthesis | `replacement_candidate`: OBV replaces `mom_12m` (BA +0.038), NATR replaces `vol_63d` (+0.026) | No | same; the replaced features were the broken ones | **H** | v201 |
| v165 | TA classification replacement shadow | `shadow_monitor`; `ta_minimal_plus_vwo_pct_b` BA +0.058, 8/8 | **Shadow** (decision 0005) | D, F01/F24, M | **H** | v201, v207 |
| v166–v170 | TA ledger, freshness, verifier, docs | Infra | Infra | — | L | — |
| bl01 | Black–Litterman τ / risk-aversion sweep | Keep τ 0.05, RA 2.5 (Δ +0.009 < 0.05) | **Retained** | **F31**, F21 | L | — (diagnostic only; F31 unfixed) |
| pb_vs_pe | Is P/B or P/E the better predictor of forward returns? | P/B vs P/E edge came from 2023+ (roe_gap R² +0.184 from 2023, −0.006 before); ~494 combinations; NW t −2.60 vs Hodrick −1.40 | No | **F25**, F12 (EPS → P/E), F15 (P/B at 2006), F09 | **H** | v204 |
| test_runtime | Faster test suite | Infra | Infra | — | L | — |
| x1 | x-series feature inventory and target sufficiency | 22 annual special-dividend snapshots; target utilities | Research-only | **F25** (raw-price `shift(−h)`, ME/BME mix, missed December specials), **F30**, F15 | **H** | v209 |
| x2 | Absolute PGR direction classification | Did not clear the base-rate gate | No | F25, F30, F01 | L | — (the base-rate framing was already honest; absolute direction is not the decision target) |
| x3 | Direct forward-return regression | Baseline-heavy; only 12m drift cleared no-change | No | F25, F30, F01 | L | — (same) |
| x4 | BVPS forecasting | Strongest early lane; beat no-change at all 4 horizons | No | **F15** (raw BVPS across 2006), F18, F12, F30 | **H** | v209 |
| x5 | BVPS × P/B decomposition | Stable anchor `no_change_pb` | No | F15, F25, F30 | M | v204 |
| x6 | Special-dividend two-stage sidecar | Low confidence (22 snapshots) | No | F08, F25, F30 | M | v209 |
| x7 | Targeted TA for x-series | `ta_minimal_plus_vwo_pct_b` cleared 2/4 horizons | No | F01/F24, F30 | **H** | v201 |
| x8 | Cross-lane synthesis | `not_ready` | — | — | — | — (summary) |
| x9 | BVPS bridge features | Better at 1m/3m, not 6m/12m | No | F15, F18, F30 | **H** | v209 |
| x10 | Capital-enhanced dividend lane | Better EV MAE than x6; low confidence | No | F08, F25, F30 | M | v209 |
| x11 | Synthesis x9/x10 | `continue_research` | — | — | — | — (summary) |
| x12 | Raw vs adjusted BVPS target audit | Adjustment helped 3m/6m; recorded the 2006 split as a −74.9 % "capital event" | No | **F15**, F30 (re-runs changed winners) | **H** | v209 |
| x13 | Adjusted decomposition | Only the 6m adjusted structural path survived | No | F15, F25, F30 | **H** | v204 |
| x14 | Indicator synthesis | One bounded 6m structural candidate | — | — | — | — (summary) |
| x15 | P/B regime overlay | No overlay beat no-change P/B | No | F15, F30 | M | v204 |
| x16 | Structural indicator package | `adjusted_structural_bvps_pb_6m` | Research-only indicator | F15, F30 | **H** | v204 |
| x17 | Persistent BVPS | Helped 3m/6m | No | F15, F18, F30 | **H** | v209 |
| x18 | Dividend-policy regime audit and targets | December 2018 policy break; Dec–Feb window | No | F08, F25 | M | v209 |
| x19 | Post-policy dividend model | Better than x10 on 3 OOS years | No | F08, F15, F30 | M | v209 |
| x20 | Dividend-lane synthesis | Occurrence one-class; only size is identifiable | — | — | — | — (summary) |
| x21 | Dividend-size target scales | `special_dividend_excess / current_bvps` best | No | F15, F30 | M | v209 |
| x22 | Dividend-size baselines | `to_current_bvps` survived | No | F15, F30 | M | v209 |
| x23 | Dividend-lane package | Size indicator candidate; occurrence under-identified | Research-only | F08, F30 | M | v209 |
| x24 | Indicator contract (bundle) | Structural 6m + dividend-size watch | Research-only | as x16/x23 | M | v209 |

## 3. Assessment

### 3.1 Conclusions most likely to change

These were rejected, promoted or tuned on features or metrics that were broken at the time.

- **Momentum and volatility (F01).**
  - v15/v18 chose the live `mom_3m`, `mom_6m`, `mom_12m`, `vol_63d` and `vwo_vxus_spread_6m`. On the weekly bars these measured 14.5-, 29- and 58-month returns and a 63-week volatility scaled by √252, and they jumped at splits.
  - v49's high-vol regime and v88's feature sweep used the same inputs.
  - v162–v165 and x7 "beat" `mom_12m`/`vol_63d` with TA features that were themselves on unadjusted weekly bars with daily windows.
- **Macro lags (F06).**
  - v134 swept lags on a DB that was already lagged once, so its "lag 0" was lag 1.
  - v58 and v155–v157 rejected macro features that reached the model with 2–4-month effective lags (1–2 months more than intended).
- **EDGAR fundamentals (F09–F12, F16–F18, F23).**
  - v15/v19's insurance features, v128's `roe` for BND, and the x-series BVPS lanes (x4, x9, x12, x17), where the 2006 split looked like a −75 % capital event.
- **The R² gate (F04).**
  - Every v37–v60 and v133–v145 verdict ranked variants by an R² whose naive contained the target. The pooled production R² was −1.1 % on that basis and +10.8 % to +13.9 % on look-ahead-free bases.
  - The differences that decided v134, v140, v141 and v143 (0–0.001 in R²) are noise.
  - v38's shrink-toward-zero won against the leaky naive.
- **CPCV (F02).** Every month was forced to DEFER, so the "mode agreement" and "would change the recommendation" statistics in v75, v94–v95 and v113–v121 compared against a constant. They carry no information.
- **Calibration (F13).** v92, v127, v130 and v135 (probability calibration) and v144 (conformal coverage 0.749 vs target 0.75) were scored in-sample. Prequential ECE is 0.14–0.19, and trailing coverage is 41–49 % against 80 %.
- **Labels (F03/F05).**
  - VOO's 2013 split error was 70.8 % of VOO's 6M target sum of squares (v11, v24, v28).
  - VGT's 10 newest labels were sign-flipped. That affects every Path B and investable-pool result from v124 on (v125, v127–v135, v128's VGT BA 0.947).

### 3.2 Not worth re-running

- **Complex model classes on ~200 monthly observations:** v44 (PCA), v45 (BayesianRidge), v53 (ARD), v54 (GPR), v91 (nonlinear classifiers). They lost by wide margins; the data regime (large P, small N) is unchanged, and AGENTS.md prefers high-bias models.
- **Outlier-taming devices whose outliers were data errors:** v41 (target winsorisation), v50 (prediction winsorisation), v55 (rank targets), v57 (rank transforms). They addressed the 100-point split errors, which the fixes removed.
- **Designs made moot:**
  - v142: EDGAR rows are now placed by filing date, so the lag is an availability rule.
  - v52: shorter test windows is an evaluation choice, not a model.
  - v59: imputation needs fell once F07/F17 were fixed.
  - v47: composite targets are covered by v206's consensus question.
  - v93: basket targets lost clearly.
- **Blocked by data that is still broken:** v51 (peer pooling) until the CB/TRV peer data (F27) is repaired; v138 and bl01 (Black–Litterman) until F31's view construction and BLP fit are fixed.
- **Derived summaries and plumbing:** regenerated automatically when their inputs are re-run.
  - summaries: v114–v117, v122, v132, v146–v148, v158, x8, x11, x14, x20;
  - plumbing: v74, v102–v109, v123–v124, v126, v139, v151–v153, v159–v161, v166–v170.
- **Superseded by the v200 baseline:** v17, v20, v22, v23, v25, v37.
- **x-series absolute-direction lanes:** x2 and x3 already used an honest base-rate gate, and absolute direction is not the decision target.
- **Process and docs:** v10, v13, v26, v29, v30–v36, v66–v69, v76–v86, v97, v136, test_runtime.

### 3.3 Research groups, order and budgets

Later steps build on earlier baselines. v201–v204 each test one feature family against the v200 specification. v205 starts from v200 plus any feature finalists. v206–v208 use the best regression and classification outputs. v209 is an independent lane that can run any time after v200. v210 is last.

| Step | Area | Revisits | Candidates (budget) | Depends on |
|---|---|---|---|---|
| v200 | Clean baseline and honest evaluation harness | v37, v60, v11/v14/v21 (live stack), v56 (descriptive), v75 | 0 (measurement only) | dividend refresh (verification N1) |
| v201 | Price and technical features | v15, v18, v43, v49, v162–v165, x7 | 10 | v200 |
| v202 | Macro/FRED features and publication timing | v134, v58, v19, v155–v157, v49 | 8 | v200 |
| v203 | Insurance fundamentals and EDGAR features | v15, v19, v142 (placement audit), F16 fiscal calendar | 9 | v200 |
| v204 | Valuation (P/B, P/E, `pb_vs_pe`, structural P/B) | pb_vs_pe, x5, x13, x15, x16 | 6 (+ ≤ 20 descriptive tests, Holm-corrected) | v200, v203 |
| v205 | Model class, regularisation, shrinkage, intervals | v38–v40, v42, v48, v70, v71, v133, v137, v140, v141, v143–v145 | 12 | v200–v204 |
| v206 | Benchmark universe and consensus weighting | v11, v12, v24, v28, v72, v75 | 6 | v205 |
| v207 | Classification and shadow layer | v46, v87–v90, v92, v96, v110, v113, v118–v121, v125, v127–v131, v135, v154, v165 | 10 | v200–v203 |
| v208 | Decision policy and tax mapping | v9, v12, v27 (sizing), v73, v94, v95, v149, v150, decision 0007 | 5 | v205–v207 |
| v209 | Dividend and BVPS x-series lanes | x1, x4, x6, x9, x10, x12, x17–x24 | 8 | v200 |
| v210 | Holdout synthesis and promotion recommendation | all finalists | ≤ 9 finalists (1 per step) | all |

Total budget: 74 development candidates, plus up to 20 descriptive valuation tests, plus at most 9 holdout finalists.

### 3.4 Guardrails (all steps)

- **Holdout.**
  - v200 freezes it: the 24 most recent month-end forecast dates whose 6M target is realised in the pinned DB.
  - Development uses only forecast dates at least 6 months before the holdout start, so no development target window overlaps it (12 months before for 12M targets).
  - The holdout is used once, in v210.
  - Honest caveat: the holdout is untouched by v200–v209, but not by the pre-v200 research that designed today's model. The v75 replay covered 2024-04 … 2026-03, v129 audited as-of dates up to 2024-03-31, and v132's temporal hold-out started 2022-01. The holdout is therefore a falsification test (a candidate must not do worse than the baseline), not proof of skill.
- **No live changes.** No step edits `config/` or production behaviour. A finalist is a JSON file. A promotion is a separate PR with a decision record under `docs/model-governance.md` and `docs/decisions/`.
- **Parallelism:** at most 2 subagents at once, in any step.
- **Provenance:** every study writes `provenance.json` (see the preamble). Every evaluated candidate goes to `outputs/candidates.csv`, including failed ones.

## 4. Shared preamble (paste first in every session)

> You are running one step of the v200 research re-run for `jhester599/pgr-vesting-decision-support`.
>
> **Background.**
> - The 2026-09-25 review (`docs/reviews/REPO_REVIEW_2026-09-25.md`) found defects in the data, targets and validation. CHANGELOG v171–v185 fixed them, and `docs/reviews/VERIFICATION_2026-09-26_claude.md` verified the fixes ("verification N1" … "N10" are its new issues).
> - Every research conclusion before v200 rests on the broken data or metrics. The plan is `docs/research/RERUN_PLAN_v200_claude.md`: read its sections 2–3 and your step.
>
> **Rules.**
> 1. **Branch and PR.** Branch from the latest `master` (`research/v2NN-<slug>` unless the session names a branch). Open one PR per step. Follow `AGENTS.md`:
>    - no K-fold, no shuffle, no scaler/imputer/selector fitted outside a training fold;
>    - walk-forward `TimeSeriesSplit` only;
>    - total returns from unadjusted prices through the repo's DRIP helpers and `monthly_relative_returns` (never `yfinance`).
> 2. **Pinned inputs.**
>    - Read `PIN_COMMIT`, `PIN_DB_SHA256`, `HOLDOUT_START`, `HOLDOUT_END` and `DEV_END` from `research/studies/v200_clean_baseline/provenance.json`. v200 itself creates them.
>    - Copy the DB with `git show $PIN_COMMIT:data/pgr_financials.db > $TMP/pin.db`, check its sha256, and open it read-only (`?mode=ro&immutable=1`).
>    - Never open the committed DB read-write. Never run a fetcher or `cli/monthly_decision.py` against it. Never call Alpha Vantage, FRED or EDGAR unless the owner asks. If the owner asks for EDGAR, send `User-Agent: Jeff Hester jeffrey.r.hester@gmail.com`, stay under 10 requests per second, and cache responses.
>    - Record `git rev-parse HEAD` and a clean-tree flag for the run that writes the committed outputs.
> 3. **Layout.**
>    - `research/studies/v2NN_<slug>/` holds `run.py`, `README.md`, `outputs/` (summaries; anything over 1 MB goes to the gitignored `outputs/detail/`) and `provenance.json`.
>    - Register the study in `research/registry.yaml` and regenerate `research/README.md` with `python research/tools/registry.py --write`.
>    - Tests go in `tests/research/test_research_v2NN_<slug>.py`. Reusable code goes in `src/pgr_vds/research_lib/`, with tests in `tests/research/`.
>    - No `sys.path` edits. Import `src`, `config` and `pgr_vds` through `pip install -e .`.
> 4. **Holdout.**
>    - Load data only through `pgr_vds.research_lib.frames`. `development_frame()` drops every target whose forecast date is after `DEV_END`, and raises if code asks for a holdout target before v210.
>    - Do not compute or look at any holdout metric, even informally.
> 5. **Metrics.** Use `pgr_vds.research_lib.evaluation` only:
>    - OOS R² against the prevailing mean of the targets realised by each forecast date, per benchmark, with the correct horizon. Pooled R² sums SSE over benchmarks, each against its own naive.
>    - Equal-weight mean rank IC, and pooled rank IC with a Driscoll–Kraay (date-clustered) p-value.
>    - Hit rate next to the base rate, with a one-sided Pesaran–Timmermann p-value.
>    - Prequential calibration (ECE with a date-block-bootstrap CI) for any probability.
>    - Clark–West against the nested baseline where it applies.
>    - Never report a hit rate without its base rate, or an IC p-value computed over pooled rows.
> 6. **Validation.**
>    - Outer loop: `TimeSeriesSplit(max_train_size=60, test_size=6, gap=8)` for 6M targets (gap 15 for 12M), matching production (60-month window, 6-month test, embargo 6 plus purge buffer 2).
>    - Every choice (feature selection, hyperparameters, shrinkage, calibration, thresholds, imputation, scaling) is made inside each training fold with a nested inner `TimeSeriesSplit` using the same gap, or prequentially from realised rows only.
> 7. **Multiple testing.**
>    - Before running anything, commit a "Pre-registration" section in the study README listing the candidates, their count (at most the step's budget), the primary metric and the success threshold.
>    - Log every evaluated candidate in `outputs/candidates.csv`.
>    - "Beats v200" means a paired date-block bootstrap (6-month blocks, 2,000 draws, seed 20260926) of the metric difference, one-sided, with Holm correction across the step's candidates.
> 8. **Success rule for regression candidates ("R-rule").** On the development period, against v200, a candidate must:
>    - raise pooled OOS R² by ≥ +1.0 pp with Holm-adjusted p < 0.05;
>    - not lower the equal-weight mean IC by more than 0.01;
>    - not lower (hit rate − base rate) by more than 1 pp;
>    - not lower any benchmark's R² by more than 3 pp;
>    - have a live-mapping policy uplift over always-50 % whose 90 % CI lower bound is ≥ −0.25 pp per decision.
>
>    At most one candidate per step becomes the finalist (`outputs/finalist.json`: the full specification, not a fitted model). If none passes, write `{"finalist": null}`.
> 9. **No live changes.** Do not edit `config/`, production modules, workflows or `artifacts/`. Promotion happens only after v210, in a separate PR under `docs/model-governance.md`.
> 10. **Parallelism.** At most 2 subagents at once.
> 11. **Finish.**
>     - Run the step's tests, then the full suite with `python -m pytest -o addopts="--tb=short" -q`, and report pytest's summary line.
>     - Confirm that the committed DB's sha256 is unchanged.
>     - Add a CHANGELOG entry ("Research v2NN — …"). Docs change only if the step touches documented behaviour.
>     - End with a short "what changed / what's left" note, including candidates used vs budget.
>
> `provenance.json` fields:
> - `study_id`, `git_commit`, `git_dirty`, `pin_commit`, `pin_db_sha256`, `max_price_date`, `max_target_date_6m`, `dev_end`, `holdout_start`, `holdout_end`, `feature_matrix_sha256`;
> - `python`, `pandas`, `numpy`, `sklearn` and `xgboost` versions;
> - `command`, `seed`, `candidates_declared`, `candidates_run`, `run_started_utc`, `run_finished_utc`.

## 5. Step prompts

### v200 — Clean baseline and honest evaluation harness

> **Hypothesis.** On the repaired data and with honest metrics, the current production specification has positive OOS R² against the prevailing mean, but no directional skill beyond the base rate. The verification's replays suggest this (R² +1 % to +6 %, PT p 0.36–0.62). The specification is: Ridge + GBT, v18 lean feature sets, prequential shrinkage, quality-weighted consensus over the 8 `PRIMARY_FORECAST_UNIVERSE` benchmarks, 6M PGR-minus-ETF total-return target. This step measures it on the development period, freezes the holdout and builds the harness. It selects nothing.
>
> **Revisits:** v37 (baseline), v60 (Clark–West, now against the prevailing mean), v11/v14/v21 (the live stack), v75 (the old "holdout" replay), and v56 (12M horizon, descriptive only).
>
> **Preconditions.** Stop and report if any fails:
> 1. `db_client.check_dividend_freshness` on the candidate pin reports no STALE ticker among the 8 primary benchmarks. Today's DB (`7c68efbd…`, `aae0be8`) fails this (verification N1). The owner must first dispatch "Weekly Data Accumulation" with `dividend_refresh: true`.
> 2. `price_integrity.find_unexplained_price_jumps` is empty.
> 3. `tests/integration/data/test_pgr_edgar_integrity.py` passes with `PGR_EDGAR_INTEGRITY_DB` pointing at the pin copy.
>
> **Inputs.**
> - `PIN_COMMIT` is the first `master` commit whose DB passes the preconditions; record it and its `PIN_DB_SHA256`.
> - Code is this branch's HEAD.
>
> **Build `src/pgr_vds/research_lib/`:**
> - `frames.py`: load the pin copy read-only, build the feature matrix with `feature_engineering._PROCESSED_PATH` redirected to a temp directory, load targets, apply as-of truncation. `development_frame()` and `holdout_frame()` (the latter raises unless `ALLOW_HOLDOUT=v210`).
> - `holdout.py`: the frozen dates.
> - `evaluation.py`:
>   - R² through `src/reporting/backtest_report.compute_oos_r_squared` with `horizon_months` and a per-benchmark `benchmark_forecast` from `src/models/prequential.prevailing_mean_forecast`;
>   - Driscoll–Kraay and Pesaran–Timmermann from `src/models/robust_inference.py` (`pesaran_timmermann_test`), the functions production uses;
>   - Clark–West, prequential ECE with a date-block CI, the paired date-block bootstrap, and Holm.
> - `splits.py`: outer and inner `TimeSeriesSplit` factories that assert gap ≥ horizon and that no test index precedes a train index.
> - `ledger.py` (`candidates.csv`) and `provenance.py`.
> - Do not modify old studies or `src/research/v37_utils.py`. New code must not call `v37_utils.compute_metrics`, which has the horizon-1 look-ahead (verification N2).
>
> **Tests (`tests/research/test_research_lib_*.py`):**
> - an oracle forecaster on overlapping 6M targets scores R² > 0 (the seed-7 series in verification N2 gives ≈ +4.9 %);
> - the naive never uses a target realised after the forecast date;
> - pooling two benchmarks does not inflate R² (N2's concatenation case);
> - the DK p-value clusters by date;
> - an always-same-sign predictor gets an undefined PT p;
> - a split with gap < horizon raises;
> - the holdout guard raises.
>
> **Holdout.**
> - `HOLDOUT` is the 24 most recent month-end forecast dates whose 6M target is realised in the pin. If the last realised 6M target is 2026-03-31, it runs from 2024-04-30 to 2026-03-31.
> - `DEV_END` = `HOLDOUT_START` − 6 months (12 months for 12M targets). The months in between are unused.
> - Write the dates to `provenance.json`, `outputs/holdout_definition.json` and `research_lib/holdout.py`.
>
> **Method.**
> 1. **Harness check.** Run the production code path (`pgr_vds.decision.signal_generation.generate_signals` and `pgr_vds.decision.health.compute_aggregate_health`) on the pin with as-of = `DEV_END`, and compare its aggregate R², ICs, hit rate and PT p with `research_lib` on the same as-of. Every metric must agree to 1e-9.
> 2. **Baseline.** Re-evaluate the production specification walk-forward on development targets only, as production does (60/6, gap 8).
> 3. **Report, per benchmark and pooled:**
>    - OOS R² (prevailing mean), equal-weight and quality-weighted mean IC, pooled IC and DK p;
>    - hit rate vs base rate and PT p; CW p;
>    - prequential ECE of the calibrated P(outperform); trailing conformal coverage;
>    - the production gates month by month over development;
>    - the live-mapping policy value vs always-50 % and always-hold (`src/models/live_policy_backtest.py`).
> 4. **Reference rows (not candidates):** zero forecast; prevailing mean; "always PGR outperforms"; equal-weight consensus; no shrinkage.
> 5. **Descriptive only:**
>    - 12M-horizon metrics;
>    - regime slices (before/after 2014, 2020, 2022);
>    - the classifier baseline for v207 (the production lean-baseline logistic and Path B: BA, covered BA, coverage, Brier, prequential ECE);
>    - the policy baseline for v208.
>
> **Budget:** 0 selection candidates.
>
> **Outputs** (`research/studies/v200_clean_baseline/outputs/`):
> - `baseline_metrics.csv`, `baseline_monthly_gates.csv`, `classifier_baseline.csv`, `policy_baseline.csv`, `holdout_definition.json`, `harness_check.json`, `summary.md`;
> - `provenance.json` in the study folder.
>
> **Success threshold:** none (this is the baseline). **Acceptance:** the harness check agrees to 1e-9, and all `research_lib` tests fail when the leaky naive is swapped in.

### v201 — Price and technical features

> **Hypothesis.** With correctly specified momentum and volatility (calendar months, split-adjusted, weekly bars), price features add OOS skill to the v200 specification. At most one TA replacement from v164/v165, recomputed on the fixed TA builder, beats the input it replaces.
>
> **Revisits:** v15/v18 (the swaps that chose `mom_12m`, `vol_63d`, `mom_3m`, `mom_6m`, `vwo_vxus_spread_6m`), v43, v49 (high-vol regime), v162–v165 and x7.
>
> **Inputs:** `PIN_COMMIT` and `PIN_DB_SHA256` from v200; development period only.
>
> **Method.** The v200 walk-forward, with only the named model's feature list changed. TA features come from `src/research/v160_ta_features.build_ta_feature_matrix(..., split_map=...)` on weekly bars. Show one 2006-05 row to prove split invariance.
>
> **Candidates (10):**
> 1. v200 without any price features (ablation);
> 2. `mom_12m` → 12-1-month momentum;
> 3. `vol_63d` → 26-week volatility;
> 4. + `high_52w`;
> 5. + `pgr_vs_kie_6m` (split-adjusted);
> 6. `ta_pgr_obv_detrended` replaces `mom_12m`;
> 7. `ta_pgr_natr_63d` replaces `vol_63d`;
> 8. `ta_minimal_replacement` (v165);
> 9. `ta_minimal_plus_vwo_pct_b` (v165);
> 10. + high-volatility regime flag (v49, from the corrected `vol_63d`).
>
> **Metrics:** as in the preamble, against v200.
>
> **Budget:** 10.
>
> **Outputs:** `research/studies/v201_price_technical/` (`outputs/candidates.csv`, `outputs/finalist.json`, `outputs/summary.md`, `provenance.json`).
>
> **Success threshold:** the R-rule. Also report the ablation (candidate 1) even if it fails: it says whether price features help at all.

### v202 — Macro/FRED features and publication timing

> **Hypothesis.** With each FRED series lagged once, by calendar month, at its minimal legal publication lag, macro features carry more signal than v58, v134 and v155–v157 found. Lags are availability rules, not tunables.
>
> **Revisits:** v134, v58, v19, v155 (WTI), v156 (USD), v157 (term premium), v49 (inverted curve).
>
> **Inputs:** the pin; development period.
>
> **Part A: audit, no selection.**
> - For each series in `config.FRED_SERIES_LAGS`, document the release schedule from FRED/BLS/Fed documentation already in the repo, or general knowledge marked as such. Do not fetch.
> - State whether the configured lag is the minimal legal one.
> - List the revision-prone series (NFCI, CPI, PPI, VMT).
> - Note the mixed aggregation (monthly-average GS2/5/10 vs month-end T10YIE/VIX; review F27).
>
> **Part B: candidates (8).**
> 1. v200 without the macro features (ablation);
> 2. the macro block at the Part A minimal lags (only if Part A finds a shorter legal lag; otherwise drop it and do not count it);
> 3. + `usd_momentum_6m`;
> 4. + `wti_return_3m`;
> 5. + `term_premium_diff_3m`;
> 6. + insurance-pricing block (`ppi_auto_ins_yoy`, `severity_index_yoy`);
> 7. `real_rate_10y` rebuilt from same-aggregation inputs (both monthly averages);
> 8. + inverted-curve regime flag.
>
> **Budget:** 8.
>
> **Outputs:** `research/studies/v202_macro_fred/` (plus `outputs/lag_audit.md`).
>
> **Success threshold:** the R-rule. Part A findings that imply a production lag change are written up as a proposal only.

### v203 — Insurance fundamentals and EDGAR features

> **Hypothesis.** With the repaired EDGAR table, PGR's own fundamentals improve OOS skill once they enter at filing date: signs, PIF definition, gaps, equity, Q4 ROE, book-yield units. Monthly YoY inputs must be made robust to the suspected 2024 fiscal-calendar change.
>
> **Revisits:** v15 and v19 (insurance features), v142 (the lag, now filing-date placement: audit only), and the live EDGAR inputs touched by F09–F11, F16 and F23.
>
> **Inputs:** the pin; development period.
>
> **Part A: audit, no selection.**
> - Test the 2024 fiscal-month hypothesis (verification N5) from the table's NPE/NPW first-month-of-quarter pattern, 2016–2026.
> - Compare monthly YoY with trailing-12-month and quarterly flow growth.
> - Confirm that every EDGAR-derived feature enters on the first business month-end on or after `filing_date` (the verification's placement check).
>
> **Part B: candidates (9).**
> 1. v200 without EDGAR features (ablation);
> 2. `npw_growth_yoy` → TTM NPW growth;
> 3. `pif_growth_yoy` → TTM-average PIF growth;
> 4. + `gainshare_estimate`;
> 5. + `roe_net_income_ttm` (8-K);
> 6. + `roe` (XBRL TTM NI / average equity);
> 7. `combined_ratio_ttm` + 3-month combined-ratio change;
> 8. + underwriting margin TTM;
> 9. + `buyback_yield`.
>
> **Budget:** 9.
>
> **Outputs:** `research/studies/v203_insurance_fundamentals/` (plus `outputs/fiscal_calendar_audit.md`).
>
> **Success threshold:** the R-rule.

### v204 — Valuation (P/B, P/E, `pb_vs_pe`, structural P/B)

> **Hypothesis.** PGR's valuation predicts 6M relative returns out of sample. The inputs are P/B against ROE, P/E, and the x-series structural P/B, all now on one share basis with correct EPS signs. The `pb_vs_pe` conclusion ("P/B beats P/E, driven by `roe_gap`") does not survive honest inference on data up to `DEV_END`.
>
> **Revisits:** `pb_vs_pe`, x5, x13, x15, x16.
>
> **Inputs:** the pin; development period only. The `pb_vs_pe` edge came from 2023+, which is partly holdout: stop at `DEV_END`.
>
> **Part A: honest re-analysis (descriptive, ≤ 20 pre-registered tests, Holm).**
> - Rebuild `pgr_valuation_monthly.csv`-style inputs in-process from the pin.
> - Report, per era (2004–2014, 2015–`DEV_END`):
>   - Hodrick 1B or non-overlapping 6M inference;
>   - Clark–West against the prevailing mean;
>   - the count of all combinations the original study tried (~494), and why this re-analysis tests ≤ 20.
>
> **Part B: candidates (6).** Added to the v200 (or v203 finalist) specification: `pb_ratio`; `pe_ratio`; `roe_gap`; `pb_vs_pe`; `pgr_pe_vs_market_pe` (the Multpl series ends 2026-04, which is fine for development); x16 `adjusted_structural_bvps_pb_6m` rebuilt on split-adjusted BVPS.
>
> **Budget:** 6 (+ ≤ 20 descriptive).
>
> **Outputs:** `research/studies/v204_valuation/`.
>
> **Success threshold:** the R-rule for Part B. Part A only reports.

### v205 — Model class, regularisation, shrinkage and intervals

> **Hypothesis.** Under honest metrics, the production regularisation and post-processing are not optimal. Shrinking toward the prevailing mean instead of zero, time-series-CV α and a shallower GBT raise OOS R² without hurting direction. Prediction intervals can be made to cover near their nominal level.
>
> **Revisits:** v38, v39, v40, v42, v48, v70, v71, v133, v137, v140, v141, v143, v144, v145.
>
> **Inputs:** the pin; development period.
>
> **Base specification:** v200's feature sets, plus the finalist feature change of v201–v204 where one met the R-rule (pre-registered: each finalist enters on its own, in step order, and is kept only if it still meets the R-rule on this base).
>
> **Candidates (12):**
> 1. shrink toward the prevailing mean (prequential α grid);
> 2. per-benchmark prequential shrinkage (v70);
> 3. prequential affine recalibration (v71);
> 4. Ridge α grid extended to 1e4 inside the nested TS CV (v133);
> 5. GBT depth 1, 25 trees, lr 0.05, subsample 0.8 (v137);
> 6. Ridge only (v40);
> 7. fixed 0.6/0.4 blend instead of prequential 1/MAE² (v141);
> 8. correlation pruning at ρ 0.80 inside folds (v143);
> 9. 48-month train window (v145), compared on common dates;
> 10. 84-month capped window (v42);
> 11. pooled panel Ridge with benchmark fixed effects (v48);
> 12. conformal intervals from prequential residual quantiles (v144).
>
> **Budget:** 12.
>
> **Outputs:** `research/studies/v205_model_regularisation/`.
>
> **Success threshold:**
> - Candidates 1–11: the R-rule.
> - Candidate 12: prequential trailing coverage within ±5 pp of nominal (80 %) with median width ≤ 1.5× v200's. Today's coverage is 41–49 %.

### v206 — Benchmark universe and consensus weighting

> **Hypothesis.** The 8-benchmark universe (v11) and the quality-weighted consensus (v72, v75) were chosen with a corrupted VOO target and in-sample weights. An honest re-evaluation prefers equal or prequential weights, and the universe choice does not change the action.
>
> **Revisits:** v11, v12, v24, v28, v72, v75.
>
> **Inputs:** the pin; development period; the v205 finalist specification, or v200 if there is none.
>
> **Candidates (6):**
> 1. equal-weight consensus;
> 2. prequential quality weights (realised rows only, re-estimated monthly);
> 3. VTI instead of VOO (v24);
> 4. without GLD and DBC (v28);
> 5. + VIG;
> 6. + VGT (labels now corrected).
>
> **Metrics:** the consensus direction's hit rate vs base rate and PT p; consensus forecast R² against its own prevailing mean; policy uplift of the live mapping against always-50 %.
>
> **Budget:** 6.
>
> **Outputs:** `research/studies/v206_universe_consensus/`.
>
> **Success threshold:**
> - Policy uplift over the v200 consensus ≥ +0.25 pp per decision, with a 90 % CI lower bound > 0.
> - (Hit rate − base rate) not lower than v200's.
> - A universe change must also keep the v11 diversification score within 0.05.

### v207 — Classification and shadow layer

> **Hypothesis.** The shadow classifiers (lean logistic, Path B composite, v128 feature map, Firth, TA variants, v113 overlay) lose most of their reported edge once VGT labels, price features and calibration are honest. At most one simplified classifier survives as a shadow signal.
>
> **Revisits:** v46, v87–v90, v92, v96, v110, v113, v118–v121, v125, v127–v131, v135, v154, v165. v91 and v93 are not re-run (section 3.2); v94/v95 go to v208.
>
> **Inputs:** the pin; development period. Features: v200's, plus v201–v203 finalists (pre-registered as in v205).
>
> **Baseline (not counted):** the production lean-baseline balanced logistic with prequential calibration, (0.30, 0.70) abstention.
>
> **Candidates (10):**
> 1. Path B composite target (corrected VGT);
> 2. Path B + temperature scaling (v130/v135 parameters fixed a priori);
> 3. v128-style benchmark feature maps re-selected inside each fold (L1 / elastic net, ≤ 12 features);
> 4. Firth logistic for VMBS and BND (v154);
> 5. `ta_minimal_replacement` (v165);
> 6. `ta_minimal_plus_vwo_pct_b` (v165);
> 7. `gemini_veto_0.50` overlay re-evaluated with nested selection (v110/v113);
> 8. pooled shared logistic (v90);
> 9. (0.15, 0.70) abstention (v131), chosen inside folds;
> 10. one prospective replay of the finalist, month by month over development, with correct as-of truncation (v118–v121).
>
> **Metrics:** covered BA, coverage, Brier, log loss, prequential ECE with CI, all against the "always actionable-sell at the base rate" naive and the baseline; paired date-block bootstrap.
>
> **Budget:** 10.
>
> **Outputs:** `research/studies/v207_classification_shadow/`, including `outputs/shadow_artifact_proposal.md`. It says keep, replace or retire for each shadow artifact production reads: the v113 overlay CSV, the v128 map CSV, Path B and the TA variants. It is a proposal only.
>
> **Success threshold:** covered BA Δ ≥ +0.03 with Holm p < 0.05; prequential ECE ≤ 0.10; coverage ≥ 0.5; Brier not worse.

### v208 — Decision policy and tax mapping

> **Hypothesis.** Given honest signals, a sell-% policy that leaves 50 % only on strong, calibrated evidence can beat always-50 % after tax. The baseline is the current ACTIONABLE mapping (decision 0007) with the v200 gates.
>
> **Revisits:** v9 (policy rules), v12/v13 policy rows, v27 (sizing), v73, v94, v95, v149, v150, decision 0007.
>
> **Inputs:** the pin; development period; the v205/v206 finalist regression outputs and the v207 finalist classifier (if any). Tax uses `src/tax` with a synthetic schedule: equal January and July vests, the scenario return 0 % as in production.
>
> **Candidates (5):**
> 1. ±1.5 % neutral band around the consensus forecast (v150);
> 2. classifier-gated mapping using the v207 finalist (v94/v95; dropped if v207 has none);
> 3. Kelly fraction 0.25, cap 0.20 with prequentially calibrated probabilities (v149);
> 4. a tiered 25/50/100 mapping that acts only when PT p < 0.05 on the trailing 60 months;
> 5. LTCG-aware timing (hold a lot to LTCG when the after-tax break-even favours it).
>
> **Metric:** mean after-tax value per vest decision relative to always-50 %, with a date-block bootstrap CI; 5th-percentile outcome; turnover.
>
> **Budget:** 5.
>
> **Outputs:** `research/studies/v208_decision_policy/`.
>
> **Success threshold:**
> - uplift ≥ +0.25 pp per decision with the 90 % CI lower bound > 0;
> - never sells more than 50 % on OUTPERFORM (decision 0007);
> - 5th-percentile outcome no worse than always-50 % by more than 1 pp.

### v209 — Dividend and BVPS research lanes (x-series)

> **Hypothesis.** The x-series BVPS and special-dividend findings partly reflected raw-price and raw-BVPS targets, the 2006 split and unversioned inputs. Rebuilt on split-adjusted BVPS and DRIP total returns with provenance, only the BVPS-growth leg keeps skill against no-change.
>
> **Revisits:** x1, x4, x6, x9, x10, x12, x17–x24. x2 and x3 are not re-run (section 3.2).
>
> **Inputs:** the pin; development period.
> - Build the feature matrix in-process; no `data/processed/feature_matrix.parquet`.
> - BVPS via `price_adjustment.restate_to_latest_share_basis`.
> - Return targets from the DRIP helpers with business-month-end dates.
> - Special dividends from `daily_dividends`, checked against the 8-K for December specials.
>
> **Candidates (8):** BVPS growth at 3m and 6m with the x9 bridge features; persistent BVPS (x17) at 3m and 6m; dividend size `special_dividend_excess / current_bvps` with the x21/x22 challengers; special-dividend occurrence (report as under-identified if one-class).
>
> **Method:**
> - Walk-forward, expanding with a cap, for monthly targets.
> - Leave-future-out annual evaluation for the dividend lane (22 snapshots).
> - Diebold–Mariano with HAC for overlapping targets.
>
> **Budget:** 8.
>
> **Outputs:** `research/studies/v209_dividend_bvps_lanes/`.
>
> **Success threshold:** ≥ 5 % MAE reduction against no-change with Holm-adjusted DM p < 0.05. A winner can only become a reporting-only indicator proposal; it never changes the recommendation.

### v210 — Holdout synthesis and promotion recommendation

> **Goal.** Test every finalist once on the untouched holdout and write a promotion recommendation.
>
> **Inputs:**
> - the pin;
> - `outputs/finalist.json` from v201–v209 (at most one each);
> - the v200 specification;
> - `ALLOW_HOLDOUT=v210`, the only step allowed to set it.
>
> **Method.**
> - For v200 and each finalist, run the frozen specification walk-forward through the holdout. Each holdout forecast trains only on targets realised by its date, as production would. Nothing is re-tuned.
> - Compute the preamble metrics on holdout dates only, and the production gates month by month.
> - Compare each finalist with v200 using paired date-block bootstrap p-values, Holm across the finalists (≤ 9).
> - Also report the months after the pin as a "prospective" window. It is not used for the decision.
>
> **Pre-registered recommendation rule.** Recommend promotion of a finalist only if all of these hold:
> 1. it met its step's development threshold;
> 2. on the holdout: ΔR² ≥ 0 against v200, Δ equal-weight IC ≥ −0.02, no benchmark ΔR² < −5 pp, and policy uplift ≥ 0;
> 3. the Holm-adjusted one-sided p for ΔR² is < 0.10.
>
> A finalist that meets 1–2 but not 3 is "consistent, not confirmed": recommend shadow monitoring, not promotion. With 24 overlapping monthly dates the test has little power; say so plainly.
>
> **Outputs** (`research/studies/v210_holdout_synthesis/`):
> - `outputs/holdout_results.csv`;
> - `outputs/promotion_recommendation.md`: plain language, with one line per finalist and the owner decisions needed;
> - a draft decision record for `docs/decisions/` (in the study folder, not merged into `docs/decisions/` by this step).
>
> **After v210.** The holdout is spent. Future promotions need a new prospective holdout: months after the pin, accrued through the monthly shadow ledgers.
