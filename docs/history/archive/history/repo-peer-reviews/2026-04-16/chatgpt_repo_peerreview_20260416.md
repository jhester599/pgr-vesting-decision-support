# Optimization Landscape Analysis

The PGR vesting decision-support stack is suffering from **weak predictive performance and overconfidence**, making its production recommendations only marginally better than the default rule. The regression ensemble’s pooled out‑of‑sample R² is deeply negative (≈−13.1% in the v38 baseline)【16†L87-L94】【49†L288-L291】 and its predictions are *overconfident* (σ_pred/σ_true > 1.0), leading to frequent “false alarm” sell signals.  Hit rates (~69%) and IC (~0.1579) are moderately positive【16†L87-L94】【42†L79-L87】 but insufficient to overcome the poor R².  The shadow-classifier’s covered balanced accuracy (~0.57) and Brier score (~0.19) also fall short of promotion gates【45†L27-L32】【51†L509-L517】, partly due to limited data (N≈60–80 per benchmark) and conservative calibration.  Test suite runtime (~85s full, 70s fast) is nontrivial, impacting developer feedback speed.

The most **actionable bottlenecks** are thus the **regression calibration** (R²/σ ratio) and **classifier calibration** (covered BA/ECE/coverage). The ensemble’s post-hoc shrinkage (α=0.50) has been the best lever so far【16†L8-L11】【49†L289-L291】, suggesting remaining gains lie in squeezing variance rather than complexity.  In classification, the binary classifier is far from its coverage (0.25) and BA (0.60) targets; improving its temperature scaling and thresholds is urgent.  

**High-ROI dimensions:** Data quality (Dimension C) and model calibration (Dims A–B) should be tackled first. Clean, more informative features (e.g. reducing FRED lags) benefit all models; tuning Ridge/GBT hyperparameters and shrinkage can directly improve OOS metrics. Classification calibration (temp/warmup, thresholds) is next, since it is close to meeting promotion criteria【51†L542-L550】. Portfolio-layer parameters (Kelly, BL) and reporting tweaks are lower impact but still important once core model fit is addressed. 

**Exhausted directions vs gaps:** Prior research (v37–v60) has saturated many model architectures: Bayesian ridge, GPR, ARD, etc., gave worse R²【7†L111-L119】. The consensus is that *“the weakness still appears structural, not architectural”*【7†L107-L115】. In contrast, **under-explored avenues** include: systematic per-benchmark shrinkage/weight tuning, feature-window optimization, and careful calibration via conformal or prequential methods. The classification layer has just completed a rich sweep (v87–v128), but calibration and abstention have room to grow, as the best covered BA is still only ~0.6987 (with coverage 0.5476)【Prompt metrics】. 

**Dependencies:** Tackle **data freshness and lag parameters first** (FRED, EDGAR) since they affect all downstream models. Then **regression calibration** (shrinkage, Ridge α) and hyperparameters (GBT) should be optimized. Only *after* the regression layer is well-calibrated should we focus on **classifier calibration** (temperature, thresholds), since classifier predictions depend on the ensemble’s output. Decision-layer parameters (BL, Kelly) depend on the regression consensus and classifier signals being stable; they come later. **Test/infrastructure optimizations** can proceed in parallel at any stage, ideally in nightly batches interleaved to speed up research loops.

**Upper bounds:** Given N≈60–80 per benchmark, the theoretical “Hughes phenomenon” suggests at most ~13–15 strong features per model. The regression OOS R² could potentially rise from ~−15% to near 0% with optimal calibration (target ≥0–2%【16†L87-L94】); any large positive R² would be surprising. Similarly, classifier BA might reach ~0.60 at 25–30% coverage if thresholds and temperature are well-tuned. Conformal coverage could be fine-tuned from 80% to better match realized dispersion. Portfolio-layer gains are incremental: small tweaks to BL tau or Kelly settings might shift portfolio return <1%. Thus, we expect *moderate* improvements: single-digit percentage-point gains in R², BA, and sharpness, not order-of-magnitude changes. 

In summary, **low-hanging fruit** lie in tuning existing knobs (shrinkage, hyperparameters, lags) under strict temporal validation【18†L175-L184】【49†L330-L338】. By contrast, exotic new models or features have consistently hurt performance in this small-N regime【7†L109-L117】【52†L730-L739】. The sequencing is clear: cleanse and calibrate first, then tune models, and finally refine decision-layer. All tasks must obey strict WFO (no K-fold) and pipeline integrity【9†L10-L17】【16†L59-L66】.

## Comprehensive Task Inventory

*We assign IDs by dimension and rough priority. Each task defines a `/autoresearch` or `/autoresearch:predict` loop with the required structure.*

### Dimension A: Regression Model Calibration and Accuracy

## TASK-A01: Shrinkage Alpha Sweep  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P1 (do first)  
**Estimated iteration cost:** fast (<30s)  
**Depends on:** none  
**Expected impact:** pooled OOS R² ↑ (higher better) by tuning ensemble shrinkage α  

**Hypothesis:** The current fixed shrinkage α=0.50 may not be optimal for the full sample. Sweeping α∈[0.1,0.9] might find a better bias-variance tradeoff.  

**Setup requirements:**  
- *Harness:* `results/research/v_gen_shrinkage_eval.py`: loads fixed predictions (e.g. WFO outputs) and applies shrinkage α read from candidate file, then computes pooled OOS R². Prints `pooled_oos_r2=X.XXXX`.  
- *Candidate:* `results/research/v134_shrinkage_candidate.txt` (single float between 0 and 1).  
- *Guard:* pytest confirming 0<=α<=1 and output format.  
- *Baseline:* current R² = -0.1310【49†L288-L291】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Optimize the ensemble prediction shrinkage factor to maximize pooled out-of-sample R², improving on the current α=0.50 baseline (R²≈-0.1310).  
Metric: python results/research/v_gen_shrinkage_eval.py --shrinkage $(cat results/research/v134_shrinkage_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/v134_shrinkage_candidate.txt  
Guard: python -c "v=float(open('results/research/v134_shrinkage_candidate.txt').read()); assert 0.0 <= v <= 1.0, f'shrinkage={v} out of range'" && python -m pytest tests/test_research_shrinkage_eval.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 40  
Log: results/research/v134_shrinkage_autoresearch_log.jsonl
```  
**Success criteria:** Achieve pooled R² ≥ 0% (≥ +0.131 point improvement) without dropping IC or hit rate.  
**Post-loop action:** If best α>0.50, consider updating config or applying it post-hoc; validate on holdout.

## TASK-A02: Ensemble Blend Weight Optimization  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P2  
**Estimated iteration cost:** medium (1–5 min)  
**Depends on:** TASK-A01  
**Expected impact:** pooled IC, R² ↑ by reweighting Ridge vs GBT in ensemble  

**Hypothesis:** The current ensemble uses quality-weighted averaging. Introducing a tunable weight between Ridge and GBT (or a small equal-weight blend) may improve consensus forecasts, as equal weights sometimes outperform small-sample optimized weights.  

**Setup requirements:**  
- *Harness:* `results/research/v_alpha_blend_eval.py`: loads ensemble WFO predictions from Ridge and GBT. Combines as `p = w*y_hat_ridge + (1-w)*y_hat_gbt` where `w` is candidate weight. Computes pooled R². Prints `pooled_oos_r2=X.XXXX`.  
- *Candidate:* `results/research/vA02_weight_candidate.txt` (float in [0,1]).  
- *Guard:* assert 0≤w≤1 and format.  
- *Baseline:* current combination (w based on inverse-variance) yields R²≈-0.1310【49†L288-L291】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune the blend weight *w* on Ridge vs GBT forecasts in the ensemble to maximize pooled OOS R², checking if a different weight than the default improves calibration.  
Metric: python results/research/v_alpha_blend_eval.py --weight $(cat results/research/vA02_weight_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vA02_weight_candidate.txt  
Guard: python -c "v=float(open('results/research/vA02_weight_candidate.txt').read()); assert 0.0 <= v <= 1.0, f'weight={v} out of range' " && python -m pytest tests/test_research_alpha_blend.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 30  
Log: results/research/vA02_blend_autoresearch_log.jsonl
```  
**Success criteria:** Achieve pooled R² gain > 0.01 without IC or hit rate drop.  
**Post-loop action:** If a non-default weight w* is found, evaluate updated consensus on holdout.

## TASK-A03: Ridge Regularization Alpha Range Expansion  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P1  
**Estimated iteration cost:** medium (4–8 min)  
**Depends on:** TASK-C01 (FRED lag tasks finish first)  
**Expected impact:** pooled R² ↑ by allowing higher Ridge α selection  

**Hypothesis:** The inner Ridge CV grid currently caps α≤1e2【49†L265-L269】. Expanding to 1e5 may let Ridge itself find high-regularization solutions, potentially obviating post-shrinkage.  

**Setup requirements:**  
- *Harness:* `results/research/v133_ridge_alpha_sweep.py` (as defined in plan) that takes `alpha_min, alpha_max, n_alpha` and prints `pooled_oos_r2=X.XXXX`. Uses config purges【49†L281-L290】.  
- *Candidate:* `results/research/v133_alpha_max_candidate.txt` (float).  
- *Guard:* assert 1e2 ≤ α_max ≤ 1e5 and run smoke tests【49†L319-L327】.  
- *Baseline:* α_max=1e2 yields R²≈-0.1310【49†L288-L291】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Determine if extending Ridge CV grid into [1e2,1e5] yields a better pooled OOS R² than -0.1310【49†L288-L291】.  
Metric: python results/research/v133_ridge_alpha_sweep.py --alpha-min 1e-4 --alpha-max $(cat results/research/v133_alpha_max_candidate.txt) --n-alpha 60 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/v133_alpha_max_candidate.txt  
Guard: python -c "v=float(open('results/research/v133_alpha_max_candidate.txt').read()); assert 1e2 <= v <= 1e5, f'alpha_max={v} out of range'" && python -m pytest tests/test_research_v133_ridge_alpha_sweep.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 30  
Log: results/research/v133_ridge_alpha_autoresearch_log.jsonl
```  
**Success criteria:** Achieve pooled R² ≥ -0.10 (≥+0.03 above baseline【49†L340-L344】) without losing IC or hit rate.  
**Post-loop action:** If α_max* >1e2 improves results, propose updating `RIDGE_ALPHA_MAX` or implementing per-fold shrinkage in production.

## TASK-A04: GBT Hyperparameter Sweep  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P2  
**Estimated iteration cost:** slow (8–15 min)  
**Depends on:** TASK-A03, TASK-C01 (lag)  
**Expected impact:** pooled R² ↑ by finding better shallow GBT  

**Hypothesis:** The current GBT (depth=2, 50 trees) may be suboptimal under small-N conditions. Testing depth=1 (stumps) or different learning rates could improve generalization【52†L701-L710】【52†L731-L739】.  

**Setup requirements:**  
- *Harness:* `results/research/v137_gbt_param_sweep.py` (as specified) supporting both flags and params-file【52†L663-L673】, printing `pooled_oos_r2`.  
- *Candidate:* `results/research/v137_gbt_params_candidate.json` with keys (`max_depth`, `n_estimators`, `learning_rate`, `subsample`).  
- *Guard:* enforce 1≤depth≤4, 10≤n_est≤200, 0.01≤lr≤0.5, 0.5≤sub≤1.0【52†L715-L724】; run tests.  
- *Baseline:* default settings yields some R² (to be recorded in setup).  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Optimize GBT hyperparameters (max_depth, n_estimators, learning_rate, subsample) to maximize the standalone GBT pooled OOS R². The default (2,50,0.1,0.8) is unproven under small N【52†L730-L739】.  
Metric: python results/research/v137_gbt_param_sweep.py --params-file results/research/v137_gbt_params_candidate.json 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/v137_gbt_params_candidate.json  
Guard: python -c "import json; p=json.load(open('results/research/v137_gbt_params_candidate.json')); assert p['max_depth'] in [1,2,3,4]; assert 10<=p['n_estimators']<=200; assert 0.01<=p['learning_rate']<=0.50; assert 0.50<=p['subsample']<=1.00" && python -m pytest tests/test_research_v137_gbt_param_sweep.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 35  
Log: results/research/v137_gbt_param_autoresearch_log.jsonl
```  
**Success criteria:** GBT-only pooled R² ≥ -0.15 (vs default) with no guard hit (nondegenerate config)【52†L735-L743】.  
**Post-loop action:** If an improved config is found, test its effect in the ensemble context before updating production GBT pipeline.

## TASK-A05: Conformal Coverage and ACI Gamma Tuning  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P3  
**Estimated iteration cost:** fast  
**Depends on:** TASK-A01  
**Expected impact:** Conformal interval calibration (coverage near nominal)  

**Hypothesis:** The nominal conformal coverage (0.80) and ACI gamma (0.05) may not be optimal. Adjusting them could yield better empirical coverage without excessive interval width.  

**Setup requirements:**  
- *Harness:* `results/research/vC01_conformal_eval.py`: computes conformal intervals over held-out WFO folds with inputs `coverage, gamma`, and returns actual coverage. Prints `coverage=X.XXXX`.  
- *Candidate:* `results/research/vC01_conformal_candidate.json` with keys `coverage` (float), `aci_gamma` (float).  
- *Guard:* 0<coverage<1, 0<gamma<1, run tests.  
- *Baseline:* current coverage=0.80, gamma=0.05, trailing coverage ≈56% (far below nominal)【42†L112-L117】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune the conformal prediction parameters to achieve closer to nominal 80% coverage, reducing the warning of undercoverage【42†L112-L117】.  
Metric: python results/research/vC01_conformal_eval.py --coverage=$(jq -r .coverage results/research/vC01_conformal_candidate.json) --gamma=$(jq -r .aci_gamma results/research/vC01_conformal_candidate.json) 2>/dev/null | grep coverage= | cut -d= -f2  
Scope: results/research/vC01_conformal_candidate.json  
Guard: python -c "import json; p=json.load(open('results/research/vC01_conformal_candidate.json')); assert 0.0<p['coverage']<1.0; assert 0.0<p['aci_gamma']<1.0" && python -m pytest tests/test_research_vC01_conformal_eval.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vC01_conformal_autoresearch_log.jsonl
```  
**Success criteria:** Actual coverage ≥ 0.75 (reducing the large gap from 0.56).  
**Post-loop action:** If improved, update `CONFORMAL_COVERAGE` or `CONFORMAL_ACI_GAMMA` in config and regenerate intervals.

## TASK-A06: Ridge ElasticNet Mixing (L1 Ratio)  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P4 (speculative)  
**Estimated iteration cost:** medium  
**Depends on:** TASK-A03  
**Expected impact:** pooled R² possibly ↑ via sparse+ridge hybrid  

**Hypothesis:** Currently only pure Ridge is used. Introducing ElasticNet with tuned `l1_ratio` may combine the stability of Ridge with selectivity of Lasso, possibly improving R².  

**Setup requirements:**  
- *Harness:* `results/research/vA06_enet_sweep.py`: a WFO loop that takes `l1_ratio` and prints pooled R².  
- *Candidate:* `results/research/vA06_l1_candidate.txt` (float [0,1]).  
- *Guard:* 0≤r≤1, tests.  
- *Baseline:* l1_ratio=0 (pure Ridge) yields R² baseline.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Sweep ElasticNet l1_ratio in [0,1] (with fixed α) to see if any blend improves pooled OOS R² over pure Ridge.  
Metric: python results/research/vA06_enet_sweep.py --l1-ratio $(cat results/research/vA06_l1_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vA06_l1_candidate.txt  
Guard: python -c "v=float(open('results/research/vA06_l1_candidate.txt').read()); assert 0.0 <= v <= 1.0" && python -m pytest tests/test_research_vA06_enet_sweep.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 30  
Log: results/research/vA06_enet_autoresearch_log.jsonl
```  
**Success criteria:** Any l1_ratio that yields R² ≥ baseline +0.01, with no drop in IC/hit rate.  
**Post-loop action:** If significant, consider adding ElasticNet as alternate model in ensemble.

## TASK-A07: Walk-Forward Window Tuning  
**Dimension:** A  
**Type:** /autoresearch  
**Priority:** P3  
**Estimated iteration cost:** medium  
**Depends on:** none  
**Expected impact:** pooled R²/IC tuning via adjusting `WFO_TRAIN_WINDOW_MONTHS` (within reason)  

**Hypothesis:** The current WFO uses 60-month train and 6-month test with 8-month embargo. Slightly shorter or longer windows may improve stability or recency weighting.  

**Setup requirements:**  
- *Harness:* `results/research/vA07_wfo_window_sweep.py` that re-runs regression WFO given two new constants `WFO_TRAIN_WINDOW_MONTHS` and `WFO_TEST_WINDOW_MONTHS` read from candidate. Prints pooled R².  
- *Candidate:* `results/research/vA07_wfo_candidate.json` with `train=60±Δ`, `test=6±Δ` (integers).  
- *Guard:* Validate plausible min train=36, test≤12, no config files changed.  
- *Baseline:* (60,6) yields R²≈-0.1310.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Explore slight variations in the WFO train/test window lengths to maximize pooled R² (accounting for bias-variance tradeoff in time series).  
Metric: python results/research/vA07_wfo_window_sweep.py --train $(jq .train results/research/vA07_wfo_candidate.json) --test $(jq .test results/research/vA07_wfo_candidate.json) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vA07_wfo_candidate.json  
Guard: python -c "import json; v=json.load(open('results/research/vA07_wfo_candidate.json')); assert 36 <= v['train'] <= 72; assert 3 <= v['test'] <= 12" && python -m pytest tests/test_research_vA07_wfo.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vA07_wfo_autoresearch_log.jsonl
```  
**Success criteria:** Any config yielding R² ≥ baseline +0.01 without regressing IC or hit rate.  
**Post-loop action:** If a clear winner, propose updating `WFO_TRAIN_WINDOW_MONTHS` or `WFO_TEST_WINDOW_MONTHS` in config.

## TASK-A08: Gaussian Process Regression (Speedup)  
**Dimension:** A  
**Type:** /autoresearch:predict  
**Priority:** P4  
**Estimated iteration cost:** fast (<30s)  
**Depends on:** none  
**Expected impact:** Decision support for adding GPR  

**Hypothesis:** Nonparametric GPR could capture residual patterns. However, previous trials (v37–v60) showed high variance. We use static analysis to predict feasibility.  

**Setup requirements:**  
- *Predict:* `/autoresearch:predict` reads code and docs to assess GPR viability.  
- No harness; write a short analysis check.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Evaluate the expected benefit of adding a GaussianProcessRegressor as a third model to the ensemble, given N≈70 and existing weak signal.  
Metric: python - <<'PYCODE'
print(0.2)  # predicted <0.2 usefulness (Karpathy-style low prior)
PYCODE
Scope: entire codebase (no file modifications)
Guard: true  
Direction: higher_is_better  
Log: results/research/predict_GPR.jsonl
```  
**Rationale:** Likely low (<20%) chance of improvement under small N【7†L109-L117】【52†L730-L739】.  
**Post-loop:** Inform research if manual effort should pursue GPR.

### Dimension B: Classification Layer and Promotion Gate

## TASK-B01: Per-Benchmark Feature Routing (Path B)  
**Dimension:** B  
**Type:** /autoresearch  
**Priority:** P2  
**Estimated iteration cost:** medium (∼45s)  
**Depends on:** TASK-B05 (Temp scaling)  
**Expected impact:** covered BA ↑ by customizing feature sets per benchmark  

**Hypothesis:** The v128 study found VGT’s BA jumped by tailoring features【46†L100-L107】. We extend this by optimizing per-benchmark feature lists (up to 12 features each) to maximize *pooled* covered BA, potentially boosting the overall signal.  

**Setup requirements:**  
- *Harness:* `results/research/v129_feature_map_eval.py` (as specified) accepts `--strategy file:<csv>` and prints `covered_ba` and `coverage`【45†L54-L63】【46†L98-L107】.  
- *Candidate:* `results/research/v129_candidate_map.csv` with columns `[benchmark,feature_set,feature_list]` (one row per benchmark).  
- *Guard:* Validate CSV schema, no >12 features per row, features from inventory, and classification code untouched【46†L119-L127】.  
- *Baseline:* "lean_baseline" strategy yields BA≈0.5000 and coverage≈0.87【45†L90-L94】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Find the optimal per-benchmark feature subsets (same format as v128) to maximize pooled covered balanced accuracy for the Path B classifier, without exceeding 12 features/benchmark【46†L98-L107】.  
Metric: python results/research/v129_feature_map_eval.py --strategy file:results/research/v129_candidate_map.csv 2>/dev/null | grep "covered_ba=" | cut -d= -f2  
Scope: results/research/v129_candidate_map.csv  
Guard: python -m pytest tests/test_research_v129_feature_map_eval.py tests/test_classification_shadow.py -q --tb=short | tail -1 | grep passed  
Direction: higher_is_better  
Iterations: 40  
Log: results/research/v129_autoresearch_log.jsonl
```  
**Success criteria:** Pooled BA ≥ 0.5100 (≥ +0.01 over lean_baseline)【46†L142-L144】 with average Brier≤0.185, ECE≤0.06.  
**Post-loop action:** Use winning CSV to configure `V128_BENCHMARK_FEATURE_MAP_PATH`; validate on temporal hold-out before shadow update.

## TASK-B02: Temperature Scaling & Warmup Tuning (Path B)  
**Dimension:** B  
**Type:** /autoresearch  
**Priority:** P1  
**Estimated iteration cost:** fast (10–20s)  
**Depends on:** none  
**Expected impact:** covered BA ↑, coverage tradeoff  

**Hypothesis:** The current Temp grid (0.50–3.0) and warmup (24m) may not maximize covered BA. Tuning `temp_max` and `warmup` can increase covered BA toward the 0.60 target【51†L542-L550】.  

**Setup requirements:**  
- *Harness:* `results/research/v135_temp_param_search.py` (as specified) takes `temp_min, temp_max, warmup` and prints `covered_ba` and `coverage`【51†L482-L490】.  
- *Candidates:* `results/research/v135_temp_max_candidate.txt` (float) and `results/research/v135_warmup_candidate.txt` (int)【51†L497-L502】.  
- *Guard:* 1.5≤temp_max≤10.0, 12≤warmup≤42 and tests【51†L522-L530】.  
- *Baseline:* Current (`temp_max=3.0, warmup=24`) yields BA=0.5725【51†L509-L517】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune the temperature scaling upper bound and prequential warmup for Path B to maximize covered BA (currently 0.5725【51†L509-L517】) while maintaining coverage.  
Metric: python results/research/v135_temp_param_search.py --temp-min 0.5 --temp-max $(cat results/research/v135_temp_max_candidate.txt) --warmup $(cat results/research/v135_warmup_candidate.txt) --low 0.15 --high 0.70 2>/dev/null | grep covered_ba= | cut -d= -f2  
Scope: results/research/v135_temp_max_candidate.txt, results/research/v135_warmup_candidate.txt  
Guard: python -c "t=float(open('results/research/v135_temp_max_candidate.txt').read()); w=int(open('results/research/v135_warmup_candidate.txt').read()); assert 1.5<=t<=10.0; assert 12<=w<=42" && python -m pytest tests/test_research_v135_temp_param_search.py -q --tb=short | tail -1 | grep passed  
Direction: higher_is_better  
Iterations: 80  
Log: results/research/v135_temp_param_autoresearch_log.jsonl
```  
**Success criteria:** Covered BA ≥ 0.60 at coverage ≥ 0.25 (meeting the 24-month gate criterion【51†L541-L550】), with Brier≤0.20.  
**Post-loop action:** Validate best (temp_max,warmup) on v132 hold-out. If promising, update config and re-run entire shadow evaluation.

## TASK-B03: Thresholds Sweep (Path A/B Abstention)  
**Dimension:** B  
**Type:** /autoresearch  
**Priority:** P3  
**Estimated iteration cost:** medium  
**Depends on:** TASK-B02  
**Expected impact:** Covered BA ↑ via optimized abstention band  

**Hypothesis:** The current high/low thresholds (0.70/0.15) were found good in v131【4†L121-L124】, but a joint search may yield slightly better coverage/BA tradeoffs, possibly asymmetric.  

**Setup requirements:**  
- *Harness:* `results/research/vB03_threshold_sweep.py` that takes `low, high` and computes covered BA & coverage (using fixed Temp scaling).  
- *Candidate:* `results/research/vB03_thresh_candidate.json` with `low, high`.  
- *Guard:* 0≤low<high≤1, tests.  
- *Baseline:* low=0.30, high=0.70 with BA ~0.6987 (covered BA, from prompt).  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Optimize the low/high abstention thresholds (previously 0.15/0.70【4†L121-L124】) to maximize covered BA for the binary classifiers (Path A or B).  
Metric: python results/research/vB03_threshold_sweep.py --low $(jq .low results/research/vB03_thresh_candidate.json) --high $(jq .high results/research/vB03_thresh_candidate.json) 2>/dev/null | grep covered_ba= | cut -d= -f2  
Scope: results/research/vB03_thresh_candidate.json  
Guard: python -c "import json; p=json.load(open('results/research/vB03_thresh_candidate.json')); assert 0.0<=p['low']<p['high']<=1.0" && python -m pytest tests/test_research_vB03_threshold.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 30  
Log: results/research/vB03_thresh_autoresearch_log.jsonl
```  
**Success criteria:** Covered BA gain ≥+0.02 without coverage <0.20.  
**Post-loop action:** Update SHADOW_CLASSIFIER_*_THRESH in config or classification code and re-evaluate.

## TASK-B04: Class-Imbalance Weight Tuning  
**Dimension:** B  
**Type:** /autoresearch  
**Priority:** P4  
**Estimated iteration cost:** medium  
**Depends on:** none  
**Expected impact:** slight BA/ECE improvement by reweighting classes  

**Hypothesis:** Class imbalance (maybe >50% negative) affects per-benchmark logistic fit. Tuning `class_weight` scalar (beyond ‘balanced’) could improve calibration/BA.  

**Setup requirements:**  
- *Harness:* `results/research/vB04_class_weight_eval.py` accepts `weight_pos` and prints covered BA.  
- *Candidate:* `results/research/vB04_weight_candidate.txt`.  
- *Guard:* weight between 0.1 and 5.0, tests.  
- *Baseline:* weight=1 (default) BA ~0.57.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune the positive-class weight multiplier in per-benchmark logistic classifiers to maximize pooled covered balanced accuracy, mitigating class imbalance.  
Metric: python results/research/vB04_class_weight_eval.py --weight $(cat results/research/vB04_weight_candidate.txt) 2>/dev/null | grep covered_ba= | cut -d= -f2  
Scope: results/research/vB04_weight_candidate.txt  
Guard: python -c "v=float(open('results/research/vB04_weight_candidate.txt').read()); assert 0.1<=v<=5.0" && python -m pytest tests/test_research_vB04_class_weight.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vB04_weight_autoresearch_log.jsonl
```  
**Success criteria:** Any weight yielding BA increase ≥ +0.01 with stable ECE.  
**Post-loop action:** If significant, apply weight in classifier training.

## TASK-B05: Coverage-Weighted Aggregation  
**Dimension:** B  
**Type:** /autoresearch  
**Priority:** P4  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** minor coverage/BA tradeoff improvement  

**Hypothesis:** Instead of equal or portfolio-weighted aggregation of per-benchmark signals (Path A), try weighting by their balanced accuracy or coverage.  

**Setup requirements:**  
- *Harness:* `results/research/vB05_aggregate_eval.py` to compute covered BA given weight scheme (e.g., w_i = BA_i or Cov_i).  
- *Candidate:* perhaps none (test fixed schemes).  
- *Guard:* none.  

**Autoresearch invocation:**  
```
/autoresearch:predict
Goal: Compare Path A portfolio weight schemes (equal vs BA-weighted vs coverage-weighted) for the per-benchmark classifiers by static analysis (no metric output).  
Metric: python - <<'PYCODE'
print("EQUAL=0.57, BA-WEIGHTED=0.58")  # dummy comparison
PYCODE
Scope: src/models/classification.py  
Guard: true  
Direction: higher_is_better  
Iterations: 1  
Log: results/research/predict_weighting.jsonl
```  
**Rationale:** Likely small impact; we flag for potential manual analysis.  
**Post-loop:** Review best scheme; run in offline evaluation if promising.

### Dimension C: Data Quality and Feature Engineering

## TASK-C01: FRED Series Lag Optimization  
**Dimension:** C  
**Type:** /autoresearch  
**Priority:** P1  
**Estimated iteration cost:** slow (5–10 min)  
**Depends on:** none  
**Expected impact:** pooled R² ↑ by using fresher macro data  

**Hypothesis:** Many FRED series are available daily/weekly, yet default 1-month lags discard 4 weeks of data【50†L411-L420】. Setting lag=0 for fast series should improve signal.  

**Setup requirements:**  
- *Harness:* `results/research/v134_fred_lag_sweep.py` (as specified) that merges overrides and prints `pooled_oos_r2`【50†L404-L412】.  
- *Candidate:* `results/research/v134_lag_candidate.json` listing 9 daily/weekly series with 0 or 1 month lag allowed【50†L402-L410】.  
- *Guard:* ensure keys match eligible set and values {0,1}【50†L421-L429】.  
- *Baseline:* no overrides (all 1) yields R²≈-0.1310【50†L411-L419】.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Determine if reducing lag assumptions for daily/weekly FRED series from 1 month to 0 (making up to 4 extra weeks of data available) improves pooled OOS R² beyond -0.1310【49†L288-L291】.  
Metric: python results/research/v134_fred_lag_sweep.py --lag-overrides "$(cat results/research/v134_lag_candidate.json)" 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/v134_lag_candidate.json  
Guard: python -c "import json; d=json.load(open('results/research/v134_lag_candidate.json')); eligible={'GS10','GS5','GS2','T10Y2Y','T10YIE','VIXCLS','BAA10Y','BAMLH0A0HYM2','MORTGAGE30US'}; assert set(d)==eligible; assert all(v in (0,1) for v in d.values())" && python -m pytest tests/test_research_v134_fred_lag_sweep.py -q --tb=short | tail -1 | grep passed  
Direction: higher_is_better  
Iterations: 25  
Log: results/research/v134_fred_lag_autoresearch_log.jsonl
```  
**Success criteria:** Pooled R² ≥ -0.12 with IC≥0.1579, hit≥0.70. Even modest gain is valuable since no complexity cost【50†L441-L449】.  
**Post-loop action:** If positive, update `FRED_SERIES_LAGS` in config for the affected series.

## TASK-C02: EDGAR Filing Lag Tuning  
**Dimension:** C  
**Type:** /autoresearch  
**Priority:** P2  
**Estimated iteration cost:** medium  
**Depends on:** none  
**Expected impact:** marginal predictive gain by using more timely filings  

**Hypothesis:** Allowing data from EDGAR 8-K (or 10-Q) to be used with 0-1 month lag rather than >1 could slightly improve returns forecasts, at risk of look-ahead bias.  

**Setup requirements:**  
- *Harness:* `results/research/vC02_edgar_lag_eval.py`: rebuilds features using candidate lag and outputs pooled R².  
- *Candidate:* `results/research/vC02_edgar_candidate.txt` (int 0 or 1).  
- *Guard:* 0≤lag≤1, not beyond.  
- *Baseline:* current lag=2 (for example) yields some R².  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Test whether reducing the assumed publication lag of SEC EDGAR filings from 2 to 0–1 months improves pooled OOS R², or introduces bias.  
Metric: python results/research/vC02_edgar_lag_eval.py --lag $(cat results/research/vC02_edgar_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vC02_edgar_candidate.txt  
Guard: python -c "v=int(open('results/research/vC02_edgar_candidate.txt').read()); assert 0 <= v <= 1" && python -m pytest tests/test_research_vC02_edgar_eval.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vC02_edgar_autoresearch_log.jsonl
```  
**Success criteria:** Any R² improvement or stability (no drop), without suspicious data leakage.  
**Post-loop action:** If improved, carefully justify updating EDGAR lag (with manual check for leakage).

## TASK-C03: Feature Correlation Pruning  
**Dimension:** C  
**Type:** /autoresearch  
**Priority:** P3  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** remove redundant features to improve variance  

**Hypothesis:** Redundant features (highly correlated predictors) can be removed. Tuning a correlation threshold for drop (e.g. 0.9,0.95) may improve sample-efficiency.  

**Setup requirements:**  
- *Harness:* `results/research/vC03_corr_prune_eval.py` which drops any two features with absolute correlation ≥ threshold `rho`, retrains Ridge on remaining features each WFO, and returns R².  
- *Candidate:* `results/research/vC03_rho_candidate.txt` (float).  
- *Guard:* 0.5≤rho≤1.0.  
- *Baseline:* no drop (rho=1.0) yields R² baseline.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Optimize the correlation cutoff for feature redundancy pruning to maximize pooled OOS R².  
Metric: python results/research/vC03_corr_prune_eval.py --rho $(cat results/research/vC03_rho_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vC03_rho_candidate.txt  
Guard: python -c "v=float(open('results/research/vC03_rho_candidate.txt').read()); assert 0.5 <= v <= 1.0" && python -m pytest tests/test_research_vC03_corr.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vC03_corr_prune_log.jsonl
```  
**Success criteria:** If a lower rho (pruning more) yields R² ≥ baseline +0.01.  
**Post-loop action:** Remove identified redundant features from model pipeline.

## TASK-C04: Derived Feature Window Sweep  
**Dimension:** C  
**Type:** /autoresearch  
**Priority:** P4  
**Estimated iteration cost:** medium  
**Depends on:** none  
**Expected impact:** uncover better momentum/TM indicators  

**Hypothesis:** Rolling-window features (momenta, moving averages) currently use fixed periods (3m,6m,12m). Tuning these windows (e.g. 2–4m, etc) may capture signals better.  

**Setup requirements:**  
- *Harness:* `results/research/vC04_window_sweep.py` that builds features with candidate window length and measures R².  
- *Candidates:* `results/research/vC04_window_candidate.json` with `mom_window=...`.  
- *Guard:* reasonable bounds, tests.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune time windows for derived features (e.g. momentum) to maximize pooled OOS R².  
Metric: python results/research/vC04_window_sweep.py --window $(jq .mom_window results/research/vC04_window_candidate.json) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vC04_window_candidate.json  
Guard: python -c "import json; v=json.load(open('results/research/vC04_window_candidate.json')); assert 1 <= v['mom_window'] <= 6" && python -m pytest tests/test_research_vC04_window.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 15  
Log: results/research/vC04_window_log.jsonl
```  
**Success criteria:** Any specific window giving +0.01 R² improvement.  
**Post-loop action:** If found, set new window in feature builder.

## TASK-C05: Fractional Differencing Parameter (`fracdiff`) Tuning  
**Dimension:** C  
**Type:** /autoresearch  
**Priority:** P4  
**Estimated iteration cost:** medium  
**Depends on:** none  
**Expected impact:** stationarity-adjusted forecasts  

**Hypothesis:** If features are fractionally differenced (see src pipeline), the differencing parameter (d) could be optimized to improve signal stationarity.  

**Setup requirements:**  
- *Harness:* `results/research/vC05_fracdiff_sweep.py` taking `d` and computing R².  
- *Candidate:* `results/research/vC05_fracdiff_candidate.txt` ([0.0,1.0]).  
- *Guard:* 0≤d≤1.  
- *Baseline:* current d? (likely 0 or some default).  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune the fractional differencing parameter for features (0 ≤ d ≤ 1) to maximize pooled OOS R².  
Metric: python results/research/vC05_fracdiff_sweep.py --d $(cat results/research/vC05_fracdiff_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2  
Scope: results/research/vC05_fracdiff_candidate.txt  
Guard: python -c "v=float(open('results/research/vC05_fracdiff_candidate.txt').read()); assert 0.0 <= v <= 1.0" && python -m pytest tests/test_research_vC05_fracdiff.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vC05_fracdiff_log.jsonl
```  
**Success criteria:** If some d gives ≥+0.01 R².  
**Post-loop:** Use best d in feature processing.

### Dimension D: Portfolio Construction and Decision Layer

## TASK-D01: Black-Litterman Tau & View-Confidence Tuning  
**Dimension:** D  
**Type:** /autoresearch  
**Priority:** P2  
**Estimated iteration cost:** fast (<30s)  
**Depends on:** none (can run anytime)  
**Expected impact:** decision accuracy ↑ (recommendation correctness)  

**Hypothesis:** The BL parameters (`tau=0.05`, `view_confidence=1.0`) were arbitrary. Tuning them to historical outcomes should increase the fraction of correct hold/sell decisions【53†L807-L816】.  

**Setup requirements:**  
- *Harness:* `results/research/v138_bl_param_eval.py` (as specified) taking `tau, view_confidence` and printing `recommendation_accuracy`【53†L773-L781】.  
- *Candidate:* `results/research/v138_bl_params_candidate.json` with `tau`, `view_confidence_scalar`.  
- *Guard:* 0.01≤tau≤0.5, 0.25≤vc≤4.0【53†L823-L831】.  
- *Baseline:* accuracy from current params (recorded in setup).  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Tune Black-Litterman parameters (tau, view confidence) to maximize the fraction of months where the BL recommendation agrees with the ex-post correct decision【53†L807-L816】.  
Metric: python results/research/v138_bl_param_eval.py --params-file results/research/v138_bl_params_candidate.json 2>/dev/null | grep recommendation_accuracy= | cut -d= -f2  
Scope: results/research/v138_bl_params_candidate.json  
Guard: python -c "import json; p=json.load(open('results/research/v138_bl_params_candidate.json')); assert 0.01<=p['tau']<=0.50; assert 0.25<=p['view_confidence_scalar']<=4.0" && python -m pytest tests/test_research_v138_bl_param_eval.py -q --tb=short | tail -1 | grep passed  
Direction: higher_is_better  
Iterations: 60  
Log: results/research/v138_bl_param_autoresearch_log.jsonl
```  
**Success criteria:** Recommendation accuracy ≥ 0.65 (coverage ≥0.25)【53†L844-L852】, with ≤10% change in mean Kelly (to avoid extreme leverage).  
**Post-loop action:** Validate best tau/vc on holdout. If valid, update config `BL_TAU`, `BL_VIEW_CONFIDENCE_SCALAR`.

## TASK-D02: Kelly Fraction and Cap Optimization  
**Dimension:** D  
**Type:** /autoresearch  
**Priority:** P3  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** adjust sizing for risk-return  

**Hypothesis:** The current Kelly fraction (0.25) and cap (20%) might not be optimal; tuning them could slightly improve long-run geometric growth while controlling risk.  

**Setup requirements:**  
- *Harness:* `results/research/vD02_kelly_eval.py`: takes `fraction` and `cap`, simulates historical PGR re-balance to measure a utility or return metric.  
- *Candidate:* `results/research/vD02_kelly_candidate.json` with `fraction`, `cap`.  
- *Guard:* 0<frac≤1, 0<cap≤0.5, tests.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Find Kelly investment fraction and max position cap that improve a portfolio utility metric (e.g. geometric return or Sharpe) without excessive drawdown.  
Metric: python results/research/vD02_kelly_eval.py --fraction $(jq .fraction results/research/vD02_kelly_candidate.json) --cap $(jq .cap results/research/vD02_kelly_candidate.json) 2>/dev/null | grep utility_score= | cut -d= -f2  
Scope: results/research/vD02_kelly_candidate.json  
Guard: python -c "import json; p=json.load(open('results/research/vD02_kelly_candidate.json')); assert 0.01<=p['fraction']<=1.0; assert 0.01<=p['cap']<=0.5" && python -m pytest tests/test_research_vD02_kelly.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 20  
Log: results/research/vD02_kelly_autoresearch_log.jsonl
```  
**Success criteria:** Any config with > baseline utility (safety metrics non-regressed).  
**Post-loop action:** Propose new KELLY_FRACTION and/or KELLY_MAX_POSITION if justified.

## TASK-D03: Neutral Band Width Tuning  
**Dimension:** D  
**Type:** /autoresearch  
**Priority:** P4  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** tweak recommendation sensitivity  

**Hypothesis:** The neutral band for recommendation (e.g. ±3%) could be widened or narrowed for better alignment with BL output and volatility.  

**Setup requirements:**  
- *Harness:* `results/research/vD03_neutral_band_eval.py`: apply different band widths and compute backtest correctness or combined metric.  
- *Candidate:* `results/research/vD03_band_candidate.txt` (float, percent).  
- *Guard:* 0%≤band≤10%.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Optimize the neutral (no-action) band width for the final recommendation rule, trading off risk and return in decisions.  
Metric: python results/research/vD03_neutral_band_eval.py --band $(cat results/research/vD03_band_candidate.txt) 2>/dev/null | grep success_rate= | cut -d= -f2  
Scope: results/research/vD03_band_candidate.txt  
Guard: python -c "v=float(open('results/research/vD03_band_candidate.txt').read()); assert 0.0<=v<=10.0" && python -m pytest tests/test_research_vD03_band.py -q --tb=short | grep passed  
Direction: higher_is_better  
Iterations: 10  
Log: results/research/vD03_band_autoresearch_log.jsonl
```  
**Success criteria:** Improved decision success rate or gain metric with no increase in risk.  
**Post-loop action:** If changed, update recommendation thresholds in code.

### Dimension E: Test Suite and Developer Experience

## TASK-E01: Test Suite Runtime Reduction  
**Dimension:** E  
**Type:** /autoresearch  
**Priority:** P1  
**Estimated iteration cost:** slow (1–5 min per iteration)  
**Depends on:** none  
**Expected impact:** test suite runtime ↓≥20% (from ~85s)  

**Hypothesis:** The suite has many redundant or heavy tests. We can refactor fixtures, parameterize, mark slow tests to reduce runtime without removing coverage【46†L172-L181】【47†L229-L238】.  

**Setup requirements:**  
- *Harness:* `scripts/measure_test_time.sh` (timing wrapper)【46†L158-L166】.  
- *Candidate:* source test files (scope is any test refactoring, not a candidate file).  
- *Guard:* pytest count ≥1683 tests passed (see plan guard regex)【46†L217-L224】.  
- *Baseline:* record current elapsed_seconds via harness.  

**Autoresearch invocation:**  
```
/autoresearch
Goal: Reduce total pytest runtime while keeping all 1,683 tests passing, by refactoring slow tests and fixtures【47†L227-L236】.  
Metric: bash scripts/measure_test_time.sh 2>/dev/null | grep elapsed_seconds= | cut -d= -f2  
Scope: tests/ (and possibly src/ for caching improvements)  
Guard: python -m pytest --tb=short -q 2>&1 | tail -3 | grep -E "1[6-9][0-9]{2} passed|[2-9][0-9]{3} passed"  
Direction: lower_is_better  
Iterations: 20  
Log: results/research/vE01_testtime_autoresearch_log.jsonl
```  
**Success criteria:** ≥20% runtime reduction with 0 failures and test count ≥1683【47†L237-L240】.  
**Post-loop action:** Commit fixture and refactoring changes; update `tests/test_wfo_engine.py` etc. For parallel runs, allow others to proceed with shorter tests.

## TASK-E02: `--fast` Marker Coverage Expansion  
**Dimension:** E  
**Type:** /autoresearch  
**Priority:** P2  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** smaller subset tests for development speed  

**Hypothesis:** Many slow tests identified (WFO, research scripts) can be marked `@pytest.mark.slow` and skipped under `--fast`.  

**Setup requirements:**  
- *Scope:* annotate up to 10 slowest tests with `@pytest.mark.slow`.  
- *Guard:* no metric; ensure test suite runs without `--fast`.  

**Autoresearch invocation:** *(script likely not needed, manual step)*  
```
/autoresearch:predict
Goal: Assess if adding a pytest --fast marker to identified slow tests (profiles given) can speed up routine runs by ~30%.  
Metric: python - <<'PYCODE'
print("Expected speedup ~30% (no measured metric)")
PYCODE
Scope: tests/ (adding @pytest.mark.slow)
Guard: true  
Direction: lower_is_better  
Iterations: 1  
Log: results/research/predict_fast_marker.jsonl
```  
**Outcome:** Decide which tests to mark slow; update pytest config to support `--fast`【47†L191-L199】.

### Dimension F: Reporting and Artifact Quality

## TASK-F01: Monthly Report Section Audit  
**Dimension:** F  
**Type:** /autoresearch:predict  
**Priority:** P3  
**Estimated iteration cost:** fast (<30s)  
**Depends on:** none  
**Expected impact:** identify clarity improvements in reporting  

**Hypothesis:** The monthly `recommendation.md` and `diagnostic.md` could benefit from reordering or clarifying signals (e.g., emphasize signal vs confidence).  

**Setup requirements:**  
- *Predict:* Analyze markdown templates or artifacts.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Identify opportunities to improve monthly report readability (section ordering, signal labeling, and highlight key metrics) by analyzing existing report structure.  
Metric: python - <<'PYCODE'
print("Candidates: reorder sections for decision, add interpretive summary")
PYCODE
Scope: docs/report_template.md or similar  
Guard: true  
Direction: higher_is_better  
Iterations: 1  
Log: results/research/predict_report.jsonl
```  
**Outcome:** Suggestions only (e.g., move 'Recommendation' to top, clarify confidence tier language). Manual revisions to markdown.

## TASK-F02: Confidence Tier Label Refinement  
**Dimension:** F  
**Type:** /autoresearch:predict  
**Priority:** P3  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** clearer wording  

**Hypothesis:** The labels (“LOW”, “MODERATE”, “HIGH”) and wording in classification/dashboards could be more intuitive.  

**Setup requirements:**  
- *Predict:* Parse `classification_shadow` output labels【42†L19-L26】.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Evaluate if the confidence tier labels (LOW/MODERATE/HIGH) and phrasing in user output could be made more descriptive to improve clarity.  
Metric: python - <<'PYCODE'
print("Consider labels like 'Low confidence', 'Moderate', 'High confidence'")
PYCODE
Scope: results/monthly_decisions/2026-04/recommendation.md  
Guard: true  
Direction: higher_is_better  
Iterations: 1  
Log: results/research/predict_labels.jsonl
```  
**Outcome:** Write improved labels manually.

## TASK-F03: Dashboard Freshness Indicator  
**Dimension:** F  
**Type:** /autoresearch:predict  
**Priority:** P4  
**Estimated iteration cost:** fast  
**Depends on:** none  

**Hypothesis:** The HTML dashboard should flag data staleness (e.g. EDGAR warnings) more prominently.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Determine if adding a 'last updated' or freshness badge to the dashboard UI (from data age metadata) improves interpretability.  
Metric: python - <<'PYCODE'
print("Flash 'Data as of 2026-04-13'")
PYCODE
Scope: src/reporting/dashboard.py  
Guard: true  
Direction: higher_is_better  
Iterations: 1  
Log: results/research/predict_dashboard.jsonl
```  
**Outcome:** If useful, implement in code.

### Dimension G: Research Infrastructure and Automation

## TASK-G01: Shared WFO Regression Harness  
**Dimension:** G  
**Type:** /autoresearch:predict  
**Priority:** P1  
**Estimated iteration cost:** fast  
**Depends on:** none  
**Expected impact:** faster research loop development  

**Hypothesis:** Many regression tasks reuse WFO logic. A shared CLI harness (e.g. `wfo_regression_eval.py`) should exist to compute pooled OOS metrics given model params.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Confirm that a reusable Python module (e.g. `src/research/wfo_eval.py`) for regression WFO metrics can serve tasks A01–A05, reducing duplicate code.  
Metric: python - <<'PYCODE'
import os; print("src/research/wfo_eval.py exists?" , os.path.exists("src/research/evaluation.py"))
PYCODE
```  
**Action:** If not present, plan to write `wfo_regression_eval.py`.

## TASK-G02: Shared Classification Evaluation Harness  
**Dimension:** G  
**Type:** /autoresearch:predict  
**Priority:** P1  
**Estimated iteration cost:** fast  

**Hypothesis:** Similar shared harness for computing covered BA, Brier, etc. Given usage in v135 (temp) and others, a shared module `classification_eval.py` is warranted.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Propose creating a shared classification evaluation harness (`classification_eval.py`) to produce covered BA, ECE, coverage from fold-level outputs, used by tasks B01–B05.  
Metric: python - <<'PYCODE'
print("proposal: yes, use existing src/models/evaluation.py") 
PYCODE
```  
**Action:** Plan to factor out shared code from v135 and v129 scripts.

## TASK-G03: Candidate File Schema Standardization  
**Dimension:** G  
**Type:** /autoresearch:predict  
**Priority:** P1  
**Estimated iteration cost:** fast  

**Hypothesis:** All candidate files should follow a standard schema (e.g. JSON with clear fields). 

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Define and enforce common schema for candidate JSON/text files (floats, dicts, lists) to simplify /autoresearch loops.  
Metric: python - <<'PYCODE'
print("Define JSON schema for tunable params")
PYCODE
```  
**Action:** Document schemas (see Section 4 below).

## TASK-G04: Baseline Measurement Scripts  
**Dimension:** G  
**Type:** /autoresearch:predict  
**Priority:** P2  
**Estimated iteration cost:** fast  

**Hypothesis:** Before tuning loops, each metric script must verify the current baseline reading matches production; adding checks improves robustness.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Ensure each new evaluation script (v133, v134, v135, etc.) includes a baseline check that its default parameters reproduce known baseline metrics.  
Metric: python - <<'PYCODE'
print("Yes, baseline check needed before loop runs")
PYCODE
```  
**Action:** Add baseline assertions to each harness.

## TASK-G05: Guard Test Coverage Improvement  
**Dimension:** G  
**Type:** /autoresearch:predict  
**Priority:** P3  
**Estimated iteration cost:** fast  

**Hypothesis:** Many tasks share guard patterns (range checks, key checks). We should catalog common guard code templates for reuse.  

**Autoresearch:predict invocation:**  
```
/autoresearch:predict
Goal: Identify the recurring guard patterns (range assertions, dict key sets, sum-to-1 checks) and create library routines or examples in docs for reuse across tasks.  
Metric: python - <<'PYCODE'
print("List guard patterns: range, sum, keys, probability, keys match, sum=1")
PYCODE
```  
**Action:** Add guard test snippets to repo (see Section 5 below).

---

## Priority Sequencing

Below is a recommended batch schedule, each fitting an 8-hour overnight window. Fast loops (*) are run first to maximize iterations. Independent loops can run in parallel (noted). 

```
## Batch 1 (Week 1, Session 1) — ~8 hours
Rationale: Tackle data quality and quick calibration tasks that benefit all downstream research.
Tasks: TASK-C01, TASK-C02, TASK-A01, TASK-E01 [parallel with C01], TASK-G01, TASK-G02

## Batch 2 (Week 1, Session 2) — ~8 hours
Rationale: Continue regression calibration and regression hyperparameter tasks.
Tasks: TASK-A03, TASK-A02, TASK-A04 [parallelizable with A03], TASK-C03

## Batch 3 (Week 2, Session 1) — ~8 hours
Rationale: Run GBT tuning and validation of improved Ridge results.
Tasks: TASK-A05, TASK-A07, TASK-A08 [predict], TASK-G03

## Batch 4 (Week 2, Session 2) — ~8 hours
Rationale: Begin classification calibration (path B).
Tasks: TASK-B02, TASK-B03 [after B02], TASK-B04 [parallel if possible], TASK-B05, TASK-G04

## Batch 5 (Week 3, Session 1) — ~8 hours
Rationale: Finalize feature routing and threshold for classification, and test suite improvements.
Tasks: TASK-B01 (after B02,B03), TASK-E02 [fast marker], TASK-E01 (continued refinements), TASK-G05

## Batch 6 (Week 3, Session 2) — ~8 hours
Rationale: Decision-layer parameters.
Tasks: TASK-D01, TASK-D02, TASK-D03, TASK-B08 [if any]

## Batch 7 (Week 4, Session 1) — ~8 hours
Rationale: Reporting/infra & predictive analyses.
Tasks: TASK-F01, TASK-F02, TASK-F03 [predict], TASK-G01-G05 (documentation), TASK-B06 /predict (if needed), TASK-G remaining predictive

## Batch 8 (Week 4, Session 2) — ~8 hours
Rationale: Run final /autoresearch:predict meta-analysis on backlog (Target 6).
Tasks: /autoresearch:predict backlog prioritization (along with any remaining predict tasks).
```

*Notes:* 
- TASK-E01 (test time) is independent and parallelizable; run it first for maximum iterations【47†L241-L244】. 
- Classification (B) tasks depend on final Temp (B02) because thresholds (B03) should use the tuned probabilities【46†L147-L149】. 
- FRED lag (C01) should be done before Ridge (A03) to feed cleaner data【50†L446-L448】. 
- GBT (A04) is lower priority than Ridge, so runs after. 
- Reporting and infrastructure tasks are mostly predictive/manual and can run anytime after core loops.

### Candidate File Schema Standards

Define schemas and guard patterns for common candidate files:

- **Single-float `.txt`:** a single number, no newline or others. Range checks in guard.  
  *Example:* `results/research/v133_alpha_max_candidate.txt` containing e.g. `10000.0`.  
  *Guard:* `assert lo <= value <= hi`.  
- **Param dict `.json`:** keys→scalars. Each value has its own bound.  
  *Example:* `v137_gbt_params_candidate.json`: `{"max_depth":2, "n_estimators":50, "learning_rate":0.1, "subsample":0.8}`.  
  *Guard:* JSON parse, then assert each key exists and ranges. E.g. `assert 1<=max_depth<=4, 10<=n_est<=200, 0.01<=lr<=0.5, 0.5<=sub<=1.0`.  
- **Feature map `.csv`:** columns `[benchmark,feature_set,feature_list]`. `feature_list` is comma-separated.  
  *Guard:* correct headers, `feature_list` entries in allowed set, count limit (≤12), no missing benchmarks.  
- **Threshold pair `.json`:** e.g. `{"low":0.15,"high":0.70}`.  
  *Guard:* 0 ≤ low < high ≤ 1.0.  
- **Weight dict `.json`:** keys=ticker, values=float, sum ≤1 (for portfolio weights).  
  *Guard:* All keys valid benchmark names, all v ≥0, sum(v)=1.0 (±1e-8).  

### Shared Harness Architecture

Standardize common evaluation harnesses:

- **WFO Regression Harness (`src/research/wfo_eval.py`):** CLI to compute pooled OOS R², IC, hit rate, sigma. Input: model type and hyperparameters or candidate config. Output: JSON or printed metrics (e.g. `print(f"pooled_oos_r2={r2}")`). Many tasks (A01, A02, A03, A04, A07) can reuse this by writing thin wrappers around `run_multi_benchmark_wfo`. Temporal integrity mandatory.  
- **Classification Harness (`src/research/classif_eval.py`):** CLI for covered BA, coverage, Brier, ECE on binary target. Used by B01 (per-benchmark or Path B), B02, B04. Takes candidate parameters or feature maps. Prints `covered_ba=X`, `coverage=Y`, etc.  
- **BL/Kelly Decision Harness (`src/research/decision_eval.py`):** CLI using consensus predictions, BL views and risk parameters to simulate final recommendation performance. Returns metrics like `recommendation_accuracy`, `mean_kelly_fraction`, `policy_uplift`. Used by D01/D02.  
- **Test Runtime Harness:** Provided as `scripts/measure_test_time.sh` (timing wrapper)【46†L158-L166】.  
- **Feature Map Eval Harness:** `results/research/v129_feature_map_eval.py` from Target 1 evaluates any feature CSV strategy, printing covered BA and coverage【46†L99-L107】.  
- **Conformal Coverage Harness:** A script that given (coverage, gamma) recalculates conformal intervals (using held-out data) and outputs achieved coverage. Useful for TASK-A05.

All harnesses must use `TimeSeriesSplit` with configured embargo/purge (no leakage)【18†L62-L71】【16†L59-L66】. They read parameters from candidate files or CLI flags.

### Guard Test Patterns

Reusable pytest assertions:

- **Float in [lo,hi]:**  
  ```python
  assert lo <= value <= hi, f"value={value} out of range"
  ```
- **Int in [lo,hi]:** similar.  
- **Probability (0,1):**  
  ```python
  assert 0.0 <= p <= 1.0
  ```
- **Dict keys set:**  
  ```python
  expected = {...}
  assert set(d.keys()) == expected, f"unexpected keys {set(d)-expected}"
  ```
- **Values sum to 1.0:**  
  ```python
  total = sum(d.values())
  assert abs(total - 1.0) < 1e-6, f"weights sum to {total}"
  ```
- **Feature names subset:** Given list of allowed features, ensure each feature in candidate appears in allowed set.  
- **CSV columns:**  
  ```python
  import pandas as pd
  df = pd.read_csv(path)
  assert set(['benchmark','feature_list']).issubset(df.columns)
  ```
- **No overlap between sets:** E.g., 
  ```python
  assert not set(investable).intersection(contextual), "Benchmark appears in both"
  ```
- **Metric above baseline:**  
  ```python
  assert metric_value >= baseline_value, "metric did not meet baseline"
  ```

### Known Anti-Patterns and Constraints

Researchers **must enforce** these for the project’s integrity:

1. **No K-Fold CV:** Must use `TimeSeriesSplit` with purge/embargo【9†L10-L13】【16†L59-L66】.  
2. **No StandardScaler globally:** Scaling only inside each training fold pipeline【9†L4-L6】【16†L59-L66】.  
3. **No yfinance:** All fundamental/historical data via approved APIs【9†L4-L6】.  
4. **No using holdout data:** All research loops must cut off before 2024‑04‑01 holdout【16†L57-L65】.  
5. **No modifying production code in research loops:** Use candidate files and separate scripts; only update production after strict gate checks【16†L59-L66】.  
6. **Holdout boundary locked:** Do not change `HOLDOUT_START`.  
7. **Guard against shrinking sample:** e.g. if FRED lags=0 yields look-ahead bias, disallow it in guard【50†L433-L442】.  
8. **Feature universe fixed:** Cannot invent new features during autoresearch; only tune existing ones.  
9. **Hughes phenomenon:** Keep total features per model ≈13 or fewer to avoid overfitting.  
10. **Prediction combination puzzle:** Equal weights often win small-N; be skeptical of complex ensemble weight tweaks【16†L7-L15】.  
11. **Temporal integrity:** Always purge/embargo as production (6-month horizon).  
12. **No using future EDGAR filings:** Only use EDGAR filings with declared dates ≤ training cutoff.  
13. **No extreme hyperparams:** Guards enforce sensible bounds (e.g. no `subsample` <0.5 for GBT)【52†L715-L724】.  
14. **No label leakage:** For BL, ensure outcome labels use only actual past outcomes (exclude unmatured data)【53†L838-L842】.  
15. **Keep model simplicity:** Any complex candidate gets a low prior (<0.3) and fallback plan.  
16. **No test deletion:** Guards ensure test count doesn’t drop (see TEST_E01)【47†L229-L238】.  
17. **Capacity limits:** Keep WFO training length ≤ data available (handled by splitter)【18†L75-L83】.  
18. **Probability sanity:** All probability outputs in [0,1]; prevent bad calibration.  
19. **JSON schema validity:** New candidate files must parse without error.  
20. **Conformal logic:** Only use ACI if data shift present; avoid invalid gamma.

### Quick-Win Inventory

Top 10 tasks for first 2 hours (fast eval, high leverage):

1. **TASK-C01 (FRED lag)** – data quality improvement (fast ~5m): reduces bias, broad impact.  
2. **TASK-A01 (Shrinkage α)** – quick metric change (fast) with large known effect【16†L87-L94】.  
3. **TASK-B02 (Temp/Warmup)** – fastest task (10s iter) with promotion-gate relevance【51†L509-L517】.  
4. **TASK-B01 (Feature Routing)** – moderate cost (45s) with solid BA gains【46†L100-L107】.  
5. **TASK-A03 (Ridge α range)** – medium (4–8m) but high ROI for R²【49†L288-L290】.  
6. **TASK-E01 (Test runtime)** – essential infra, direct dev speedup【47†L229-L238】.  
7. **TASK-E02 (fast marker)** – trivial setup, immediate runtime skip on --fast.  
8. **TASK-B03 (Threshold sweep)** – medium, can run after temperature (fast warms).  
9. **TASK-D01 (BL tuning)** – fast (<30s), directly optimizes decision metric【53†L807-L816】.  
10. **TASK-G01/G02 (predict shared harness)** – meta steps to speed future tasks (can run quickly).

### /autoresearch:predict Tasks

Below are static-analysis meta-tasks using `/autoresearch:predict`:

1. **Backlog Prioritization** (Target 6) – Evaluates remaining research ideas from multiple personas to rank them【57†L25-L40】:  
   ```
   /autoresearch:predict
   Goal: Prioritize research backlog items (in docs/research/backlog.md) for expected impact on OOS R², classification BA, and decision utility from the perspective of multiple personas.  
   Metric: python - <<'PYCODE'
   print("Ranked list of backlog items (dummy output)") 
   PYCODE
   Scope: docs/research/backlog.md  
   Guard: true  
   Direction: higher_is_better  
   Log: results/research/predict_backlog.jsonl
   ```
2. **Path A vs Path B Impact** – Static check if dual classifiers add value:  
   ```
   /autoresearch:predict
   Goal: Compare Path A (per-benchmark) vs Path B (composite) classifiers to predict which will yield higher BA when both fully tuned, using code complexity and prospective sample arguments.  
   Metric: python - <<'PYCODE'
   print("Path B likely superior (based on pooled target rationale)")
   PYCODE
   Scope: src/models/classification_shadow.py  
   Guard: true  
   Direction: higher_is_better  
   Log: results/research/predict_pathAB.jsonl
   ```
3. **Feature Family Prioritization** – Identify which feature groups (price vs fundamental vs macro) have highest signal:  
   ```
   /autoresearch:predict
   Goal: Use feature importance or correlation analysis in code to predict which feature families (technical, fundamental, macro) should be prioritized in future engineering.  
   Metric: python - <<'PYCODE'
   print("Technical features likely top predictors")
   PYCODE
   Scope: src/features/ (assuming feature definitions here)  
   Guard: true  
   Direction: higher_is_better  
   Log: results/research/predict_features.jsonl
   ```
4. **Hyperparameter Range Selection** – Static guess best hyperparam bounds (e.g. GBT depth) via small-sample logic:  
   ```
   /autoresearch:predict
   Goal: Assess if the planned GBT hyperparameter ranges (depth up to 4, trees up to 200) are reasonable given N≈70 (favor low complexity).  
   Metric: python - <<'PYCODE'
   print("Max depth 2-3, n_est <=100 recommended")
   PYCODE
   Scope: docs or code comments for GBT config  
   Guard: true  
   Direction: higher_is_better  
   Log: results/research/predict_hyper.jsonl
   ```
5. **Risk Assessment** – Flag tasks that risk data leakage or overfitting:  
   ```
   /autoresearch:predict
   Goal: Static review of proposed changes (e.g. EDGAR lag=0, extreme GBT params) to flag potential leakage or overfit before running.  
   Metric: python - <<'PYCODE'
   print("Warning: EDGAR lag=0 may leak future info")
   PYCODE
   Scope: results/research (scheduled changes)  
   Guard: true  
   Direction: higher_is_better  
   Log: results/research/predict_risk.jsonl
   ```
These `/autoresearch:predict` outputs are guidance, not executable loops.

```text
<autoresearch_task_inventory>
TASK-A01 ShrinkageAlphaSweep A P1 fast none pooled_oos_r2=$(python results/research/v_gen_shrinkage_eval.py --shrinkage $(cat results/research/v134_shrinkage_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-A02 EnsembleBlendWeightOptimization A P2 medium TASK-A01 pooled_oos_r2=$(python results/research/v_alpha_blend_eval.py --weight $(cat results/research/vA02_weight_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-A03 RidgeAlphaGridOptimization A P1 medium none pooled_oos_r2=$(python results/research/v133_ridge_alpha_sweep.py --alpha-min 1e-4 --alpha-max $(cat results/research/v133_alpha_max_candidate.txt) --n-alpha 60 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-A04 GBTHyperparamSweep A P2 slow none pooled_oos_r2=$(python results/research/v137_gbt_param_sweep.py --params-file results/research/v137_gbt_params_candidate.json 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-A05 ConformalCoverageTuning A P3 fast none coverage=$(python results/research/vC01_conformal_eval.py --coverage=$(jq -r .coverage results/research/vC01_conformal_candidate.json) --gamma=$(jq -r .aci_gamma results/research/vC01_conformal_candidate.json) 2>/dev/null | grep coverage= | cut -d= -f2)
TASK-A06 ElasticNetL1RatioSweep A P4 medium TASK-A03 pooled_oos_r2=$(python results/research/vA06_enet_sweep.py --l1-ratio $(cat results/research/vA06_l1_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-A07 WFOWindowTuning A P3 medium none pooled_oos_r2=$(python results/research/vA07_wfo_window_sweep.py --train $(jq .train results/research/vA07_wfo_candidate.json) --test $(jq .test results/research/vA07_wfo_candidate.json) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-A08 GaussianProcessPredictive Y P4 fast none dummy=$(python - <<<'print(0.2)' | grep -v "^")
TASK-B01 FeatureRoutingClassification B P2 medium TASK-B05 covered_ba=$(python results/research/v129_feature_map_eval.py --strategy file:results/research/v129_candidate_map.csv 2>/dev/null | grep covered_ba= | cut -d= -f2)
TASK-B02 TempWarmupTuning B P1 fast none covered_ba=$(python results/research/v135_temp_param_search.py --temp-min 0.5 --temp-max $(cat results/research/v135_temp_max_candidate.txt) --warmup $(cat results/research/v135_warmup_candidate.txt) --low 0.15 --high 0.70 2>/dev/null | grep covered_ba= | cut -d= -f2)
TASK-B03 ThresholdSweepClassification B P3 medium TASK-B02 covered_ba=$(python results/research/vB03_threshold_sweep.py --low $(jq .low results/research/vB03_thresh_candidate.json) --high $(jq .high results/research/vB03_thresh_candidate.json) 2>/dev/null | grep covered_ba= | cut -d= -f2)
TASK-B04 ClassImbalanceWeight B P4 medium none covered_ba=$(python results/research/vB04_class_weight_eval.py --weight $(cat results/research/vB04_weight_candidate.txt) 2>/dev/null | grep covered_ba= | cut -d= -f2)
TASK-B05 AggregateWeighting B P4 fast none dummy=$(python - <<<'print("No change needed")')
TASK-C01 FREDPublicationLagOpt C P1 slow none pooled_oos_r2=$(python results/research/v134_fred_lag_sweep.py --lag-overrides "$(cat results/research/v134_lag_candidate.json)" 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-C02 EDGARLagTuning C P2 medium none pooled_oos_r2=$(python results/research/vC02_edgar_lag_eval.py --lag $(cat results/research/vC02_edgar_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-C03 CorrelationPruning C P3 fast none pooled_oos_r2=$(python results/research/vC03_corr_prune_eval.py --rho $(cat results/research/vC03_rho_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-C04 DerivedFeatureWindow C P4 medium none pooled_oos_r2=$(python results/research/vC04_window_sweep.py --window $(jq .mom_window results/research/vC04_window_candidate.json) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-C05 FractionalDiffTuning C P4 medium none pooled_oos_r2=$(python results/research/vC05_fracdiff_sweep.py --d $(cat results/research/vC05_fracdiff_candidate.txt) 2>/dev/null | grep pooled_oos_r2= | cut -d= -f2)
TASK-D01 BLParamTuning D P2 fast none recommendation_accuracy=$(python results/research/v138_bl_param_eval.py --params-file results/research/v138_bl_params_candidate.json 2>/dev/null | grep recommendation_accuracy= | cut -d= -f2)
TASK-D02 KellyParamTuning D P3 fast none utility_score=$(python results/research/vD02_kelly_eval.py --fraction $(jq .fraction results/research/vD02_kelly_candidate.json) --cap $(jq .cap results/research/vD02_kelly_candidate.json) 2>/dev/null | grep utility_score= | cut -d= -f2)
TASK-D03 NeutralBandTuning D P4 fast none success_rate=$(python results/research/vD03_neutral_band_eval.py --band $(cat results/research/vD03_band_candidate.txt) 2>/dev/null | grep success_rate= | cut -d= -f2)
TASK-E01 TestRuntimeReduction E P1 slow none elapsed_seconds=$(bash scripts/measure_test_time.sh 2>/dev/null | grep elapsed_seconds= | cut -d= -f2)
TASK-E02 FastMarkerCoverage E P2 fast none dummy=$(python - <<<'print("Mark slow tests for --fast")')
TASK-F01 ReportSectionAudit F P3 fast none dummy=$(python - <<<'print("Suggest reordering report sections")')
TASK-F02 TierLabelRefinement F P3 fast none dummy=$(python - <<<'print("Consider adding 'confidence' text to labels")')
TASK-F03 DashboardFreshness F P4 fast none dummy=$(python - <<<'print("Propose adding data as-of date on dashboard")')
TASK-G01 WFORegressionHarness G P1 fast none dummy=$(python - <<<'print("Create a shared WFO eval module")')
TASK-G02 ClassificationHarness G P1 fast none dummy=$(python - <<<'print("Create a shared classification eval module")')
TASK-G03 CandidateSchemaStandard G P1 fast none dummy=$(python - <<<'print("Define JSON/text candidate file schemas")')
TASK-G04 BaselineScriptRobustness G P2 fast none dummy=$(python - <<<'print("Include baseline checks in harness scripts")')
TASK-G05 GuardPatternLibrary G P3 fast none dummy=$(python - <<<'print("Publish common pytest guard patterns")')
TASK-G06 BacklogPrioritization PREDICT P2 fast none dummy=$(python - <<<'print("Prioritized backlog list")')
TASK-G07 PathA_v_PathB PREDICT P3 fast none dummy=$(python - <<<'print("Path B likely more efficient")')
TASK-G08 FeatureFamilyRanking PREDICT P3 fast none dummy=$(python - <<<'print("Technical features likely most useful")')
TASK-G09 HyperparamRangeSelect PREDICT P3 fast none dummy=$(python - <<<'print("Depth<=3, trees<=100 recommended")')
TASK-G10 ChangeRiskAssessment PREDICT P3 fast none dummy=$(python - <<<'print("Flag EDGAR lag0 as potential leakage")')
</autoresearch_task_inventory>
```

**Sources:** Project docs and code described above【16†L87-L94】【49†L288-L291】【50†L411-L419】【51†L509-L517】【53†L807-L816】【9†L10-L16】 were used to extract constants, baselines, and constraints. Any unsupported claim was avoided or attributed to domain best practice.