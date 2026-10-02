v202 tested eight fixed insurance and valuation feature blocks against the accepted v200 forecast on the same historical development dates. None qualified as a research finalist. The best six-month block, property-excluded PIF growth (I2), raised R² by 0.066, but the evidence was too uncertain after campaign correction and its probability scores worsened. The owner’s vest decision and monthly email do not change.

## What was tested

The [preregistration](outputs/registered.json) reserves campaign slots 7–14. Each candidate changes one block in the production Ridge and unchanged shallow GBT ensemble; there are no combinations or later searches. The same 50 Ridge penalties, 10 shrinkage values, GBT settings, mature-past weights, calibration and interval method apply to candidate and v200 control.

| Slot | Block | Feature treatment |
|---|---|---|
| 7 / I1 | Underwriting | Twelve-month combined ratio and annual change replace the incumbent ratio. |
| 8 / I2 | PIF | Calendar YoY growth in agency auto, direct auto and commercial policies excludes property-definition drift. |
| 9 / I3 | Gainshare | Positive trailing underwriting margin times stable PIF growth adds a dividend-mechanics proxy. |
| 10 / I4 | Investments | Twelve-month investment-income growth and percent-unit book yield replace incumbent versions. |
| 11 / I5 | Book value | Latest share-basis BVPS growth replaces incumbent growth. |
| 12 / I6 | ROE | Trailing monthly net income divided by average common equity adds ROE without treating Q4 as an annual quarter. |
| 13 / I7 | Valuation | Split-consistent P/B, positive-earnings P/E and their log spread are added. The spread cancels price algebraically and is an earnings-to-book ratio. |
| 14 / I8 | Premiums/rates | Trailing-12-month NPW growth and its insurance-PPI gap replace single-month NPW growth and the incumbent rate-gap field. |

Features use complete monthly report calendars and become usable only on or after recorded filing dates. The registered missing-date fallback is report month plus two business month ends. No consumed development row needed it: the [sensitivity table](outputs/fallback_sensitivity.json) has zero affected rows under one-, two- or three-month fallback. PPI uses the frozen one-calendar-month publication rule; observed historical release timestamps and vintages are absent. Later amendments to stored rows cannot be reconstructed as first-reported values.

The A3 audit used NPW/NPE seasonality across 2016–2026 only to define I8, without holdout return metrics. Median February-minus-March ratio was +0.098 in 2016–2023; March-minus-February was +0.225 in 2024–2026. Stored `npw_growth_yoy` standard deviation was 0.0757 before 2024 and 0.1979 in 2024. This confirms a shifted fiscal-month pattern, so I8 uses trailing-12-month NPW growth. [The audit](outputs/fiscal_month_audit.json) was frozen before fitting.

## Primary six-month development result

Matched v200 R² is 0.0105, equal-weight benchmark IC −0.0416, hit rate 0.651 versus the mature-past majority rule's 0.588, ECE 0.172 and nominal 80% interval coverage 0.761. Every candidate uses its 942 rows and 144 monthly origins. The first-step column is D9's descriptive estimate of the ΔR² needed to clear the first Holm step at that candidate's own spread: 3.01 times the standard deviation of 2,000 date-block ΔR² draws. It is no additional test or success route.

| Candidate | R² | ΔR² vs v200 | Raw / Holm p | Approx first-step ΔR² | Equal-weight IC | Hit | ECE | 80% cover | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| I1 | −0.004 | −0.015 | 1.000 / 1.000 | 0.148 | −0.080 | 0.641 | 0.201 | 0.770 | ΔR², Holm, IC, direction, ECE |
| I2 | 0.077 | +0.066 | 0.074 / 1.000 | 0.117 | −0.006 | 0.652 | 0.159 | 0.763 | Holm, Brier, log loss |
| I3 | 0.057 | +0.046 | 0.158 / 1.000 | 0.118 | −0.016 | 0.628 | 0.178 | 0.753 | Holm, direction, ECE, coverage |
| I4 | 0.053 | +0.043 | 0.154 / 1.000 | 0.115 | −0.023 | 0.650 | 0.180 | 0.775 | Holm, direction, ECE |
| I5 | 0.012 | +0.002 | 1.000 / 1.000 | 0.027 | −0.064 | 0.653 | 0.191 | 0.767 | ΔR², Holm, IC, direction, calibration |
| I6 | 0.036 | +0.025 | 0.247 / 1.000 | 0.113 | −0.055 | 0.642 | 0.187 | 0.774 | Holm, IC, direction, ECE |
| I7 | −0.117 | −0.127 | 1.000 / 1.000 | 0.289 | +0.012 | 0.639 | 0.159 | 0.762 | ΔR², Holm, direction |
| I8 | −0.017 | −0.027 | 1.000 / 1.000 | 0.145 | −0.091 | 0.613 | 0.181 | 0.784 | ΔR², Holm, IC, direction, calibration |

The registered primary requires ΔR² at least +0.010, paired moving-date-block campaign-Holm p below .05, IC loss no more than .01, and no deterioration in hit skill, Brier, log loss, ECE or interval coverage distance from .80. Blocks are six months, all benchmarks for a date travel together, seed 20260926 and 2,000 draws. Pending and cancelled slots stay p=1 in the 38-candidate family. There is no active-policy proposal. [Metrics](outputs/metrics.json), [paired comparisons](outputs/comparison.json), [multiplicity](outputs/multiplicity.json), [fold ledger](outputs/fold_ledger.csv) and [closeout](outputs/closeout.json) contain unrounded values and every gate.

Twelve-month results are diagnostics only. I4's ΔR² was +0.074 with raw paired diagnostic p 0.0155; v200 12-month R² was −0.191. This cannot rescue its failed six-month primary. The 372 scored rows are only 66 overlapping monthly origins, or 5.5 nonoverlapping years.

## Frozen v201 comparison and valuation re-analysis

The six v201 attempt 2 recipes were frozen before v202. Selecting among their preregistered out-of-sample predictions inside each downstream three-fold inner history yielded a comparator on only 78 six-month dates (624 benchmark rows); earlier folds and all 12-month folds lacked enough upstream history and remain unscorable. On that common six-month subset its R² was −0.191, v200's was −0.078, and I2's was +0.020. I2's descriptive ΔR² against this fold-causal v201 comparator was +0.210. Smaller support and the failed preregistered primary prevent a finalist. [Matched comparison](outputs/v201_matched_comparison.json) and [selection ledger](outputs/v201_selection_h6.csv) show the calculation. The full-development v201 P1 result (+0.058 ΔR², raw p 0.087, Holm p 1.0) is context; a globally selected recipe was excluded from honest inference.

The separate A4 audit preregistered six descriptive tests: inverse P/B, inverse P/E and their spread in 2004–2014 and 2015–development end, with six-month date-block inference and their own Holm correction. The first computation incorrectly inherited VOO's 2010 start; a source correction then failed to require identical P/B and P/E dates. Both six-test outputs are preserved as discarded attempts. The [final matched-support audit](outputs/descriptive_pb_pe_matched.json) uses the same PGR return label from VGT's longer development rows and only dates where all three signals exist. In 2004–2014, inverse P/B IC was +0.692 and inverse-spread IC +0.509 on the same 105 dates (both Holm p 0.003); inverse P/E was near zero. In 2015–development end, inverse P/B IC was +0.366 on 98 matched dates (Holm p 0.070), while P/E and spread did not clear Holm. These are overlapping-label associations, not out-of-sample forecasts or promotion claims. [The support amendments](outputs/a4_matched_support_amendment.json) record the correction before the final computation; they add no candidate slots.

## Source checks and limitations

The [source audit](outputs/source_audit.json) found complete monthly keys from August 2004 to August 2023, four leap-February rows, 11 negative-net-income months, book yields in percent units (1.6–5.6), and Q4 quarterly net income matching three monthly values in 15 comparable Q4s to floating precision. The 2006 four-for-one split changes raw weekly close from 107.20 to 27.24; multiplying post-split close by four gives a +1.64% continuous price move. Targets retain v200's unadjusted-price, manual-split, fractional-share DRIP definition.

The [identity detail](outputs/source_identity_detail.json) found assets minus liabilities minus recorded equity above 1% of assets in 36 months, and recorded BVPS more than $0.10 away from recorded equity divided by common shares in 66 months starting March 2018 (up to $0.85). Source rows are not repaired here. This weakens I5/I7 interpretation, although neither passed forecast gates. PIF components are stored repaired values, not first-reported vintages; `pif_total` differs from first reports in 50 months from July 2019, as [v205's audit](../v205_dividend_bvps/outputs/source_vintage_audit.json) found. The accepted March 2026 VWO dividend provider gap remains quarantined in v200's 12 affected targets and does not enter development results.

All model transforms, pruning, medians, scaling, thresholds and priors use earlier training history. Outer `TimeSeriesSplit` has sliding 60/120-month training windows, six-month tests, gaps 12/24 (h-month purge plus h-month embargo). Each outer training window has three inner chronological six-month folds with the same 2h gap and at least 24/60 usable unique training months. Label-end and availability checks apply to scored origins. No K-fold, CPCV, LOO, shuffled fit or full-sample scaler is used. These forecasts have no trading-cost or policy P&L estimate.

## Reproduction and verification

The accepted [v200 lock](../v200_clean_baseline/outputs/baseline_lock.json) SHA256 is `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`. It binds repaired code `c3b4798b90a43f9dbd616981861e9ff01fa2de72` and database blob `ed7997f6f540f664e59dd44a4079616d74a8e8cb`, SHA256 `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`. Seed parents and R2-lite/R3b IDs remain in that lock. v202 exported the exact DB to external scratch and opened it immutable. The tracked checkout DB was unchanged at SHA256 `86329ac84fa546379781d2ca951845ea7b27583200f4cd75cac1139ec1af815d` before and after fitting. The registry check used historical add-only entry verification.

The first execution attempt stopped before fitting because Git normalized preregistration line endings. Scoped byte-preservation rules were committed, and exact committed/working equality passed before the 16 registered candidate-horizon fits. The first full-suite attempt stopped at collection with 26 import errors because `PYTHONPATH` put `src` ahead of the repository's `research` namespace. A copied isolated Python 3.12.14 venv was installed editable for this checkout and the required full command rerun with repository root on `PYTHONPATH`. No candidate, grid or lag was changed after results appeared.

The final `python -m pytest -o addopts="--tb=short" -q` run exited 0: **2712 passed, 1 skipped, 2 xfailed, 146 warnings in 379.10s (0:06:19)**. The first collection attempt exited 2 with 26 environment import errors; there were no inherited test failures in the corrected final run. The seven independent v202 math fixtures passed after observed red failures, and mypy found no issues in the new `research_lib` module. [Final pytest log](outputs/full_pytest_final.log) and [exit file](outputs/full_pytest_final.exit) preserve pytest's own result.

## What changed / what is left

What changed: one research-only study, exact preregistration, tests, features, predictions, metrics, ledgers, provenance and registry entry. What is left: v207's one-time quarantine decision under D2 and any separate data repair for equity/BVPS identity. Nothing is promoted here.
