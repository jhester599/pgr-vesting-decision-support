The incumbent has little demonstrated forecasting edge under this stricter
research protocol. Its six-month R² is +0.0105, with uncertainty spanning
zero; its twelve-month R² is −0.1914. Calibration and directional evidence
do not justify stronger trading decisions. The exactly pinned development
forecasts reproduced byte for byte and can serve as the v201–v206 research
comparator. This study changes research artifacts only.

The approved input is R3b code
`c3b4798b90a43f9dbd616981861e9ff01fa2de72`, with DB bytes extracted from
`ed7997f6f540f664e59dd44a4079616d74a8e8cb:data/pgr_financials.db`, SHA256
`38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`.
[baseline_lock.json](outputs/baseline_lock.json) binds those exact inputs,
the parent seed, R2-lite rebuild/run and PR identifiers, R3b closeout,
dependencies, source inputs and final forecasts. It records the separate
research execution commit, `b0a590bd2e9df975c420166d96cc3d919eb4934e`.
The branch began at fetched master
`e4500daa92adf839415cd5aca25c8e3d8274bb33`; its production source and config
match the approved code pin. No data repair or provider call occurred here.

The original seed's required dividend check was red. Its 20 stale tickers
were VTI, VOO, VGT, VHT, VFH, VIS, VDE, VPU, KIE, VXUS, VEA, VWO, VIG,
SCHD, BND, BNDX, VCIT, VMBS, VNQ and ALL. The approved repair passes required
freshness, split/DRIP, month/quarter accounting, duplicate-month and live-input
checks. All 3,816 development targets match independent raw-price, manual
split and fractional-share reinvestment calculations; maximum difference is
`1.1102230246251565e-15`. The accepted March 2026 VWO provider gap affects
exactly [12 targets](outputs/vwo_accepted_gap.csv), all quarantined. GLD is
an audited nonpayer. Unresolved CB prices and optional stale research
features are excluded from the incumbent's frozen catalog.

Actual filing dates gate EDGAR features. FRED uses the frozen conservative
calendar publication lags; historical release times and vintages are absent.
The repaired historical EDGAR values also lack original per-row retrieval
timestamps. These limitations are disclosed rather than certified away.
Weekly source bars require DRIP at the last observed unadjusted close on or
before an ex-date, matching the approved stored target convention. The
[source availability ledger](outputs/source_availability_ledger.csv) records
the assumptions and filing/first-use gates. Research extraction disables
legacy full-frame count pruning, retains the fixed catalog, and fits each
imputer/scaler within its own chronological training history.

The registry, ordered candidate grids and partition hashes were committed
before fitting. There was one incumbent blueprint and zero alternatives:
the unchanged Ridge 50 penalties `logspace(-4,4,50)`, fixed depth-2/50-tree
GBT, and the distinct 10 mature-label shrinkage values. Outer training/test/
gap sizes are 60/6/12 months for 6M and 120/6/24 for 12M. All three inner
folds have test size 6, the same horizon gap and minimum mature support
24/60 months. Monthly dates are split before benchmark rows are expanded.
Unsupported folds stay unscorable; no gaps or support thresholds were reduced.

Only labels with both target end and availability before 2023-09-29 are
fitted or scored. Development, boundary-purged and quarantine partitions
contain 3,816, 180 and 540 rows respectively. Their SHA256 hashes were sealed
before fitting. The quarantine union begins at the earlier 12M boundary;
dates whose outcomes cross it are purged. Monthly annual-horizon labels
overlap and are not independent annual events. No quarantine metric was
computed. [Step 13V](../../../docs/reviews/2026-09-26_step13v_comparison.md)
is provisional historical context only: its full-history September replay
was not run. Any reproduction belongs in the registered v207 batch.

The R² comparator uses all earlier labels matured by each origin, including
training history. IC and directional inference use moving monthly-date
blocks of length 6/12, all benchmarks kept together, seed 20260926 and 2,000
replicates. There is one preregistered primary 6M loss-improvement reference
test; 12M remains secondary. Its conservative 38-test Holm envelope assigns
unused slots p=1. No campaign alternative was fitted; all 38 actual campaign
slots remain pending/unused. Secondary safeguards supply no alternate route
to promotion. [multiplicity.json](outputs/multiplicity.json) records this
distinction explicitly. Forecasts are gross targets; no active policy,
transaction cost model or trading return is claimed.

| Development metric | 6M primary reference | 12M secondary |
|---|---:|---:|
| OOS rows / unique origins / benchmarks | 942 / 144 / 8 | 372 / 66 / 8 |
| Origin window | 2011-03–2023-02 | 2017-03–2022-08 |
| Latest outcome availability | 2023-08-31 | 2023-08-31 |
| Honest R² | +0.010543 | −0.191446 |
| R² 95% date-block interval | [−0.131847, +0.123321] | [−0.726424, +0.053108] |
| Equal-weight IC / panel IC | −0.041563 / −0.026101 | −0.071516 / −0.026013 |
| IC p, equal-weight / panel | 0.620690 / 0.776112 | 0.593703 / 0.831584 |
| Forecast hit / mature past majority hit | 0.650743 / 0.588110 | 0.612903 / 0.763441 |
| Observed positive-label base rate | 0.708068 | 0.787634 |
| Directional-skill raw / adjusted p | 0.029485 / 1.000000 | 1.000000 / 1.000000 |
| Primary raw / adjusted p | 0.216892 / 1.000000 | secondary loss p=1.000000 |
| Mature prequential Brier / log loss / ECE | 0.267000 / 0.821413 / 0.171946 | 0.273357 / 1.440522 / 0.317500 |
| Calibration rows / unevaluated warmup | 730 / 212 | 129 / 243 |
| Nominal 80% ACI coverage / scored rows | 0.760920 / 870 | 0.745098 / 255 |
| Interval warmup / shrinkage warmup rows | 72 / 58 | 117 / 88 |
| Scorable / unsupported outer benchmark-folds | 157 / 35 | 62 / 26 |

Hit rates use row support; the paired directional test averages within dates
before its block resampling, so its skill estimate need not equal the
difference of those two row-weighted hit rates. Unsupported 6M outer pairs
include 15 with insufficient outer labels and 20 with insufficient inner
history; the 12M counts are 3 and 23. [metrics.json](outputs/metrics.json)
also contains the full IC confidence intervals. All predictions, mature
naive forecasts, component weights, selected penalties, residuals and warmup
flags are saved; [fold_ledger.csv](outputs/fold_ledger.csv) records support.

At the development as-of 2022-08-31, actual `generate_signals` plus
`compute_aggregate_health` matches independent input assembly exactly in
aggregate R², equal-weight and pooled IC, hit rate and PT p for both horizons
(maximum difference 0, required tolerance 1e-9). The matched-support protocol
bridge fixes the repaired input and endpoint. It changes research gaps,
nested transformation/penalty selection and the 12M training window together;
it is not an isolated experiment on gap length alone. Production-health IC
uses pre-shrink `z`; both matched-support IC columns instead use issued
`y_hat`. No bridge parameter or metric was searched.

| Matched development support at 2022-08-31 | Strict research R² | Production-settings R² |
|---|---:|---:|
| 6M, 828 rows / 132 dates | +0.008296 | +0.133871 |
| 12M, 276 rows / 54 dates | −0.146719 | −0.147884 |

The [bridge table](outputs/bridge_table.csv) preserves the A1 equivalence
rows separately from these matched-support diagnostics. Production 6M gap
8 and 12M gap 15 are used only for this development bridge.

Fixed endpoint controls are frozen separately, with exact units and
availability rules in the lock and [control support](outputs/control_support.json):

- PathB is the same six-benchmark composite event, class 1 for relative 6M
  return below −.03, with the production balanced logistic C=.5 and fixed
  mature-prediction temperature grid. Raw support is 72 dates; calibrated
  support is 35, with 37 unevaluated calibration warmup dates. Raw/calibrated
  Brier is .183120/.161680, log loss .907307/.493313 and ECE .212011/.107776.
  Raw hit/base hit is .805556/.763889; calibrated hit/base hit is
  .800000/.657143 on its smaller matched support. Skill raw p is
  .308346/.159420, adjusted p=1 for both. These are separate binary
  diagnostics, not regression improvement evidence.
- Past 12M cash predicts next 12M cash, in USD per origin-date share with
  split-adjusted share counts. It has 262 scored forecasts and 11 warmup
  rows; descriptive R² against the mature target mean is .039015, MAE .797459.
- The latest-filed current BVPS predicts exact report-month+12 growth with
  actual target filing availability. Its mature mean control has 205 scored
  forecasts and 12 warmup rows; descriptive MAE is .121862 in fractional
  growth units. Overlapping labels are not independent annual events.
- The unbenchmarked PGR 6M split/DRIP return control is a signed asset
  return, not an absolute-value magnitude. It has 274 scored forecasts and
  5 warmup rows; mature-mean descriptive MAE is .118205.
- The x23/x18 annual excess-dividend/current-November-BVPS endpoint uses
  December 1 through next February 28/29, post-policy November origins, and
  the causal prior-24M median of positive payments ≤.25. Only three positive
  annual events are supported (2019–2021 November origins), with two finite
  control forecasts and one warmup. A zero-excess year is excluded from this
  conditional-positive endpoint. No ordinary-payment fallback is invented
  for an earlier unsupported origin. This endpoint is unscorable; there is
  no annual performance score, p-value or promotion claim.

Cash/BVPS/asset controls are descriptive endpoint comparators, not trained
model evidence or active-policy proposals. Their units are never pooled.
The sparse annual endpoint cannot establish an independent yearly edge for
v204/v205. Calibration/coverage acceptance gates apply to active proposals;
v200 freezes this weak incumbent comparator without requiring positive R².

Every new mathematical module has independent expected outputs observed red
before implementation, then green. The old v37 helper fails the overlapping
conditional-mean oracle fixture (−.241379 versus honest +.769968) and the
two-benchmark pooling fixture (+.998840 versus honest −3). These two known
defects remain strict expected failures and the helper is not reused.
An exact realised-outcome oracle scores 1 with either formula; the first
fixture intentionally includes fixed realised forecast error to reveal the
wrong maturity comparator. It does not claim a perfect oracle can fail R².

Attempts and discarded approaches are preserved in
[verification logs](outputs/verification/) and [closeout.json](outputs/closeout.json).
Initial pytest temporary-directory permissions were resolved by an isolated
task runtime. The first real preflight stopped without fitting when SQLite
migration rows lacked a row factory; this was repaired in code and rerun.
Independent review rejected full-frame feature pruning, the initial annual
endpoint interpretation, an uncorrected ACI quantile, missing availability
gates, silent unsupported histories, a blocked-README error, incomplete
source pins and missing control summaries. Each correction occurred before
any fitting. There was no feature, metric, window or threshold search after
results. Legacy production bridge all-NaN warnings are retained in its log.

The exact requested full command was
`python -m pytest -o addopts="--tb=short" -q`. Pytest's own summary is:
`2658 passed, 1 skipped, 2 xfailed, 105 warnings in 345.67s (0:05:45)`, exit 0.
There are no inherited test failures in this run; legacy warnings and the
skip are reported, and the two xfails document the old helper defects.
The v200 fixtures separately report `73 passed, 2 xfailed`, exit 0.
The second read-only run reproduces all nine core files byte for byte;
maximum numeric difference is 0 against tolerance 1e-10. The production
bridge is independently assembled at the same development date in the
first run and is not a second quarantine replay.

Tracked DB SHA256 before and after both execution and the full suite is
`38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`.
Runtime/core dependencies are exact-locked and verified before execution.
The [complete environment](outputs/dependency_environment.json) additionally
records transitive and tooling versions; no package installation changed
between preregistration, either execution and this complete capture. Later
sessions must verify these versions and must not silently upgrade them.
Scoped Git attributes preserve exact CSV/output/new-code bytes across
checkouts; inherited source hashes allow only Git CRLF/LF equivalence.
Unrelated `.codex/` work remains untouched. There is no fetcher, provider,
email, live model, live feature or recommendation-policy change.

To reproduce, install this checkout normally, use the exact dependency
versions in `outputs/runtime_lock.json`, and run
`python research/studies/v200_clean_baseline/run.py --skip-bridge --scratch <external-dir> --output-dir <external-output-dir>`.
The runner verifies the approved Git/DB and every consumed file pin before
fitting and writes a blocked README on missing/failed preflight. DB copies
and regenerated caches stay outside the tracked data path. Do not silently
refreeze an executed register or upgrade dependencies. Later sessions must
copy and verify the accepted lock and its forecast pins, and record their
own execution commit separately.

What changed: an installed causal research namespace, exact repair and
forecast locks, independent tests, development baseline/controls, the
production bridge, support/availability/inference ledgers and reproducible
research artifacts. What is left: run only separately authorized later
studies against these exact accepted inputs; v207 opens the retrospective
quarantine once after finalists are frozen. Any promotion needs its own
governance PR under the amended D2 rule. Sparse annual support and historical
vintage limitations remain unresolved evidence limits.
