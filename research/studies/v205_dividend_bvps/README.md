## Owner summary

**What was tested.** v205 rebuilt the x-series dividend and book-value
(BVPS) research on the clean v200 inputs. The rebuild restates share counts
across splits, reinvests dividends as fractional shares, counts December
special dividends and matches report months by calendar. It then tested six
preregistered forecasting ideas against the fixed forecasts v200 already
uses for the same targets:

- three dividend ideas: D1, D2 and D3;
- three BVPS ideas: B1, B2 and B3.

**Result in one sentence.** No idea cleared its preregistered bar, so
nothing advances to v207 and nothing changes in the live system.

**What it means for the vest decision.** Nothing changes. The clean
rebuild changes the historical *labels* a great deal. For example, raw BVPS
treated the 2006 split as a −70% to −75% collapse, and raw-price returns
missed dividends at 98 of 279 origins. But clean inputs did not produce a
model that forecasts next year's dividend cash, BVPS growth or PGR's 6-month
return better than simple past-only rules.

The nearest miss was D2, a dividend-cash model using past cash, combined
ratio, calendar-matched PIF growth and trailing-12-month NPW growth:

- its average error was 9.0% lower than "next year's cash = last year's
  cash", just short of the required 10%;
- its evidence was not statistically reliable: p = .27 before and 1.0 after
  campaign correction.

The BVPS models were clearly worse than a past-average forecast.

**Metric definitions.**

- **MAE** is the average absolute forecast error, in the target's own units.
- **Honest R²** is 1 − Σ(outcome − forecast)² / Σ(outcome − past mean)². The
  past mean uses only outcomes already known at each forecast date. Above 0
  means the forecast beat that running average.
- **ΔR²** is the candidate's R² minus its control's R² on the same rows.
- **IC** is the rank correlation between forecasts and outcomes.
- **Hit rate** is the share of dates where the forecast got the direction
  right. The events are:
  - cash: next-year cash above last year's;
  - BVPS growth and return: above zero.
- **Base hit rate** comes from always predicting whichever direction was
  more common among outcomes already known.
- **Brier score** is the mean squared error of the event probability.
- **ECE** is the average gap between predicted and observed frequencies,
  over 10 bins.
- **80% coverage** is how often the outcome fell inside the nominal 80%
  interval.
- **p-values** come from a one-sided moving-block bootstrap: 2,000 draws,
  blocks of 12 months (6 for B3), seed 20260926.
- **Adjusted p** is Holm-corrected across the 38-test campaign. v205 fills
  slots 29–34, and slots not yet used by other studies count as p = 1.

## Results (development only, before 2023-09-29)

| Candidate | Target (h) | Scored dates | MAE vs control | ΔMAE | Honest R² vs control | ΔR² | raw / Holm p | Hit vs control / base | Brier vs control | 80% cover vs control | Passes |
|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| D1 past cash + CR | next-12M cash (12) | 78 (2016-03–2022-08) | 1.919 / 1.376 | −39.5% | −0.273 / +0.250 | −0.523 | 1 / 1 | .500 / .487 / .513 | .474 / .271 | .055 / .218 | no |
| D2 + PIF/NPW | next-12M cash (12) | 66 (2017-03–2022-08) | 1.442 / 1.585 | **+9.0%** | +0.442 / +0.251 | **+0.191** | .268 / 1 | .667 / .409 / .591 | .271 / .305 | .302 / .279 | no |
| D3 Gainshare/book yield | annual excess/BVPS | 3 events | — | — | — | — | 1 / 1 | — | — | — | closed |
| B1 BVPS lags + ROE | next-12M BVPS growth (12) | 72 (2016-10–2022-09) | .187 / .128 | −46.0% | −1.420 / 0 | −1.420 | 1 / 1 | .389 / .806 / .806 | .408 / .233 | .184 / .592 | no |
| B2 + CR/PIF/NPW | next-12M BVPS growth (12) | 66 (2017-04–2022-09) | .186 / .132 | −40.8% | −1.088 / 0 | −1.088 | 1 / 1 | .379 / .788 / .788 | .330 / .261 | .233 / .605 | no |
| B3 structural x16 | 6M PGR DRIP return (6) | 144 (2011-03–2023-02) | .130 / .097 | −34.5% | −0.944 / 0 | −0.944 | 1 / 1 | .785 / .813 / .813 | .153 / .195 | .732 / .811 | no |

In the table:

- ΔMAE is the relative reduction, so positive is better.
- For the BVPS and return lanes, the control forecast *is* the past mean, so
  the control's honest R² is exactly 0.

Gates each lane failed ([metrics.json](outputs/metrics.json),
[comparison.json](outputs/comparison.json)):

- D2 failed only the 10% MAE gate and the adjusted-p gate. Its ΔR², hit rate,
  Brier score, coverage and support gates passed.
- D1 also failed ΔR², Brier score and coverage.
- B1 and B2 failed every performance and calibration gate.
- B3 failed MAE, p, ΔR², direction and coverage. Its Brier score (.153 vs
  .195) and ECE (.137 vs .227) were better than the control's, but that does
  not rescue the primary loss.

No winner was nominated; v205 sends **no finalist** to v207.

## What each registered candidate was

The register, features, grids and rules were committed in
[preregistration.json](outputs/preregistration.json) before any fit
(commit `8c2fe9b`; pre-fit amendment A1 at `e2d3906`, below). Every
candidate is a Ridge regression. Its median imputer and scaler are fitted
inside each training history, and its penalty is chosen from {1, 10, 100}
by summed absolute error in three inner chronological folds (exact ties
choose the larger penalty). There was no algorithm tournament and no search
after results.

**Dividend lane** (units: USD cash per one origin-date PGR share).

- **D1** uses past-12M cash, the prior-year past-12M cash and trailing
  12-month combined ratio.
- **D2** adds calendar-matched PIF year-over-year growth and trailing-12M
  NPW growth. The trailing NPW growth follows the A3/F16 recommendation over
  monthly NPW YoY.
- D1 and D2 forecast the change from past-12M cash, and the fixed control
  (past-12M cash) is added back.
- **D3** revisits the x23 survivor. The x23 contract is:
  - post-policy November origins;
  - December 1 to next February 28/29 cash;
  - minus the median positive payment ≤ $0.25 in the prior 24 months;
  - divided by the latest filed BVPS.

  Features would have been Gainshare estimate and book yield (percent,
  F10), compared only with v200's past-only same-label mean. Only three
  positive annual events exist before the boundary (2019, 2020, 2021
  Novembers):
  - 2018 has no causal ordinary baseline;
  - 2022 had zero excess;
  - later years fall in the quarantine.

  The preregistered rule therefore closed D3 without fitting
  ([annual_support.csv](outputs/annual_support.csv)).

**BVPS lane.**

- **B1** uses split-consistent BVPS growth over 12, 3 and 6 months and
  trailing ROE.
- **B2** adds combined ratio, PIF growth and NPW growth.
- The target of B1 and B2 is the latest filed BVPS to the exact report
  month + 12, available on the future report's filing date. The control is
  v200's past-only prevailing mean growth.
- **B3** freezes x16's `adjusted_structural_bvps_pb_6m` mapping: Ridge on
  x9's `bvps_lags` block forecasts split-consistent dividend-adjusted 6M
  BVPS growth. With P/B held at its current value (x15 found no overlay beat
  it), the forecast 6M PGR return equals that growth. It is scored on the
  split/DRIP 6M PGR return against v200's mature mean return.
- B3 differs from the archive in two ways:
  - The x16 fixed penalty of 1,000 is outside this campaign's {1, 10, 100}
    budget.
  - The calendar flags use the origin month rather than x9's lagged feature
    month.
- B3's inner penalty choice scores the adjusted-BVPS training label, not the
  return.

Ordinary/special dividend components are only a descriptive decomposition
([dividend_decomposition.csv](outputs/dividend_decomposition.csv)), not an
occurrence model. Its rule is: before December 2018 is the annual variable
dividend; after that, payments ≤ $0.25 are regular and larger ones special.
The February 2019 payment of $2.514 therefore lands in "special", although
it was the last payment under the old policy.

## The hypothesis: handling versus complexity

The hypothesis was that clean handling changes the x-series conclusions more
than algorithm complexity. It is supported for the *labels* and not rescued
for the *models*.

The [target audit](outputs/target_audit.json) compares archived-style
handling with the clean labels (label arithmetic only):

- **BVPS 12M growth.** Raw per-share BVPS mis-states 12 origins from 2005-06
  to 2006-05 by up to 0.914. This is F15's split cliff: x12 recorded it as a
  "capital event".
- **PGR 6M return.** Raw close ratios differ by more than 1 point from the
  split/DRIP return at 98 of 279 origins (maximum 0.809, 2006 split).
- **Next-12M cash.** Unrestated per-share cash differs at 21 origins before
  the 2006 split.
- **December specials.** A January–March window (x1, F25) misses the $1.50
  December 2021 special: its Q1 cash is $1.50 below the December–February
  cash. The $1.00 December 2010 special would also be missed, but it falls
  outside the post-policy endpoint.

Earlier x-series "wins" rested partly on those label errors. On clean
labels with honest nested folds, the only surviving direction is modest
dividend-cash persistence smoothing (D2). It is not significant.

## Inputs, pins and availability

Pinned inputs:

- **Baseline and seed.** The accepted v200
  [baseline lock](../v200_clean_baseline/outputs/baseline_lock.json)
  (SHA256 `c558ecb5…02ffb0`) binds:
  - R3b code `c3b4798b90a43f9dbd616981861e9ff01fa2de72`;
  - DB SHA256 `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`,
    exported with `git show ed7997f:data/pgr_financials.db` outside the
    repository and opened read-only and immutable;
  - the parent seed `aae0be8…` / `7c68ef…`.
- **Consumed v200 outputs.** The control forecasts, runtime and dependency
  locks, partitions and control support are pinned by exact SHA256 and
  checked against v200's output manifest
  ([preflight.json](outputs/preflight.json)).
- **Provider calls.** There were no Alpha Vantage, FRED or EDGAR calls, no
  fetchers and no email.

Registry exception:

- The v200 lock also pins `research/registry.yaml`, which this study must
  extend.
- The runner verifies that file at v200's execution commit and requires all
  114 pinned entries unchanged. Every other consumed file must match its pin
  in the working tree.
- The first attempt to verify the lock without this exception stopped before
  fitting ([attempt 2](outputs/verification/attempt2_blocked_registry_pin.json)).

Runtime:

- This container had Python 3.11. v205 installed Python 3.12.14 and the full
  v200 package set at exact versions in an isolated virtual environment.
- `verify_runtime` passed. Nothing was upgraded.

Hand-checked targets:

- All 772 v200 control labels were recomputed by hand from raw sources and
  match within 8.9e-16. End and availability dates match exactly.
- v200's prevailing means match within 2.2e-16 once v200's one pre-output
  history label (the 1999-11 origin) is included. The first attempt, without
  it, stopped before fitting
  ([attempt 1](outputs/verification/attempt1_blocked_naive_history.json)).
- Returns use one raw share, explicit split ratios and fractional-share
  reinvestment at the last observed raw close on or before each ex-date
  (v200's adapter). Cash and BVPS are restated per origin-date share.

Features ([features.csv](outputs/features.csv), hashed in the
preregistration):

- They are rolling monthly and use only reports whose month *and* actual
  filing date precede the origin.
- Trailing and YoY windows require every calendar month in the window; a
  missing month gives NaN, never a 13-month change (F16).
- Dollar-valued features, targets and offsets are restated to the share
  basis of each fold's first test origin before fitting, and forecasts are
  restated back.

Vintage disclosure:

- EDGAR values are the repaired current table, gated by filing date, as in
  v200.
- Against first-reported values, BVPS, combined ratio, NPW, ROE and book
  yield are unchanged in every compared month.
- `pif_total` differs in 50 months from 2019-07. This is step 3b's
  single-definition PIF rebuild (F11), not look-ahead, but D2 and B2's PIF
  feature is not a historical vintage
  ([source_vintage_audit.json](outputs/source_vintage_audit.json)).
- Historical FRED vintages are irrelevant here: no FRED series is used.

## Validation protocol and support

Fold settings:

| Horizon | Outer split (train / test / gap) | Purge + embargo | Inner folds | Minimum usable history |
|---|---|---|---|---|
| 12M | TimeSeriesSplit 120 / 6 / 24 | 12 + 12 months | 3, each test 6, gap 24 | 60 months |
| 6M | TimeSeriesSplit 60 / 6 / 12 | 6 + 6 months | 3, each test 6, gap 12 | 24 months |

How the folds were built and checked:

- The outer fold also needs the same minimum mature support.
- Dates are the contiguous business month-ends spanning v200's origins for
  each endpoint, so candidates share v200's targets, support and naive
  forecast.
- Training labels must have ended and arrived (by filing date where
  relevant) before the inner or outer test origin. Inner validation labels
  must have arrived by the outer origin.
- Unsupported folds are unscorable and were never shortened
  ([fold_ledger.csv](outputs/fold_ledger.csv)). Scorable outer folds out of
  all outer folds were:

  | D1 | D2 | B1 | B2 | B3 |
  |---:|---:|---:|---:|---:|
  | 13 / 21 | 11 / 21 | 12 / 12 | 11 / 12 | 24 / 34 |

  Early folds lack the EDGAR history; the ledger names the reason for each.
- Development labels are sealed in the
  [partition lock](outputs/partition_lock.json). Every label ends and is
  available before 2023-09-29, and v200's quarantine definition is inherited
  unchanged.
- No quarantine label, feature or metric was read
  ([access ledger](outputs/access_ledger.json)).

Support is still thin:

- Monthly 12M labels overlap. The scored 12M lanes span 66–78 months, only
  5.5–6.5 independent 12-month blocks, just above the preregistered minimum
  of 5.
- Bootstrap blocks run over consecutive *scored* dates.
- For cash, v200's prevailing mean averages labels across the 2002 and 2006
  share bases, as preregistered in v200. That inflates both sides' R² about
  equally, so ΔR² is the comparable figure.

Calibration:

- Calibration is prequential. It uses only matured residuals of the same
  stream, with at least 12, and warmup rows stay unevaluated.
- Low cash-lane coverage (.05–.30) reflects the lumpy special-dividend era
  (2019–2021 specials of $2.35–$6.10). No interval method was tuned after
  seeing it.

## Attempts, review and negative results

All attempts are recorded in [candidate_ledger.json](outputs/candidate_ledger.json)
and [verification/](outputs/verification/).

**Blocked preflights.** Two preflight attempts blocked before any fit (see
Inputs, pins and availability).

**Independent review.** An independent read-only review of the frozen code,
before any fit, found:

- no temporal leakage;
- a correct share-basis round trip;
- one crash path: `evaluate` negated an object warmup column. This was
  already cast in the committed targets.
- D3's closure was hard-coded rather than enforced from its rule.

**Amendment A1** (commit `e2d3906`, before fitting) responded to that
review:

- `evaluate` casts the warmup column explicitly;
- D3's closure is enforced from its rule outcome;
- the rules text states the outer minimum support and B3's differences;
- a synthetic runner test was added.

Candidates, features, grids and thresholds did not change.

**Negative results.**

- **B1 and B2.** Their Ridge fits learned mean-reversion in recent BVPS
  growth from the 2004–2016 windows (2008 drawdown and recovery). They then
  forecast falling growth through the persistent 2017–2020 rise, which gave
  IC −0.64 and −0.70.
- **B3.** It adds no return skill (IC .016, p .91) and loses to the mature
  mean on MAE.
- **D1.** It over-extrapolates the specials.

There were no discarded comparisons, retries or extra variants: six slots,
five fitted, one closed.

**Multiplicity** ([multiplicity.json](outputs/multiplicity.json)). The raw
p-values are D1 1, D2 .268, D3 1, B1 1, B2 1 and B3 1. Holm over the 38-slot
family gives 1 for all; v207 completes the campaign adjustment.

## Verification

- **Tests.** New mathematical code
  (`src/pgr_vds/research_lib/xseries.py`) has 19 hand-calculated
  expected-output fixtures in `tests/research/test_v205_xseries.py`. The
  runner has 2 synthetic fixtures in `tests/research/test_v205_runner.py`,
  including a 4-for-1 split inside every training window.
  - All were observed red before implementation
    ([red logs](outputs/verification/)).
  - Two fixtures were red only by mutation, and the logs say so: the IC
    fixture, added after drafting, and the share-basis fixture, added when
    the need appeared.
  - All 21 are now green.
- **Reproducibility.** A second, independent execution into an external
  directory reproduced all seven core outputs byte for byte
  ([reproducibility.json](outputs/reproducibility.json)).
- **Full suite.** `python -m pytest -o addopts="--tb=short" -q` reports:
  PYTEST_SUMMARY
- **Tracked DB.** SHA256 of `data/pgr_financials.db` was
  `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6` before
  execution, after execution and after the full suite (unchanged).
- **Scope.** There was no live config, live feature, model, recommendation
  policy or monthly-email change.

To reproduce, install this checkout with the exact versions in
`research/studies/v200_clean_baseline/outputs/runtime_lock.json` (Python
3.12.14), then run:

```text
python research/studies/v205_dividend_bvps/run.py --preregister --scratch <external-dir>
python research/studies/v205_dividend_bvps/run.py --execute --scratch <external-dir> --output-dir <external-output-dir>
```

Execute refuses to fit unless the committed preregistration is
byte-identical to the rebuilt one. `finalize.py` writes provenance and the
output manifest from saved outputs only.

## What changed / what is left

**What changed.**

- An installed, tested causal library for split-consistent cash, BVPS and
  adjusted-BVPS targets, filing-gated rolling monthly features, nested
  chronological Ridge, block inference and prequential calibration.
- A preregistered, reproducible v205 study with hand-verified targets, a
  descriptive target audit, ledgers, predictions and a closeout.
- A registry entry.

**What is left.**

- v205 nominates **no finalist** for v207, so the campaign's six-finalist
  cap is unaffected by this lane.
- The annual excess-dividend endpoint cannot be tested honestly until more
  post-policy years mature.
- A D2-style cash-persistence idea would be a new registered candidate in a
  later campaign, not a retry here.
- Promotion of anything would still need a separate governance PR after the
  v207 quarantine rule (D2).
