# R5 event/tax closeout — v190 (2026-09-27)

Scope: R5 only, under the [amended plan](PRE_V200_FIX_PROMPTS_codex.md).
Read AGENTS.md, original F19/F22 in
[the review](REPO_REVIEW_2026-09-25.md), their adjacent/PARTIAL limits in
[independent verification](VERIFICATION_2026-09-26.md), and the corrections
in [the second verification](VERIFICATION_2026-09-26_claude.md).
Historical CHANGELOG claims are not verification.

## Before implementation: contract and quantified fixture

Latest master: `73a3d09bdb5d6242b0fc6761e150ce248dbe115e`.
Branch: `codex/R5-event-tax`. Unrelated `.codex/` is preserved.
Tracked DB SHA256 before tests:
`f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d`.
No v200 baseline lock exists on this base. Remediation uses only synthetic
event/lot data, including the fixture below; no holdout debugging or replay.

The event contract is explicit in `VestingEvent.horizon_6m_end` and
`horizon_12m_end`: calendar months from the event, with day clamping. The
monthly model target instead runs BME to BME. Selecting the first target
on/after a mid-month event cannot measure its stated holding period.

| Quantity | Synthetic fixture date/value |
|---|---|
| Vest event | 2020-01-15 |
| Feature/forecast anchor | 2019-12-31 (last feature row <= event) |
| Last observable start bar, both tickers | 2020-01-10; PGR 100, VTI 100 |
| Stated 6M event endpoint | 2020-07-15 |
| Event ending bar | 2020-07-10; PGR 80, VTI 110 |
| Currently selected monthly target origin | 2020-01-31; PGR 70, VTI 100 |
| Selected target endpoint | 2020-07-31; PGR 210, VTI 120 |
| Split | PGR 2:1, 2020-01-21 |
| Dividend | PGR 3/share, 2020-01-24, raw close 60 |
| Evaluation as-of for mature fixture | 2021-01-15 |

Independent arithmetic, before changing code: one PGR share becomes two;
dividend cash is `2 * 3 = 6`; DRIP adds `6 / 60 = 0.1` shares.
Event terminal value is `2.1 * 80 = 168`, so PGR returns 68% and VTI
returns `110 / 100 - 1 = 10%`: relative return **58%**. The monthly
target is `210 / 70 - 1 - (120 / 100 - 1) = 180%` (earlier actions
cancel in that subperiod). Origin and endpoint are both 16 calendar days
late; actual start/end bars are 21 days late. Return difference: **122 pp**.
This confirms a timing defect on an independently specified synthetic
contract; it does not estimate the defect's historical investment impact.

Weekly prices approximate execution. January 17's raw PGR close 120 is
later than the event and cannot be called an available event-date close.
The convention uses the last observable bar <= event, and last bar <=
the stated endpoint. Monthly forecast horizon and realised event holding
period must remain separately labelled; monthly targets must not be rebuilt.

Tax evidence before repair: for a 2023-03-01 acquisition the existing
calendar helper gives first LTCG day 2024-03-02. Old rebalancer code uses
`365 - holding_days`: February 29 displays zero instead of two days,
and March 1 (366 days held) drops the warning instead of displaying one.
Calendar eligibility, not an alert-zone override, must determine these days.

## Final behavior and output comparison

`compute_event_outcome` uses the existing raw-price total-return machinery
and canonical `split_history`, with separate PGR/benchmark outcomes. The
endpoint must be <= evaluation as-of, and both feeds need a finite positive
covering quote dated >= endpoint and <= as-of. Valuation uses the preceding
quote, never that covering quote if it is later than the endpoint. Missing
starting quotes, invalid prices or incomplete horizons yield no completed
backtest result, so they cannot be counted as misses. Both 6M and 12M
endpoints are explicit. No price/target table or training label is rewritten.

Monthly stability retains the exact-origin BME target contract and checks
its own BME endpoint. Dataclass results, `run_full_backtest` and the supported
CSV exporter label the forecast anchor, outcome convention, nominal dates,
actual bars and asset returns. The forecast remains a monthly-horizon
forecast; event directional agreement is descriptive holding-period evidence.

[Synthetic event/lot changes](R5_synthetic_event_lot_diff.csv) are hand-derived.
On a pretax $100 allocation smoke fixture, always-50% changes from $210
(the wrong monthly window) to $139 (the event window); selling 25% changes
from $255 to $153.50. Their difference changes from $45 to $14.50 solely
because of matched outcome timing. These are fictional prices and fixed
mocked forecasts, not historical policy-performance estimates. Forecast
20%, sell fraction 25%, training series and monthly relative target 180%
remain unchanged. Warning corrections change **no utility arithmetic**:
rates, actual lot shares/basis, lot ordering and sell mapping are preserved.
No historical utility artifact is rewritten, and no untouched/quarantined
row is inspected or used as new promotion evidence.

Default STCG alerts continue after 180 days until the existing
`ltcg_eligible_date` (anniversary plus one day). The legacy default
`STCG_ZONE_MAX_DAYS=365` now maps to that calendar boundary. Nondefault
configured caps and explicit `zone_max_days` retain their original
holding-age alert-window units; they never set eligibility or the day count.
Unvested/empty lots stay excluded. No owner tax assumptions or thresholds
were introduced. For February 29, 2024 acquisition the first eligible date
is independently asserted as March 1, 2025; for July 1, 2025 it is July 2,
2026.

## Isolation and commands

Windows PowerShell; CPython 3.12.14, pytest 9.0.2, pandas 3.0.6. R5's external
venv reuses the verifier's installed dependencies via a path-only `.pth` and
has its own offline editable scratch installation (`--no-deps
--no-build-isolation`). No package download or provider call. Python startup
audit hooks block socket connect/DNS in test processes and their children;
provider/e-mail credentials are removed from the test environment.

External root: `C:\Users\Jeff\AppData\Local\Temp\pgr-R5-20260927`.
Clones `scratch`, `counterfactual`, and clean `master-tax` have independent
Git objects (`git clone --no-hardlinks`), each initially at the recorded
master. All mutable DB work is in external synthetic fixtures or temporary
DB copies. The tracked original is never opened with SQLite; hash reads and
copying are the only original-DB operations. Existing full-suite artifact
and copied-decision tests are offline smoke verification of already inspected
history, not promotion evidence or R5 holdout debugging. No fetcher/decision
runs against the original, no real e-mail, and no DB change to commit.

Interpreter below (`python` in the commands):
`C:\Users\Jeff\AppData\Local\Temp\pgr-R5-20260927\venv\Scripts\python.exe`.
`run_verification.py` in that external root copies only the scoped files,
captures the exact argv, working directory, summaries, exit codes and source/
clone DB hashes in `runs.jsonl`, with individual `.log` and `.exit` files.
`PYTHONPATH` includes the chosen external clone and network guard;
`PYTEST_DEBUG_TEMPROOT` is an external R5 directory. `R5_SCRATCH` selects the
counterfactual clone; `R5_REVERSAL` selects the reversals described below.

```powershell
python -m pytest -o addopts="--tb=short" -q tests/unit/backtest/test_backtest_engine.py tests/unit/tax/test_stcg_boundary.py tests/unit/tax/test_tax_hand_computed.py tests/unit/processing/test_total_return.py
python -m pytest -o addopts="--tb=short" -q tests/unit/reporting/test_backtest_report.py tests/unit/backtest/test_monthly_backtest.py
python -m pytest -o addopts="--tb=short" -q
python scripts/checks/check_doc_links.py
```

## Red/green evidence

Original implementation was tested before production edits. Six required
regression names (tax parameterizations give nine cases):

- `test_vest_event_return_uses_event_window_not_next_month_target`
- `test_event_outcome_requires_complete_horizon`
- `test_event_return_uses_only_observable_start_price`
- `test_event_window_split_dividend_matches_hand_calculation`
- `test_stcg_warning_uses_calendar_eligibility_across_leap_year`
- `test_stcg_warning_stops_on_first_ltcg_day`

The original-red command was the base pytest command above with the backtest
and STCG test modules and `-k` joining those six exact names with `or`.
Its valid defect run (`confirmed-red`) reported **`9 failed, 70 deselected
in 1.58s`**, pytest exit **1**. The first four fail on 1.8 versus independently
calculated .58 (or a falsely available outcome); tax cases fail on 0 versus
2/1 days or a prematurely absent warning. Initial repair: **`9 passed,
70 deselected in 1.42s`**, exit **0**. The final maturity test is stricter:
there must be no completed result before July 15, rather than a NaN result
that could still enter downstream hit-rate counts.

Final scratch counterfactuals retain revised tests and restore the defect;
each run is followed by restored-code green verification:

| Counterfactual / command suffix after base pytest command | Exact summary | Exit |
|---|---|---|
| Restore original event target-selection block from `73a3d09`, preserving new public helper/metadata APIs; backtest module `-k` the four event names above plus `test_event_twelve_month_outcome_uses_stated_end or test_event_outcome_leaves_monthly_targets_unchanged` | `6 failed, 41 deselected in 1.63s` | 1 |
| Restore original rebalancer from `73a3d09`; `tests/unit/tax/test_stcg_boundary.py` (includes revised 1/66/16/166-day assertions) | `10 failed, 34 passed in 1.34s` | 1 |
| Restore original exporter from `73a3d09`; `tests/unit/backtest/test_backtest_engine.py::test_event_csv_labels_model_anchor_and_holding_period` | `1 failed in 1.38s` | 1 |
| Revert valid-price coverage guard to `dropna()`; `tests/unit/backtest/test_backtest_engine.py::test_event_outcome_requires_valid_covering_price` | `2 failed in 1.40s` | 1 |
| Restored final code, requested focused command | `128 passed in 2.72s` | 0 |
| Monthly/reporting command above | `39 passed in 2.09s` | 0 |

The exporter-label regression first failed with `KeyError: forecast_anchor`
before its edit (`1 failed in 1.47s`, exit 1). Review edge regressions for
zero/infinite covering prices and preserving nondefault configured alert
caps were also observed red (`3 failed in 1.47s`, exit 1), then green
(`3 passed in 1.24s`, exit 0). No mathematical expectation is computed by
the implementation under test. Missing-feed/start regressions, explicit
12M arithmetic (89% PGR minus 10% VTI = 79%), preservation of the monthly
table, matched policy allocations, CSV labels and lot filtering supplement
the required tests.

Setup failures are not defect evidence: the first run had an inaccessible
default pytest temp directory (`70 deselected, 9 errors in 1.51s`, exit 1).
Two synthetic-table setup attempts used incorrect dividend table names
(`5 failed, 70 deselected, 4 errors in 1.55s` and `... in 5.92s`, exits 1).
An unnamed stub target series initially caused skipped predictions; naming
it fixed the fixture before accepting the red evidence. The first valid red
log hit a console-encoding error; the pytest exit and summary were captured,
then UTF-8 output was configured. No repair is claimed from setup errors.

## Full suite and remaining work

Final requested full-suite command:

```text
1 failed, 2583 passed, 1 skipped, 115 warnings in 304.00s (0:05:03)
exit code: 1
```

This is not a passing full-suite gate. The interim full suite reported
`1 failed, 2577 passed, 1 skipped, 117 warnings in
326.77s (0:05:26)`, exit 1. Its failure is unchanged master
`tests/unit/tax/test_property_tax_boundaries.py::test_optimize_sale_orders_losses_then_ltcg_then_stcg`.
Clean external master at `73a3d09` reproduces the same node:
`1 failed in 0.51s`, exit 1. Both `capital_gains.py` and the property test
have zero diff from that base. The floating-point example (price 20, basis
`20.000000000000004`, shares `409.8506658027625`) is already documented in
[R1](R1_safety_closeout.md) and [R4](R4_shadow_closeout.md). No counterexample
was deleted, seed selected, test skipped or unrelated tax repair added.

The first full-suite attempt had a verification-environment namespace
collision (`26 errors in 6.83s`, exit 2): adding `scratch/src` to PYTHONPATH
shadowed the root `research` namespace. Removing that extra path and using
the scratch editable install resolved collection; no repository import
workaround was added.

A link-check attempt in the copied scratch tree initially omitted the newly
added synthetic CSV from the driver's scoped copy list (`351 files, 1 broken
links`, exit 1). Adding the artifact to that list produced the fresh final
`[doc-links] 351 files, 0 broken links`, exit 0. The repository link itself
was valid; no link assertion or repository checker was weakened.

New/changed Python lines pass PEP 8 (`ruff --isolated --select E,W
--line-length 79` filtered to changed lines): **0 new violations**, exit 0;
64 inherited findings are outside R5. Repository-configured ruff passed.
Documentation links: `[doc-links] 351 files, 0 broken links`, exit 0.
`git diff --check`: exit 0. A read-only reviewer identified unavailable-result
counting, missing CSV labels, invalid covering quotes and configured alert
caps; each was corrected and rechecked with no remaining R5 source blocker.

Tracked DB SHA256 **after** focused/full/counterfactual tests:
`f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d` —
identical to before. External scratch/counterfactual/master DB copies also
retain that hash. Full-suite original/clone before/after hashes are captured
in the driver records. No DB or historical verification report changed.

R5 confirms and repairs the demonstrated event offset and calendar warning
defects. Weekly execution/DRIP-price approximation remains. Before v206,
always-50% and candidate policies must share `compute_event_outcome` and a
common available-event mask. No historical policy utility was recalculated;
old event utility figures cannot silently become correctly aligned evidence.
The inherited full-suite tax blocker needs a separate authorized repair;
this PR remains draft until that gate is resolved. No production model
promotion or investment-performance improvement is established.
