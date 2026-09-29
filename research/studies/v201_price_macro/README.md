v201 is blocked before fitting. Registering the required new study changes
`research/registry.yaml`, but v200's accepted lock pins its earlier bytes and
its verifier reads the current checkout. No candidate forecast or performance
comparison was computed. This gives no new basis to change the owner's vest
decision; the incumbent remains in place.

The accepted v200 lock is present and clean, and its complete output manifest,
exact dependency versions and D7 Git/DB pins passed the initial read-only
checks on latest master `363ef5e3ec607aa2c5776ab296de18d25ed9ee4a`.
After adding the mandatory v201 registry entry, the actual preparation run
failed with `ValueError: SHA256 mismatch for source text: research/registry.yaml`.
The expected registry digest is
`410cd237f7e0d32639db8bf9b8193a3e8ca6541e52aa2fd01fd6f962f0275806`.
[preflight.json](outputs/preflight.json) and the retained
[preparation log](outputs/verification/prepare_failed.log) record the failure.
The accepted lock was neither changed nor relabelled provisional.

The owner's explicit stop condition was applied: no fitting after a failed
preflight. Six blueprints were declared, but all remain blocked and unscored.
The reduced [runner](run.py) reproduces the failing gate and has no fitting
implementation. An initial unexecuted fitting scaffold was discarded after
this failure. It produced no predictions or metrics. No alternative snapshot,
mutable DB path, provider call, data write, backfill or post-result search was
used to bypass the gate.

The declared six one-factor procedures are listed in
[candidate_ledger.json](outputs/candidate_ledger.json), including removed and
added columns for each model: P1 calendar 3/6/12M momentum; P2 13-week
volatility plus 52-week-close-high distance; P3 VOO-relative 12-month EMA
(distance from a trailing smoothed price ratio) and 6-month RSI (a bounded
ratio of recent gains to total absolute changes), chosen from the v162
inventory before metrics; M1 slope/real-yield change; M2 VIX/NFCI/high-yield
credit; M3 insurance-PPI YoY minus the mean used-car/medical-CPI YoY cost gap.
These declarations are not accepted executed preregistrations. Six planned
slots are reserved, none fitted. All 38 p=1 values in the multiplicity ledger
are conservative pending/unused placeholders, not measured p-values.

Independent mathematical fixtures were observed red before the new installed
`research_lib.price_macro` module existed (7 failed, exit 1), then green
(7 passed, exit 0). They cover split continuity, calendar momentum, 13-week
sample volatility times sqrt(52), 52-week highs, monthly EMA/RSI, one calendar
availability rule, missing periods, future perturbations and a known paired
loss improvement. Runner gate fixtures were also red before implementation
(3 failed, exit 1). A first combined green attempt had one test regex mismatch
(`preregister` versus `preregistration`); correcting that expectation yielded
10 passed, exit 0. All attempts are retained. The mathematical recipes are
tested preparation only; no evidence of forecasting skill is claimed.

The exact control remains R3b code
`c3b4798b90a43f9dbd616981861e9ff01fa2de72` and DB extracted from
`ed7997f6f540f664e59dd44a4079616d74a8e8cb:data/pgr_financials.db`, SHA256
`38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`.
The exact accepted baseline lock SHA256 is
`c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`.
Its byte-identical [copy](outputs/v200_baseline_lock.json) retains parent seed
hashes and R2-lite/R3b migration/rebuild identifiers. v201 records its own
code commit separately in [provenance.json](provenance.json).

The VWO March 2026 dividend gap is accepted provider data. The 12 affected
v200 targets are quarantined; they were not loaded or scored here. No holdout
metric was computed. Earlier v75, v129, v132 and both step13V replays saw that
quarantine; it cannot be described as untouched historical evidence.

Macro observations have latest-vintage values and no historical publication
timestamps. Proposed rules therefore use calendar availability proxies:
one month for selected series except NFCI's existing conservative two-month
rule, applied once. Missing/stale periods remain missing and already lagged
inputs are rejected. No historical vintage or publication precision is
invented. The corrected/archived mom12, April 2020 VIX and rate-gap examples,
fold ledgers, candidate forecasts, calibration and performance results were
not generated because preparation stopped first. Their output ledgers state
this absence explicitly. Synthetic fixture examples are not DB evidence.

The planned evaluation would use v200's identical development targets,
support, folds and mature naive forecast, unchanged 50-penalty Ridge and
10-value past-only shrinkage grids, fixed GBT and tested causal adapters.
Primary 6M/secondary 12M outer windows are 60/120 months, test size 6, gap
12/24, with three inner folds, test size 6, gap 2*h and mature minimum
24/60 months. There is no relaxed-gap or reduced-support fallback.
RÂ² is error reduction versus a past mature mean; Î”RÂ² compares candidate and
control using the same denominator. IC is rank association, equal-weight
across benchmarks or across the panel. Hit rate is direction accuracy versus
a past-learned majority rule; base rate is the observed positive fraction.
Brier/log loss/ECE measure probability error/calibration; nominal-80% coverage
is the fraction inside prequential intervals, with warmup unevaluated.
These metrics are defined for future execution, not measured in this session.
The declared success rule is Î”RÂ² >= .010, Holm38 paired date-block p < .05,
IC loss <= .01 and no directional/calibration deterioration. No finalist was
selected, and no promotion is proposed.

Independent review exposed a missing-final-week aggregation defect: pandas
monthly `last()` could skip a NaN endpoint and reuse an earlier weekly feature.
A new fixture was observed red (1 failed, exit 1), then `last(skipna=False)`
fixed it (11 focused tests passed, exit 0). No real feature frame or forecast
was constructed before or after this correction. Review otherwise found
consistent blocked ledgers and five byte-identical v200 copies.

The first full suite reported `2 failed, 2667 passed, 1 skipped, 2 xfailed,
120 warnings in 350.87s (0:05:50)`, exit 1. Both were documentation-link checks
that ran before the newly referenced v201 artifacts had been written. This
was this session's preparation-order error, not an inherited defect. The
link checker now reports 356 files / 0 broken links; the final full suite was
rerun after completing artifacts and the helper correction. A separate
focused invocation without isolated TEMP/TMP produced 10 Windows temporary-
directory setup errors; the isolated run passes. These attempts are recorded
in verification_attempts.json; none authorized fitting.

The final command `python -m pytest -o addopts="--tb=short" -q` reports
`2670 passed, 1 skipped, 2 xfailed, 99 warnings in 333.95s (0:05:33)`, exit 0.
[Full log](outputs/verification/full_pytest_final.log) and
[exit/hash record](outputs/verification/full_pytest_exit.json) are retained.
No inherited failures remain. The skip, legacy warnings and two inherited
strict v37 expected failures are disclosed. Strict PEP8 E/F checks with
79-column lines, mypy (29 source files with the locked Python3.12), registry
(115 studies / 0 problems), documentation links (356 files / 0 broken) and
no-new-sys.path checks pass. The exact runtime is reused without upgrades.

Tracked DB SHA256 before and after preflight and both full-suite attempts:
`38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`.
The tested v201 source is committed separately as `c1a08828b3984d9d275891c83b55fad407ffe9c5`.
Tests ran on those source bytes before this commit, based on master
`363ef5e3ec607aa2c5776ab296de18d25ed9ee4a` with a dirty working tree; no
fitting execution commit exists. Final provenance pins all tested source
bytes and distinguishes them from the R3b baseline code.
No data, live config, feature, model, policy or monthly email changed.
Unrelated `.codex/` work was preserved.

To reproduce the blocker with the exactly locked runtime and normal package
installation: `python research/studies/v201_price_macro/run.py --scratch
<external-dir>`. It exits nonzero before fitting.

What changed: a documented fail-closed v201 session, registry/README entry,
exact pin copies, declared recipes, tested preparation helpers and gate tests.
What is left: review how downstream sessions verify v200's frozen input
snapshot while separately pinning their own new registry/code, or explicitly
accept a successor lock; then authorize a fresh v201 execution. This session
does not weaken pins or certify that change. v207 remains the only quarantine
opening, and any eventual promotion requires a separate governance PR.
