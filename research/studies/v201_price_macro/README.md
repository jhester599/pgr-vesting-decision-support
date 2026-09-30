v201 attempt 2 completed the six frozen price/technical and macro comparisons on the exact accepted v200 inputs. None of the six candidates clears every preregistered gate. No finalist advances to v207 and the incumbent remains in place. Nothing changes in the owner's vest decision or monthly email.

Attempt 1's blocked session is preserved byte for byte in [the archive](outputs/attempt1/README.md), with its [archive manifest](outputs/attempt1/archive_manifest.json). The original root outputs remain historical blocked evidence; all continuation results are in [attempt 2](outputs/attempt2/).

## What was tested

Six one-factor blueprints, campaign slots 1–6, with no combinations or extra search:

- P1: calendar 3/6/12M momentum replaces the momentum block.
- P2: adjusted 13-week volatility plus 52-week-close-high distance replaces volatility.
- P3: frozen VOO-relative EMA12 distance and simple RSI6 replace mom6/mom12.
- M1: observed-period slope and six-month real-yield change replace the rate block.
- M2: VIX/NFCI/high-yield credit at one calendar publication rule replace the stress block.
- M3: insurance-PPI YoY minus the mean used-car/medical-CPI YoY cost gap replaces rate adequacy.

[Preregistration](outputs/attempt2/registered.json) freezes exact removed/added columns for both models, grids, lags, thresholds and tie order before fitting. [Matched recipe examples](outputs/attempt2/feature_examples.json) explicitly show that v200 had already repaired mom12, April 2020 VIX and the gap. [Documentary archive examples](outputs/attempt2/archived_finding_examples.json) distinguish the original F01/F06/F07 defects. Those archived reports are context, not a defective control or new quarantine calculation.

The [matched recipe audit](outputs/attempt2/matched_recipe_audit.json) confirms identical repaired momentum and stress levels. P1 adds mom3/mom6 to Ridge's incumbent mom12 block, so its gain is a feature-block addition. M2 adds no new information; its small forecast difference may reflect the frozen replacement-column ordering and GBT tie handling. It supplies no evidence of a macro repair gain. The rate gap has one missing-status difference because both cost components are required. No recipe was changed or refitted after this audit.

## Primary 6M development results

Matched v200: R² 0.011, equal-weight IC -0.042, hit/base 0.651/0.588, ECE 0.172, 80% coverage 0.761.

| Candidate | R² | ΔR² vs v200 | raw / Holm p | equal-weight IC | hit / past base hit | ECE / 80% cover | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---|
| P1 | 0.068 | 0.058 | 0.0870 / 1.0000 | 0.012 | 0.661 / 0.588 | 0.175 / 0.768 | holm_primary, log_loss, ece |
| P2 | 0.010 | -0.000 | 1.0000 / 1.0000 | -0.028 | 0.660 / 0.588 | 0.162 / 0.771 | delta_r2, holm_primary, log_loss |
| P3 | 0.065 | 0.054 | 0.0645 / 1.0000 | -0.024 | 0.663 / 0.588 | 0.161 / 0.771 | holm_primary, log_loss |
| M1 | 0.045 | 0.034 | 0.2674 / 1.0000 | -0.040 | 0.666 / 0.588 | 0.174 / 0.762 | holm_primary, log_loss, ece |
| M2 | 0.009 | -0.001 | 1.0000 / 1.0000 | -0.037 | 0.651 / 0.588 | 0.165 / 0.764 | delta_r2, holm_primary, log_loss |
| M3 | -0.058 | -0.068 | 1.0000 / 1.0000 | -0.108 | 0.614 / 0.588 | 0.172 / 0.759 | delta_r2, holm_primary, ic, hit_rate, directional_skill, coverage_80 |

R² means 1 − squared forecast error / squared error of the realised past mean. The current answer never enters that mean. ΔR² uses exactly matched rows and the same denominator. IC is rank correlation; equal-weight IC averages benchmarks, while panel IC describes the whole panel. Hit rate is direction accuracy; its control always predicts the majority direction learned from matured labels. Base rate is the observed positive fraction, reported in [metrics](outputs/attempt2/metrics.json).

The primary threshold is ΔR² ≥ .010 and paired date-block Holm p < .05, IC loss ≤ .01, and no direction/calibration deterioration. Before fitting, the conservative safeguard interpretation was frozen: hit rate and directional skill cannot fall; Brier/log loss/ECE cannot rise; coverage cannot move farther from .80. Missing safeguards fail. Only numerical roundoff uses epsilon 1e-12. No active-policy proposal is made.

## Secondary 12M diagnostics

Matched v200 R² -0.191, equal-weight IC -0.072. These diagnostics never rescue a failed primary candidate.

| Candidate | R² | ΔR² | paired diagnostic p | equal-weight IC |
|---|---:|---:|---:|---:|
| P1 | -0.228 | -0.036 | 1.0000 | -0.022 |
| P2 | -0.217 | -0.026 | 1.0000 | -0.075 |
| P3 | -0.190 | 0.001 | 0.7816 | 0.036 |
| M1 | -0.188 | 0.004 | 0.4228 | 0.043 |
| M2 | -0.197 | -0.005 | 1.0000 | -0.011 |
| M3 | -0.131 | 0.060 | 0.1774 | 0.107 |

## Uncertainty and calibration

All benchmarks travel together in monthly-date bootstrap blocks of length h (6 or 12), seed 20260926, 2,000 replicates. Scored dates are a complete monthly grid. Monthly annual labels overlap; 66 scored 12M months represent only 5.5 annual blocks, not 66 independent annual events. The 144 6M months represent 24 six-month blocks. There is no row-pooled significance test.

| Candidate / horizon | ΔR² 95% interval | equal-weight IC 95% interval | panel IC / clustered p | direction skill / clustered p |
|---|---|---|---:|---:|
| P1 / 6M | -0.005, 0.149 | -0.176, 0.195 | 0.029 / 0.7636 | 0.095 / 0.0185 |
| P1 / 12M | -0.073, -0.002 | -0.204, 0.266 | -0.025 / 0.8431 | -0.159 / 1.0000 |
| P2 / 6M | -0.028, 0.023 | -0.210, 0.142 | -0.012 / 0.9040 | 0.090 / 0.0205 |
| P2 / 12M | -0.051, 0.009 | -0.188, 0.195 | -0.042 / 0.7461 | -0.141 / 1.0000 |
| P3 / 6M | -0.005, 0.135 | -0.202, 0.142 | -0.010 / 0.9310 | 0.094 / 0.0125 |
| P3 / 12M | -0.030, 0.117 | -0.167, 0.256 | -0.089 / 0.4583 | -0.129 / 1.0000 |
| M1 / 6M | -0.015, 0.136 | -0.203, 0.148 | -0.014 / 0.8821 | 0.098 / 0.0175 |
| M1 / 12M | -0.019, 0.046 | -0.168, 0.192 | -0.074 / 0.5892 | -0.144 / 1.0000 |
| M2 / 6M | -0.006, 0.003 | -0.221, 0.133 | -0.023 / 0.7971 | 0.080 / 0.0280 |
| M2 / 12M | -0.020, 0.007 | -0.198, 0.224 | -0.006 / 0.9710 | -0.149 / 1.0000 |
| M3 / 6M | -0.178, 0.021 | -0.276, 0.075 | -0.076 / 0.4353 | 0.044 / 0.1374 |
| M3 / 12M | -0.136, 0.333 | -0.241, 0.256 | 0.167 / 0.1804 | -0.112 / 1.0000 |

Directional-skill and IC p-values are secondary diagnostics, not alternate success paths. [Multiplicity](outputs/attempt2/multiplicity.json) adjusts the six registered primary tests together with the six already reported v205 primaries across the 38-slot family; pending slots have p=1. Endpoints are not pooled. v207 completes the campaign adjustment.

| Candidate / horizon | rows / unique months | probability / interval scored rows | Brier / log loss | ECE / 80% coverage |
|---|---:|---:|---:|---:|
| P1 / 6M | 942 / 144 | 730 / 870 | 0.261 / 0.822 | 0.175 / 0.768 |
| P1 / 12M | 372 / 66 | 129 / 255 | 0.276 / 1.401 | 0.297 / 0.741 |
| P2 / 6M | 942 / 144 | 730 / 870 | 0.263 / 0.827 | 0.162 / 0.771 |
| P2 / 12M | 372 / 66 | 129 / 255 | 0.279 / 1.547 | 0.269 / 0.749 |
| P3 / 6M | 942 / 144 | 730 / 870 | 0.258 / 0.838 | 0.161 / 0.771 |
| P3 / 12M | 372 / 66 | 129 / 255 | 0.239 / 0.874 | 0.218 / 0.757 |
| M1 / 6M | 942 / 144 | 730 / 870 | 0.266 / 0.830 | 0.174 / 0.762 |
| M1 / 12M | 372 / 66 | 129 / 255 | 0.282 / 1.116 | 0.276 / 0.729 |
| M2 / 6M | 942 / 144 | 730 / 870 | 0.265 / 0.826 | 0.165 / 0.764 |
| M2 / 12M | 372 / 66 | 129 / 255 | 0.272 / 1.585 | 0.312 / 0.749 |
| M3 / 6M | 942 / 144 | 730 / 870 | 0.264 / 0.801 | 0.172 / 0.759 |
| M3 / 12M | 372 / 66 | 129 / 255 | 0.371 / 1.930 | 0.318 / 0.671 |

Brier and log loss measure probability error; ECE measures probability calibration. Interval coverage is the fraction inside nominal 80% intervals. Warmup stays missing and unevaluated. Calibrators, intervals, weights and shrinkage use only matured past forecasts/outcomes, using the identical accepted v200 procedures. These are forecast comparisons without a trading policy or cost-adjusted P&L; transaction costs and live-policy thresholds are not evaluated.

## Chronology, availability and pins

Outer TimeSeriesSplit is sliding 60/120 training months, six test months, gap 12/24 (h purge plus h embargo). Three inner TimeSeriesSplit folds each have six test months and gap 2h, with the same maximum training window and minimum usable 24/60 months. All label ends/arrival dates are checked against every test origin. Unsupported v200 folds stay unscorable with the same gaps. [Fold ledger](outputs/attempt2/fold_ledger.csv) records every candidate, learned penalty and training support. Fold-local medians/scales are learned only on training data. Targets, endpoints, folds, support and realised naive forecasts match v200 exactly.

The unchanged 50 Ridge penalties, ten past-only shrinkages, GBT depth2/50 trees/.1 learning rate/.8 subsample/seed42 and incumbent mature-past ensemble/calibration are shared with the control. Only the declared feature block changes. No globally future-selected upstream recipe is used.

The accepted [baseline lock](../v200_clean_baseline/outputs/baseline_lock.json), SHA256 `c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0`, binds repaired code `c3b4798b90a43f9dbd616981861e9ff01fa2de72` and DB Git `ed7997f6f540f664e59dd44a4079616d74a8e8cb`, SHA256 `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`, plus seed parents and R2-lite/R3b repair IDs. The DB is exported to external scratch and opened immutable; no provider backfill occurs. The accepted VWO March 2026 provider gap remains; its 12 affected targets are quarantined.

The registry-only verification amendment reads exact bytes at v200 execution commit `b0a590bd2e9df975c420166d96cc3d919eb4934e`, verifies its original hash and unchanged entries, then checks every other input in the current checkout. The accepted lock and pinned v200 code are not rewritten. The study separately pins its own registry snapshot, source and all consumed earlier files by SHA256.

Raw macro storage has no publication timestamps or historical vintages. [Availability ledger](outputs/attempt2/availability_ledger.csv) applies one exact calendar-period rule (one month except the frozen two-month NFCI rule). Missing or stale periods stay missing; no already lagged cache is accepted and no forward fill is used. This is a conservative calendar proxy, not a certified actual-release/vintage backtest. Actual release certification remains a limitation for future promotion. EDGAR features are inherited from v200's filing-gated features, with its stated vintage limits. Raw weekly prices, manual splits and fractional-share dividend reinvestment define the unchanged targets; calendar features use past split-neutral prices. Weekly month-end valuation remains an approximation.

Maximum computed feature origin is 2023-08-31; development outcome ends/availability are before 2023-09-29. [Access ledger](outputs/attempt2/access_ledger.json) records zero quarantine labels/metrics. Quarantine definitions/access-ledger hashes are inherited. v207 alone opens the retrospective quarantine once after freezing finalists; any eventual promotion needs its separate governance PR and D2 forward evidence.

## Attempts and verification

Attempt 1 stopped before all fitting. Attempt 2's [attempt ledger](outputs/attempt2/attempts.json) records pre-fit checks/amendments, red/green fixtures, review and execution. The six recipes were declared originally and committed for execution before fitting at `acc3ce98357af4d4a997daac5426537d19cfd3a0`. No new feature, lag, threshold recipe, combination, discarded fitted comparison or metric-driven retry was tried. Negative results and every failed safeguard are retained.

The mathematical/runner fixtures have observed red/green evidence in [verification](outputs/attempt2/verification/). New fixtures cover historical pins and tampering, splits on both assets, missing months and future perturbations, exact targets/control support, label ends, safeguard pass/fail and finalist tie order. Independent pre-fit review found no blocker; its example-label clarification was fixed before execution. Scoped mypy, strict PEP8, registry, docs and no-new-sys.path checks are recorded.

Required full command: `python -m pytest -o addopts="--tb=short" -q`:
`2704 passed, 1 skipped, 2 xfailed, 133 warnings in 331.54s (0:05:31)`, exit 0.
[Full log](outputs/attempt2/verification/full_pytest.log). Inherited pytest failures: none. The skip, legacy warnings and two strict v37 xfails are disclosed. Broader initial mypy/Windows path probes and introduced verification errors are recorded separately rather than relabelled as inherited pytest failures.

Tracked DB SHA256 before preparation, after execution and after full pytest is unchanged: `38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6`. Own model code commit is recorded separately from repaired baseline code; dirty status and exact source/runtime/dependency/input/output hashes are in [provenance](provenance.json). Unrelated `.codex/` work remains preserved. No fetchers, provider calls, email, DB migrations/data writes, live configuration, recommendations or model changes occurred.

To reproduce with the exact v200 runtime and installed package, check out the frozen code and verify registered input hashes, then run `python research/studies/v201_price_macro/run.py --execute --scratch <external-dir> --output-dir <external-output-dir>`. Execution refuses uncommitted/changed preregistration or mismatched inputs. Preparation is separate: `--prepare --scratch <external-dir>`; do not overwrite historical committed inputs casually.

Cross-platform reproduction (independent review, 2026-09-30). Preparation ran on Windows, so two exact-equality gates stop `--execute` before fitting on a default Linux checkout. First, 27 plain-text inputs in the preregistration carry CRLF digests; a `core.autocrlf=true` checkout reproduces them, and `run.py`, `research_lib` and data artifacts keep their exact bytes. Second, `np.logspace(-4, 4, 50)` differs from the registered Ridge grid in 3 of 50 values by 1 ULP, so the procedure check fails; [linux_reproduction_harness.py](outputs/attempt2/verification/linux_reproduction_harness.py) sets the grid in place to the registered values and calls the unmodified `execute()`. `run.py` is itself a preregistered input and was not changed. With Python 3.12.14 and the exact v200 packages on Linux, the fold ledger reproduced byte for byte, predictions within 3.6e-14 and metrics within 2.2e-15, with identical comparisons, p-values, dispositions and no finalist ([record](outputs/attempt2/verification/linux_reproduction.json)). The committed outputs are the Windows execution and were not replaced. `tests/research/test_v201_continuation_runner.py` checks that every registered input still has its registered content, up to line endings for text.

What changed: v201's blocker was repaired without loosening baseline pins, attempt 1 was preserved, and all six development candidates were tested and closed out with uncertainty and provenance. What is left: no v201 finalist; v207 campaign synthesis and genuinely unused forward evidence remain. Historical macro publication/vintage certification remains unresolved.
