# v127 Path B Calibration Sweep

Date: 2026-04-12
Status: implemented on `codex/v122-classifier-audit-note`

## Purpose

v126 established that Path B has better matched covered balanced accuracy than
Path A, but weaker calibration:

- Path A matched v126: `balanced_accuracy_covered = 0.5000`, `brier = 0.2058`,
  `log_loss = 0.6132`, `ece = 0.1269`
- Path B raw v126: `balanced_accuracy_covered = 0.6450`, `brier = 0.2188`,
  `log_loss = 0.8207`, `ece = 0.2234`

The natural follow-up was to test whether Path B's discrimination could be kept
while repairing its reliability.

## Implementation

Files:
- `results/research/v127_path_b_calibration.py`
- `tests/test_research_v127_path_b_calibration.py`

Inputs:
- `results/research/v125_portfolio_target_fold_detail.csv` from the matched v126
  remediation pass

Outputs:
- `results/research/v127_path_b_calibration_results.csv`
- `results/research/v127_path_b_calibration_detail.csv`
- `results/research/v127_path_b_calibration_summary.md`

Calibration candidates implemented:
1. `path_b_raw_v126`
2. `path_b_platt_v127`
3. `path_b_temp_v127`

Method rules:
- strictly prequential: candidate `t` can only use observations `< t`
- warmup before calibration activation: `24` matched OOS observations
- no K-fold CV
- no full-sample scaling before temporal separation

## Results

### Raw Path B reference

- `balanced_accuracy_covered = 0.6450`
- `brier_score = 0.2188`
- `log_loss = 0.8207`
- `ece = 0.2234`

### Prequential Platt scaling

- `balanced_accuracy_covered = 0.5000`
- `brier_score = 0.2158`
- `log_loss = 0.6638`
- `ece = 0.1387`

Interpretation:
- strongest ECE improvement of the tested candidates
- calibration improves materially
- covered balanced accuracy collapses back to Path A territory

### Prequential temperature scaling

- `balanced_accuracy_covered = 0.5725`
- `brier_score = 0.1917`
- `log_loss = 0.6010`
- `ece = 0.1570`

Interpretation:
- best Brier/log-loss outcome
- better discrimination retention than Platt
- still gives back too much covered balanced accuracy versus raw Path B

## Decision

No v127 calibration candidate is adopted as the next Path B default.

Why:
- both calibration candidates improve reliability
- neither preserves enough covered balanced accuracy to clear the adoption gate
- therefore raw Path B remains the stronger discriminator, but not a
  promotion-ready probability model

## Recommended next work

1. Keep Path A as the primary shadow architecture.
2. Keep raw Path B as a diagnostic research path only.
3. If Path B research continues, the next sensible experiments are:
   - asymmetric decision thresholds rather than probability replacement
   - benchmark-specific feature subsetting for Path A / Path B comparison
   - more flexible but still conservative calibration forms applied to Path B's
     logit scores

## Verification commands

- `python -m pytest tests/test_research_v127_path_b_calibration.py`
- `python results/research/v127_path_b_calibration.py`

## Handoff notes for Claude Code

The most important takeaway is not "calibration failed" in the abstract. It is:

- Path B still appears to have useful ranking/discrimination signal.
- Path B does not yet produce trustworthy decision probabilities.
- Platt scaling is the strongest ECE repair.
- Temperature scaling is the strongest compromise on Brier/log-loss.
- Neither candidate is good enough to replace raw v126 Path B in the branch.

If Claude continues from here, treat `v127_path_b_calibration_detail.csv` as the
working table for any next-step threshold or calibration experiments.
