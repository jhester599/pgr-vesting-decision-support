# v164 Technical Analysis Synthesis Summary

Created: 2026-04-18

## Status

`v160-v164` establishes the technical-analysis research scaffold and
pre-registration artifacts. It does not promote any TA feature into production
or shadow reporting.

## Completed

- Archived the three external TA research reports under
  `docs/archive/history/v160-ta-research-reports/`.
- Added a pure pandas/numpy TA feature factory in
  `src/research/v160_ta_features.py`.
- Added a broad-screen harness in `results/research/v162_ta_broad_screen.py`.
- Added survivor-confirmation helpers in
  `results/research/v163_ta_survivor_confirm.py`.
- Added focused pytest coverage for feature math, screening inventory,
  add/replace mode construction, survivor caps, correlation pruning, and
  deterministic candidate JSON.

## Candidate Outcome

Recommendation: `monitor_only`

Reason: the code and pre-registration framework are in place, but the empirical
v162 broad screen has not been run in this implementation pass. No TA feature
has yet earned shadow integration or replacement status.

## Next Empirical Commands

```bash
python results/research/v162_ta_broad_screen.py
python results/research/v163_ta_survivor_confirm.py
```

After those artifacts exist, review `v162_ta_broad_screen_summary.csv`,
`v163_ta_survivor_confirm_summary.csv`, and `v163_ta_survivor_candidate.json`
before deciding whether to write a separate shadow-candidate plan.
