## Summary
- wire the v7.1 three-scenario tax engine into the live vesting recommendation path
- surface v7.4 runtime governance metrics in the monthly workflow and diagnostic report
- add regression coverage for both integrations

## What changed
- execute `compute_three_scenarios()` inside `generate_recommendation()` and populate `VestingRecommendation.three_scenario`
- compute real obs/feature governance metrics during monthly signal generation
- run a representative CPCV check in the monthly workflow and publish the result in `diagnostic.md`
- add tests covering populated three-scenario output and runtime diagnostic reporting

## Validation
- `python -m pytest tests/test_stcg_boundary.py tests/test_diagnostic_report.py tests/test_monthly_report_tax.py tests/test_v74_guards.py -q`
- `python -m pytest -q`
