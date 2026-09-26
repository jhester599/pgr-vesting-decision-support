# Monthly charts

Rewritten every month by the "Regenerate monthly charts" step of
`monthly_decision.yml`, which runs `scripts/repurchase_timeseries_charts.py`
and `scripts/capital_return_charts.py` and commits exactly the files named in
its `MONTHLY_CHARTS` list (the `CHART_FILES` of the two scripts). The monthly
email attaches four of them.

Moved here from `results/research/pgr_*.png` in v180.
