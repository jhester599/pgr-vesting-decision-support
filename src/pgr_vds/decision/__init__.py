"""The monthly sell/hold decision (was ``scripts/monthly_decision.py``).

Review 2026-09-25, section 5, phase 5 split the 4,080-line script into:

- ``schedule``: as-of date and recommendation-layer mode;
- ``refresh``: the FRED data refresh;
- ``signal_generation``: features, the WFO ensemble, calibration, conformal
  intervals, the live consensus and the simpler-baseline cross-check;
- ``health``: realised-only OOS health, the recommendation-mode gate, the
  model-health snapshot, drift and retrain audit, the policy backtest and the
  manifest warnings;
- ``tax_lots``: vest dates, the tax context, the provisional vest scenario and
  guidance for held lots;
- ``portfolio``: redeploy guidance and the Black-Litterman diagnostic;
- ``rendering``, ``recommendation_report``, ``diagnostic_report``: the
  reports and the calibration plot;
- ``artifacts``: output folders, CSVs, the decision log, the shadow ledgers
  and the run manifest;
- ``pipeline``: ``main``, which runs the steps in order.

The command-line entry point is ``cli/monthly_decision.py``.
"""
