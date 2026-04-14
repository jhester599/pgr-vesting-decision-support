# Backlog Scoring Rubric

Each persona scores every candidate from 1 to 10 on the following axes:

- Signal quality ROI: expected metric improvement per effort unit.
- Promotion gate proximity: whether the item directly advances a shadow or production promotion criterion.
- Data risk: lower scores for candidates requiring fragile external data or timing assumptions.
- Model complexity cost: lower scores for candidates that add persistent branching or hard-to-maintain hyperparameters.
- Reversibility: higher scores when the change can be tested and rolled back cleanly.

Consensus scoring notes:
- Start from the average of the four persona totals scaled back to 10.
- Apply a modest penalty when persona disagreement exceeds 2 points.
- Favor already-unblocked items over items gated by elapsed shadow time.
