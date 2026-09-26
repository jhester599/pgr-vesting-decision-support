# x22 Research Memo

## Scope

x22 challenges the x21 target-scale result with harder annual
baselines so we can tell whether the gain is mostly from scaling or
from the feature-driven size model.

## Results

- Best row: `x10_capital_generation` / `to_current_bvps` / `ridge_scaled` (dollar MAE 3.666, scaled MAE 0.074).

## Interpretation

- If a baseline wins, target scaling matters more than feature depth.
- If an x21 ridge row still wins, the current-BVPS normalization looks
  like a real modeling improvement rather than a cosmetic transform.
