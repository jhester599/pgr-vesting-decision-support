# x8 Synthesis Memo

## Scope

x8 synthesizes checked-in x2-x7 research artifacts. It does not
train models, refresh data, or alter production/monthly/shadow paths.

## Shadow Readiness

- Status: `not_ready`.
- Rationale: classification evidence is mixed and horizon-specific; direct-return benchmarks remain baseline-heavy; decomposition still depends on no-change P/B; special-dividend sample is very small.

## Cross-Lane Findings

- Absolute classification remains research-only. x2 did not clear
  the base-rate gate, while x7 targeted TA replacements improved
  selected 3m/6m classification evidence.
- Direct forward-return modeling remains a benchmark lane. The
  checked-in x3 summary shows only limited no-change gate clearance.
- BVPS forecasting is the strongest structural leg. The x5
  recombination still relies on no-change P/B as the stable anchor.
- Special-dividend forecasting should remain an annual sidecar
  because the November snapshot sample is very small.

## Gate Counts

- x2 base-rate gate true rows: 0 (0 horizons).
- x3 no-change gate true rows: 3 (1 horizon).
- x4 BVPS no-change gate true rows: 14 (4 horizons).
- x7 best cleared horizons: 2.

## Leading Sidecars

- TA leader: `ta_minimal_plus_vwo_pct_b` cleared 2/4 horizons.
- Special-dividend leader: `historical_rate__ridge_positive_excess` with 18 annual observations and expected-value MAE 1.497.

## Recommendation

Do not wire x-series into shadow artifacts yet. The next research
step should be a narrow x9 robustness pass: confirm x7 3m/6m TA
evidence, pressure-test BVPS/PB recombination, and keep the
special-dividend model framed as low-confidence annual research.
