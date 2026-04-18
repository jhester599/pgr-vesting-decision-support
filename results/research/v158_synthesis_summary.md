# v158 Synthesis: v153-v157 Classification and Feature Research

Date: 2026-04-17
Branch: codex/v153-v158-classification-feature-research

## Experiment Outcomes

| Experiment | Benchmarks | Winner? | Delta BA_cov | Recommendation |
|---|---|---|---|---|
| CLS-02 Firth logistic | All 8 | VMBS, BND | VMBS +0.0412, BND +0.0704 | adopt_firth_for_thin_benchmarks |
| FEAT-02 WTI momentum | DBC, VDE | none | DBC +0.0051, VDE +0.0206 | no_benefit |
| FEAT-01 USD momentum | BND, VXUS, VWO | none | BND -0.0767, VXUS 0.0000, VWO +0.0087 | no_benefit |
| FEAT-03 Term prem diff | All 8 | none | best: VDE +0.0169 | no_benefit |

## Key Findings

**Firth logistic (CLS-02):** The Firth-penalized logistic clearly outperforms the
standard balanced-weight logistic for VMBS (+0.04 BA_cov) and BND (+0.07 BA_cov).
All 8 benchmarks were classified as thin (avg_train_positives < 30), but Firth only
helped the bond benchmarks materially. VXUS, VWO, and GLD showed negative deltas,
suggesting that Firth's regularization interacts differently with higher-frequency
sell events in equity/EM benchmarks. Recommendation: adopt Firth for VMBS and BND
only; retain standard logistic for other benchmarks.

**WTI momentum (FEAT-02):** Neither DBC (+0.005) nor VDE (+0.021) cleared the 0.04
threshold. The feature is present in the matrix but adds no meaningful lift for the
covered balanced accuracy at the lean-baseline level. Recommendation: no adoption.
A follow-up could combine WTI with Firth for DBC/VDE, but this was not tested here.

**USD momentum (FEAT-01):** Both features were materialized in the DB. BND was hurt
(-0.077) while VWO showed marginal improvement (+0.009). USD momentum does not
improve classification for the tested benchmarks at the lean-baseline level.
Recommendation: no adoption. The peer review hypothesis that currency moves aid
BND/VXUS classification was not confirmed.

**Term premium diff (FEAT-03):** Best improvement was VDE at +0.017, just below the
0.02 threshold. BND regressed sharply (-0.088). Term premium differencing as a
standalone addition to the lean baseline shows no consistent benefit.
Recommendation: no adoption in this form. A threshold-based gate (sell when
term_premium_diff_3m > 0.25%) was not tested and remains an open follow-up.

## Promotion Decision

Shadow adoption criteria: BA_covered delta >= 0.02 for at least one target benchmark,
with no regression on untargeted benchmarks.

Experiments meeting criteria: **CLS-02 Firth logistic** (VMBS +0.04, BND +0.07)

Shadow adoption plan for v159 (next cycle):
- Adopt Firth-penalized logistic for VMBS and BND in the Path A classifier
- Retain standard balanced-weight logistic for all other benchmarks
- Surface both logistic variants side-by-side in monthly shadow artifacts
- Follow the `autoresearch_followon_v150` reporting-lane pattern

All other experiments (WTI, USD, term premium diff) are recorded as no-benefit.

## Next Queue

1. v159: Wire Firth logistic winners (VMBS, BND) into the shadow classification lane
2. BL-01: Black-Litterman tau/view tuning (highest remaining decision-layer task,
   deferred since v153)
3. FEAT-02 follow-up: test WTI + Firth combined for DBC/VDE (optional; low priority)
4. CLS-03: Path A vs Path B production decision (still time-locked on 24 matured months)
