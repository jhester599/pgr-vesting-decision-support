# bl01 — tau sweep eval

<!-- Registry entry: research/registry.yaml (id: bl01). -->

**Question.** BL-01: Monte Carlo parameter sweep for Black-Litterman tau / risk_aversion tuning.

**Date.** 2026-04-18 (first commit).

**Status.** `retained`, feeds `config.model.BL_TAU / BL_RISK_AVERSION`.

**Notes.** keep_incumbent (tau 0.05, risk aversion 2.5).

**Closeout / record.** [BL01_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/bl01_tau_sweep_eval/bl01_tau_sweep_eval.py
```

## Outputs

`outputs/` (1 files, committed): `bl01_tau_candidate.json`

## Tests

- [`tests/research/test_bl01_sweep.py`](../../../tests/research/test_bl01_sweep.py)
