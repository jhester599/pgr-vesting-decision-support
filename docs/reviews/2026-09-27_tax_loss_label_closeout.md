# Tax-loss label fix: closeout (v192)

Owner decision D6 in [the pre-v200 prompts](PRE_V200_FIX_PROMPTS_codex.md#owner-decisions-and-execution-plan-2026-09-27).
Base: `master` at `697947c9f10ecdb049769bd8219edee1b294e02a` (after R5, PR #142).
Run on Linux, Python 3.11.15, in a venv with `pip install -e ".[dev]"`.

## The failure

[R1](R1_safety_closeout.md), [R4](R4_shadow_closeout.md) and
[R5](R5_event_tax_closeout.md) each recorded the same full-suite failure on
unchanged master:
`tests/unit/tax/test_property_tax_boundaries.py::test_optimize_sale_orders_losses_then_ltcg_then_stcg`.
R1 escalated it to the owner. It blocked the pre-v200 gate "full suite
passing".

`optimize_sale` (`src/tax/capital_gains.py`) sorts lots into loss, LTCG and
STCG groups by gain **per share**, but labelled each sold lot by the sign of
its **total-dollar** gain, `shares * price - shares * basis`. At price 20,
basis `20.000000000000004` and `409.8506658027625` shares, the per-share gain
is `-3.55e-15`. Both totals round to the same float, so the dollar gain is
exactly `0.0`. The lot was sorted as a loss but labelled LTCG, which the
property rejects (`assert 20.000000000000004 <= 20.0`).

The failing example lived only in the Windows machine's local Hypothesis
database. On Linux the unchanged test passed (`11 passed in 3.25s` for the
file), because random generation does not find this example by itself.

## The fix

The label now uses the same per-share test as the sort:
`elif _gain_per_share(lot) < 0:` in place of `elif gain < 0:`. For the
failing lot the label changes from LTCG to LOSS. Its tax is unchanged, since
`0.0 x rate = 0.0`. A lot's label changes only when its per-share gain is
negative and its dollar gain rounds to exactly zero. When the per-share gain
is zero or more, the dollar gain is never negative, because multiplying by a
positive number preserves the order of floats. Tax rates, lot data, the DB,
model inputs and the monthly e-mail are unchanged.

## Tests (red, then green)

- `test_property_tax_boundaries.py`: the counterexample is pinned with
  Hypothesis `@example`, so it runs on every OS and every run, not only
  where the local example database holds it. The property itself is
  unchanged.
- `test_capital_gains.py::TestLotSelectionPriority::test_rounded_away_loss_is_still_labelled_loss`:
  a hand-computed case. It asserts the rounding premise, labels
  `["LOSS", "LTCG"]`, zero gain and tax on the loss lot, and total tax
  `10 x (20 - 10) x 0.20 = 20.0`.

| Code | Command | Summary | Exit |
|---|---|---|---|
| Original, pinned example only | the property node | `1 failed in 0.61s` | 1 |
| Original | both nodes above | `2 failed in 0.47s` | 1 |
| Fixed | both nodes above | `2 passed in 1.17s` | 0 |
| Fixed | `tests/unit/tax tests/unit/portfolio` | `217 passed in 5.86s` | 0 |
| Fixed, docs edited mid-run | full suite, `python -m pytest -q` | `1 failed, 2584 passed, 1 skipped, 146 warnings in 922.25s (0:15:22)` | 1 |
| Fixed, files stable | `tests/integration/pipeline/test_dry_run_read_only.py` | `5 passed, 1 warning in 146.57s (0:02:26)` | 0 |
| Fixed, committed tree | full suite, `python -m pytest -q` | FULLSUITE2 | FULLEXIT2 |

The first full run's one failure was
`test_monthly_decision_dry_run_leaves_db_and_tracked_files_unchanged`. It
hashes every tracked file before and after a dry run, and it flagged
`PRE_V200_FIX_PROMPTS_codex.md`, which this session was editing while the
suite ran. Rerun alone with the files stable, the module passed. The tax
node passed in that run.

Other checks: `ruff check .` gave `All checks passed!`. The changed lines
have 0 findings under `ruff --isolated --select E,W --line-length 79`.
`git diff --check` exited 0. `python scripts/checks/check_doc_links.py`
gave `[doc-links] 352 files, 0 broken links`, exit 0.

`data/pgr_financials.db` SHA256 was
`f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d` before
and after the full suite.

## Limits

The full suite ran on Linux only. The Windows exit-zero gate needs one run on
the owner's machine, where the retained example first failed. The pinned
`@example` makes that run deterministic. This fix changes a tax label in a
floating-point edge case. It does not establish any investment-performance
improvement.
