# Project History

Everything here is a record of finished work: plans, closeouts, result
summaries, external reviews and retired code. None of it describes how the
system works today. For that, start at [`docs/README.md`](../README.md).

Until review 2026-09-25 (section 5, phase 4) these records were spread over
five folders (`docs/plans/`, `docs/superpowers/`, `docs/closeouts/`,
`docs/results/`, `docs/archive/`) plus a root `archive/` for retired code.
They were moved here with `git mv`, sub-folders unchanged, so
`git log --follow` still shows each file's full history. Links inside the
records were rewritten to the new paths, and `scripts/checks/check_doc_links.py`
checks every Markdown file under this folder.

## Contents

| Folder | Files | Era | What it holds |
|---|---:|---|---|
| [`plans/`](plans/) | 28 | v8–v34 | Codex execution plans, one per version (`codex-v9-plan.md` … `codex-v34-plan.md`). [README](plans/README.md) |
| [`superpowers/plans/`](superpowers/plans/) | 47 | v37–x24 (2026-04-10 → 2026-04-24) | Dated implementation plans, execution logs and results summaries |
| [`superpowers/specs/`](superpowers/specs/) | 2 | 2026-04-10 | Design specs that fed those plans |
| [`closeouts/`](closeouts/) | 32 | v9–v170, BL-01 | Per-version closeout and handoff notes (`V<n>_CLOSEOUT_AND_V<n+1>_NEXT.md`) |
| [`results/`](results/) | 24 | v9–v29 | Result summaries and syntheses for the studies whose raw outputs are in [`research/legacy/`](../../research/legacy/README.md). [README](results/README.md) |
| [`archive/`](archive/) | 45 | 2026-03 → 2026-04 | Early plans, session notes, the pre-v2 changelog, and external reports under [`archive/history/`](archive/history/) (peer reviews, repo peer reviews, v15/v160/x1 research reports, redeploy-portfolio reports, the original agent prompts) |
| [`retired-code/`](retired-code/README.md) | 1 | v11–v24 | Index of the retired v11–v24 study scripts and their test, deleted from the tree in phase 4; each is one `git show` away |
| [`repo-hygiene-review-2026-04-19.md`](repo-hygiene-review-2026-04-19.md) | 1 | 2026-04-19 | Documentation and archive audit that deferred these moves until a link checker existed |

## Where things go now

| Record | Location |
|---|---|
| A promotion decision (what went live, and why) | [`docs/decisions/`](../decisions/README.md), one file each |
| An audit or repository review | [`docs/reviews/`](../reviews/) |
| A research study: script, outputs, conclusion | `research/studies/<id>_<slug>/` (index: [`research/README.md`](../../research/README.md)) |
| A version's changes | [`CHANGELOG.md`](../../CHANGELOG.md) |
| A plan or closeout for a multi-version workstream | here: [`superpowers/plans/`](superpowers/plans/) or [`closeouts/`](closeouts/), same naming as the existing files |

Do not edit these records to match today's code. If a record is wrong in a way
that matters, say so in the doc that supersedes it and link back.

## Finding things

- **Why is the live model the way it is?** Read the decision files in
  [`docs/decisions/`](../decisions/README.md); each links to the plans and
  closeouts behind it.
- **What did version vN change?** Search [`CHANGELOG.md`](../../CHANGELOG.md),
  then the closeout named `VN_CLOSEOUT_*` in [`closeouts/`](closeouts/).
- **What did an external reviewer say?** [`archive/history/`](archive/history/)
  groups peer reviews by date.
