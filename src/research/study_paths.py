"""Where research studies keep their files (review 2026-09-25, section 5, phase 3).

Each study lives in ``research/studies/<id>_<slug>/``: its script(s), a
README, and ``outputs/``. Output file names start with the study id
(``v38_shrinkage_results.csv``), so a file name alone locates its folder.
Large per-fold detail files go to ``outputs/detail/``, which is gitignored.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
STUDIES_DIR = RESEARCH_DIR / "studies"
LEGACY_DIR = RESEARCH_DIR / "legacy"

_STUDY_ID = re.compile(r"^((?:v|x|bl)\d+)_")


@lru_cache(maxsize=None)
def study_dir(study_id: str) -> Path:
    """Return ``research/studies/<study_id>_<slug>/``.

    Raises:
        LookupError: if no folder, or more than one, has that id.
    """
    matches = sorted(p for p in STUDIES_DIR.glob(f"{study_id}_*") if p.is_dir())
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise LookupError(f"study id {study_id!r}: expected one folder, found {found}")
    return matches[0]


def study_output_path(filename: str, *, detail: bool = False) -> Path:
    """Return the path of a study output, located by the id prefix of its name.

    Args:
        filename: output file name starting with a study id, e.g.
            ``v38_shrinkage_best_results.csv``.
        detail: put the file in the gitignored ``outputs/detail/`` folder
            (for per-fold detail files over 1 MB, which are not committed).

    Raises:
        ValueError: if the name does not start with a study id.
        LookupError: if no study folder has that id.
    """
    match = _STUDY_ID.match(filename)
    if match is None:
        raise ValueError(f"{filename!r} does not start with a study id (v38_, x15_, bl01_)")
    outputs = study_dir(match.group(1)) / "outputs"
    return outputs / "detail" / filename if detail else outputs / filename
