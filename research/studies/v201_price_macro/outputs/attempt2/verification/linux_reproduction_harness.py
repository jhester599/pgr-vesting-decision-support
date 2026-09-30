"""Reproduce v201 attempt 2 on Linux with the unmodified execute().

Review evidence only (2026-09-30); it never replaces committed outputs.
Run from a ``core.autocrlf=true`` checkout with the exact v200 runtime:
``python linux_reproduction_harness.py <checkout> <scratch> <output>``.

``np.logspace(-4, 4, 50)`` differs from the Windows-registered Ridge grid
in 3 of 50 values by 1 ULP, which fails the procedure check. The shared
grid array is set in place to the registered values, so the procedure
check and every fit use exactly the preregistered penalties.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import pgr_vds.research_lib.adapters as adapters


def main() -> None:
    root, scratch, output = (Path(value) for value in sys.argv[1:4])
    study = root / "research/studies/v201_price_macro"
    registered = json.loads(
        (study / "outputs/attempt2/registered.json").read_text(
            encoding="utf-8"
        )
    )
    adapters.RIDGE_GRID[:] = np.array(registered["ridge_grid"])
    spec = importlib.util.spec_from_file_location(
        "v201_run", study / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RIDGE_GRID is adapters.RIDGE_GRID
    module.execute(scratch, output)
    print("EXECUTE_DONE")


if __name__ == "__main__":
    main()
