"""PGR vesting decision support: the installable package (review 2026-09-25, section 5).

``src/pgr_vds/`` is the review's target package. Phase 5 moved the monthly
decision logic (``pgr_vds.decision``) and the EDGAR monthly 8-K pipeline
(``pgr_vds.ingestion.edgar_monthly``) here; the older packages under
``src/`` (``src.models``, ``src.database``, ...) move in later.

The package is importable as ``pgr_vds`` only. ``src`` is a package too, so
the same files could also load as ``src.pgr_vds``, which would give every
module two copies and two sets of globals (monkeypatches and caches would
silently miss one). That import fails instead.
"""

from __future__ import annotations

if __name__ != "pgr_vds":
    raise ImportError(
        f"import pgr_vds, not {__name__}: the package lives at src/pgr_vds/ "
        "but is installed as a top-level package (pip install -e .)."
    )
