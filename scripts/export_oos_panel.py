"""Export the realised-only OOS panel of the production ensemble (read-only).

Review 2026-09-25, step 6 (WP8, F20). The live ACTIONABLE sell-% mapping is
backtested on the same prequential OOS record the monthly gate scores
(``build_prequential_panel``). This script rebuilds that record exactly as
``monthly_decision._generate_signals`` does for one as-of date and writes one
row per (benchmark, OOS date) to CSV:

    benchmark, date, y_true, z, alpha, y_hat, naive

The DB is opened read-only and its sha256 is checked before and after. Run it
against a copy of ``data/pgr_financials.db``.

Usage:
    python scripts/export_oos_panel.py --db-path /tmp/copy.db \
        --as-of 2026-09-21 --out tests/fixtures/live_mapping_oos_panel.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import db_client  # noqa: E402
from src.models.prequential import PANEL_BASE_COLUMNS  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", required=True, help="Path to a copy of the DB.")
    parser.add_argument("--as-of", required=True, help="As-of date (YYYY-MM-DD).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    from scripts.monthly_decision import _generate_signals

    db_path = Path(args.db_path)
    before = _sha256(db_path)
    conn = db_client.get_connection(str(db_path), read_only=True)
    try:
        _, _, diagnostics = _generate_signals(conn, date.fromisoformat(args.as_of))
    finally:
        conn.close()
    after = _sha256(db_path)
    if before != after:
        raise RuntimeError(f"DB changed during the export: {before} -> {after}")

    panel = diagnostics.get("prequential_panel")
    if panel is None or panel.empty:
        raise RuntimeError("No prequential OOS panel was produced.")
    out = panel[list(PANEL_BASE_COLUMNS)].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.10g")
    print(
        f"Wrote {len(out):,} rows ({out['date'].nunique()} dates, "
        f"{out['benchmark'].nunique()} benchmarks) to {args.out}; DB sha256 {after[:12]} unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
