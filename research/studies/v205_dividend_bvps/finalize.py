"""Write v205 provenance, closeout and output manifest from saved outputs.

Reads only committed code, the v200 lock and this study's outputs. It fits
nothing, scores nothing and opens no database, so the executed ``run.py``
stays byte-identical to the commit that produced the forecasts.

    python research/studies/v205_dividend_bvps/finalize.py \
        --reproduction <external-output-dir> [--pytest-log <log>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from pgr_vds.research_lib.provenance import (
    INPUT_DB_COMMIT,
    INPUT_DB_SHA256,
    PARENT_SEED_COMMIT,
    PARENT_SEED_SHA256,
    runtime_lock,
    sha256_file,
    source_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
OUTPUTS = STUDY / "outputs"
CORE = [
    "predictions.csv",
    "fold_ledger.csv",
    "calibration_streams.csv",
    "metrics.json",
    "multiplicity.json",
    "comparison.json",
    "candidate_ledger.json",
]
CODE = [
    "research/studies/v205_dividend_bvps/run.py",
    "src/pgr_vds/research_lib/xseries.py",
    "src/pgr_vds/research_lib/adapters.py",
    "src/pgr_vds/research_lib/metrics.py",
    "src/pgr_vds/research_lib/temporal.py",
    "src/pgr_vds/research_lib/provenance.py",
    "tests/research/test_v205_xseries.py",
    "tests/research/test_v205_runner.py",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def load(name: str) -> Any:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def save(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def reproducibility(reproduction: Path) -> dict[str, Any]:
    """Compare the second independent execution byte for byte."""
    files = {}
    for name in CORE:
        first = sha256_file(OUTPUTS / name)
        second = sha256_file(reproduction / name)
        files[name] = {
            "sha256": first,
            "second_run_sha256": second,
            "identical": first == second,
        }
    return {
        "method": "second --execute into an external directory with a "
        "fresh verified DB export",
        "files": files,
        "all_identical": all(item["identical"] for item in files.values()),
    }


def closeout(metrics: dict[str, Any], comparison: dict[str, Any]) -> dict:
    rows = {}
    for name, summary in metrics.items():
        rows[name] = {
            "status": summary["status"],
            "passes": summary["disposition"]["passes"],
            "failed_gates": summary["disposition"]["failed"],
            "relative_mae_reduction": summary.get("relative_mae_reduction"),
            "delta_r2": summary.get("delta_r2"),
            "raw_p": summary.get("raw_p"),
            "adjusted_p": summary.get("adjusted_p"),
        }
    return {
        "study": "v205_dividend_bvps",
        "disposition": "no winner; all lanes remain research-only",
        "winner": comparison["winner"],
        "candidates": rows,
        "v207_finalist": None,
        "promotion": False,
        "live_changes": "none: no config, feature, model, policy or email "
        "change",
        "negative_results": [
            "B1/B2 Ridge BVPS-growth models lose to the past-only mean "
            "(MAE +46%/+41%) with strongly negative IC.",
            "B3 structural adjusted-BVPS/no-change-P/B mapping loses to the "
            "mature mean 6M PGR DRIP return (MAE +35%).",
            "D1 (past cash + combined ratio) loses to past-12M cash.",
            "D2 improves MAE 9.0% and R2 by .19 but misses the 10% MAE bar "
            "and its primary test (raw p .268, Holm p 1).",
            "D3 closed: 3 positive annual events.",
        ],
        "what_is_left": [
            "v207 opens the quarantine once; v205 contributes no finalist.",
            "The annual excess-dividend endpoint needs more post-policy "
            "years before any honest test is possible.",
            "Any rerun of D2's idea is a new candidate needing its own "
            "registration; no post-result search was done here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--pytest-log", type=Path)
    arguments = parser.parse_args()
    record = load("run_record.json")
    preregistration = load("preregistration.json")
    metrics = load("metrics.json")
    comparison = load("comparison.json")
    repro = reproducibility(arguments.reproduction)
    save(OUTPUTS / "reproducibility.json", repro)
    save(OUTPUTS / "closeout.json", closeout(metrics, comparison))
    execution = record["execution_code_commit"]
    changed_since = [
        path for path in CODE
        if git("diff", "--name-only", execution, "--", path)
    ]
    lock = ROOT / "research/studies/v200_clean_baseline/outputs"
    provenance = {
        "study": "v205_dividend_bvps",
        "as_of": preregistration["as_of"],
        "development_boundary": preregistration["development_boundary"],
        "max_available_label": max(
            item["last_available"]
            for item in preregistration["partition_lock"]["endpoints"].values()
        ),
        "input_db": {
            "git_commit": INPUT_DB_COMMIT,
            "sha256": INPUT_DB_SHA256,
            "access": "git show export outside the repository, verified, "
            "opened read-only immutable",
        },
        "parent_seed": {
            "git_commit": PARENT_SEED_COMMIT,
            "db_sha256": PARENT_SEED_SHA256,
        },
        "baseline_lock": {
            "path": "research/studies/v200_clean_baseline/outputs/"
            "baseline_lock.json",
            "sha256": sha256_file(lock / "baseline_lock.json"),
            "registry_exception": "research/registry.yaml verified at the "
            "v200 execution commit; all pinned entries unchanged, v205 "
            "appended (outputs/preflight.json)",
        },
        "code": {
            "preregistration_commit": record["preregistration_commit"],
            "execution_code_commit": execution,
            "execution_tree_note": "execute requires the committed, clean "
            "preregistration; source files listed here are compared with "
            "the execution commit",
            "sources_changed_since_execution": changed_since,
            "source_sha256": {path: source_sha256(ROOT / path)
                              for path in CODE},
        },
        "runtime": {
            "execution": record["runtime"],
            "finalize": runtime_lock(),
            "v200_runtime_lock_sha256": sha256_file(
                lock / "runtime_lock.json"
            ),
            "v200_dependency_environment_sha256": sha256_file(
                lock / "dependency_environment.json"
            ),
            "installation_note": "Python 3.12.14 and the complete v200 "
            "package set installed at exact versions in an isolated venv; "
            "verify_runtime passed",
        },
        "extraction_sql": preregistration["extraction_sql"],
        "targets": preregistration["endpoints"],
        "target_checks": preregistration["target_checks"],
        "sources": {
            "splits": "split_history PGR, explicit ratios only",
            "dividends": "daily_dividends PGR by ex-date; December specials "
            "included by the December-February window",
            "edgar": "pgr_edgar_monthly repaired current values gated by "
            "actual filing_date; first-reported comparison in "
            "source_vintage_audit.json",
            "fred": "not used",
            "provider_calls": 0,
        },
        "procedure": preregistration["rules"],
        "candidates": preregistration["candidates"],
        "budget": {"campaign_candidates": 6, "alpha_grid": [1, 10, 100],
                   "fits_beyond_register": 0},
        "seed": 20260926,
        "bootstrap_replicates": 2000,
        "partition_lock_sha256": sha256_file(OUTPUTS / "partition_lock.json"),
        "access_ledger_sha256": sha256_file(OUTPUTS / "access_ledger.json"),
        "v200_quarantine": preregistration["partition_lock"]["v200_quarantine"],
        "tracked_db_sha256": {
            "before": record["tracked_db_sha256_before"],
            "after_execution": record["tracked_db_sha256_after"],
        },
        "reproducibility": repro["all_identical"],
    }
    if arguments.pytest_log is not None:
        lines = arguments.pytest_log.read_text("utf-8").splitlines()
        provenance["full_pytest"] = [
            line for line in lines if " passed" in line or "exit=" in line
        ][-2:]
    save(STUDY / "provenance.json", provenance)
    manifest = {
        str(path.relative_to(STUDY)).replace("\\", "/"): sha256_file(path)
        for path in sorted(OUTPUTS.rglob("*"))
        if path.is_file() and path.name != "output_manifest.json"
    }
    manifest["provenance.json"] = sha256_file(STUDY / "provenance.json")
    manifest["README.md"] = sha256_file(STUDY / "README.md")
    save(OUTPUTS / "output_manifest.json", {"sha256": manifest})


if __name__ == "__main__":
    main()
