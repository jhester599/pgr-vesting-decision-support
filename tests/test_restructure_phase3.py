"""Review 2026-09-25, section 5, phase 3: one folder per study (WP13, F32).

- every study lives in ``research/studies/<id>_<slug>/`` (script, README,
  ``outputs/``), and its test in ``tests/research/``;
- ``research/registry.yaml`` lists every study folder, and
  ``research/README.md`` is generated from it;
- ``results/v9..v28`` are unchanged under ``research/legacy/``;
- ``results/`` holds no code, and ``*_detail.csv`` files over 1 MB are no
  longer committed (they go to a gitignored ``outputs/detail/``);
- production reads the promoted study outputs from their new paths.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = REPO_ROOT / "research" / "studies"
ONE_MB = 1024 * 1024


def _tracked(*paths: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [p for p in out.splitlines() if p]


def _registry():
    from research.tools import registry

    return registry


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_results_contains_no_python_files() -> None:
    assert [p for p in _tracked("results") if p.endswith(".py")] == []


def test_research_scripts_left_scripts() -> None:
    assert _tracked("scripts/research") == []
    assert [p for p in _tracked("scripts") if "experiments" in Path(p).name] == []


def test_results_research_is_gone() -> None:
    assert _tracked("results/research") == []


@pytest.mark.parametrize(
    "folder, script",
    [
        ("v38_shrinkage", "v38_shrinkage.py"),
        ("v72_quality_weighted_consensus", "v72_quality_weighted_consensus.py"),
        ("v75_holdout_shadow_replay", "v75_holdout_shadow_replay.py"),
        ("v128_benchmark_feature_search", "v128_benchmark_feature_search.py"),
        ("v129_feature_map_eval", "v129_feature_map_eval.py"),
        ("v129_feature_map_eval", "v129_vgt_robustness_audit.py"),
        ("v131_threshold_sweep_eval", "v131_threshold_sweep_eval.py"),
        ("v132_threshold_validation", "v132_threshold_validation.py"),
        ("v134_fred_lag_sweep", "v134_fred_lag_sweep.py"),
        ("v142_edgar_lag_eval", "v142_edgar_lag_eval.py"),
        ("x15_pb_regime_overlay", "x15_pb_regime_overlay.py"),
        ("x24_indicator_contract", "x24_indicator_contract.py"),
        ("pb_vs_pe_analysis", "pb_vs_pe_analysis.py"),
        ("v9_experiments", "feature_experiments.py"),
    ],
)
def test_live_setting_and_x_series_studies_have_a_folder(folder: str, script: str) -> None:
    study = STUDIES_DIR / folder
    assert (study / script).is_file()
    assert (study / "README.md").is_file()


def test_every_study_folder_has_readme_and_outputs_or_legacy_link() -> None:
    problems = []
    for study in sorted(p for p in STUDIES_DIR.iterdir() if p.is_dir()):
        if not (study / "README.md").is_file():
            problems.append(f"{study.name}: no README.md")
        has_outputs = (study / "outputs").is_dir()
        readme = (study / "README.md").read_text(encoding="utf-8") if (study / "README.md").is_file() else ""
        if not has_outputs and "research/legacy/" not in readme:
            problems.append(f"{study.name}: no outputs/ and no legacy link")
    assert problems == []


def test_study_tests_live_in_tests_research() -> None:
    top_level = [
        p for p in _tracked("tests")
        if re.match(r"tests/(test_research_|test_v\d+_research|test_bl01|test_pb_vs_pe)", p)
    ]
    assert top_level == []
    assert (REPO_ROOT / "tests" / "research" / "__init__.py").is_file()
    assert (REPO_ROOT / "tests" / "research" / "test_research_v38_shrinkage.py").is_file()


def test_study_scripts_resolve_the_repo_root() -> None:
    """Scripts moved one level deeper must climb one level further."""
    wrong = []
    scripts = [rel for rel in _tracked("research/studies") if rel.endswith(".py")]
    assert len(scripts) >= 100
    for rel in scripts:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for depth in re.findall(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]", text):
            if depth != "3":
                wrong.append(f"{rel}: parents[{depth}]")
        for chain in re.findall(r"((?:os\.path\.dirname\()+)os\.path\.abspath\(__file__\)", text):
            if chain.count("dirname") != 4:
                wrong.append(f"{rel}: {chain.count('dirname')} x dirname")
    assert wrong == []


def test_moved_tests_resolve_the_repo_root() -> None:
    wrong = []
    moved = _tracked("tests/research")
    assert len(moved) >= 100
    for rel in moved:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for depth in re.findall(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]", text):
            if depth != "2":
                wrong.append(f"{rel}: parents[{depth}]")
    assert wrong == []


_OLD_LOCATION = re.compile(
    r"results/research|results\.research|scripts/research|scripts\.research|"
    r"\"results\"\s*[/,]\s*\"research\""
)


def test_code_no_longer_points_at_old_study_locations() -> None:
    offenders = []
    for rel in _tracked("*.py", "*.yml", "*.toml", "*.sh"):
        if rel.startswith(("archive/", "research/legacy/")) or "/outputs/" in rel:
            continue
        if rel.startswith("tests/test_restructure_phase"):
            continue  # these assert that the old locations are gone
        tree_text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if rel.endswith(".py"):
            # docstrings and comments may record where a file used to live
            code_lines = _code_lines(tree_text)
        else:
            code_lines = tree_text.splitlines()
        for line in code_lines:
            if _OLD_LOCATION.search(line):
                offenders.append(f"{rel}: {line.strip()[:100]}")
    assert offenders == []


def _code_lines(text: str) -> list[str]:
    """Return the source lines outside docstrings and comments."""
    tree = ast.parse(text)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ) and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    lines = []
    for n, line in enumerate(text.splitlines(), 1):
        if n in doc_lines or line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_every_study_folder_is_registered() -> None:
    registry = _registry()
    assert registry.validate(registry.load_registry()) == []


def test_registry_rejects_an_unregistered_folder(tmp_path: Path) -> None:
    registry = _registry()
    studies = tmp_path / "studies"
    (studies / "v1_known").mkdir(parents=True)
    (studies / "v2_stray").mkdir()
    entry = {
        "id": "v1", "slug": "known", "date": "2026-01-01", "question": "q?",
        "status": "closed", "promoted_to": None, "closeout": None,
    }
    errors = registry.validate([entry], studies_dir=studies, repo_root=tmp_path)
    assert errors == ["research/studies/v2_stray: folder is not in the registry"]


def test_registry_rejects_bad_entries(tmp_path: Path) -> None:
    registry = _registry()
    studies = tmp_path / "studies"
    (studies / "v1_known").mkdir(parents=True)
    entry = {
        "id": "v1", "slug": "known", "date": "2026-1-1", "question": "q?",
        "status": "promoted", "promoted_to": None, "closeout": "docs/missing.md",
    }
    errors = registry.validate([entry], studies_dir=studies, repo_root=tmp_path)
    assert any("date" in e for e in errors)
    assert any("needs promoted_to" in e for e in errors)
    assert any("closeout" in e for e in errors)


def test_research_readme_is_generated_from_registry() -> None:
    registry = _registry()
    expected = registry.render_readme(registry.load_registry())
    actual = (REPO_ROOT / "research" / "README.md").read_text(encoding="utf-8")
    assert actual == expected


@pytest.mark.parametrize(
    "study_id, status",
    [
        ("v38", "promoted"),
        ("v72", "promoted"),
        ("v128", "shadow"),
        ("v129", "shadow"),
        ("v131", "retained"),
        ("v132", "retained"),
        ("v134", "retained"),
        ("v142", "retained"),
        ("v113", "shadow"),
        ("v141", "shadow"),
    ],
)
def test_live_setting_studies_record_what_they_set(study_id: str, status: str) -> None:
    entries = {e["id"]: e for e in _registry().load_registry()}
    assert entries[study_id]["status"] == status
    assert entries[study_id]["promoted_to"]


def test_ci_runs_the_registry_check() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python research/tools/registry.py" in ci


# ---------------------------------------------------------------------------
# Legacy results
# ---------------------------------------------------------------------------

LEGACY = ["v9", "v11", "v12", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "v24", "v27", "v28"]


@pytest.mark.parametrize("version", LEGACY)
def test_legacy_result_folders_moved(version: str) -> None:
    assert _tracked(f"results/{version}") == []
    assert _tracked(f"research/legacy/{version}") != []


def test_legacy_readme_lists_every_folder() -> None:
    text = (REPO_ROOT / "research" / "legacy" / "README.md").read_text(encoding="utf-8")
    for version in LEGACY:
        assert f"`{version}/`" in text


# ---------------------------------------------------------------------------
# Detail files
# ---------------------------------------------------------------------------


def test_no_committed_detail_csv_over_one_mb_outside_legacy() -> None:
    big = [
        p for p in _tracked("*_detail*.csv")
        if not p.startswith("research/legacy/")
        and (REPO_ROOT / p).stat().st_size > ONE_MB
    ]
    assert big == []


def test_detail_output_folders_are_gitignored() -> None:
    probe = "research/studies/v128_benchmark_feature_search/outputs/detail/x_detail.csv"
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", probe], cwd=REPO_ROOT
    )
    assert result.returncode == 0


def test_large_detail_writers_use_the_detail_folder() -> None:
    from src.research.study_paths import study_output_path

    v128 = study_output_path("v128_regularized_selection_detail.csv", detail=True)
    assert v128.parent.name == "detail"
    assert v128.parent.parent == STUDIES_DIR / "v128_benchmark_feature_search" / "outputs"
    v128_src = (STUDIES_DIR / "v128_benchmark_feature_search" / "v128_benchmark_feature_search.py").read_text(encoding="utf-8")
    assert "detail=True" in v128_src
    v162_src = (STUDIES_DIR / "v162_ta_broad_screen" / "v162_ta_broad_screen.py").read_text(encoding="utf-8")
    assert '"detail" / "v162_ta_broad_screen_detail.csv"' in v162_src


# ---------------------------------------------------------------------------
# Resolver and production reads
# ---------------------------------------------------------------------------


def test_study_output_path_resolves_by_id() -> None:
    from src.research.study_paths import study_dir, study_output_path

    assert study_dir("v128") == STUDIES_DIR / "v128_benchmark_feature_search"
    assert study_output_path("v38_shrinkage_best_results.csv") == (
        STUDIES_DIR / "v38_shrinkage" / "outputs" / "v38_shrinkage_best_results.csv"
    )
    with pytest.raises(LookupError):
        study_dir("v999")
    with pytest.raises(ValueError):
        study_output_path("no_study_prefix.csv")


def test_production_study_inputs_exist() -> None:
    import config
    from src.models.classification_gate_overlay import DEFAULT_OVERLAY_RESULTS_PATH
    from src.reporting.shadow_followon import load_followon_candidate_bundle

    assert (REPO_ROOT / config.V128_BENCHMARK_FEATURE_MAP_PATH).is_file()
    assert (REPO_ROOT / DEFAULT_OVERLAY_RESULTS_PATH).is_file()
    bundle = load_followon_candidate_bundle()
    assert set(bundle) == {
        "v141_blend_weight", "v143_corr_prune", "v144_conformal", "v149_kelly",
        "v150_neutral_band",
    }
