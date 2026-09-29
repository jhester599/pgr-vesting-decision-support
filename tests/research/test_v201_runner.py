"""Fail-closed orchestration and preregistration contracts."""

import importlib.util
from pathlib import Path

import pytest


def runner() -> object:
    path = Path(__file__).resolve().parents[2] / (
        "research/studies/v201_price_macro/run.py"
    )
    assert path.is_file(), "The gated v201 runner must exist"
    spec = importlib.util.spec_from_file_location("v201_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absent_lock_blocks_before_any_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runner()
    monkeypatch.setattr(module, "BASELINE", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        module.preflight(tmp_path / "output", tmp_path / "scratch")
    assert "blocked" in (tmp_path / "output/README.md").read_text()
    assert not (tmp_path / "output/predictions.csv").exists()


def test_exact_six_blueprints_and_p3_inventory_names() -> None:
    module = runner()
    assert list(module.BLUEPRINTS) == ["P1", "P2", "P3", "M1", "M2", "M3"]
    assert module.BLUEPRINTS["P3"]["inventory"] == [
        "ta_ratio_ema_gap_12m_voo",
        "ta_ratio_rsi_6m_voo",
    ]
    assert module.LAGS["NFCI"] == 2
    assert all(lag == 1 for name, lag in module.LAGS.items() if name != "NFCI")
    assert module.BLUEPRINTS["P1"]["remove"] == ["mom_3m", "mom_6m", "mom_12m"]


def test_unregistered_execution_is_refused(tmp_path: Path) -> None:
    module = runner()
    with pytest.raises(ValueError, match="preregistration"):
        module.require_register(tmp_path)
