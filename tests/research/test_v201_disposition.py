"""Independent known pass/fail outputs for the frozen finalist gates."""

from __future__ import annotations

import importlib


def test_all_gates_and_failed_primary_cannot_use_secondary_success() -> None:
    module = importlib.import_module(
        "pgr_vds.research_lib.price_macro_disposition"
    )
    control = {
        "equal_weight_ic": 0.12,
        "hit_rate": 0.59,
        "directional_skill": 0.09,
        "brier": 0.21,
        "log_loss": 0.6,
        "ece": 0.1,
        "coverage_80": 0.78,
    }
    candidate = {
        "equal_weight_ic": 0.11,
        "hit_rate": 0.6,
        "directional_skill": 0.1,
        "brier": 0.2,
        "log_loss": 0.5,
        "ece": 0.09,
        "coverage_80": 0.8,
    }
    comparison = {"delta_r2": 0.011, "holm38_p": 0.049}
    assert module.candidate_disposition(candidate, control, comparison)[
        "passes"
    ]
    failed = module.candidate_disposition(
        candidate,
        control,
        {"delta_r2": 0.009, "holm38_p": 0.0001},
    )
    assert not failed["passes"]
    assert not failed["gates"]["delta_r2"]
    assert not module.candidate_disposition(
        candidate,
        control,
        {"delta_r2": 0.02, "holm38_p": 0.05},
    )["passes"]
    unknown = {**candidate, "ece": None}
    assert not module.candidate_disposition(unknown, control, comparison)[
        "passes"
    ]


def test_finalist_uses_only_passed_primary_and_frozen_tie_order() -> None:
    module = importlib.import_module(
        "pgr_vds.research_lib.price_macro_disposition"
    )
    rows = {
        "P1": {"passes": True, "delta_r2": 0.02},
        "P2": {"passes": True, "delta_r2": 0.02},
        "M1": {"passes": False, "delta_r2": 0.5},
    }
    assert module.nominate_finalist(["P1", "P2", "M1"], rows) == "P1"
    assert module.nominate_finalist(["M1"], rows) is None
