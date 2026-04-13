"""Tests for v129 dual-track benchmark-specific feature map loader."""
from __future__ import annotations

import pytest
from pathlib import Path

from config.features import DUAL_TRACK_LEAN_BASELINE_OVERRIDES
from src.models.v129_feature_map import (
    DualTrackFeatureMapError,
    load_v128_feature_map,
    resolve_benchmark_features,
)

_LEAN = ["f1", "f2", "f3"]
_REAL_MAP = Path("results/research/v128_benchmark_feature_map.csv")


def test_load_returns_dict_with_benchmarks() -> None:
    fmap = load_v128_feature_map(_REAL_MAP)
    assert isinstance(fmap, dict)
    assert "BND" in fmap


def test_bnd_resolves_to_benchmark_specific_features() -> None:
    fmap = load_v128_feature_map(_REAL_MAP)
    features = resolve_benchmark_features("BND", fmap, lean_baseline=_LEAN, overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
    assert "pb_ratio" in features
    assert features != _LEAN


def test_vgt_always_resolves_to_lean_baseline() -> None:
    fmap = load_v128_feature_map(_REAL_MAP)
    features = resolve_benchmark_features("VGT", fmap, lean_baseline=_LEAN, overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
    assert features == _LEAN


def test_gld_resolves_to_lean_baseline_not_in_switched_set() -> None:
    fmap = load_v128_feature_map(_REAL_MAP)
    features = resolve_benchmark_features("GLD", fmap, lean_baseline=_LEAN, overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
    assert features == _LEAN


def test_unknown_benchmark_falls_back_to_lean_baseline() -> None:
    features = resolve_benchmark_features("UNKNOWN", {}, lean_baseline=_LEAN, overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
    assert features == _LEAN


def test_missing_path_raises_dual_track_error() -> None:
    with pytest.raises(DualTrackFeatureMapError):
        load_v128_feature_map(Path("nonexistent/path.csv"))


def test_load_only_includes_switched_benchmarks() -> None:
    fmap = load_v128_feature_map(_REAL_MAP)
    # GLD switched_from_baseline=False, so not in the returned dict
    assert "GLD" not in fmap
    # BND switched_from_baseline=True
    assert "BND" in fmap


def test_features_are_list_of_strings() -> None:
    fmap = load_v128_feature_map(_REAL_MAP)
    for benchmark, features in fmap.items():
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)
