"""v129 dual-track shadow: benchmark-specific feature map loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.features import DUAL_TRACK_LEAN_BASELINE_OVERRIDES


class DualTrackFeatureMapError(RuntimeError):
    """Raised when the v128 feature map CSV cannot be loaded."""


def load_v128_feature_map(path: Path | str) -> dict[str, list[str]]:
    """Load v128 benchmark feature map; return only switched benchmarks.

    Returns a dict mapping benchmark ticker -> list of feature names.
    Only benchmarks where switched_from_baseline is True are included.
    Benchmarks not in the result should fall back to lean_baseline via
    resolve_benchmark_features().

    Raises:
        DualTrackFeatureMapError: if the CSV is missing or unparseable.
    """
    try:
        df = pd.read_csv(Path(path))
    except FileNotFoundError as exc:
        raise DualTrackFeatureMapError(
            f"v128 feature map not found at {path!r}. "
            "Run results/research/v128_benchmark_feature_search.py first."
        ) from exc
    except Exception as exc:
        raise DualTrackFeatureMapError(f"Failed to parse v128 feature map: {exc}") from exc

    required = {"benchmark", "selected_features", "switched_from_baseline"}
    missing = required - set(df.columns)
    if missing:
        raise DualTrackFeatureMapError(
            f"v128 feature map is missing columns: {missing}"
        )

    switched = df[df["switched_from_baseline"].astype(str).str.lower().isin({"true", "1", "yes"})]
    result: dict[str, list[str]] = {}
    for _, row in switched.iterrows():
        benchmark = str(row["benchmark"])
        features = [f.strip() for f in str(row["selected_features"]).split("|") if f.strip()]
        result[benchmark] = features
    return result


def resolve_benchmark_features(
    benchmark: str,
    feature_map: dict[str, list[str]],
    *,
    lean_baseline: list[str],
    overrides: frozenset[str] = DUAL_TRACK_LEAN_BASELINE_OVERRIDES,
) -> list[str]:
    """Return the feature list for a benchmark respecting the lean-baseline override set.

    If the benchmark is in overrides (e.g. VGT), always returns lean_baseline.
    Otherwise returns the benchmark-specific list from feature_map, or lean_baseline
    if the benchmark is not in the map.
    """
    if benchmark in overrides:
        return list(lean_baseline)
    return list(feature_map.get(benchmark, lean_baseline))
