"""Tests for v131 Path B composite-target classifier module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.path_b_classifier import (
    build_composite_return_series,
    fit_path_b_classifier,
    apply_prequential_temperature_scaling,
    PATH_B_THRESHOLD,
)


def test_path_b_threshold_is_003() -> None:
    assert PATH_B_THRESHOLD == pytest.approx(0.03)


def test_apply_prequential_temperature_scaling_warmup_returns_raw() -> None:
    """Before warmup window, returns raw probs unchanged."""
    probs = np.array([0.6, 0.7, 0.5])
    labels = np.array([1, 0, 1])
    result = apply_prequential_temperature_scaling(probs, labels, warmup=5)
    np.testing.assert_array_almost_equal(result, probs)


def test_apply_prequential_temperature_scaling_shape_preserved() -> None:
    n = 30
    probs = np.random.default_rng(42).uniform(0.1, 0.9, n)
    labels = np.random.default_rng(42).integers(0, 2, n)
    result = apply_prequential_temperature_scaling(probs, labels, warmup=10)
    assert result.shape == probs.shape


def test_apply_prequential_temperature_scaling_output_in_01() -> None:
    n = 40
    probs = np.random.default_rng(7).uniform(0.05, 0.95, n)
    labels = np.random.default_rng(7).integers(0, 2, n)
    result = apply_prequential_temperature_scaling(probs, labels, warmup=10)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_fit_path_b_classifier_returns_float_or_none() -> None:
    """fit_path_b_classifier returns a float probability or None."""
    X = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=100),
                      "f2": np.random.default_rng(1).normal(size=100)})
    y = pd.Series((np.random.default_rng(2).normal(size=100) < -0.03).astype(int))
    result = fit_path_b_classifier(X, y, feature_cols=["f1", "f2"])
    assert result is None or isinstance(result, float)
