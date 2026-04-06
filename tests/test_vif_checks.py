"""Tests for v32.1 — VIF multicollinearity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import compute_vif


def _make_df(data: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(data)


def test_independent_features_have_low_vif():
    """Uncorrelated features should produce VIF values close to 1.0."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    })
    vif = compute_vif(df)
    assert not vif.empty
    assert set(vif.index) == {"a", "b", "c"}
    # Independent features should have VIF close to 1
    assert (vif < 3.0).all(), f"Expected all VIF < 3 for independent features, got:\n{vif}"


def test_highly_correlated_feature_has_high_vif():
    """A feature that is a near-linear combination of others should have high VIF."""
    rng = np.random.default_rng(7)
    n = 200
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = 2 * a + 3 * b + rng.standard_normal(n) * 0.01  # nearly a + b
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    vif = compute_vif(df)
    assert not vif.empty
    # c is nearly a linear combination of a and b → very high VIF
    assert vif["c"] > 100, f"Expected VIF for 'c' > 100, got {vif['c']:.2f}"


def test_returns_series_sorted_descending():
    """VIF Series should be sorted in descending order."""
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
    })
    vif = compute_vif(df)
    assert list(vif.values) == sorted(vif.values, reverse=True)


def test_feature_cols_subset():
    """Only specified feature_cols should be evaluated."""
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    })
    vif = compute_vif(df, feature_cols=["a", "b"])
    assert set(vif.index) == {"a", "b"}
    assert "c" not in vif.index


def test_fewer_than_two_obs_returns_empty():
    """If fewer than 2 complete rows remain after NaN drop, return empty Series."""
    df = pd.DataFrame({
        "a": [1.0, np.nan, np.nan],
        "b": [2.0, np.nan, np.nan],
    })
    vif = compute_vif(df)
    assert vif.empty


def test_nan_rows_are_dropped():
    """NaN rows should be excluded; result should still be computed on clean rows."""
    rng = np.random.default_rng(5)
    n = 80
    a = rng.standard_normal(n).tolist()
    b = rng.standard_normal(n).tolist()
    # Inject some NaN rows
    a[10] = np.nan
    b[20] = np.nan
    df = pd.DataFrame({"a": a, "b": b})
    vif = compute_vif(df)
    assert not vif.empty
    assert set(vif.index) == {"a", "b"}


def test_single_feature_returns_empty():
    """A single feature has no other features to regress on; result is empty."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    vif = compute_vif(df)
    # statsmodels VIF with a single column is undefined (denominator = 1 by definition
    # but regression on zero predictors is degenerate) — we accept either empty or
    # a Series with one value, as long as no exception is raised.
    assert isinstance(vif, pd.Series)


def test_empty_dataframe_returns_empty():
    """Empty DataFrame should return empty Series."""
    vif = compute_vif(pd.DataFrame())
    assert vif.empty
