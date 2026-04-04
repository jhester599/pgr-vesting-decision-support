from __future__ import annotations

import math

from src.research.diversification import classify_correlation_bucket, diversification_score


def test_classify_correlation_bucket_thresholds() -> None:
    assert classify_correlation_bucket(0.80) == "highly_correlated"
    assert classify_correlation_bucket(0.60) == "moderately_correlated"
    assert classify_correlation_bucket(0.20) == "diversifying"
    assert classify_correlation_bucket(float("nan")) == "unknown"


def test_diversification_score_rewards_lower_correlation() -> None:
    low_corr = diversification_score(0.10, 0.05)
    high_corr = diversification_score(0.80, 0.75)
    assert low_corr > high_corr
    assert 0.0 <= high_corr <= 1.0
    assert 0.0 <= low_corr <= 1.0
