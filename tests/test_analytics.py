"""Tests for analytics module."""

import numpy as np
import pytest
from scipy.stats import norm

from market_abm.analytics import (
    compute_autocorrelation,
    compute_return_statistics,
    hill_estimator,
    validate_stylized_facts,
)


class TestComputeReturnStatistics:
    def test_normal_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 5000)
        stats = compute_return_statistics(returns)
        assert abs(stats['mean']) < 0.001
        assert abs(stats['kurtosis']) < 1.0  # normal kurtosis ≈ 0
        assert stats['n'] == 5000

    def test_fat_tailed_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_t(df=3, size=5000) * 0.01
        stats = compute_return_statistics(returns)
        assert stats['kurtosis'] > 1.0  # should be significantly > 0

    def test_keys_present(self):
        returns = np.random.default_rng(0).normal(0, 1, 100)
        stats = compute_return_statistics(returns)
        expected = {'mean', 'std', 'skewness', 'kurtosis', 'jb_statistic',
                    'jb_pvalue', 'min', 'max', 'n'}
        assert expected == set(stats.keys())


class TestComputeAutocorrelation:
    def test_output_shape(self):
        returns = np.random.default_rng(42).normal(0, 1, 500)
        result = compute_autocorrelation(returns, nlags=20)
        assert len(result['acf_returns']) == 21  # lag 0 through 20
        assert len(result['acf_abs_returns']) == 21
        assert len(result['acf_squared_returns']) == 21

    def test_lag_zero_is_one(self):
        returns = np.random.default_rng(42).normal(0, 1, 500)
        result = compute_autocorrelation(returns, nlags=10)
        assert result['acf_returns'][0] == pytest.approx(1.0)

    def test_white_noise_low_acf(self):
        returns = np.random.default_rng(42).normal(0, 1, 2000)
        result = compute_autocorrelation(returns, nlags=10)
        # For white noise, ACF at lag > 0 should be small
        assert all(abs(r) < 0.1 for r in result['acf_returns'][1:])


class TestHillEstimator:
    def test_normal_returns_high_tail_index(self):
        """Normal distribution has light tails → high tail index."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1, 10000)
        alpha = hill_estimator(returns)
        assert alpha > 3.0  # Normal tails are very thin

    def test_heavy_tailed_returns(self):
        """Student-t(3) should give tail index near 3."""
        rng = np.random.default_rng(42)
        returns = rng.standard_t(df=3, size=10000)
        alpha = hill_estimator(returns)
        assert 1.5 < alpha < 5.0  # Rough range for t(3)

    def test_custom_k(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1, 1000)
        alpha = hill_estimator(returns, k=50)
        assert np.isfinite(alpha)


class TestValidateStylizedFacts:
    def test_returns_all_facts(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 2000)
        results = validate_stylized_facts(returns)
        expected_keys = {'fat_tails', 'volatility_clustering',
                         'no_return_autocorrelation', 'non_normality', 'tail_index'}
        assert expected_keys == set(results.keys())

    def test_each_fact_has_passed_key(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 2000)
        results = validate_stylized_facts(returns)
        for fact, info in results.items():
            assert 'passed' in info
            assert isinstance(info['passed'], bool)
