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


from market_abm.agents import AgentType
from market_abm.analytics import compute_portfolio_metrics


class _MockAgent:
    def __init__(self, agent_type, cash, inventory, initial_wealth):
        self.agent_type = agent_type
        self.cash = cash
        self.inventory = inventory
        self.initial_wealth = initial_wealth


class TestPortfolioMetrics:
    def test_zero_pnl_at_start(self):
        agents = [_MockAgent(AgentType.NOISE, 10000.0, 10, 11000.0)
                  for _ in range(5)]
        result = compute_portfolio_metrics(agents, AgentType.NOISE, 100.0)
        assert result['mean_pnl'] == pytest.approx(0.0)
        assert result['n'] == 5

    def test_positive_pnl_when_price_rises(self):
        agents = [_MockAgent(AgentType.FUNDAMENTAL, 9900.0, 11, 11000.0)]
        result = compute_portfolio_metrics(agents, AgentType.FUNDAMENTAL, 110.0)
        assert result['mean_pnl'] == pytest.approx(110.0)

    def test_empty_type_returns_zeros(self):
        agents = [_MockAgent(AgentType.NOISE, 10000.0, 10, 11000.0)]
        result = compute_portfolio_metrics(agents, AgentType.TREND, 100.0)
        assert result['n'] == 0
        assert result['mean_pnl'] == 0.0

    def test_sharpe_zero_when_symmetric(self):
        agents = [
            _MockAgent(AgentType.NOISE, 10100.0, 10, 11000.0),
            _MockAgent(AgentType.NOISE, 9900.0, 10, 11000.0),
        ]
        result = compute_portfolio_metrics(agents, AgentType.NOISE, 100.0)
        assert result['sharpe'] == pytest.approx(0.0)


def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _arch(n, seed, a0=1e-5, a1=0.6):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = np.sqrt(a0 + a1 * x[i - 1] ** 2) * rng.normal()
    return x


class TestPerLagAutocorrelationCheck:
    def test_white_noise_passes(self):
        res = validate_stylized_facts(
            np.random.default_rng(3).normal(0, 0.01, 5000))
        assert res['no_return_autocorrelation']['passed']
        assert res['no_return_autocorrelation']['failing_lags'] == []

    def test_white_noise_pass_rate_matches_nominal_level(self):
        # Five independent 5% lag tests: about 77% of white-noise series pass.
        passes = [
            validate_stylized_facts(
                np.random.default_rng(s).normal(0, 0.01, 5000)
            )['no_return_autocorrelation']['passed']
            for s in range(200)
        ]
        assert 0.65 < np.mean(passes) < 0.88

    def test_negative_ar1_fails_and_reports_lag(self):
        # The old mean-|ACF| over 5 lags against 3x the band passed this.
        res = validate_stylized_facts(_ar1(-0.15, 5000, seed=1))
        r = res['no_return_autocorrelation']
        assert not r['passed']
        assert 1 in r['failing_lags']
        assert r['acf_lag1'] < -r['confidence_band']

    def test_positive_ar1_fails(self):
        res = validate_stylized_facts(_ar1(0.3, 3000, seed=2))
        assert not res['no_return_autocorrelation']['passed']
        assert 1 in res['no_return_autocorrelation']['failing_lags']


class TestVolatilityClusteringCheck:
    def test_arch_series_shows_clustering(self):
        res = validate_stylized_facts(_arch(5000, seed=5))
        vc = res['volatility_clustering']
        assert vc['passed']
        assert vc['acf_squared_lag1'] > vc['confidence_band']
        assert vc['acf_squared_lag5'] < vc['acf_squared_lag1']

    def test_white_noise_has_no_clustering(self):
        res = validate_stylized_facts(
            np.random.default_rng(3).normal(0, 0.01, 5000))
        assert not res['volatility_clustering']['passed']


class TestRunExperimentUsesAllReturns:
    def test_volatility_matches_all_returns_including_zeros(self):
        from market_abm.analytics import run_experiment
        from market_abm.config import DEFAULT_PARAMS
        from market_abm.model import MarketModel
        params = {**DEFAULT_PARAMS, 'steps': 400, 'n_agents': 8, 'seed': 3}
        model = MarketModel(params)
        model.run()
        r = model.output.variables.MarketModel['log_return'].values
        assert (r == 0.0).sum() > 0, "scenario must contain zero returns"
        out = run_experiment(params)
        assert out['volatility'] == pytest.approx(float(np.std(r)))


class TestValidateStylizedFactsBadInput:
    def test_nan_input_raises(self):
        r = np.random.default_rng(0).normal(0, 0.01, 500)
        r[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_stylized_facts(r)

    def test_inf_input_raises(self):
        r = np.random.default_rng(0).normal(0, 0.01, 500)
        r[10] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            validate_stylized_facts(r)

    def test_constant_input_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            validate_stylized_facts(np.zeros(500))

    def test_too_short_input_raises_value_error_not_index_error(self):
        with pytest.raises(ValueError, match="at least"):
            validate_stylized_facts(np.array([0.01, -0.02, 0.01]))
        with pytest.raises(ValueError, match="at least"):
            validate_stylized_facts(np.random.default_rng(0).normal(size=6))

    def test_short_but_valid_series_still_works(self):
        res = validate_stylized_facts(
            np.random.default_rng(1).normal(0, 0.01, 15))
        assert 'no_return_autocorrelation' in res
