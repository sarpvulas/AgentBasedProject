"""Tests for strategy switching."""

import numpy as np
import pytest

import agentpy as ap

from market_abm.agents import Strategy, Trader
from market_abm.strategy_switching import (
    compute_exponential_ma_fitness,
    compute_strategy_fitness,
    multinomial_logit_probabilities,
    switch_strategies,
)


class TestComputeStrategyFitness:
    def test_returns_all_strategies(self):
        result = compute_strategy_fitness(
            price=101.0, prev_price=100.0,
            fundamental=102.0, prev_fundamental=101.0,
            past_returns=np.array([0.01, 0.005]),
            params={'phi': 1.0, 'chi': 1.5},
        )
        assert set(result.keys()) == set(Strategy)

    def test_fundamentalist_profits_when_correct(self):
        """If fundamental > price and price rises, fundamentalist profits."""
        result = compute_strategy_fitness(
            price=105.0, prev_price=100.0,
            fundamental=110.0, prev_fundamental=110.0,
            past_returns=np.array([0.01]),
            params={'phi': 1.0, 'chi': 1.5},
        )
        assert result[Strategy.FUNDAMENTALIST] > 0

    def test_noise_fitness_is_zero(self):
        result = compute_strategy_fitness(
            price=105.0, prev_price=100.0,
            fundamental=110.0, prev_fundamental=110.0,
            past_returns=np.array([0.01]),
            params={'phi': 1.0, 'chi': 1.5},
        )
        assert result[Strategy.NOISE] == 0.0

    def test_zero_price_returns_zeros(self):
        result = compute_strategy_fitness(
            price=0.0, prev_price=100.0,
            fundamental=100.0, prev_fundamental=100.0,
            past_returns=np.array([]),
            params={},
        )
        assert all(v == 0.0 for v in result.values())


class TestExponentialMAFitness:
    def test_ema_blending(self):
        current = {s: 1.0 for s in Strategy}
        prev = {s: 0.0 for s in Strategy}
        result = compute_exponential_ma_fitness(current, prev, alpha=0.5)
        for s in Strategy:
            assert result[s] == pytest.approx(0.5)

    def test_alpha_one_ignores_history(self):
        current = {s: 2.0 for s in Strategy}
        prev = {s: 10.0 for s in Strategy}
        result = compute_exponential_ma_fitness(current, prev, alpha=1.0)
        for s in Strategy:
            assert result[s] == pytest.approx(2.0)

    def test_alpha_zero_ignores_current(self):
        current = {s: 2.0 for s in Strategy}
        prev = {s: 10.0 for s in Strategy}
        result = compute_exponential_ma_fitness(current, prev, alpha=0.0)
        for s in Strategy:
            assert result[s] == pytest.approx(10.0)


class TestMultinomialLogit:
    def test_probabilities_sum_to_one(self):
        fitness = {Strategy.FUNDAMENTALIST: 0.1, Strategy.CHARTIST: 0.2,
                   Strategy.NOISE: -0.1}
        probs = multinomial_logit_probabilities(fitness, beta=1.0)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_beta_zero_gives_uniform(self):
        fitness = {Strategy.FUNDAMENTALIST: 100.0, Strategy.CHARTIST: -100.0,
                   Strategy.NOISE: 0.0}
        probs = multinomial_logit_probabilities(fitness, beta=0.0)
        for p in probs.values():
            assert p == pytest.approx(1.0 / 3, abs=1e-10)

    def test_high_beta_concentrates(self):
        fitness = {Strategy.FUNDAMENTALIST: 1.0, Strategy.CHARTIST: 0.0,
                   Strategy.NOISE: 0.0}
        probs = multinomial_logit_probabilities(fitness, beta=100.0)
        assert probs[Strategy.FUNDAMENTALIST] > 0.99

    def test_numerical_stability_large_values(self):
        """Should not overflow with large fitness values."""
        fitness = {Strategy.FUNDAMENTALIST: 1000.0, Strategy.CHARTIST: 999.0,
                   Strategy.NOISE: 998.0}
        probs = multinomial_logit_probabilities(fitness, beta=1.0)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(np.isfinite(p) for p in probs.values())


class TestSwitchStrategies:
    def _make_traders(self, n=100):
        model = ap.Model({'n_agents': n, 'steps': 1, 'phi': 1.0, 'chi': 1.5,
                          'noise_sigma': 0.05, 'chartist_memory': 10})
        model.setup()
        traders = [Trader(model) for _ in range(n)]
        for t in traders:
            t.setup()
            t.strategy = Strategy.NOISE
        return traders

    def test_all_switch_to_dominant(self):
        """With switch_prob=1.0 and one dominant strategy, most should switch."""
        traders = self._make_traders(200)
        probs = {Strategy.FUNDAMENTALIST: 0.98, Strategy.CHARTIST: 0.01,
                 Strategy.NOISE: 0.01}
        rng = np.random.default_rng(42)
        switch_strategies(traders, probs, rng, switch_prob=1.0)
        fund_count = sum(1 for t in traders if t.strategy == Strategy.FUNDAMENTALIST)
        assert fund_count > 150

    def test_preserves_trader_count(self):
        traders = self._make_traders(50)
        probs = {s: 1 / 3 for s in Strategy}
        rng = np.random.default_rng(42)
        switch_strategies(traders, probs, rng, switch_prob=1.0)
        assert len(traders) == 50

    def test_low_switch_prob_few_changes(self):
        """With low switch_prob, most agents should keep their strategy."""
        traders = self._make_traders(200)
        probs = {Strategy.FUNDAMENTALIST: 0.98, Strategy.CHARTIST: 0.01,
                 Strategy.NOISE: 0.01}
        rng = np.random.default_rng(42)
        switch_strategies(traders, probs, rng, switch_prob=0.05)
        # Most should still be NOISE (their initial strategy)
        noise_count = sum(1 for t in traders if t.strategy == Strategy.NOISE)
        assert noise_count > 150  # ~95% stay
