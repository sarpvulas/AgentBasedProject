"""Tests for Trader agents."""

import numpy as np
import pytest

import agentpy as ap

from market_abm.agents import Strategy, Trader


def _make_trader(strategy=Strategy.FUNDAMENTALIST, **params):
    """Helper: create a single Trader within a minimal model."""
    defaults = {
        'phi': 1.0, 'chi': 1.5, 'noise_sigma': 0.05,
        'chartist_memory': 10, 'n_agents': 1, 'steps': 1,
    }
    defaults.update(params)
    model = ap.Model(defaults)
    model.setup()  # initializes model.random
    trader = Trader(model)
    trader.setup()
    trader.strategy = strategy
    return trader


class TestStrategy:
    def test_enum_members(self):
        assert Strategy.FUNDAMENTALIST.value != Strategy.CHARTIST.value
        assert len(Strategy) == 3


class TestTrader:
    def test_fundamentalist_demand_undervalued(self):
        trader = _make_trader(Strategy.FUNDAMENTALIST, phi=1.0)
        # fundamental > price → positive demand
        d = trader.compute_demand(price=90.0, fundamental=100.0,
                                  past_returns=np.array([]))
        assert d > 0

    def test_fundamentalist_demand_overvalued(self):
        trader = _make_trader(Strategy.FUNDAMENTALIST, phi=1.0)
        d = trader.compute_demand(price=110.0, fundamental=100.0,
                                  past_returns=np.array([]))
        assert d < 0

    def test_fundamentalist_demand_at_fair_value(self):
        trader = _make_trader(Strategy.FUNDAMENTALIST, phi=1.0)
        d = trader.compute_demand(price=100.0, fundamental=100.0,
                                  past_returns=np.array([]))
        assert d == pytest.approx(0.0)

    def test_chartist_demand_positive_trend(self):
        trader = _make_trader(Strategy.CHARTIST, chi=1.5)
        returns = np.array([0.01, 0.02, 0.01])
        d = trader.compute_demand(price=100.0, fundamental=100.0,
                                  past_returns=returns)
        assert d > 0

    def test_chartist_demand_negative_trend(self):
        trader = _make_trader(Strategy.CHARTIST, chi=1.5)
        returns = np.array([-0.02, -0.01, -0.03])
        d = trader.compute_demand(price=100.0, fundamental=100.0,
                                  past_returns=returns)
        assert d < 0

    def test_chartist_demand_empty_returns(self):
        trader = _make_trader(Strategy.CHARTIST)
        d = trader.compute_demand(price=100.0, fundamental=100.0,
                                  past_returns=np.array([]))
        assert d == 0.0

    def test_noise_demand_is_stochastic(self):
        trader = _make_trader(Strategy.NOISE, noise_sigma=0.05)
        demands = [trader.compute_demand(100.0, 100.0, np.array([])) for _ in range(100)]
        assert np.std(demands) > 0

    def test_phi_scaling(self):
        t1 = _make_trader(Strategy.FUNDAMENTALIST, phi=1.0)
        t2 = _make_trader(Strategy.FUNDAMENTALIST, phi=2.0)
        d1 = t1.compute_demand(90.0, 100.0, np.array([]))
        d2 = t2.compute_demand(90.0, 100.0, np.array([]))
        assert d2 == pytest.approx(2.0 * d1)
