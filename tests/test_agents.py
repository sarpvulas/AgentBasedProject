"""Tests for Trader agents."""

import numpy as np
import pytest

import agentpy as ap

from market_abm.agents import AgentType, Trader


def _make_trader(agent_type=AgentType.NOISE, **params):
    """Helper: create a single Trader within a minimal model."""
    defaults = {
        'initial_cash': 10000.0, 'initial_inventory': 10,
        'fundamental_initial': 100.0, 'fundamental_sensitivity': 1.0,
        'trend_threshold': 0.0, 'n_agents': 1, 'steps': 1,
    }
    defaults.update(params)
    model = ap.Model(defaults)
    model.setup()
    trader = Trader(model)
    trader.setup()
    trader.agent_type = agent_type
    return trader


class TestTraderSetup:
    def test_initial_portfolio(self):
        trader = _make_trader()
        assert trader.cash == 10000.0
        assert trader.inventory == 10
        assert trader.initial_wealth == 11000.0


class TestNoiseTrader:
    def test_returns_order_or_none(self):
        trader = _make_trader(AgentType.NOISE)
        rng = np.random.default_rng(42)
        results = [trader.decide(100.0, 99.0, 100.0, 99.0, 101.0, 1, rng)
                   for _ in range(100)]
        orders = [r for r in results if r is not None]
        nones = [r for r in results if r is None]
        assert len(orders) > 0
        assert len(nones) > 0

    def test_cannot_sell_with_zero_inventory(self):
        trader = _make_trader(AgentType.NOISE)
        trader.inventory = 0
        rng = np.random.default_rng(42)
        for _ in range(200):
            order = trader.decide(100.0, 99.0, 100.0, 99.0, 101.0, 1, rng)
            if order is not None:
                assert order.side != "sell"

    def test_cannot_buy_without_cash(self):
        trader = _make_trader(AgentType.NOISE)
        trader.cash = 0.0
        rng = np.random.default_rng(42)
        for _ in range(200):
            order = trader.decide(100.0, 99.0, 100.0, 99.0, 101.0, 1, rng)
            if order is not None:
                assert order.side != "buy"


class TestFundamentalTrader:
    def test_buys_when_undervalued(self):
        trader = _make_trader(AgentType.FUNDAMENTAL,
                              fundamental_sensitivity=10.0)
        rng = np.random.default_rng(42)
        buys = sum(1 for _ in range(100)
                   if (o := trader.decide(90.0, 89.0, 100.0, 89.0, 91.0,
                                          1, rng)) is not None
                   and o.side == "buy")
        assert buys > 50

    def test_sells_when_overvalued(self):
        trader = _make_trader(AgentType.FUNDAMENTAL,
                              fundamental_sensitivity=10.0)
        rng = np.random.default_rng(42)
        sells = sum(1 for _ in range(100)
                    if (o := trader.decide(110.0, 109.0, 100.0, 109.0, 111.0,
                                           1, rng)) is not None
                    and o.side == "sell")
        assert sells > 50

    def test_prefers_limit_orders(self):
        trader = _make_trader(AgentType.FUNDAMENTAL,
                              fundamental_sensitivity=10.0)
        rng = np.random.default_rng(42)
        limits, total = 0, 0
        for _ in range(200):
            o = trader.decide(90.0, 89.0, 100.0, 89.0, 91.0, 1, rng)
            if o is not None:
                total += 1
                if o.order_type == "limit":
                    limits += 1
        assert limits / total > 0.6


class TestTrendTrader:
    def test_buys_on_uptrend(self):
        trader = _make_trader(AgentType.TREND)
        rng = np.random.default_rng(42)
        order = None
        for _ in range(50):
            order = trader.decide(101.0, 100.0, 100.0, 99.0, 102.0, 1, rng)
            if order is not None:
                break
        assert order is not None
        assert order.side == "buy"

    def test_sells_on_downtrend(self):
        trader = _make_trader(AgentType.TREND)
        rng = np.random.default_rng(42)
        order = None
        for _ in range(50):
            order = trader.decide(99.0, 100.0, 100.0, 98.0, 100.0, 1, rng)
            if order is not None:
                break
        assert order is not None
        assert order.side == "sell"

    def test_holds_when_flat(self):
        trader = _make_trader(AgentType.TREND, trend_threshold=0.5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            order = trader.decide(100.1, 100.0, 100.0, 99.0, 101.0, 1, rng)
            assert order is None

    def test_prefers_market_orders(self):
        trader = _make_trader(AgentType.TREND)
        rng = np.random.default_rng(42)
        markets, total = 0, 0
        for _ in range(200):
            o = trader.decide(102.0, 100.0, 100.0, 99.0, 103.0, 1, rng)
            if o is not None:
                total += 1
                if o.order_type == "market":
                    markets += 1
        assert markets / total > 0.6
