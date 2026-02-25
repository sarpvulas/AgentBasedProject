"""Integration tests for MarketModel."""

import numpy as np
import pytest

from market_abm.agents import AgentType
from market_abm.config import DEFAULT_PARAMS
from market_abm.model import MarketModel


class TestMarketModelSetup:
    def test_creates_correct_agent_count(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 5, 'n_agents': 50})
        model.run()
        assert len(model.traders) == 50

    def test_initial_type_fractions(self):
        model = MarketModel({
            **DEFAULT_PARAMS, 'steps': 1, 'n_agents': 100,
            'frac_fundamental': 0.5, 'frac_trend': 0.3,
        })
        model.setup()
        counts = {}
        for t in model.traders:
            counts[t.agent_type] = counts.get(t.agent_type, 0) + 1
        assert counts[AgentType.FUNDAMENTAL] == 50
        assert counts[AgentType.TREND] == 30
        assert counts[AgentType.NOISE] == 20


class TestMarketModelRun:
    def test_smoke_test(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 100, 'n_agents': 30})
        results = model.run()
        assert results is not None

    def test_records_expected_columns(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 50, 'n_agents': 30})
        results = model.run()
        data = results.variables.MarketModel
        expected = {'price', 'fundamental', 'log_return', 'best_bid',
                    'best_ask', 'spread', 'volume', 'frac_fundamental',
                    'frac_trend', 'frac_noise', 'wealth_fundamental',
                    'wealth_trend', 'wealth_noise'}
        assert expected.issubset(set(data.columns))

    def test_price_stays_positive(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 200, 'n_agents': 50})
        results = model.run()
        prices = results.variables.MarketModel['price']
        assert (prices > 0).all()

    def test_wealth_conservation(self):
        """Total cash and inventory conserved across all agents."""
        params = {**DEFAULT_PARAMS, 'steps': 100, 'n_agents': 30}
        model = MarketModel(params)
        model.run()
        total_cash = sum(t.cash for t in model.traders)
        total_inv = sum(t.inventory for t in model.traders)
        assert total_cash == pytest.approx(
            params['n_agents'] * params['initial_cash'], abs=0.01)
        assert total_inv == params['n_agents'] * params['initial_inventory']

    def test_reproducibility(self):
        params = {**DEFAULT_PARAMS, 'steps': 50, 'n_agents': 30, 'seed': 42}
        m1 = MarketModel(params)
        r1 = m1.run()
        m2 = MarketModel(params)
        r2 = m2.run()
        p1 = r1.variables.MarketModel['price'].values
        p2 = r2.variables.MarketModel['price'].values
        np.testing.assert_array_equal(p1, p2)

    def test_reports_summary_stats(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 200, 'n_agents': 50})
        results = model.run()
        reporters = results.reporters
        assert 'kurtosis' in reporters.columns

    def test_intervention_schedule(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 50, 'n_agents': 30})
        model.schedule_intervention(25, -0.10)
        model.run()
        hist = model.fundamental.history
        assert len(hist) > 25

    def test_no_negative_cash_or_inventory(self):
        """Settlement validation prevents negative portfolios."""
        params = {
            **DEFAULT_PARAMS, 'steps': 200, 'n_agents': 30,
            'initial_cash': 500.0, 'stale_order_age': 20,
        }
        model = MarketModel(params)
        model.run()
        for t in model.traders:
            assert t.cash >= 0, f"Agent {t.id} has negative cash: {t.cash}"
            assert t.inventory >= 0, (
                f"Agent {t.id} has negative inventory: {t.inventory}")

    def test_invalid_fractions_raise_error(self):
        """frac_fundamental + frac_trend > 1 should raise ValueError."""
        params = {
            **DEFAULT_PARAMS, 'steps': 1, 'n_agents': 10,
            'frac_fundamental': 0.6, 'frac_trend': 0.6,
        }
        with pytest.raises(ValueError, match=r"> 1\.0"):
            model = MarketModel(params)
            model.run()
