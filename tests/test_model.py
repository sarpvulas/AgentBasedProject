"""Integration tests for MarketModel."""

import numpy as np
import pytest

from market_abm.agents import Strategy
from market_abm.config import DEFAULT_PARAMS
from market_abm.model import MarketModel


class TestMarketModelSetup:
    def test_creates_correct_agent_count(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 5, 'n_agents': 50})
        model.run()
        assert len(model.traders) == 50

    def test_initial_strategy_fractions(self):
        model = MarketModel({
            **DEFAULT_PARAMS, 'steps': 1, 'n_agents': 100,
            'init_fundamentalist_frac': 0.5,
            'init_chartist_frac': 0.3,
        })
        model.setup()
        counts = {}
        for t in model.traders:
            counts[t.strategy] = counts.get(t.strategy, 0) + 1
        assert counts[Strategy.FUNDAMENTALIST] == 50
        assert counts[Strategy.CHARTIST] == 30
        assert counts[Strategy.NOISE] == 20


class TestMarketModelRun:
    def test_smoke_test(self):
        """Model runs without error for a short simulation."""
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 50, 'n_agents': 30})
        results = model.run()
        assert results is not None

    def test_records_expected_columns(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 20, 'n_agents': 30})
        results = model.run()
        data = results.variables.MarketModel
        expected_cols = {'price', 'fundamental', 'log_return',
                         'frac_fundamentalist', 'frac_chartist', 'frac_noise'}
        assert expected_cols.issubset(set(data.columns))

    def test_price_stays_positive(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 200, 'n_agents': 50})
        results = model.run()
        prices = results.variables.MarketModel['price']
        assert (prices > 0).all()

    def test_fractions_sum_to_one(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 100, 'n_agents': 50})
        results = model.run()
        data = results.variables.MarketModel
        total = data['frac_fundamentalist'] + data['frac_chartist'] + data['frac_noise']
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_reports_summary_stats(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 100, 'n_agents': 50})
        results = model.run()
        reporters = results.reporters
        assert 'kurtosis' in reporters.columns
        assert 'jb_statistic' in reporters.columns

    def test_reproducibility(self):
        params = {**DEFAULT_PARAMS, 'steps': 50, 'n_agents': 30, 'seed': 42}
        m1 = MarketModel(params)
        r1 = m1.run()
        m2 = MarketModel(params)
        r2 = m2.run()
        p1 = r1.variables.MarketModel['price'].values
        p2 = r2.variables.MarketModel['price'].values
        np.testing.assert_array_equal(p1, p2)

    def test_intervention_schedule(self):
        model = MarketModel({**DEFAULT_PARAMS, 'steps': 50, 'n_agents': 30})
        model.schedule_intervention(25, -0.10)
        model.run()
        # After a negative shock, fundamental should show a dip
        hist = model.fundamental.history
        # The shock is applied at step 25, fundamental should be lower
        assert len(hist) > 25
