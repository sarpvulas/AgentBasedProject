"""Tests for MarketMaker."""

import numpy as np
import pytest

from market_abm.market_maker import MarketMaker


class TestMarketMaker:
    def test_initial_state(self):
        mm = MarketMaker(100.0, 0.01)
        assert mm.price == 100.0
        assert mm.price_history == [100.0]
        assert mm.return_history == []

    def test_positive_demand_raises_price(self):
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        mm.update_price(aggregate_demand=100.0, n_agents=100)
        assert mm.price > 100.0

    def test_negative_demand_lowers_price(self):
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        mm.update_price(aggregate_demand=-100.0, n_agents=100)
        assert mm.price < 100.0

    def test_zero_demand_no_change(self):
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        mm.update_price(aggregate_demand=0.0, n_agents=100)
        assert mm.price == pytest.approx(100.0)

    def test_price_always_positive(self):
        """Even with extreme negative demand, price stays positive."""
        rng = np.random.default_rng(42)
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.001, rng=rng)
        for _ in range(500):
            mm.update_price(aggregate_demand=-1000.0, n_agents=100)
        assert mm.price > 0

    def test_history_tracking(self):
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        mm.update_price(10.0, 100)
        mm.update_price(-5.0, 100)
        assert len(mm.price_history) == 3
        assert len(mm.return_history) == 2

    def test_get_past_returns(self):
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        for d in [10, -5, 20, -15, 8]:
            mm.update_price(d, 100)
        last3 = mm.get_past_returns(3)
        assert len(last3) == 3
        np.testing.assert_array_equal(last3, mm.return_history[-3:])

    def test_get_past_returns_empty(self):
        mm = MarketMaker(100.0, 0.01)
        result = mm.get_past_returns(5)
        assert len(result) == 0

    def test_get_past_returns_short_history(self):
        mm = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        mm.update_price(10.0, 100)
        result = mm.get_past_returns(5)
        assert len(result) == 1

    def test_log_price(self):
        mm = MarketMaker(100.0, 0.01)
        assert mm.log_price == pytest.approx(np.log(100.0))

    def test_lambda_scaling(self):
        """Larger lambda → larger price impact."""
        mm1 = MarketMaker(100.0, 0.01, noise_sigma=0.0)
        mm2 = MarketMaker(100.0, 0.05, noise_sigma=0.0)
        mm1.update_price(50.0, 100)
        mm2.update_price(50.0, 100)
        # mm2 should have moved further from 100
        assert abs(mm2.price - 100.0) > abs(mm1.price - 100.0)
