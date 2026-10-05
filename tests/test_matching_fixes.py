"""Model-level tests for self-trade prevention, quantity-aware partial fills
and aggressor capacity pre-checks.

The scenarios drive the real MarketModel.step() with scripted agent orders and
only read public state (order book lists, trade_history, agent portfolios),
so they run unchanged against the pre-fix code.
"""

import pytest

from market_abm.config import DEFAULT_PARAMS
from market_abm.model import MarketModel
from market_abm.order_book import Order


def make_model(n_agents=4, **over):
    model = MarketModel({**DEFAULT_PARAMS, 'steps': 1, 'n_agents': n_agents,
                         **over})
    model.setup()
    return model


def step_with(model, orders, t=0):
    """Run one model step in which each agent submits its scripted order."""
    model.t = t
    for trader in model.traders:
        order = orders.get(trader.id)
        trader.decide = lambda *a, _o=order, **k: _o
    model.step()


def replay(model):
    p = model.p
    cash = {t.id: p['initial_cash'] for t in model.traders}
    inv = {t.id: p['initial_inventory'] for t in model.traders}
    for tr in model.order_book.trade_history:
        q = getattr(tr, 'quantity', 1)
        cash[tr.buyer_id] -= tr.price * q
        cash[tr.seller_id] += tr.price * q
        inv[tr.buyer_id] += q
        inv[tr.seller_id] -= q
    for t in model.traders:
        assert t.cash == pytest.approx(cash[t.id])
        assert t.inventory == inv[t.id]


class TestSelfTrades:
    def test_agent_does_not_match_its_own_resting_ask(self):
        m = make_model()
        a, b = m.traders[0], m.traders[1]
        step_with(m, {a.id: Order(a.id, "sell", "limit", 100.0, 1, 0)}, t=0)
        step_with(m, {b.id: Order(b.id, "sell", "limit", 101.0, 1, 1)}, t=1)
        step_with(m, {a.id: Order(a.id, "buy", "market", 0.0, 1, 2)}, t=2)
        trades = m.order_book.trade_history
        assert [(t.buyer_id, t.seller_id, t.price) for t in trades] == [
            (a.id, b.id, 101.0)]
        # a's own ask keeps its place on the book
        assert [o.agent_id for o in m.order_book.asks] == [a.id]

    def test_priority_kept_for_other_agents(self):
        m = make_model()
        a, b, c, d = m.traders
        step_with(m, {b.id: Order(b.id, "sell", "limit", 100.0, 1, 0)}, t=0)
        step_with(m, {c.id: Order(c.id, "sell", "limit", 100.0, 1, 1)}, t=1)
        step_with(m, {d.id: Order(d.id, "buy", "market", 0.0, 1, 2)}, t=2)
        trade = m.order_book.trade_history[0]
        assert trade.seller_id == b.id  # earlier arrival at the same price

    def test_no_self_trades_over_a_full_run(self):
        for seed in (1, 42):
            model = MarketModel({**DEFAULT_PARAMS, 'seed': seed})
            model.run()
            same = [t for t in model.order_book.trade_history
                    if t.buyer_id == t.seller_id]
            assert same == [], f"seed {seed}: {len(same)} self-trades"


class TestPartialFills:
    def test_market_buy_takes_requested_quantity(self):
        m = make_model()
        a, b = m.traders[0], m.traders[1]
        step_with(m, {a.id: Order(a.id, "sell", "limit", 100.0, 5, 0)}, t=0)
        step_with(m, {b.id: Order(b.id, "buy", "market", 0.0, 3, 1)}, t=1)
        assert b.inventory == DEFAULT_PARAMS['initial_inventory'] + 3
        assert a.inventory == DEFAULT_PARAMS['initial_inventory'] - 3
        assert b.cash == pytest.approx(DEFAULT_PARAMS['initial_cash'] - 300)
        assert [o.quantity for o in m.order_book.asks] == [2]
        replay(m)

    def test_sweep_across_levels_and_volume_in_units(self):
        m = make_model()
        a, b, c = m.traders[:3]
        step_with(m, {a.id: Order(a.id, "sell", "limit", 100.0, 2, 0)}, t=0)
        step_with(m, {b.id: Order(b.id, "sell", "limit", 101.0, 2, 1)}, t=1)
        step_with(m, {c.id: Order(c.id, "buy", "market", 0.0, 3, 2)}, t=2)
        assert c.inventory == DEFAULT_PARAMS['initial_inventory'] + 3
        assert c.cash == pytest.approx(
            DEFAULT_PARAMS['initial_cash'] - (200 + 101))
        assert m.order_book.last_trade_price == 101.0
        assert m.order_book.volume_history[-1] == 3
        replay(m)

    def test_limit_remainder_rests(self):
        m = make_model()
        a, b = m.traders[0], m.traders[1]
        step_with(m, {a.id: Order(a.id, "sell", "limit", 100.0, 2, 0)}, t=0)
        step_with(m, {b.id: Order(b.id, "buy", "limit", 100.0, 5, 1)}, t=1)
        assert b.inventory == DEFAULT_PARAMS['initial_inventory'] + 2
        assert [(o.agent_id, o.quantity) for o in m.order_book.bids] == [
            (b.id, 3)]
        assert m.order_book.asks == []
        replay(m)

    @pytest.mark.parametrize('seed', [1, 7])
    def test_replay_with_quantity_greater_than_one(self, seed):
        # Small endowments force capacity-limited (partial) fills.
        model = MarketModel({**DEFAULT_PARAMS, 'seed': seed, 'steps': 1500,
                             'order_size': 5, 'initial_cash': 900.0,
                             'initial_inventory': 3})
        model.run()
        trades = model.order_book.trade_history
        assert any(t.quantity > 1 for t in trades)
        assert any(t.quantity < 5 for t in trades)
        assert all(t.buyer_id != t.seller_id for t in trades)
        replay(model)
        for t in model.traders:
            assert t.cash >= -1e-6
            assert t.inventory >= 0
        assert sum(t.inventory for t in model.traders) == 100 * 3


class TestAggressorCapacity:
    def test_broke_buyer_does_not_drop_resting_ask(self):
        m = make_model()
        broke, seller = m.traders[0], m.traders[1]
        broke.cash = 120.0   # can afford the last price (100), not the ask
        step_with(m, {seller.id: Order(seller.id, "sell", "limit", 150.0, 1,
                                       0)}, t=0)
        step_with(m, {broke.id: Order(broke.id, "buy", "market", 0.0, 1, 1)},
                  t=1)
        assert [(o.agent_id, o.price) for o in m.order_book.asks] == [
            (seller.id, 150.0)]
        assert m.order_book.trade_history == []
        assert m.order_book.last_trade_price == DEFAULT_PARAMS[
            'fundamental_initial']
        assert broke.cash == 120.0

    def test_broke_buyer_takes_what_it_can_afford(self):
        m = make_model()
        buyer, seller = m.traders[0], m.traders[1]
        buyer.cash = 250.0
        step_with(m, {seller.id: Order(seller.id, "sell", "limit", 100.0, 5,
                                       0)}, t=0)
        step_with(m, {buyer.id: Order(buyer.id, "buy", "market", 0.0, 5, 1)},
                  t=1)
        assert buyer.inventory == DEFAULT_PARAMS['initial_inventory'] + 2
        assert buyer.cash == pytest.approx(50.0)
        assert [o.quantity for o in m.order_book.asks] == [3]

    def test_no_unsettled_trades_over_full_runs(self):
        # Seeds 1 and 2 had voided trades before the pre-check.
        for seed in (1, 2):
            model = MarketModel({**DEFAULT_PARAMS, 'seed': seed})
            model.run()
            assert model.n_unsettled == 0
