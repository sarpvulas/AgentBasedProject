"""Tests for OrderBook."""

import pytest

from market_abm.order_book import Order, OrderBook, Trade


def submit(ob, order, **kw):
    """Submit and return the first fill (or None); qty-1 orders fill at most once."""
    trades = ob.submit_order(order, **kw)
    assert len(trades) <= 1
    return trades[0] if trades else None


class TestOrderBookBasics:
    def test_empty_book(self):
        ob = OrderBook()
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.spread is None
        assert ob.last_trade_price is None

    def test_initial_price(self):
        ob = OrderBook(initial_price=100.0)
        assert ob.last_trade_price == 100.0

    def test_add_limit_buy(self):
        ob = OrderBook()
        order = Order(agent_id=1, side="buy", order_type="limit",
                      price=99.0, timestamp=1)
        trade = submit(ob, order)
        assert trade is None
        assert ob.best_bid == 99.0

    def test_add_limit_sell(self):
        ob = OrderBook()
        order = Order(agent_id=1, side="sell", order_type="limit",
                      price=101.0, timestamp=1)
        trade = submit(ob, order)
        assert trade is None
        assert ob.best_ask == 101.0

    def test_spread_calculation(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="buy", order_type="limit",
                              price=99.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        assert ob.spread == pytest.approx(2.0)


class TestMarketOrders:
    def test_market_buy_executes_at_ask(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        trade = submit(ob, Order(agent_id=2, side="buy",
                                      order_type="market", price=0.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 101.0
        assert trade.buyer_id == 2
        assert trade.seller_id == 1

    def test_market_sell_executes_at_bid(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="buy", order_type="limit",
                              price=99.0, timestamp=1))
        trade = submit(ob, Order(agent_id=2, side="sell",
                                      order_type="market", price=0.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 99.0
        assert trade.buyer_id == 1
        assert trade.seller_id == 2

    def test_market_buy_no_asks_fails(self):
        ob = OrderBook()
        trade = submit(ob, Order(agent_id=1, side="buy",
                                      order_type="market", price=0.0,
                                      timestamp=1))
        assert trade is None

    def test_market_sell_no_bids_fails(self):
        ob = OrderBook()
        trade = submit(ob, Order(agent_id=1, side="sell",
                                      order_type="market", price=0.0,
                                      timestamp=1))
        assert trade is None


class TestLimitOrderCrossing:
    def test_limit_buy_crosses_ask(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=100.0, timestamp=1))
        trade = submit(ob, Order(agent_id=2, side="buy",
                                      order_type="limit", price=101.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 100.0

    def test_limit_sell_crosses_bid(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="buy", order_type="limit",
                              price=100.0, timestamp=1))
        trade = submit(ob, Order(agent_id=2, side="sell",
                                      order_type="limit", price=99.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 100.0

    def test_no_crossing_when_bid_below_ask(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        trade = submit(ob, Order(agent_id=2, side="buy",
                                      order_type="limit", price=99.0,
                                      timestamp=2))
        assert trade is None
        assert ob.best_bid == 99.0
        assert ob.best_ask == 101.0


class TestOrderPriority:
    def test_bids_sorted_by_price_descending(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="buy", order_type="limit",
                              price=98.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="buy", order_type="limit",
                              price=99.0, timestamp=2))
        submit(ob, Order(agent_id=3, side="buy", order_type="limit",
                              price=97.0, timestamp=3))
        assert ob.best_bid == 99.0

    def test_asks_sorted_by_price_ascending(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=103.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="sell", order_type="limit",
                              price=101.0, timestamp=2))
        submit(ob, Order(agent_id=3, side="sell", order_type="limit",
                              price=102.0, timestamp=3))
        assert ob.best_ask == 101.0

    def test_fifo_at_same_price(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=100.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="sell", order_type="limit",
                              price=100.0, timestamp=2))
        trade = submit(ob, Order(agent_id=3, side="buy",
                                      order_type="market", price=0.0,
                                      timestamp=3))
        assert trade.seller_id == 1


class TestStaleOrderCleanup:
    def test_cancel_stale_orders(self):
        ob = OrderBook()
        submit(ob, Order(agent_id=1, side="buy", order_type="limit",
                              price=99.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="sell", order_type="limit",
                              price=101.0, timestamp=5))
        ob.cancel_stale_orders(current_step=12, max_age=10)
        assert ob.best_bid is None
        assert ob.best_ask == 101.0


class TestEndStep:
    def test_records_history(self):
        ob = OrderBook(initial_price=100.0)
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="buy", order_type="market",
                              price=0.0, timestamp=1))
        ob.end_step()
        assert ob.price_history == [101.0]
        assert ob.volume_history == [1]

    def test_resets_step_trades(self):
        ob = OrderBook(initial_price=100.0)
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        submit(ob, Order(agent_id=2, side="buy", order_type="market",
                              price=0.0, timestamp=1))
        ob.end_step()
        ob.end_step()
        assert ob.volume_history == [1, 0]


class TestVoidTrade:
    def _cross(self, ob, price, ts):
        submit(ob, Order(agent_id=1, side="sell", order_type="limit",
                              price=price, timestamp=ts))
        return submit(ob, Order(agent_id=2, side="buy",
                                     order_type="market", price=0.0,
                                     timestamp=ts))

    def test_void_restores_initial_price_and_volume(self):
        ob = OrderBook(initial_price=100.0)
        trade = self._cross(ob, 105.0, 1)
        assert ob.last_trade_price == 105.0
        ob.void_trade(trade)
        assert ob.last_trade_price == 100.0
        assert ob.trade_history == []
        ob.end_step()
        assert ob.volume_history == [0]

    def test_void_restores_previous_trade_price(self):
        ob = OrderBook(initial_price=100.0)
        self._cross(ob, 101.0, 1)
        trade = self._cross(ob, 107.0, 2)
        ob.void_trade(trade)
        assert ob.last_trade_price == 101.0
        assert len(ob.trade_history) == 1

    def test_void_rejects_non_latest_trade(self):
        ob = OrderBook(initial_price=100.0)
        first = self._cross(ob, 101.0, 1)
        self._cross(ob, 102.0, 2)
        with pytest.raises(ValueError):
            ob.void_trade(first)


def O(agent, side, kind, price, qty, ts):
    return Order(agent_id=agent, side=side, order_type=kind, price=price,
                 quantity=qty, timestamp=ts)


class TestQuantityAndSelfTrade:
    def test_limit_crossing_only_own_order_is_dropped_not_rested(self):
        ob = OrderBook()
        ob.submit_order(O(1, "sell", "limit", 100.0, 1, 1))
        assert ob.submit_order(O(1, "buy", "limit", 101.0, 1, 2)) == []
        assert ob.bids == []          # would have crossed the book
        assert len(ob.asks) == 1

    def test_partial_fill_then_remainder_rests(self):
        ob = OrderBook()
        ob.submit_order(O(1, "sell", "limit", 100.0, 2, 1))
        trades = ob.submit_order(O(2, "buy", "limit", 100.0, 5, 2))
        assert [t.quantity for t in trades] == [2]
        assert [(o.agent_id, o.quantity) for o in ob.bids] == [(2, 3)]

    def test_market_remainder_is_discarded(self):
        ob = OrderBook()
        ob.submit_order(O(1, "sell", "limit", 100.0, 2, 1))
        trades = ob.submit_order(O(2, "buy", "market", 0.0, 5, 2))
        assert sum(t.quantity for t in trades) == 2
        assert ob.bids == [] and ob.asks == []

    def test_capacity_limits_fill_and_drops_unfundable_resting_order(self):
        ob = OrderBook()
        ob.submit_order(O(1, "sell", "limit", 100.0, 3, 1))   # owner has 0
        ob.submit_order(O(3, "sell", "limit", 101.0, 3, 1))
        cap = {1: 0, 2: 2, 3: 10}
        trades = ob.submit_order(
            O(2, "buy", "market", 0.0, 5, 2),
            capacity=lambda agent, side, price: cap[agent])
        assert [(t.seller_id, t.quantity) for t in trades] == [(3, 2)]
        assert [(o.agent_id, o.quantity) for o in ob.asks] == [(3, 1)]

    def test_void_restores_partial_quantity_at_original_priority(self):
        ob = OrderBook(initial_price=100.0)
        ob.submit_order(O(1, "sell", "limit", 100.0, 3, 1))
        ob.submit_order(O(2, "sell", "limit", 100.0, 3, 1))
        trades = ob.submit_order(O(3, "buy", "market", 0.0, 3, 2),
                                 settle=lambda t: False)
        assert trades == []
        assert [(o.agent_id, o.quantity) for o in ob.asks] == [(1, 3), (2, 3)]
        assert ob.trade_history == []
