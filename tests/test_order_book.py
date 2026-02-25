"""Tests for OrderBook."""

import pytest

from market_abm.order_book import Order, OrderBook, Trade


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
        trade = ob.submit_order(order)
        assert trade is None
        assert ob.best_bid == 99.0

    def test_add_limit_sell(self):
        ob = OrderBook()
        order = Order(agent_id=1, side="sell", order_type="limit",
                      price=101.0, timestamp=1)
        trade = ob.submit_order(order)
        assert trade is None
        assert ob.best_ask == 101.0

    def test_spread_calculation(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="buy", order_type="limit",
                              price=99.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        assert ob.spread == pytest.approx(2.0)


class TestMarketOrders:
    def test_market_buy_executes_at_ask(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        trade = ob.submit_order(Order(agent_id=2, side="buy",
                                      order_type="market", price=0.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 101.0
        assert trade.buyer_id == 2
        assert trade.seller_id == 1

    def test_market_sell_executes_at_bid(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="buy", order_type="limit",
                              price=99.0, timestamp=1))
        trade = ob.submit_order(Order(agent_id=2, side="sell",
                                      order_type="market", price=0.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 99.0
        assert trade.buyer_id == 1
        assert trade.seller_id == 2

    def test_market_buy_no_asks_fails(self):
        ob = OrderBook()
        trade = ob.submit_order(Order(agent_id=1, side="buy",
                                      order_type="market", price=0.0,
                                      timestamp=1))
        assert trade is None

    def test_market_sell_no_bids_fails(self):
        ob = OrderBook()
        trade = ob.submit_order(Order(agent_id=1, side="sell",
                                      order_type="market", price=0.0,
                                      timestamp=1))
        assert trade is None


class TestLimitOrderCrossing:
    def test_limit_buy_crosses_ask(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=100.0, timestamp=1))
        trade = ob.submit_order(Order(agent_id=2, side="buy",
                                      order_type="limit", price=101.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 100.0

    def test_limit_sell_crosses_bid(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="buy", order_type="limit",
                              price=100.0, timestamp=1))
        trade = ob.submit_order(Order(agent_id=2, side="sell",
                                      order_type="limit", price=99.0,
                                      timestamp=2))
        assert trade is not None
        assert trade.price == 100.0

    def test_no_crossing_when_bid_below_ask(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        trade = ob.submit_order(Order(agent_id=2, side="buy",
                                      order_type="limit", price=99.0,
                                      timestamp=2))
        assert trade is None
        assert ob.best_bid == 99.0
        assert ob.best_ask == 101.0


class TestOrderPriority:
    def test_bids_sorted_by_price_descending(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="buy", order_type="limit",
                              price=98.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="buy", order_type="limit",
                              price=99.0, timestamp=2))
        ob.submit_order(Order(agent_id=3, side="buy", order_type="limit",
                              price=97.0, timestamp=3))
        assert ob.best_bid == 99.0

    def test_asks_sorted_by_price_ascending(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=103.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="sell", order_type="limit",
                              price=101.0, timestamp=2))
        ob.submit_order(Order(agent_id=3, side="sell", order_type="limit",
                              price=102.0, timestamp=3))
        assert ob.best_ask == 101.0

    def test_fifo_at_same_price(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=100.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="sell", order_type="limit",
                              price=100.0, timestamp=2))
        trade = ob.submit_order(Order(agent_id=3, side="buy",
                                      order_type="market", price=0.0,
                                      timestamp=3))
        assert trade.seller_id == 1


class TestStaleOrderCleanup:
    def test_cancel_stale_orders(self):
        ob = OrderBook()
        ob.submit_order(Order(agent_id=1, side="buy", order_type="limit",
                              price=99.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="sell", order_type="limit",
                              price=101.0, timestamp=5))
        ob.cancel_stale_orders(current_step=12, max_age=10)
        assert ob.best_bid is None
        assert ob.best_ask == 101.0


class TestEndStep:
    def test_records_history(self):
        ob = OrderBook(initial_price=100.0)
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="buy", order_type="market",
                              price=0.0, timestamp=1))
        ob.end_step()
        assert ob.price_history == [101.0]
        assert ob.volume_history == [1]

    def test_resets_step_trades(self):
        ob = OrderBook(initial_price=100.0)
        ob.submit_order(Order(agent_id=1, side="sell", order_type="limit",
                              price=101.0, timestamp=1))
        ob.submit_order(Order(agent_id=2, side="buy", order_type="market",
                              price=0.0, timestamp=1))
        ob.end_step()
        ob.end_step()
        assert ob.volume_history == [1, 0]
