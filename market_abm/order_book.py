"""Simple limit order book with top-of-book matching."""

from dataclasses import dataclass


@dataclass
class Order:
    agent_id: int
    side: str          # "buy" or "sell"
    order_type: str    # "market" or "limit"
    price: float
    quantity: int = 1
    timestamp: int = 0


@dataclass
class Trade:
    price: float
    buyer_id: int
    seller_id: int
    timestamp: int


class OrderBook:
    """Top-of-book limit order book with price-time priority.

    Supports market and limit orders. Market orders execute immediately
    against the best resting quote. Limit orders that cross the spread
    execute at the resting order's price. Non-crossing limits rest on
    the book.
    """

    def __init__(self, initial_price: float | None = None):
        self.bids: list[Order] = []
        self.asks: list[Order] = []
        self.last_trade_price: float | None = initial_price
        self.trade_history: list[Trade] = []
        self.price_history: list[float | None] = []
        self.spread_history: list[float | None] = []
        self.volume_history: list[int] = []
        self._step_trades: list[Trade] = []

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def submit_order(self, order: Order) -> Trade | None:
        if order.order_type == "market":
            return self._execute_market_order(order)
        else:
            return self._submit_limit_order(order)

    def _execute_market_order(self, order: Order) -> Trade | None:
        if order.side == "buy":
            if not self.asks:
                return None
            resting = self.asks.pop(0)
            trade = Trade(price=resting.price, buyer_id=order.agent_id,
                          seller_id=resting.agent_id,
                          timestamp=order.timestamp)
        else:
            if not self.bids:
                return None
            resting = self.bids.pop(0)
            trade = Trade(price=resting.price, buyer_id=resting.agent_id,
                          seller_id=order.agent_id,
                          timestamp=order.timestamp)
        self._record_trade(trade)
        return trade

    def _submit_limit_order(self, order: Order) -> Trade | None:
        if order.side == "buy":
            if self.asks and order.price >= self.asks[0].price:
                resting = self.asks.pop(0)
                trade = Trade(price=resting.price, buyer_id=order.agent_id,
                              seller_id=resting.agent_id,
                              timestamp=order.timestamp)
                self._record_trade(trade)
                return trade
            else:
                self.bids.append(order)
                self.bids.sort(key=lambda o: (-o.price, o.timestamp))
                return None
        else:
            if self.bids and order.price <= self.bids[0].price:
                resting = self.bids.pop(0)
                trade = Trade(price=resting.price, buyer_id=resting.agent_id,
                              seller_id=order.agent_id,
                              timestamp=order.timestamp)
                self._record_trade(trade)
                return trade
            else:
                self.asks.append(order)
                self.asks.sort(key=lambda o: (o.price, o.timestamp))
                return None

    def _record_trade(self, trade: Trade):
        self.last_trade_price = trade.price
        self.trade_history.append(trade)
        self._step_trades.append(trade)

    def cancel_stale_orders(self, current_step: int, max_age: int):
        self.bids = [o for o in self.bids
                     if current_step - o.timestamp <= max_age]
        self.asks = [o for o in self.asks
                     if current_step - o.timestamp <= max_age]

    def end_step(self):
        self.price_history.append(self.last_trade_price)
        self.spread_history.append(self.spread)
        self.volume_history.append(len(self._step_trades))
        self._step_trades = []
