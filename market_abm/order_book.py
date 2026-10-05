"""Limit order book with price-time priority, partial fills and self-trade prevention."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Order:
    agent_id: int
    side: str          # "buy" or "sell"
    order_type: str    # "market" or "limit"
    price: float
    quantity: int = 1
    timestamp: int = 0
    seq: int = 0       # arrival sequence assigned when the order rests


@dataclass
class Trade:
    price: float
    buyer_id: int
    seller_id: int
    timestamp: int
    quantity: int = 1
    # The resting order this fill consumed, kept so void_trade can restore it.
    resting: Order | None = field(default=None, repr=False, compare=False)


# capacity(agent_id, side, price) -> largest quantity the agent can settle at
# that price (cash for a buy, inventory for a sell).
CapacityFn = Callable[[int, str, float], int]
# settle(trade) -> True if cash and inventory moved, False if it could not.
SettleFn = Callable[[Trade], bool]


class OrderBook:
    """Limit order book with price-time priority.

    Market orders sweep the opposite side until filled or the side is empty;
    unfilled market quantity is discarded. Limit orders that cross trade at
    the resting order's price and rest any unfilled remainder.

    An order never matches a resting order of the same agent: those are
    skipped and the next order in price-time priority is used. A limit order
    whose remainder would still cross an opposite resting order (only
    possible when that order is the agent's own, or the agent ran out of
    cash/inventory) is dropped rather than rested, so the book is never
    crossed.

    If ``capacity`` is given, each fill is limited to what both parties can
    settle at the trade price, and a resting order whose owner cannot settle
    at all is removed. If ``settle`` is given it is called after each fill;
    returning False voids that fill (restoring the resting order) and stops
    matching. That is a safety net, since ``capacity`` should prevent it.
    """

    def __init__(self, initial_price: float | None = None):
        self.bids: list[Order] = []
        self.asks: list[Order] = []
        self._initial_price = initial_price
        self.last_trade_price: float | None = initial_price
        self.trade_history: list[Trade] = []
        self.price_history: list[float | None] = []
        self.spread_history: list[float | None] = []
        self.volume_history: list[int] = []
        self._step_trades: list[Trade] = []
        self._seq = 0

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

    @staticmethod
    def _key(side: str):
        if side == "buy":
            return lambda o: (-o.price, o.timestamp, o.seq)
        return lambda o: (o.price, o.timestamp, o.seq)

    def _rest(self, order: Order):
        self._seq += 1
        order.seq = self._seq
        book = self.bids if order.side == "buy" else self.asks
        book.append(order)
        book.sort(key=self._key(order.side))

    @staticmethod
    def _crosses(order: Order, resting: Order) -> bool:
        if order.order_type == "market":
            return True
        if order.side == "buy":
            return order.price >= resting.price
        return order.price <= resting.price

    def submit_order(self, order: Order,
                     capacity: CapacityFn | None = None,
                     settle: SettleFn | None = None) -> list[Trade]:
        """Match an order and return the fills (possibly empty)."""
        book = self.asks if order.side == "buy" else self.bids
        remaining = order.quantity
        trades: list[Trade] = []

        for resting in list(book):
            if remaining <= 0:
                break
            if not self._crosses(order, resting):
                break  # sorted by priority: nothing further can cross
            if resting.agent_id == order.agent_id:
                continue  # self-trade prevention: skip own resting order
            qty = min(remaining, resting.quantity)
            if capacity is not None:
                if capacity(resting.agent_id, resting.side,
                            resting.price) <= 0:
                    book.remove(resting)  # owner can no longer settle it
                    continue
                own_cap = capacity(order.agent_id, order.side, resting.price)
                if own_cap <= 0:
                    break
                qty = min(qty,
                          own_cap,
                          capacity(resting.agent_id, resting.side,
                                   resting.price))
            buy = order.side == "buy"
            trade = Trade(price=resting.price,
                          buyer_id=order.agent_id if buy else resting.agent_id,
                          seller_id=resting.agent_id if buy else order.agent_id,
                          timestamp=order.timestamp, quantity=qty,
                          resting=resting)
            resting.quantity -= qty
            if resting.quantity <= 0:
                book.remove(resting)
            self._record_trade(trade)
            if settle is not None and not settle(trade):
                self.void_trade(trade)
                break
            trades.append(trade)
            remaining -= qty

        if order.order_type == "limit" and remaining > 0:
            if not any(self._crosses(order, o) for o in book):
                self._rest(Order(order.agent_id, order.side, "limit",
                                 order.price, remaining, order.timestamp))
        return trades

    def _record_trade(self, trade: Trade):
        self.last_trade_price = trade.price
        self.trade_history.append(trade)
        self._step_trades.append(trade)

    def void_trade(self, trade: Trade):
        """Remove a trade that could not be settled (the most recent one).

        Restores the consumed quantity to the resting order at its original
        priority, and last_trade_price to the previous executed trade (or the
        initial price), so unsettled trades never reach price or volume.
        """
        if not self.trade_history or self.trade_history[-1] is not trade:
            raise ValueError("can only void the most recent trade")
        self.trade_history.pop()
        if self._step_trades and self._step_trades[-1] is trade:
            self._step_trades.pop()
        resting = trade.resting
        if resting is not None:
            book = self.bids if resting.side == "buy" else self.asks
            resting.quantity += trade.quantity
            if not any(o is resting for o in book):
                book.append(resting)
                book.sort(key=self._key(resting.side))
        self.last_trade_price = (self.trade_history[-1].price
                                 if self.trade_history
                                 else self._initial_price)

    def cancel_stale_orders(self, current_step: int, max_age: int):
        self.bids = [o for o in self.bids
                     if current_step - o.timestamp <= max_age]
        self.asks = [o for o in self.asks
                     if current_step - o.timestamp <= max_age]

    def end_step(self):
        self.price_history.append(self.last_trade_price)
        self.spread_history.append(self.spread)
        self.volume_history.append(sum(t.quantity for t in self._step_trades))
        self._step_trades = []
