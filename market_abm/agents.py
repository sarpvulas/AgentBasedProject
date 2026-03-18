"""Trader agents with heterogeneous strategies and portfolios."""

from enum import Enum, auto

import agentpy as ap
import numpy as np

from .order_book import Order


class AgentType(Enum):
    FUNDAMENTAL = auto()
    TREND = auto()
    NOISE = auto()



class Trader(ap.Agent):
    """Single trader that decides buy/sell/hold and submits orders."""

    def setup(self):
        p = self.model.p
        self.agent_type: AgentType = AgentType.NOISE
        self.cash: float = p.get('initial_cash', 10000.0)
        self.inventory: int = p.get('initial_inventory', 10)
        init_price = p.get('fundamental_initial', 100.0)
        self.initial_wealth: float = self.cash + self.inventory * init_price
        self._ema_vol: float = 0.0  # smoothed realized volatility

    def decide(self, price: float, prev_price: float | None,
               fundamental: float, best_bid: float | None,
               best_ask: float | None, step: int,
               rng: np.random.Generator) -> Order | None:
        """Return an Order or None (hold)."""
        if self.agent_type == AgentType.NOISE:
            return self._noise_decide(price, best_bid, best_ask, step, rng)
        elif self.agent_type == AgentType.FUNDAMENTAL:
            return self._fundamental_decide(price, fundamental, step, rng)
        else:
            return self._trend_decide(price, prev_price, step, rng)

    def _noise_decide(self, price, best_bid, best_ask, step, rng):
        roll = rng.random()
        if roll < 0.3:
            side = "buy"
        elif roll < 0.6:
            side = "sell"
        else:
            return None

        if side == "buy" and self.cash < price:
            return None
        if side == "sell" and self.inventory <= 0:
            return None

        if rng.random() < 0.3:
            return Order(agent_id=self.id, side=side, order_type="market",
                         price=0.0, quantity=1, timestamp=step)
        else:
            spread = ((best_ask - best_bid)
                      if (best_bid is not None and best_ask is not None)
                      else 1.0)
            spread = max(spread, 0.01)
            jitter = rng.uniform(-spread, spread)
            limit_price = max(0.01, price + jitter)
            return Order(agent_id=self.id, side=side, order_type="limit",
                         price=round(limit_price, 2), quantity=1,
                         timestamp=step)

    def _fundamental_decide(self, price, fundamental, step, rng):
        if price <= 0:
            return None
        mispricing = (fundamental - price) / price
        sensitivity = self.model.p.get('fundamental_sensitivity', 1.0)
        action_prob = min(abs(mispricing) * sensitivity, 1.0)

        if rng.random() > action_prob:
            return None

        side = "buy" if mispricing > 0 else "sell"

        if side == "buy" and self.cash < price:
            return None
        if side == "sell" and self.inventory <= 0:
            return None

        # All limit orders (no market) — fundamental agents provide
        # liquidity rather than demanding it. This prevents instant
        # mean-reversion bounces that destroy volatility clustering.
        if side == "buy":
            limit_price = price + rng.uniform(0, 1) * (fundamental - price)
        else:
            limit_price = price - rng.uniform(0, 1) * (price - fundamental)
        limit_price = max(0.01, limit_price)
        return Order(agent_id=self.id, side=side, order_type="limit",
                     price=round(limit_price, 2), quantity=1,
                     timestamp=step)

    def _trend_decide(self, price, prev_price, step, rng):
        if prev_price is None or prev_price <= 0 or price <= 0:
            return None

        ret = (price - prev_price) / prev_price

        # Update smoothed realized volatility (EMA of |return|).
        # This creates volatility clustering: high recent vol →
        # trend followers more active → sustains high vol.
        lookback = self.model.p.get('trend_lookback', 20)
        alpha = 2.0 / (lookback + 1)
        self._ema_vol = alpha * abs(ret) + (1 - alpha) * self._ema_vol

        threshold = self.model.p.get('trend_threshold', 0.0)
        if abs(ret) <= threshold:
            return None

        # Activity scales with both return magnitude AND recent volatility.
        # The vol multiplier creates the positive feedback loop.
        sensitivity = self.model.p.get('trend_sensitivity', 10.0)
        vol_boost = 1.0 + 200.0 * self._ema_vol  # amplify in volatile regimes
        action_prob = min(abs(ret) * sensitivity * vol_boost, 1.0)
        if rng.random() > action_prob:
            return None

        side = "buy" if ret > 0 else "sell"

        if side == "buy" and self.cash < price:
            return None
        if side == "sell" and self.inventory <= 0:
            return None

        # 60/40 market/limit split (down from 80/20) — reduces
        # outsized price impact on thin books (helps tail index)
        if rng.random() < 0.6:
            return Order(agent_id=self.id, side=side, order_type="market",
                         price=0.0, quantity=1, timestamp=step)
        else:
            if side == "buy":
                limit_price = price * 1.001
            else:
                limit_price = price * 0.999
            return Order(agent_id=self.id, side=side, order_type="limit",
                         price=round(max(0.01, limit_price), 2),
                         quantity=1, timestamp=step)
