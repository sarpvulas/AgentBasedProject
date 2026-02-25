"""MarketModel — single-asset LOB market with heterogeneous agents."""

from collections import Counter

import agentpy as ap
import numpy as np
from scipy.stats import jarque_bera, kurtosis as sp_kurtosis

from .agents import AgentType, Trader
from .analytics import compute_portfolio_metrics
from .config import DEFAULT_PARAMS
from .fundamental import FundamentalProcess
from .order_book import OrderBook


class MarketModel(ap.Model):
    """Single-asset market where agents trade through a limit order book.

    Simulation loop per step:
      1. Evolve fundamental value (O-U process)
      2. Shuffle agents (random arrival order)
      3. Each agent decides buy/sell/hold and submits orders
      4. Trades settle immediately (cash and inventory transfer)
      5. Cancel stale limit orders
      6. Record observables
    """

    def setup(self):
        for key, val in DEFAULT_PARAMS.items():
            if key not in self.p:
                self.p[key] = val

        seed = self.p.get('seed', None)
        self.rng = np.random.default_rng(seed)

        self.fundamental = FundamentalProcess(
            initial_value=self.p['fundamental_initial'],
            kappa=self.p['kappa'],
            mu=self.p['mu'],
            sigma=self.p['fundamental_sigma'],
            rng=self.rng,
        )

        self.order_book = OrderBook(
            initial_price=self.p['fundamental_initial']
        )

        self.traders = ap.AgentList(self, self.p['n_agents'], Trader)
        self._assign_types()
        self._agent_lookup = {t.id: t for t in self.traders}
        self._prev_price = self.p['fundamental_initial']
        self._interventions: dict[int, float] = {}

    def _assign_types(self):
        n = self.p['n_agents']
        frac_sum = self.p['frac_fundamental'] + self.p['frac_trend']
        if frac_sum > 1.0:
            raise ValueError(
                f"frac_fundamental + frac_trend = {frac_sum:.2f} > 1.0")
        n_fund = int(n * self.p['frac_fundamental'])
        n_trend = int(n * self.p['frac_trend'])
        n_noise = n - n_fund - n_trend

        types = (
            [AgentType.FUNDAMENTAL] * n_fund
            + [AgentType.TREND] * n_trend
            + [AgentType.NOISE] * n_noise
        )
        self.rng.shuffle(types)
        for trader, atype in zip(self.traders, types):
            trader.agent_type = atype

    def step(self):
        self._prev_price = (self.order_book.last_trade_price
                            or self._prev_price)

        # 1. Evolve fundamental
        self.fundamental.step()
        if self.t in self._interventions:
            self.fundamental.apply_shock(self._interventions[self.t])

        # 2. Market state
        price = (self.order_book.last_trade_price
                 or self.p['fundamental_initial'])
        prev_price = self._prev_price
        fundamental = self.fundamental.value
        best_bid = self.order_book.best_bid
        best_ask = self.order_book.best_ask

        # 3. Shuffle agents
        agent_order = list(self.traders)
        self.rng.shuffle(agent_order)

        # 4. Agent decisions
        for trader in agent_order:
            order = trader.decide(price, prev_price, fundamental,
                                  best_bid, best_ask, self.t, self.rng)
            if order is not None:
                trade = self.order_book.submit_order(order)
                if trade is not None:
                    self._settle_trade(trade)
                    best_bid = self.order_book.best_bid
                    best_ask = self.order_book.best_ask

        # 5. Cleanup
        self.order_book.cancel_stale_orders(
            self.t, self.p['stale_order_age'])

        # 6. End step
        self.order_book.end_step()

    def _settle_trade(self, trade):
        buyer = self._agent_lookup[trade.buyer_id]
        seller = self._agent_lookup[trade.seller_id]
        if buyer.cash < trade.price or seller.inventory < 1:
            return  # Cancel trade — agent can't cover (stale resting order)
        buyer.cash -= trade.price
        seller.cash += trade.price
        buyer.inventory += 1
        seller.inventory -= 1

    def update(self):
        price = (self.order_book.last_trade_price or self._prev_price)

        self.record('price', price)
        self.record('fundamental', self.fundamental.value)

        if self._prev_price > 0 and price > 0:
            self.record('log_return', np.log(price / self._prev_price))
        else:
            self.record('log_return', 0.0)

        self.record('best_bid', self.order_book.best_bid or 0.0)
        self.record('best_ask', self.order_book.best_ask or 0.0)
        self.record('spread', self.order_book.spread or 0.0)
        self.record('volume',
                     self.order_book.volume_history[-1]
                     if self.order_book.volume_history else 0)

        counts = Counter(t.agent_type for t in self.traders)
        n = self.p['n_agents']
        self.record('frac_fundamental',
                     counts.get(AgentType.FUNDAMENTAL, 0) / n)
        self.record('frac_trend', counts.get(AgentType.TREND, 0) / n)
        self.record('frac_noise', counts.get(AgentType.NOISE, 0) / n)

        for atype in AgentType:
            typed = [t for t in self.traders if t.agent_type == atype]
            if typed:
                mean_w = np.mean([t.cash + t.inventory * price
                                  for t in typed])
            else:
                mean_w = 0.0
            self.record(f'wealth_{atype.name.lower()}', mean_w)

    def end(self):
        prices = [p for p in self.order_book.price_history
                  if p is not None]
        if len(prices) < 10:
            return

        returns = np.diff(np.log(prices))
        if len(returns) < 10:
            return

        kurt = float(sp_kurtosis(returns, fisher=True))
        jb_stat, jb_pval = jarque_bera(returns)

        self.report('kurtosis', kurt)
        self.report('jb_statistic', float(jb_stat))
        self.report('jb_pvalue', float(jb_pval))
        self.report('mean_return', float(np.mean(returns)))
        self.report('std_return', float(np.std(returns)))
        self.report('n_steps', len(returns))

        valid_spreads = [s for s in self.order_book.spread_history
                         if s is not None]
        if valid_spreads:
            self.report('mean_spread', float(np.mean(valid_spreads)))

        last_price = (self.order_book.last_trade_price
                      or self.p['fundamental_initial'])
        for atype in AgentType:
            metrics = compute_portfolio_metrics(
                list(self.traders), atype, last_price)
            name = atype.name.lower()
            self.report(f'{name}_mean_pnl', metrics['mean_pnl'])
            self.report(f'{name}_sharpe', metrics['sharpe'])

    def schedule_intervention(self, step: int, pct_change: float):
        if not hasattr(self, '_interventions'):
            self._interventions = {}
        self._interventions[step] = pct_change
