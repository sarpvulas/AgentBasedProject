"""MarketModel — orchestrator for the heterogeneous-agent financial market ABM."""

from collections import Counter

import agentpy as ap
import numpy as np
from scipy.stats import jarque_bera, kurtosis as sp_kurtosis

from .agents import Strategy, Trader
from .config import DEFAULT_PARAMS
from .fundamental import FundamentalProcess
from .market_maker import MarketMaker
from .strategy_switching import (
    compute_exponential_ma_fitness,
    compute_strategy_fitness,
    multinomial_logit_probabilities,
    switch_strategies,
)


class MarketModel(ap.Model):
    """Heterogeneous-agent single-asset financial market model.

    Simulation loop per step:
      1. Evolve fundamental value
      2. Each trader computes demand based on current strategy
      3. Market maker updates price from aggregate demand
      4. Compute strategy fitness & EMA
      5. Strategy switching via multinomial logit
      6. Record observables
    """

    def setup(self):
        # Merge defaults with user-supplied parameters
        for key, val in DEFAULT_PARAMS.items():
            if key not in self.p:
                self.p[key] = val

        # Random generator (seeded via agentpy's self.random)
        seed = self.p.get('seed', None)
        self.rng = np.random.default_rng(seed)

        # Fundamental value process
        self.fundamental = FundamentalProcess(
            initial_value=self.p['fundamental_initial'],
            drift=self.p['fundamental_drift'],
            volatility=self.p['fundamental_volatility'],
            rng=self.rng,
        )

        # Market maker
        self.market_maker = MarketMaker(
            initial_price=self.p['price_initial'],
            lambda_mm=self.p['lambda_mm'],
            noise_sigma=self.p['mm_noise_sigma'],
            gamma=self.p.get('gamma', 0.0),
            rng=self.rng,
        )

        # Traders
        self.traders = ap.AgentList(self, self.p['n_agents'], Trader)
        self._assign_initial_strategies()

        # EMA fitness state
        self.ema_fitness = {s: 0.0 for s in Strategy}

        # Intervention schedule: {step_number: pct_change}
        self._interventions: dict[int, float] = {}

        # Previous state for fitness computation
        self._prev_price = self.market_maker.price
        self._prev_fundamental = self.fundamental.value

    def _assign_initial_strategies(self):
        """Assign strategies to traders based on initial fraction parameters."""
        n = self.p['n_agents']
        n_fund = int(n * self.p['init_fundamentalist_frac'])
        n_chart = int(n * self.p['init_chartist_frac'])
        n_noise = n - n_fund - n_chart

        strategies = (
            [Strategy.FUNDAMENTALIST] * n_fund
            + [Strategy.CHARTIST] * n_chart
            + [Strategy.NOISE] * n_noise
        )
        self.rng.shuffle(strategies)
        for trader, strat in zip(self.traders, strategies):
            trader.strategy = strat

    def step(self):
        """Execute one simulation step."""
        # Save previous state
        self._prev_price = self.market_maker.price
        self._prev_fundamental = self.fundamental.value

        # 1. Evolve fundamental
        self.fundamental.step()

        # 1b. Apply scheduled intervention
        current_step = self.t
        if current_step in self._interventions:
            self.fundamental.apply_shock(self._interventions[current_step])

        # 2. Compute demands
        price = self.market_maker.price
        fundamental = self.fundamental.value
        past_returns = self.market_maker.get_past_returns(self.p['chartist_memory'])

        demands = []
        for trader in self.traders:
            d = trader.compute_demand(price, fundamental, past_returns)
            demands.append(d)

        aggregate_demand = sum(demands)

        # 3. Market maker updates price (with endogenous volatility)
        chartist_frac = sum(1 for t in self.traders if t.strategy == Strategy.CHARTIST) / self.p['n_agents']
        self.market_maker.update_price(aggregate_demand, self.p['n_agents'],
                                       chartist_frac=chartist_frac)

        # 4. Fitness & EMA
        new_past_returns = self.market_maker.get_past_returns(self.p['chartist_memory'])
        raw_fitness = compute_strategy_fitness(
            price=self.market_maker.price,
            prev_price=self._prev_price,
            fundamental=fundamental,
            prev_fundamental=self._prev_fundamental,
            past_returns=new_past_returns,
            params=dict(self.p),
        )
        self.ema_fitness = compute_exponential_ma_fitness(
            raw_fitness, self.ema_fitness, self.p['fitness_ema_alpha']
        )

        # 5. Strategy switching with herding
        if current_step % self.p['switching_interval'] == 0:
            counts = Counter(t.strategy for t in self.traders)
            n = self.p['n_agents']
            fractions = {s: counts.get(s, 0) / n for s in Strategy}
            probs = multinomial_logit_probabilities(
                self.ema_fitness, self.p['beta'],
                fractions=fractions,
                herding=self.p.get('herding', 0.0),
            )
            switch_strategies(list(self.traders), probs, self.rng,
                             switch_prob=self.p.get('switch_prob', 0.1))

    def update(self):
        """Record observables after each step."""
        # Strategy fractions
        counts = Counter(t.strategy for t in self.traders)
        n = self.p['n_agents']

        self.record('price', self.market_maker.price)
        self.record('fundamental', self.fundamental.value)
        self.record('log_return', self.market_maker.return_history[-1]
                     if self.market_maker.return_history else 0.0)
        self.record('frac_fundamentalist', counts.get(Strategy.FUNDAMENTALIST, 0) / n)
        self.record('frac_chartist', counts.get(Strategy.CHARTIST, 0) / n)
        self.record('frac_noise', counts.get(Strategy.NOISE, 0) / n)

    def end(self):
        """Compute and report summary statistics."""
        returns = np.array(self.market_maker.return_history)
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

    def schedule_intervention(self, step: int, pct_change: float):
        """Schedule a shock to the fundamental value at a given step.

        Parameters
        ----------
        step : int
            The simulation step at which to apply the shock.
        pct_change : float
            Fractional change, e.g. -0.10 for a 10% drop.
        """
        if not hasattr(self, '_interventions'):
            self._interventions = {}
        self._interventions[step] = pct_change
