"""Trader agents with heterogeneous strategies."""

from enum import Enum, auto

import agentpy as ap
import numpy as np


class Strategy(Enum):
    FUNDAMENTALIST = auto()
    CHARTIST = auto()
    NOISE = auto()


class Trader(ap.Agent):
    """Single trader that computes demand based on its current strategy."""

    def setup(self):
        p = self.model.p
        self.strategy: Strategy = Strategy.NOISE  # overridden during model setup
        self.phi: float = p.get('phi', 1.0)
        self.chi: float = p.get('chi', 1.5)
        self.noise_sigma: float = p.get('noise_sigma', 0.05)
        self.chartist_memory: int = p.get('chartist_memory', 10)

        # Fitness tracking (one per strategy)
        self.fitness = {s: 0.0 for s in Strategy}

    def compute_demand(self, price: float, fundamental: float,
                       past_returns: np.ndarray) -> float:
        """Compute signed demand given current market state."""
        if self.strategy == Strategy.FUNDAMENTALIST:
            return self._fundamentalist_demand(price, fundamental)
        elif self.strategy == Strategy.CHARTIST:
            return self._chartist_demand(past_returns)
        else:
            return self._noise_demand()

    def _fundamentalist_demand(self, price: float, fundamental: float) -> float:
        """Mean-reversion toward fundamental: phi * (V - P) / P."""
        return self.phi * (fundamental - price) / price

    def _chartist_demand(self, past_returns: np.ndarray) -> float:
        """Trend-following: chi * sum(weighted recent log-returns).

        Uses exponentially decaying weights so more recent returns matter more.
        """
        if len(past_returns) == 0:
            return 0.0
        n = len(past_returns)
        # Exponential weights: most recent gets highest weight
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()
        return self.chi * np.dot(weights, past_returns)

    def _noise_demand(self) -> float:
        """Random demand: N(0, sigma)."""
        return self.model.random.normalvariate(0.0, self.noise_sigma)
