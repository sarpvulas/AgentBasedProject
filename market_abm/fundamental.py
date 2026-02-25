"""Fundamental value process — mean-reverting Ornstein-Uhlenbeck."""

import numpy as np


class FundamentalProcess:
    """Exogenous fundamental value with mean reversion.

    F(t+1) = F(t) + kappa * (mu - F(t)) + sigma * epsilon(t)
    where epsilon ~ N(0, 1).
    """

    def __init__(self, initial_value: float, kappa: float, mu: float,
                 sigma: float, rng: np.random.Generator | None = None):
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()
        self.value = initial_value
        self.history: list[float] = [initial_value]

    def step(self) -> float:
        """Advance fundamental value by one period."""
        shock = self.rng.normal()
        self.value += self.kappa * (self.mu - self.value) + self.sigma * shock
        self.history.append(self.value)
        return self.value

    def apply_shock(self, pct_change: float) -> float:
        """Apply a one-time percentage shock to the fundamental."""
        self.value *= (1.0 + pct_change)
        return self.value
