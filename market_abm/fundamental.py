"""Fundamental value process — geometric random walk."""

import numpy as np


class FundamentalProcess:
    """Exogenous fundamental value evolving as a geometric random walk.

    V(t+1) = V(t) * exp(drift + volatility * epsilon)
    where epsilon ~ N(0, 1).
    """

    def __init__(self, initial_value: float, drift: float, volatility: float,
                 rng: np.random.Generator | None = None):
        self.drift = drift
        self.volatility = volatility
        self.rng = rng or np.random.default_rng()
        self.value = initial_value
        self.history: list[float] = [initial_value]

    def step(self) -> float:
        """Advance fundamental value by one period."""
        shock = self.rng.normal()
        self.value *= np.exp(self.drift + self.volatility * shock)
        self.history.append(self.value)
        return self.value

    def apply_shock(self, pct_change: float) -> float:
        """Apply a one-time percentage shock to the fundamental.

        Parameters
        ----------
        pct_change : float
            Fractional change, e.g. 0.05 for +5%.
        """
        self.value *= (1.0 + pct_change)
        return self.value

    @property
    def log_value(self) -> float:
        return np.log(self.value)
