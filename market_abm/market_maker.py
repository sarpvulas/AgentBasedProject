"""Market maker: sets price based on excess demand."""

import numpy as np


class MarketMaker:
    """Market maker that adjusts price based on excess demand with endogenous volatility.

    P(t+1) = P(t) * exp(lambda * D / sqrt(N) * (1 + gamma * n_c) + epsilon)

    where D is aggregate excess demand, N is the number of traders,
    n_c is the fraction of chartists (amplifying price impact during herding),
    gamma controls the endogenous volatility amplification,
    and epsilon is a small market-maker noise term.

    The sqrt(N) normalization (instead of N) is crucial: it preserves the
    variance structure of aggregate demand, preventing CLT from smoothing
    away the fat tails that emerge from heterogeneous trading strategies.
    """

    def __init__(self, initial_price: float, lambda_mm: float,
                 noise_sigma: float = 0.0, gamma: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.lambda_mm = lambda_mm
        self.noise_sigma = noise_sigma
        self.gamma = gamma
        self.rng = rng or np.random.default_rng()

        self.price = initial_price
        self.price_history: list[float] = [initial_price]
        self.return_history: list[float] = []

    def update_price(self, aggregate_demand: float, n_agents: int,
                     chartist_frac: float = 0.0) -> float:
        """Update price based on normalized excess demand.

        Parameters
        ----------
        aggregate_demand : float
            Sum of all agent demands.
        n_agents : int
            Total number of agents.
        chartist_frac : float
            Current fraction of chartist agents (for endogenous volatility).
        """
        noise = self.rng.normal(0.0, self.noise_sigma) if self.noise_sigma > 0 else 0.0

        # Endogenous volatility amplifier: more chartists → more price impact
        vol_multiplier = 1.0 + self.gamma * chartist_frac

        # sqrt(N) normalization preserves heterogeneity effects
        log_return = self.lambda_mm * aggregate_demand / np.sqrt(n_agents) * vol_multiplier + noise

        # Clamp to prevent explosive dynamics
        log_return = np.clip(log_return, -0.10, 0.10)

        new_price = self.price * np.exp(log_return)
        self.return_history.append(log_return)
        self.price = new_price
        self.price_history.append(new_price)
        return new_price

    def get_past_returns(self, n: int) -> np.ndarray:
        """Return the last n log-returns (or fewer if history is shorter)."""
        if not self.return_history:
            return np.array([])
        return np.array(self.return_history[-n:])

    @property
    def log_price(self) -> float:
        return np.log(self.price)
