"""Heterogeneous Agent Financial Market ABM."""

__version__ = "0.1.0"

from .agents import Strategy, Trader
from .config import DEFAULT_PARAMS
from .fundamental import FundamentalProcess
from .market_maker import MarketMaker
from .model import MarketModel
from .strategy_switching import (
    compute_exponential_ma_fitness,
    compute_strategy_fitness,
    multinomial_logit_probabilities,
    switch_strategies,
)

__all__ = [
    "DEFAULT_PARAMS",
    "FundamentalProcess",
    "MarketMaker",
    "MarketModel",
    "Strategy",
    "Trader",
    "compute_exponential_ma_fitness",
    "compute_strategy_fitness",
    "multinomial_logit_probabilities",
    "switch_strategies",
]
