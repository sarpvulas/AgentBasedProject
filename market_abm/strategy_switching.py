"""Multinomial logit strategy switching with EMA fitness and herding."""

from collections import Counter

import numpy as np

from .agents import Strategy, Trader


def compute_strategy_fitness(price: float, prev_price: float,
                             fundamental: float, prev_fundamental: float,
                             past_returns: np.ndarray,
                             params: dict) -> dict[Strategy, float]:
    """Compute hypothetical one-step profit for each strategy.

    Each strategy's fitness is the profit it *would* have earned if it had
    held a position proportional to its demand signal from the previous step.
    Profit = demand(t-1) * log_return(t).
    """
    if prev_price <= 0 or price <= 0:
        return {s: 0.0 for s in Strategy}

    log_return = np.log(price / prev_price)

    # Fundamentalist demand at t-1
    fund_demand = params.get('phi', 1.0) * (prev_fundamental - prev_price) / prev_price
    fund_profit = fund_demand * log_return

    # Chartist demand at t-1 (uses returns up to t-1)
    if len(past_returns) > 1:
        chart_returns = past_returns[:-1] if len(past_returns) > 1 else past_returns
        chart_demand = params.get('chi', 1.5) * np.mean(chart_returns)
    else:
        chart_demand = 0.0
    chart_profit = chart_demand * log_return

    # Noise: expected profit is 0
    noise_profit = 0.0

    return {
        Strategy.FUNDAMENTALIST: fund_profit,
        Strategy.CHARTIST: chart_profit,
        Strategy.NOISE: noise_profit,
    }


def compute_exponential_ma_fitness(current_fitness: dict[Strategy, float],
                                   prev_ema: dict[Strategy, float],
                                   alpha: float) -> dict[Strategy, float]:
    """EMA smoothing: F_ema(t) = alpha * F(t) + (1-alpha) * F_ema(t-1)."""
    return {
        s: alpha * current_fitness[s] + (1.0 - alpha) * prev_ema[s]
        for s in Strategy
    }


def multinomial_logit_probabilities(fitness: dict[Strategy, float],
                                    beta: float,
                                    fractions: dict[Strategy, float] | None = None,
                                    herding: float = 0.0) -> dict[Strategy, float]:
    """Convert fitness to switching probabilities via multinomial logit with herding.

    P(strategy s) = exp(beta * F_s + herding * n_s) / sum_j exp(beta * F_j + herding * n_j)

    where n_s is the current fraction of agents using strategy s.
    The herding term creates positive feedback: popular strategies attract more agents.
    Numerically stabilized by subtracting the max before exponentiation.
    """
    attractiveness = np.array([
        beta * fitness[s] + herding * (fractions.get(s, 1/3) if fractions else 1/3)
        for s in Strategy
    ])
    attractiveness -= attractiveness.max()  # numerical stability
    exp_vals = np.exp(attractiveness)
    probs = exp_vals / exp_vals.sum()
    return {s: float(p) for s, p in zip(Strategy, probs)}


def switch_strategies(traders: list[Trader],
                      probabilities: dict[Strategy, float],
                      rng: np.random.Generator,
                      switch_prob: float = 0.1) -> None:
    """Reassign strategies with individual-level stochastic switching.

    Each agent independently decides whether to reconsider their strategy
    (with probability switch_prob). Those who reconsider draw a new strategy
    from the multinomial logit probabilities.
    """
    strategies = list(Strategy)
    prob_array = np.array([probabilities[s] for s in strategies])
    prob_array /= prob_array.sum()

    reconsider = rng.random(len(traders)) < switch_prob
    n_reconsidering = int(reconsider.sum())

    if n_reconsidering > 0:
        new_choices = rng.choice(len(strategies), size=n_reconsidering, p=prob_array)
        idx = 0
        for i, trader in enumerate(traders):
            if reconsider[i]:
                trader.strategy = strategies[new_choices[idx]]
                idx += 1
