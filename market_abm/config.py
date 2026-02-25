"""Default parameters for the single-asset LOB market ABM."""

DEFAULT_PARAMS = {
    # Simulation
    'steps': 5000,
    'seed': 42,

    # Agents
    'n_agents': 100,
    'frac_fundamental': 0.33,
    'frac_trend': 0.33,
    # remainder -> noise traders
    'initial_cash': 10000.0,
    'initial_inventory': 10,

    # Fundamental process (Ornstein-Uhlenbeck)
    'fundamental_initial': 100.0,
    'mu': 100.0,               # long-run mean
    'kappa': 0.01,             # mean-reversion speed
    'fundamental_sigma': 0.5,  # shock volatility

    # Agent behavior
    'fundamental_sensitivity': 1.0,  # scales action probability
    'trend_threshold': 0.0,          # min price change to trigger trend

    # Order book
    'stale_order_age': 10,  # max age before limit order cancellation
}
