"""Default parameters for the heterogeneous-agent financial market ABM."""

DEFAULT_PARAMS = {
    # Simulation
    'steps': 2000,
    'seed': 42,

    # Agents
    'n_agents': 50,
    'init_fundamentalist_frac': 0.35,
    'init_chartist_frac': 0.35,
    # remainder → noise traders

    # Fundamental process (geometric random walk)
    'fundamental_initial': 100.0,
    'fundamental_drift': 0.0,        # μ per step
    'fundamental_volatility': 0.002, # σ per step

    # Fundamentalist demand
    'phi': 0.4,  # mean-reversion strength toward fundamental

    # Chartist demand
    'chi': 2.5,            # trend-following strength
    'chartist_memory': 3,  # lookback window (steps)

    # Noise trader demand
    'noise_sigma': 0.30,   # std of noise demand shock

    # Market maker
    'price_initial': 100.0,
    'lambda_mm': 0.012,    # price adjustment speed (scaled for sqrt(N))
    'mm_noise_sigma': 0.003,  # market-maker noise
    'gamma': 3.5,          # endogenous volatility amplification by chartist fraction

    # Strategy switching
    'beta': 10.0,           # intensity of choice (logit temperature⁻¹)
    'fitness_ema_alpha': 0.35, # EMA smoothing for fitness
    'switching_interval': 1,   # switch every N steps
    'switch_prob': 0.10,       # per-agent probability of reconsidering strategy
    'herding': 2.0,            # social influence strength
}
