# Market ABM

A heterogeneous agent-based model of a financial market with a limit order book. Built for the King's College ABM lecture.

Three trader types — noise, fundamental, and trend-following — submit limit and market orders to a continuous double auction. Price emerges endogenously from order matching. The model reproduces key stylized facts of financial returns: fat tails, volatility clustering, and absence of return autocorrelation.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
market_abm/
  agents.py          # Noise, fundamental, trend-following agents
  order_book.py      # Limit order book with price-time priority
  fundamental.py     # Ornstein-Uhlenbeck fundamental value process
  model.py           # AgentPy model wiring agents + LOB
  analytics.py       # Return statistics and stylized facts validation
  visualization.py   # Matplotlib chart functions
  config.py          # Default parameters
app.py               # Streamlit dashboard with simulation + guide tabs
tests/               # Pytest suite
```

## Dashboard

The Streamlit dashboard has two tabs:

- **Simulation** — run the model with configurable parameters and view price charts, return distributions, autocorrelation panels, PnL by strategy, and stylized facts validation
- **Guide** — educational reference explaining every parameter, chart, and statistical test with a simple/technical toggle

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `n_agents` | Total number of traders | 100 |
| `frac_fundamental` | Share of value investors | 0.33 |
| `frac_trend` | Share of momentum traders | 0.33 |
| `mu` | Long-run fundamental mean | 100 |
| `kappa` | Mean reversion speed | 0.01 |
| `fundamental_sigma` | Fundamental volatility | 0.50 |
| `fundamental_sensitivity` | Mispricing reaction strength | 1.0 |
| `trend_threshold` | Minimum move for trend signal | 0.5 |
| `stale_order_age` | Steps before order cancellation | 10 |
| `steps` | Simulation length | 5000 |

## Tests

```bash
pytest tests/ -q
```
