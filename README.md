# Market ABM

[![tests](https://github.com/sarpvulas/AgentBasedProject/actions/workflows/tests.yml/badge.svg)](https://github.com/sarpvulas/AgentBasedProject/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A heterogeneous agent-based model of a financial market with a limit order book, built in Python for the King's College London agent-based modelling course (MSc Computational Finance).

## TL;DR

Real prices show fat-tailed returns and clustered volatility, and simple models with one kind of trader usually do not. This repo simulates 100 noise, fundamental and trend-following traders who trade one asset through a continuous double auction, so the price comes out of order matching rather than being assumed. With the default parameters the simulated returns are strongly fat-tailed (excess kurtosis 15.5 on seed 42, 14.9 on average over 30 seeds) and show weak, irregular volatility clustering, but they also show a negative lag-1 autocorrelation in every seed, so not every stylized fact is reproduced (see Results and Limitations).

![Price vs fundamental value, seed 42](docs/img/price_vs_fundamental.png)

## Results

Default parameters, 5,000 steps (5,001 observations, because step 0 is recorded), seed 42 (`python scripts/make_figures.py` prints these):

| Statistic | Seed 42 | 30 seeds (1-30): mean, min to max |
|---|---|---|
| Excess kurtosis of per-step log returns | 15.5 | 14.9, 7.2 to 48.4 |
| ACF of returns, lag 1 | -0.038 | -0.067, -0.099 to -0.038 |
| ACF of squared returns, lag 1 | 0.022 | 0.080, 0.0005 to 0.258 |
| Hill tail index (top 5% of abs returns) | 2.13 | 2.03, 1.78 to 2.37 |

All statistics use all 5,001 per-step returns, including the zero returns from steps without a trade (29 on seed 42; 18 to 33 across the 30 seeds). `run_experiment` and the dashboard use the same series.

What this shows:

- Fat tails: yes. Excess kurtosis is positive and large in all 30 seeds (the built-in check passes in 30 of 30). The QQ plot below shows the excess comes from the tails; the centre of the distribution is flatter than normal. Three seeds exceed 20 (21.9, 32.8 and 48.4), which pulls the mean up; the other 27 are between 7.2 and 19.5.
- Volatility clustering: weak and irregular, not a clean decay. ACF of squared returns is positive at lag 1 in all 30 seeds but above the 95% band (about 0.028) in 24. The check (lag-1 ACF of squared returns above the band and lower at lag 5) passes in 20 of 30 seeds. Seed 42 fails it: its lag-1 value is 0.022 (inside the band), then 0.125, 0.108, 0.029, 0.043 at lags 2 to 5, while ACF of absolute returns is above the band at all of lags 1 to 5 (0.049, 0.076, 0.086, 0.032, 0.041).
- No return autocorrelation: not reproduced. The per-lag check (every lag 1 to 5 inside the 95% band) passes in 0 of 30 seeds. Lag-1 ACF is negative and outside the band in all 30 seeds; lags 1 and 2 fail in all 30, lag 3 in 26, lag 4 in 17, lag 5 in 15. A bounce between resting bid and ask prices is a plausible cause (not tested; the book is two-sided at the end of only 23.8% of steps on seed 42).
- Hill index around 2 is at the low end of the 2 to 6 range the code treats as plausible (the check passes in 17 of 30 seeds), using a 5% tail on only 5,001 observations; treat it as indicative.
- The price tracks the fundamental closely in 29 of 30 seeds (mean absolute gap 1.88 on seed 42; 30-seed mean 2.82, range 1.73 to 24.1), with seed 17 as the exception (mean gap 24, price up to 176).
- Thin-book mechanics probably drive the extreme tails. The book is two-sided at the end of only about 24% of steps, and single jumps through stale far quotes dominate the largest returns. In seed 16 (kurtosis 48.4) the largest returns are single moves of 13 to 17%, for example 94.07 to 107.52 after a step with a spread of 19.13; removing the largest return still leaves kurtosis 37.3. The same mechanism may also explain the negative lag-1 ACF (untested).

![Return distribution and QQ plot, seed 42](docs/img/return_distribution.png)

## Model

Each step:

1. The fundamental value follows an Ornstein-Uhlenbeck process, `F(t+1) = F(t) + kappa (mu - F(t)) + sigma eps`.
2. Agents are shuffled and act one at a time. Each submits at most one order of `order_size` units (default 1; market or limit) or holds.
3. Matching uses price-time priority across levels. A market order, or a limit order that crosses the spread, trades against resting orders at their prices until it is filled or nothing crosses; each fill can be partial. An agent never trades with its own resting orders (they are skipped, others keep their priority). A limit remainder rests; a market remainder is dropped. A limit order whose remainder would still cross only because of the agent's own resting order is dropped rather than rested, so the book is never crossed.
4. Each fill is limited to what both sides can cover at the trade price (cash for the buyer, inventory for the seller) and settles immediately. A resting order whose owner can no longer cover it is removed. If a fill still fails to settle, it is voided, the resting order is restored at its original priority, and it does not set the price or count as volume (a safety net; 0 voids on seeds 1-30).
5. Resting limit orders older than `stale_order_age` steps are cancelled.

| Agent | Rule |
|---|---|
| Noise | Buys 30%, sells 30%, holds 40%. Half market orders; half limit orders priced around the last price with jitter up to one spread. |
| Fundamental | Trades toward the fundamental with probability `min(abs(mispricing) * fundamental_sensitivity, 1)`; mostly limit orders priced between price and fundamental. |
| Trend | Acts when the last return exceeds `trend_threshold`, with probability scaled by `trend_sensitivity`; mostly market orders in the direction of the move. |

Agents start with 10,000 cash and 10 units, and cannot buy without cash or sell without inventory. The price is the last trade price. Volume is counted in units traded.

```
market_abm/
  agents.py          noise, fundamental, trend-following agents
  order_book.py      limit order book: price-time priority, partial fills, self-trade prevention
  fundamental.py     Ornstein-Uhlenbeck fundamental value
  model.py           AgentPy model wiring agents + book
  analytics.py       return statistics, stylized-facts checks, multi-seed runs
  visualization.py   Matplotlib charts
  config.py          default parameters
app.py               Streamlit dashboard (simulation + guide tabs)
notebooks/           single run, parameter sweep, stylized facts, sensitivity, perturbation (no stored outputs)
scripts/make_figures.py   regenerates docs/img and prints the table above (about 30 s)
tests/               pytest suite
```

Stack: Python 3.12, AgentPy, NumPy, pandas, SciPy, statsmodels, Matplotlib, Streamlit.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py          # dashboard
pytest tests/ -q              # tests (100 at the time of writing)
python scripts/make_figures.py
```

The dashboard has a Simulation tab (parameters, price charts, return distribution, autocorrelation, PnL by strategy, stylized-facts checks) and a Guide tab explaining each parameter and chart.

## Parameters

Defaults from `market_abm/config.py`:

| Parameter | Description | Default |
|---|---|---|
| `steps` | Simulation length | 5000 |
| `seed` | Random seed | 42 |
| `n_agents` | Total number of traders | 100 |
| `frac_fundamental` | Share of fundamental traders | 0.33 |
| `frac_trend` | Share of trend followers (rest are noise) | 0.33 |
| `initial_cash` / `initial_inventory` | Starting endowment per agent | 10000 / 10 |
| `fundamental_initial`, `mu` | Starting and long-run fundamental value | 100 |
| `kappa` | Mean reversion speed | 0.01 |
| `fundamental_sigma` | Fundamental shock volatility | 0.5 |
| `fundamental_sensitivity` | Fundamental trader reaction to mispricing | 2.0 |
| `trend_threshold` | Minimum fractional return for a trend signal | 0.005 |
| `trend_sensitivity` | Trend trader reaction to return size | 5.0 |
| `order_size` | Units per order submitted by every agent | 1 |
| `stale_order_age` | Steps before a resting order is cancelled | 10 |

## Reproducibility

- One seed (`seed`, default 42) drives a single NumPy `Generator` used for the fundamental, agent types, arrival order and all agent decisions. The same seed and parameters give identical price paths (covered by `test_same_seed_same_prices`).
- Results above were produced with AgentPy 0.1.5, NumPy 2.4.4, pandas 3.0.3, SciPy 1.17.0 and statsmodels 0.14.0 on Python 3.12.2. An independent review on a fresh install (NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, statsmodels 0.15.0) reproduced identical numbers. `requirements.txt` only gives lower bounds, so other versions may differ.
- The 30-seed numbers use seeds 1 to 30 with default parameters.
- Defaults changed in the matching fixes, so seed-42 paths from earlier versions of this repo are not reproduced. With `order_size = 1`, what changed: agents no longer match their own resting orders; aggressors are limited to what they can pay or deliver at the actual trade price instead of the last price; a resting order whose owner can no longer cover it is removed instead of consuming the aggressor's order for the step; a limit order that would cross only the agent's own resting order is dropped instead of rested. Earlier numbers (seed 42 / 30-seed mean) were: kurtosis 10.9 / 12.6, lag-1 ACF of returns -0.092 / -0.071, lag-1 ACF of squared returns 0.172 / 0.068, Hill 2.09 / 2.08, self-trades 0.65% of trades.

## Limitations

- One asset. Orders carry a quantity, sweep several price levels and fill partially, but every agent submits the same fixed `order_size` (default 1), so no partial fill can occur at default settings. Partial fills, multi-level sweeps and capacity limits are covered by tests with larger sizes, not by the headline results.
- Resting limit orders do not reserve cash or inventory. An owner who has spent the cash by the time the order is hit has the order removed rather than filled.
- A limit order that would cross only the agent's own resting order is dropped instead of rested. This keeps the book uncrossed but is a modelling choice, not market practice (venues usually cancel or reduce one of the two orders). A market order behaves differently: it skips the agent's own order and keeps trading against the next one.
- Returns are lumpy: the histogram shows a flat-topped centre with a few large jumps, so kurtosis is driven by the tails rather than a peaked centre.
- The stylized-fact checks are strict or approximate in places. The per-lag return check tests five lags at 5% each, so even white noise fails at least one lag in about 23% of series (tested). The confidence band for squared returns is the same `1.96/sqrt(n)` band, which is approximate for a non-Gaussian series. The KS normality test standardizes with sample parameters, so its p-values are too lenient. `validate_stylized_facts` raises `ValueError` for non-finite, constant or very short input instead of returning a vacuous pass.
- No calibration to real market data; the "plausible range" checks are rules of thumb, not tests against data.
- Tests check the checks on synthetic series (white noise, AR(1), ARCH) and the model's conservation and matching rules; they do not assert the stylized-fact outputs of the model itself.
- Notebook 04 needs SALib; with SALib 1.5.1 and pandas 3 the parameter names must be a NumPy array (done in the notebook). The notebooks were re-run end to end on the current model after the changes, but their outputs are not stored.

## Credits and license

Built for the King's College London agent-based modelling course (MSc Computational Finance) by Sarp Vulaş (Dubai). MIT license, see [LICENSE](LICENSE).
