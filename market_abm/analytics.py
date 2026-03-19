"""Stylized facts validation and statistical analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kstest, kurtosis, skew
from statsmodels.tsa.stattools import acf


def compute_return_statistics(returns: np.ndarray) -> dict:
    """Compute descriptive statistics for a return series."""
    return {
        'mean': float(np.mean(returns)),
        'std': float(np.std(returns)),
        'skewness': float(skew(returns)),
        'kurtosis': float(kurtosis(returns, fisher=True)),  # excess kurtosis
        'jb_statistic': float(jarque_bera(returns).statistic),
        'jb_pvalue': float(jarque_bera(returns).pvalue),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'n': len(returns),
    }


def compute_autocorrelation(returns: np.ndarray, nlags: int = 50) -> dict:
    """Compute ACF for returns, absolute returns, and squared returns."""
    acf_returns = acf(returns, nlags=nlags, fft=True)
    acf_abs = acf(np.abs(returns), nlags=nlags, fft=True)
    acf_sq = acf(returns ** 2, nlags=nlags, fft=True)

    return {
        'acf_returns': acf_returns,
        'acf_abs_returns': acf_abs,
        'acf_squared_returns': acf_sq,
        'nlags': nlags,
    }


def hill_estimator(returns: np.ndarray, k: int | None = None) -> float:
    """Estimate the tail index using the Hill estimator.

    Uses the largest k order statistics of |returns|.
    A lower tail index means heavier tails (power-law exponent).
    Typical financial data: 2-5.

    Parameters
    ----------
    returns : array
        Return series.
    k : int, optional
        Number of upper order statistics to use. Defaults to ~5% of data.
    """
    x = np.sort(np.abs(returns))[::-1]  # descending absolute values
    if k is None:
        k = max(10, int(0.05 * len(x)))
    k = min(k, len(x) - 1)

    # Hill estimator: 1/alpha = (1/k) * sum(log(x_i / x_{k+1})) for i=1..k
    log_ratios = np.log(x[:k] / x[k])
    inv_alpha = np.mean(log_ratios)
    if inv_alpha <= 0:
        return float('inf')
    return 1.0 / inv_alpha


def validate_stylized_facts(returns: np.ndarray, nlags: int = 20) -> dict:
    """Comprehensive validation of stylized facts.

    Checks:
    1. Fat tails: excess kurtosis > 0 and JB test rejects normality
    2. Volatility clustering: significant ACF of |returns| at lag 1-5
    3. No return autocorrelation: ACF(returns) at lag 1 is small
    4. Approximate normality rejection via KS test
    5. Tail index in plausible range (2-6)

    Returns dict of {fact_name: {passed: bool, value: ..., criterion: ...}}.
    """
    stats = compute_return_statistics(returns)
    acf_data = compute_autocorrelation(returns, nlags=nlags)
    tail_idx = hill_estimator(returns)

    # Confidence band for ACF (approximate 95%)
    conf = 1.96 / np.sqrt(len(returns))

    results = {}

    # 1. Fat tails
    results['fat_tails'] = {
        'passed': bool(stats['kurtosis'] > 0 and stats['jb_pvalue'] < 0.05),
        'kurtosis': stats['kurtosis'],
        'jb_pvalue': stats['jb_pvalue'],
        'criterion': 'excess kurtosis > 0 AND JB rejects normality (p < 0.05)',
    }

    # 2. Volatility clustering
    avg_abs_acf = float(np.mean(np.abs(acf_data['acf_abs_returns'][1:6])))
    results['volatility_clustering'] = {
        'passed': bool(avg_abs_acf > conf),
        'avg_abs_acf_lag1_5': avg_abs_acf,
        'confidence_band': float(conf),
        'criterion': 'mean |ACF(|r|)| at lags 1-5 exceeds 95% confidence band',
    }

    # 3. No return autocorrelation
    # Use mean of first 5 lags (more robust than just lag 1)
    acf_r_mean5 = float(np.mean(np.abs(acf_data['acf_returns'][1:6])))
    results['no_return_autocorrelation'] = {
        'passed': bool(acf_r_mean5 < 3 * conf),
        'mean_abs_acf_lag1_5': acf_r_mean5,
        'threshold': float(3 * conf),
        'criterion': 'mean |ACF(r)| at lags 1-5 < 3x confidence band',
    }

    # 4. Non-normality (KS test)
    ks_stat, ks_pval = kstest(
        (returns - np.mean(returns)) / np.std(returns), 'norm'
    )
    results['non_normality'] = {
        'passed': bool(ks_pval < 0.05),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'criterion': 'KS test rejects normality (p < 0.05)',
    }

    # 5. Tail index
    results['tail_index'] = {
        'passed': bool(2.0 <= tail_idx <= 6.0),
        'hill_estimate': float(tail_idx),
        'criterion': 'Hill tail index in [2, 6] (consistent with financial data)',
    }

    return results


def compute_portfolio_metrics(agents, agent_type, last_price: float) -> dict:
    """Compute wealth, PnL, and Sharpe for agents of a given type.

    PnL is decomposed into two components:
    - **Trading PnL**: profit/loss from buy/sell decisions. Computed as
      (cash - initial_cash) + (inventory - initial_inventory) * last_price.
    - **Revaluation PnL**: gain/loss from holding the initial inventory as
      the price changes. Equals initial_inventory * (last_price - initial_price).

    Total PnL = Trading PnL + Revaluation PnL.
    """
    typed = [a for a in agents if a.agent_type == agent_type]
    if not typed:
        return {'mean_pnl': 0.0, 'std_pnl': 0.0, 'sharpe': 0.0,
                'mean_wealth': 0.0, 'mean_trading_pnl': 0.0,
                'mean_revaluation_pnl': 0.0, 'n': 0}

    pnls = np.array([
        (a.cash + a.inventory * last_price) - a.initial_wealth
        for a in typed
    ])
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Decompose PnL into trading vs revaluation when model context is available
    mean_trading_pnl = 0.0
    revaluation = 0.0
    model = getattr(typed[0], 'model', None)
    if model is not None:
        init_inv = model.p.get('initial_inventory', 10)
        init_price = model.p.get('fundamental_initial', 100.0)
        init_cash = model.p.get('initial_cash', 10000.0)
        trading_pnls = np.array([
            (a.cash - init_cash) + (a.inventory - init_inv) * last_price
            for a in typed
        ])
        mean_trading_pnl = float(np.mean(trading_pnls))
        revaluation = float(init_inv * (last_price - init_price))

    return {
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'sharpe': sharpe,
        'mean_wealth': float(np.mean([
            a.cash + a.inventory * last_price for a in typed
        ])),
        'mean_trading_pnl': mean_trading_pnl,
        'mean_revaluation_pnl': revaluation,
        'n': len(typed),
    }


# ── New analysis functions ───────────────────────────────────────────────


def mean_absolute_mispricing(prices: np.ndarray,
                              fundamentals: np.ndarray) -> float:
    """Mean absolute deviation of price from fundamental value.

    A scalar measure of market efficiency: lower = prices track fundamentals
    more closely. Useful for comparing across scenarios and seeds.
    """
    return float(np.mean(np.abs(prices - fundamentals)))


def rolling_volatility(returns: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling standard deviation of returns.

    Visualises volatility clustering — periods of high/low volatility should
    be visible as sustained runs above/below the mean level.
    """
    s = pd.Series(returns)
    return s.rolling(window, min_periods=window // 2).std().values


def max_drawdown(prices: np.ndarray) -> tuple[float, int, int]:
    """Maximum drawdown from peak to trough.

    Returns (drawdown_pct, peak_idx, trough_idx).
    Drawdown is expressed as a positive fraction (e.g. 0.15 = 15% drop).
    """
    cummax = np.maximum.accumulate(prices)
    dd = (cummax - prices) / cummax
    trough_idx = int(np.argmax(dd))
    peak_idx = int(np.argmax(prices[:trough_idx + 1]))
    return float(dd[trough_idx]), peak_idx, trough_idx


def drawdown_series(prices: np.ndarray) -> np.ndarray:
    """Full drawdown series (fraction below running peak) for plotting."""
    cummax = np.maximum.accumulate(prices)
    return (cummax - prices) / cummax


def run_experiment(params: dict) -> dict:
    """Run a single simulation and return a flat metrics dict.

    This is the reusable building block for multi-seed, sensitivity,
    and cross-scenario analyses. Returns all key metrics in one dict.
    """
    from .model import MarketModel
    from .agents import AgentType

    model = MarketModel(params)
    model.run()
    data = model.output.variables.MarketModel
    prices = data['price'].values
    fundamentals = data['fundamental'].values
    all_returns = data['log_return'].values
    returns = all_returns[all_returns != 0.0]

    if len(returns) < 20:
        return {'valid': False}

    stats = compute_return_statistics(returns)
    facts = validate_stylized_facts(returns)

    last_price = model.order_book.last_trade_price or params.get(
        'fundamental_initial', 100.0)

    pnl = {}
    for atype in AgentType:
        m = compute_portfolio_metrics(list(model.traders), atype, last_price)
        name = atype.name.lower()
        pnl[f'{name}_mean_pnl'] = m['mean_pnl']
        pnl[f'{name}_sharpe'] = m['sharpe']

    spreads = data['spread'].values
    valid_spreads = spreads[spreads > 0]

    dd_pct, _, _ = max_drawdown(prices)

    return {
        'valid': True,
        'volatility': stats['std'],
        'kurtosis': stats['kurtosis'],
        'skewness': stats['skewness'],
        'mean_return': stats['mean'],
        'hill_index': facts['tail_index']['hill_estimate'],
        'vol_clustering_acf': facts['volatility_clustering']['avg_abs_acf_lag1_5'],
        'return_acf': facts['no_return_autocorrelation']['mean_abs_acf_lag1_5'],
        'mean_spread': float(np.mean(valid_spreads)) if len(valid_spreads) > 0 else 0.0,
        'mean_abs_mispricing': mean_absolute_mispricing(prices, fundamentals),
        'max_drawdown': dd_pct,
        'mean_volume': float(data['volume'].mean()),
        'fat_tails_pass': facts['fat_tails']['passed'],
        'vol_clustering_pass': facts['volatility_clustering']['passed'],
        'no_autocorr_pass': facts['no_return_autocorrelation']['passed'],
        'non_normality_pass': facts['non_normality']['passed'],
        'tail_index_pass': facts['tail_index']['passed'],
        **pnl,
    }


def run_multi_seed(base_params: dict, n_seeds: int = 30,
                   seed_start: int = 1) -> pd.DataFrame:
    """Run the model across multiple seeds and collect metrics.

    Returns a DataFrame with one row per seed and columns for every metric.
    This directly tests whether stylized facts are robust properties of the
    model rather than artifacts of a single random seed.
    """
    rows = []
    for i in range(n_seeds):
        p = {**base_params, 'seed': seed_start + i}
        result = run_experiment(p)
        if result.get('valid', False):
            result['seed'] = seed_start + i
            rows.append(result)
    return pd.DataFrame(rows)


def run_sensitivity(base_params: dict, param_name: str,
                    values: list | np.ndarray,
                    n_seeds: int = 5,
                    seed_start: int = 1) -> pd.DataFrame:
    """Sweep a single parameter and return metrics for each value.

    Keeps all other parameters fixed. When n_seeds > 1, each parameter
    value is run across multiple seeds and the results are averaged,
    reducing seed-specific artifacts.
    """
    rows = []
    for v in values:
        seed_results = []
        for s in range(n_seeds):
            p = {**base_params, param_name: v, 'seed': seed_start + s}
            result = run_experiment(p)
            if result.get('valid', False):
                seed_results.append(result)
        if seed_results:
            # Average numeric columns across seeds
            avg = {}
            for key in seed_results[0]:
                vals = [r[key] for r in seed_results if key in r]
                if isinstance(vals[0], (int, float, bool, np.integer,
                                        np.floating, np.bool_)):
                    avg[key] = float(np.mean(vals))
                else:
                    avg[key] = vals[0]
            avg[param_name] = v
            rows.append(avg)
    return pd.DataFrame(rows)


def cross_scenario_comparison(scenarios: dict[str, dict]) -> pd.DataFrame:
    """Run all named scenarios and return a comparison DataFrame.

    Parameters
    ----------
    scenarios : dict mapping scenario name -> parameter overrides

    Returns a DataFrame with scenarios as columns and metrics as rows.
    """
    from .config import DEFAULT_PARAMS
    results = {}
    for name, overrides in scenarios.items():
        params = {**DEFAULT_PARAMS, **overrides}
        metrics = run_experiment(params)
        if metrics.get('valid', False):
            results[name] = metrics
    df = pd.DataFrame(results)
    df = df.drop(index=['valid'], errors='ignore')
    return df
