"""Stylized facts validation and statistical analysis."""

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
