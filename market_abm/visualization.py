"""Matplotlib plotting functions for LOB model outputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .analytics import compute_autocorrelation, compute_return_statistics


def plot_price_and_fundamental(data: pd.DataFrame,
                                ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data['price'], label='Market Price',
            alpha=0.9, linewidth=0.8)
    ax.plot(data.index, data['fundamental'], label='Fundamental',
            alpha=0.7, linewidth=0.8, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')
    ax.set_title('Price vs Fundamental Value')
    ax.legend()
    return ax


def plot_return_distribution(returns: np.ndarray,
                              ax_hist: plt.Axes | None = None,
                              ax_qq: plt.Axes | None = None
                              ) -> tuple[plt.Axes, plt.Axes]:
    if ax_hist is None or ax_qq is None:
        _, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 4))

    standardized = (returns - np.mean(returns)) / np.std(returns)
    ax_hist.hist(standardized, bins=80, density=True, alpha=0.7,
                 color='steelblue', edgecolor='none', label='Simulated')
    x = np.linspace(-5, 5, 200)
    ax_hist.plot(x, norm.pdf(x), 'r-', linewidth=1.5, label='Normal')
    ax_hist.set_xlabel('Standardized Return')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Return Distribution')
    ax_hist.legend()
    ax_hist.set_xlim(-5, 5)

    sorted_returns = np.sort(standardized)
    theoretical = norm.ppf(np.linspace(0.001, 0.999, len(sorted_returns)))
    ax_qq.scatter(theoretical, sorted_returns, s=2, alpha=0.5,
                  color='steelblue')
    lim = max(abs(theoretical.min()), abs(theoretical.max()))
    ax_qq.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')
    ax_qq.set_title('QQ Plot vs Normal')
    return ax_hist, ax_qq


def plot_autocorrelation_panel(returns: np.ndarray, nlags: int = 50,
                                axes: list | None = None) -> list[plt.Axes]:
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    acf_data = compute_autocorrelation(returns, nlags=nlags)
    conf = 1.96 / np.sqrt(len(returns))
    lags = np.arange(nlags + 1)

    titles = ['ACF of Returns', 'ACF of |Returns|', 'ACF of Returns squared']
    keys = ['acf_returns', 'acf_abs_returns', 'acf_squared_returns']

    for ax, title, key in zip(axes, titles, keys):
        ax.bar(lags[1:], acf_data[key][1:], width=0.6,
               color='steelblue', alpha=0.7)
        ax.axhline(conf, color='red', linestyle='--',
                    linewidth=0.8, alpha=0.6)
        ax.axhline(-conf, color='red', linestyle='--',
                    linewidth=0.8, alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title(title)
    return list(axes)


def plot_spread_over_time(data: pd.DataFrame,
                           ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(data.index, data['spread'], linewidth=0.8,
            color='darkorange', alpha=0.8)
    mean_spread = data['spread'].replace(0, np.nan).mean()
    ax.axhline(mean_spread, color='red', linestyle='--',
               linewidth=0.8, alpha=0.6,
               label=f'Mean: {mean_spread:.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Spread')
    ax.set_title('Bid-Ask Spread Over Time')
    ax.legend()
    return ax


def plot_volume_over_time(data: pd.DataFrame,
                           ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(data.index, data['volume'], width=1.0,
           color='steelblue', alpha=0.6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Trades')
    ax.set_title('Volume Per Step')
    return ax


def plot_pnl_by_strategy(metrics: dict[str, dict],
                          ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    means = [metrics[n]['mean_pnl'] for n in names]
    stds = [metrics[n]['std_pnl'] for n in names]
    colors = ['#2196F3', '#F44336', '#9E9E9E']
    ax.bar(names, means, yerr=stds, color=colors[:len(names)],
           alpha=0.8, capsize=5)
    ax.set_ylabel('PnL')
    ax.set_title('PnL by Strategy')
    ax.axhline(0, color='black', linewidth=0.5)
    return ax


def plot_wealth_evolution(data: pd.DataFrame,
                           ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    wealth_cols = {
        'wealth_fundamental': ('Fundamental', '#2196F3'),
        'wealth_trend': ('Trend', '#F44336'),
        'wealth_noise': ('Noise', '#9E9E9E'),
    }
    for col, (label, color) in wealth_cols.items():
        if col in data.columns:
            ax.plot(data.index, data[col], label=label,
                    color=color, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Wealth')
    ax.set_title('Wealth Evolution by Strategy')
    ax.legend()
    return ax
