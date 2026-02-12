"""Reusable matplotlib plotting functions for model outputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from .analytics import compute_autocorrelation, compute_return_statistics


def plot_price_and_fundamental(data: pd.DataFrame, ax: plt.Axes | None = None,
                                **kwargs) -> plt.Axes:
    """Plot price and fundamental value time series."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data['price'], label='Market Price', alpha=0.9, linewidth=0.8)
    ax.plot(data.index, data['fundamental'], label='Fundamental', alpha=0.7,
            linewidth=0.8, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')
    ax.set_title('Price vs Fundamental Value')
    ax.legend()
    return ax


def plot_strategy_fractions(data: pd.DataFrame, ax: plt.Axes | None = None,
                             **kwargs) -> plt.Axes:
    """Plot strategy fraction evolution as stacked area chart."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    cols = ['frac_fundamentalist', 'frac_chartist', 'frac_noise']
    labels = ['Fundamentalist', 'Chartist', 'Noise']
    colors = ['#2196F3', '#F44336', '#9E9E9E']

    ax.stackplot(data.index, *[data[c] for c in cols],
                 labels=labels, colors=colors, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction')
    ax.set_title('Strategy Composition')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    return ax


def plot_return_distribution(returns: np.ndarray, ax_hist: plt.Axes | None = None,
                              ax_qq: plt.Axes | None = None,
                              **kwargs) -> tuple[plt.Axes, plt.Axes]:
    """Plot return histogram with normal overlay, and QQ plot."""
    if ax_hist is None or ax_qq is None:
        fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
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

    # QQ plot
    sorted_returns = np.sort(standardized)
    theoretical = norm.ppf(np.linspace(0.001, 0.999, len(sorted_returns)))
    ax_qq.scatter(theoretical, sorted_returns, s=2, alpha=0.5, color='steelblue')
    lim = max(abs(theoretical.min()), abs(theoretical.max()))
    ax_qq.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')
    ax_qq.set_title('QQ Plot vs Normal')
    ax_qq.set_aspect('equal')

    return ax_hist, ax_qq


def plot_autocorrelation_panel(returns: np.ndarray, nlags: int = 50,
                                axes: list | None = None,
                                **kwargs) -> list[plt.Axes]:
    """Three-panel ACF plot: returns, |returns|, returns²."""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    acf_data = compute_autocorrelation(returns, nlags=nlags)
    conf = 1.96 / np.sqrt(len(returns))
    lags = np.arange(nlags + 1)

    titles = ['ACF of Returns', 'ACF of |Returns|', 'ACF of Returns²']
    keys = ['acf_returns', 'acf_abs_returns', 'acf_squared_returns']

    for ax, title, key in zip(axes, titles, keys):
        ax.bar(lags[1:], acf_data[key][1:], width=0.6, color='steelblue', alpha=0.7)
        ax.axhline(conf, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(-conf, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title(title)

    return list(axes)


def plot_simulation_dashboard(data: pd.DataFrame, returns: np.ndarray | None = None,
                               figsize: tuple = (16, 14),
                               **kwargs) -> plt.Figure:
    """Full 4-panel simulation dashboard.

    Panels:
      1. Price vs Fundamental
      2. Strategy Fractions
      3. Return Distribution (histogram + QQ)
      4. Autocorrelation Panel (3 sub-panels)
    """
    if returns is None:
        returns = data['log_return'].values

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    # Row 1: Price + Fundamental
    ax_price = fig.add_subplot(gs[0, :])
    plot_price_and_fundamental(data, ax=ax_price)

    # Row 2: Strategy fractions
    ax_strat = fig.add_subplot(gs[1, :])
    plot_strategy_fractions(data, ax=ax_strat)

    # Row 3: Return distribution
    ax_hist = fig.add_subplot(gs[2, :2])
    ax_qq = fig.add_subplot(gs[2, 2])
    plot_return_distribution(returns, ax_hist=ax_hist, ax_qq=ax_qq)

    # Row 4: ACF panel
    ax_acf = [fig.add_subplot(gs[3, i]) for i in range(3)]
    plot_autocorrelation_panel(returns, axes=ax_acf)

    # Add summary stats as suptitle
    stats = compute_return_statistics(returns)
    fig.suptitle(
        f"Market ABM Dashboard  |  Kurt={stats['kurtosis']:.2f}  "
        f"Skew={stats['skewness']:.3f}  JB p={stats['jb_pvalue']:.4f}",
        fontsize=13, fontweight='bold', y=1.01
    )

    return fig
