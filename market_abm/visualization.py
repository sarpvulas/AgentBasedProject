"""Matplotlib plotting functions for LOB model outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .analytics import (compute_autocorrelation, compute_return_statistics,
                         rolling_volatility, drawdown_series)


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


# ── New visualisation functions ──────────────────────────────────────────


def plot_rolling_volatility(returns: np.ndarray, window: int = 50,
                             ax: plt.Axes | None = None) -> plt.Axes:
    """Rolling-window standard deviation of returns.

    Shows volatility clustering visually: sustained runs of high and low
    volatility appear as hills and valleys rather than random fluctuations.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    rv = rolling_volatility(returns, window=window)
    ax.plot(rv, linewidth=0.8, color='darkorange', alpha=0.85)
    mean_vol = np.nanmean(rv)
    ax.axhline(mean_vol, color='red', linestyle='--', linewidth=0.8,
               alpha=0.6, label=f'Mean: {mean_vol:.4f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Rolling Std Dev')
    ax.set_title(f'Rolling Volatility (window={window})')
    ax.legend()
    return ax


def plot_drawdown(prices: np.ndarray,
                   ax: plt.Axes | None = None) -> plt.Axes:
    """Drawdown from running peak, expressed as percentage below peak.

    Drawdown quantifies the worst peak-to-trough decline. In the trend-heavy
    scenario you expect deeper and more persistent drawdowns due to
    momentum-driven bubble-crash dynamics.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    dd = drawdown_series(prices) * 100  # percent
    ax.fill_between(range(len(dd)), dd, alpha=0.4, color='#F44336')
    ax.plot(dd, linewidth=0.6, color='#D32F2F', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown from Peak')
    ax.invert_yaxis()
    return ax


def plot_multi_seed_boxplots(seed_df: pd.DataFrame,
                              metrics: list[str] | None = None,
                              axes: list[plt.Axes] | None = None
                              ) -> list[plt.Axes]:
    """Box plots of key metrics across multiple random seeds.

    Demonstrates that the model's statistical properties are robust
    and not artifacts of a single random seed.
    """
    if metrics is None:
        metrics = ['kurtosis', 'hill_index', 'vol_clustering_acf',
                   'volatility', 'mean_abs_mispricing', 'max_drawdown']
    # Filter to metrics that exist in the dataframe
    metrics = [m for m in metrics if m in seed_df.columns]
    n = len(metrics)
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]
    labels = {
        'kurtosis': 'Excess Kurtosis',
        'hill_index': 'Hill Tail Index',
        'vol_clustering_acf': 'ACF(|r|) Lags 1-5',
        'volatility': 'Return Std Dev',
        'mean_abs_mispricing': 'Mean |P - F|',
        'max_drawdown': 'Max Drawdown',
    }
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#607D8B']
    for ax, metric, color in zip(axes, metrics, colors):
        vals = seed_df[metric].dropna()
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        bp['medians'][0].set_color('black')
        ax.set_title(labels.get(metric, metric), fontsize=10)
        ax.set_xticks([])
        mean_val = vals.mean()
        ax.axhline(mean_val, color=color, linestyle=':', linewidth=0.8,
                   alpha=0.7)
        ax.text(1.35, mean_val, f'{mean_val:.3f}', fontsize=8,
                va='center', color=color)
    return list(axes)


def plot_multi_seed_pass_rates(seed_df: pd.DataFrame,
                                ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart showing the pass rate of each stylized fact across seeds."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    fact_cols = ['fat_tails_pass', 'vol_clustering_pass', 'no_autocorr_pass',
                 'non_normality_pass', 'tail_index_pass']
    labels = ['Fat Tails', 'Vol. Clustering', 'No Autocorr.', 'Non-Normality',
              'Tail Index']
    existing = [(c, l) for c, l in zip(fact_cols, labels) if c in seed_df.columns]
    if not existing:
        return ax
    cols, labs = zip(*existing)
    rates = [seed_df[c].mean() * 100 for c in cols]
    colors = ['#4CAF50' if r >= 80 else '#FF9800' if r >= 50 else '#F44336'
              for r in rates]
    bars = ax.bar(labs, rates, color=colors, alpha=0.8, edgecolor='white')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{rate:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Stylized Fact Pass Rates Across Seeds')
    ax.axhline(80, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
    return ax


def plot_sensitivity_line(sweep_df: pd.DataFrame, param_name: str,
                           metric_name: str, ax: plt.Axes | None = None,
                           color: str = '#2196F3',
                           ylabel: str | None = None) -> plt.Axes:
    """Line plot of a metric against a swept parameter value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep_df[param_name], sweep_df[metric_name],
            marker='o', markersize=4, linewidth=1.5, color=color, alpha=0.85)
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel(ylabel or metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} vs {param_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.2)
    return ax
