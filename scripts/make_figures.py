"""Regenerate README figures and print the stylized-fact numbers.

Usage: python scripts/make_figures.py  (seed 42, default parameters)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_abm import DEFAULT_PARAMS, MarketModel
from market_abm.analytics import (compute_autocorrelation,
                                  compute_return_statistics, hill_estimator)
from market_abm.visualization import (plot_price_and_fundamental,
                                      plot_return_distribution,
                                      plot_autocorrelation_panel)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 42


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = MarketModel({**DEFAULT_PARAMS, "seed": SEED})
    model.run()
    data = model.output.variables.MarketModel
    returns = data["log_return"].values

    fig, ax = plt.subplots(figsize=(10, 3.5))
    plot_price_and_fundamental(data, ax=ax)
    ax.set_title(f"Price vs fundamental (seed {SEED}, {len(data)} observations)")
    fig.tight_layout()
    fig.savefig(OUT / "price_vs_fundamental.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_return_distribution(returns, ax_hist=axes[0], ax_qq=axes[1])
    fig.tight_layout()
    fig.savefig(OUT / "return_distribution.png", dpi=130)
    plt.close(fig)

    stats = compute_return_statistics(returns)
    acf = compute_autocorrelation(returns, nlags=5)
    print("seed", SEED, "n", stats["n"])
    print("std", stats["std"], "excess kurtosis", stats["kurtosis"],
          "skew", stats["skewness"])
    print("ACF(r) lag1", acf["acf_returns"][1])
    print("ACF(r^2) lags1-5", acf["acf_squared_returns"][1:6])
    print("ACF(|r|) lags1-5", acf["acf_abs_returns"][1:6])
    print("95% band", 1.96 / np.sqrt(len(returns)))
    print("Hill (top 5%)", hill_estimator(returns))


if __name__ == "__main__":
    main()
