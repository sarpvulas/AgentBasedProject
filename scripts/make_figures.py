"""Regenerate README figures and print the stylized-fact numbers.

Usage: python scripts/make_figures.py  (seed 42 figures and statistics,
then the 30-seed table over seeds 1-30; default parameters throughout).
All statistics use all per-step log returns, zeros included.
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
                                  compute_return_statistics, hill_estimator,
                                  validate_stylized_facts)
from market_abm.visualization import (plot_price_and_fundamental,
                                      plot_return_distribution,
                                      plot_autocorrelation_panel)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 42


def seed_row(seed):
    model = MarketModel({**DEFAULT_PARAMS, "seed": seed})
    model.run(display=False)
    data = model.output.variables.MarketModel
    r = data["log_return"].values
    acf = compute_autocorrelation(r, nlags=5)
    facts = validate_stylized_facts(r)
    trades = model.order_book.trade_history
    return {
        "seed": seed,
        "kurtosis": compute_return_statistics(r)["kurtosis"],
        "acf_r_lag1": acf["acf_returns"][1],
        "acf_r2_lag1": acf["acf_squared_returns"][1],
        "acf_r2_lag5": acf["acf_squared_returns"][5],
        "hill": hill_estimator(r),
        "band": 1.96 / np.sqrt(len(r)),
        "zero_returns": int((r == 0).sum()),
        "mean_abs_mispricing": float(np.mean(
            np.abs(data["price"] - data["fundamental"]))),
        "n_trades": len(trades),
        "self_trades": sum(t.buyer_id == t.seller_id for t in trades),
        "unsettled": model.n_unsettled,
        "two_sided_frac": float(np.mean(
            [s is not None for s in model.order_book.spread_history])),
        "fat_tails": facts["fat_tails"]["passed"],
        "vol_clustering": facts["volatility_clustering"]["passed"],
        "no_autocorr": facts["no_return_autocorrelation"]["passed"],
        "tail_index": facts["tail_index"]["passed"],
        "failing_lags": facts["no_return_autocorrelation"]["failing_lags"],
    }


def seed_table(seeds=range(1, 31)):
    import pandas as pd
    df = pd.DataFrame([seed_row(s) for s in seeds])
    print("\n30-seed table (seeds %d-%d): mean, min to max" % (
        min(seeds), max(seeds)))
    for col in ["kurtosis", "acf_r_lag1", "acf_r2_lag1", "acf_r2_lag5",
                "hill", "zero_returns", "n_trades", "two_sided_frac",
                "mean_abs_mispricing"]:
        print(f"{col:>15s}: mean {df[col].mean():.4f}, "
              f"min {df[col].min():.4f}, max {df[col].max():.4f}")
    print("kurtosis > 0 in", int((df["kurtosis"] > 0).sum()), "seeds")
    print("ACF(r^2) lag1 > 0 in", int((df.acf_r2_lag1 > 0).sum()), "seeds;",
          "above band in", int((df.acf_r2_lag1 > df.band).sum()))
    print("ACF(r) lag1 < -band in", int((df.acf_r_lag1 < -df.band).sum()),
          "seeds; |lag1| within band in",
          int((df.acf_r_lag1.abs() <= df.band).sum()))
    print("self-trades:", int(df.self_trades.sum()),
          "unsettled fills:", int(df.unsettled.sum()))
    for k in ["fat_tails", "vol_clustering", "no_autocorr", "tail_index"]:
        print(f"check {k}: passes in {int(df[k].sum())} of {len(df)} seeds")
    from collections import Counter
    print("failing lags (count of seeds):",
          dict(sorted(Counter(l for f in df.failing_lags for l in f).items())))
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = MarketModel({**DEFAULT_PARAMS, "seed": SEED})
    model.run(display=False)
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
    print("zero returns", int((returns == 0).sum()), "of", len(returns))
    trades = model.order_book.trade_history
    print("trades", len(trades), "self-trades",
          sum(t.buyer_id == t.seller_id for t in trades))
    print("two-sided book at end of step:", float(np.mean(
        [x is not None for x in model.order_book.spread_history])))
    print("mean |price - fundamental|",
          float(np.mean(np.abs(data["price"] - data["fundamental"]))))
    seed_table()


if __name__ == "__main__":
    main()
