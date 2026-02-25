"""Streamlit dashboard for the single-asset LOB market ABM."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Guide Tab CSS & Helpers ──────────────────────────────────────────────

GUIDE_CSS = """
<style>
@keyframes guideReveal {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.guide-card {
    background: transparent;
    border-bottom: 1px solid rgba(0,0,0,0.06);
    padding: 1.1rem 0.25rem 1.2rem;
    animation: guideReveal 0.4s ease both;
}
.guide-card:last-child { border-bottom: none; }
.guide-card-header {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
}
.guide-card-header .gc-icon {
    display: inline-block;
    margin-right: 0.35rem;
    font-style: normal;
}
.guide-card-body {
    line-height: 1.7;
    font-size: 0.94rem;
    color: #1a1a1a;
}
.guide-card-body code {
    background: #1a1a2e;
    color: #e0e0f0;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
    font-size: 0.84rem;
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
}
.guide-card-body b, .guide-card-body strong {
    font-weight: 600;
    color: #0a0a0a;
}
.guide-card-body ul {
    margin: 0.5rem 0 0.2rem;
    padding-left: 1.1rem;
    list-style: none;
}
.guide-card-body li {
    margin-bottom: 0.35rem;
    position: relative;
    padding-left: 0.2rem;
}
.guide-card-body li::before {
    content: '\\2013';
    position: absolute;
    left: -1.1rem;
    color: #bbb;
}
.guide-card-fx {
    margin-top: 0.55rem;
    font-size: 0.84rem;
    color: #666;
    letter-spacing: 0.01em;
}
.guide-section-title {
    margin-top: 2rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #111;
}
.guide-section-title h3 {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #111;
    margin: 0;
}
.guide-group-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #aaa;
    margin-top: 1.2rem;
    margin-bottom: 0.1rem;
}
.stagger-0  { animation-delay: 0s; }
.stagger-1  { animation-delay: 0.05s; }
.stagger-2  { animation-delay: 0.1s; }
.stagger-3  { animation-delay: 0.15s; }
.stagger-4  { animation-delay: 0.2s; }
.stagger-5  { animation-delay: 0.25s; }
.stagger-6  { animation-delay: 0.3s; }
.stagger-7  { animation-delay: 0.35s; }
.stagger-8  { animation-delay: 0.4s; }
.stagger-9  { animation-delay: 0.45s; }
.stagger-10 { animation-delay: 0.5s; }
.stagger-11 { animation-delay: 0.55s; }
</style>
"""


def guide_card(icon, title, body, color="blue", stagger=0):
    """Return an HTML guide card string."""
    return (
        f'<div class="guide-card stagger-{stagger}">'
        f'<div class="guide-card-header">'
        f'<span class="gc-icon">{icon}</span>{title}</div>'
        f'<div class="guide-card-body">{body}</div></div>'
    )


def guide_section(icon, title):
    """Return a section heading HTML string."""
    return (
        f'<div class="guide-section-title">'
        f'<h3>{icon} {title}</h3></div>'
    )


# ── Guide Content Dictionaries ───────────────────────────────────────────

OVERVIEW_CONTENT = {
    "simple": (
        "This model simulates a stock market with three types of traders "
        "who buy and sell through a <b>limit order book</b> — just like a "
        "real exchange. Each trader follows a different strategy, and the "
        "price emerges naturally from their interactions. No one sets the "
        "price; it comes from supply and demand."
    ),
    "technical": (
        "A heterogeneous agent-based model (HAM) with a continuous double "
        "auction mechanism (limit order book). Three agent types — noise, "
        "fundamental, and trend-following — submit limit and market orders. "
        "The price is determined endogenously by order matching. The "
        "fundamental value follows an Ornstein-Uhlenbeck process. Agent "
        "fractions are exogenous and fixed for each run."
    ),
}

TRADER_CONTENT = {
    "noise": {
        "icon": "🎲",
        "title": "Noise Traders",
        "color": "blue",
        "simple": (
            "These traders act <b>randomly</b> — they buy, sell, or sit out "
            "based on a coin flip. They represent uninformed activity in real "
            "markets (retail speculation, liquidity trades)."
            "<ul>"
            "<li>30% chance to buy, 30% to sell, 40% to do nothing</li>"
            "<li>Split evenly between limit orders and market orders</li>"
            "<li>They add <b>noise</b> to the price, making it harder for "
            "others to read the signal</li>"
            "</ul>"
        ),
        "technical": (
            "Random agents with uniform action probabilities: P(buy)=0.3, "
            "P(sell)=0.3, P(hold)=0.4. Order type is 50/50 limit vs market. "
            "Limit prices are set with random jitter around the bid-ask midpoint."
            "<ul>"
            "<li>Serve as the liquidity backbone — without noise, the book thins</li>"
            "<li>Their randomness prevents the market from locking into "
            "deterministic cycles</li>"
            "<li>Fraction = <code>1 - frac_fundamental - frac_trend</code></li>"
            "</ul>"
        ),
    },
    "fundamental": {
        "icon": "📊",
        "title": "Fundamental Traders",
        "color": "green",
        "simple": (
            "These traders know the <b>true value</b> of the asset. When the "
            "price is too low, they buy; when it's too high, they sell. Think "
            "of them as patient value investors."
            "<ul>"
            "<li>They act like a rubber band pulling the price toward the "
            "fundamental value</li>"
            "<li>Mostly use limit orders (80%) — they're patient</li>"
            "<li>Stronger when the sensitivity parameter is higher</li>"
            "</ul>"
        ),
        "technical": (
            "Trade based on perceived mispricing: "
            "<code>signal = (F(t) - P(t)) / P(t)</code>. Buy when signal "
            "exceeds a threshold, sell when below. "
            "Order type: 80% limit, 20% market."
            "<ul>"
            "<li>Limit prices placed randomly between current price and "
            "fundamental value — capturing the mispricing spread</li>"
            "<li><code>fundamental_sensitivity</code> scales the signal; "
            "higher values mean smaller mispricings trigger trades</li>"
            "<li>Primary stabilizing force — they anchor the price to F(t)</li>"
            "</ul>"
        ),
    },
    "trend": {
        "icon": "📈",
        "title": "Trend Followers",
        "color": "orange",
        "simple": (
            "These traders chase momentum — if the price is going up, they "
            "buy; if it's falling, they sell. Think of them as technical "
            "analysts riding the wave."
            "<ul>"
            "<li>They <b>amplify</b> price moves, creating bigger swings</li>"
            "<li>Mostly use market orders (80%) — they want in NOW</li>"
            "<li>Only act when the move is big enough (above the threshold)</li>"
            "</ul>"
        ),
        "technical": (
            "Momentum signal: <code>ret = P(t) - P(t-1)</code>. Buy when "
            "<code>ret > threshold</code>, sell when "
            "<code>ret < -threshold</code>. "
            "Order type: 80% market, 20% limit."
            "<ul>"
            "<li>Market-order dominance makes them aggressive price movers</li>"
            "<li><code>trend_threshold</code> filters out small fluctuations "
            "— set to 0 and they react to every tick</li>"
            "<li>Primary destabilizing force — positive feedback loops "
            "create volatility clustering and momentum</li>"
            "</ul>"
        ),
    },
}

PARAM_CONTENT = {
    "n_agents": {
        "icon": "👥", "title": "Number of Agents", "color": "blue",
        "simple": (
            "Total number of traders in the market. More agents means more "
            "orders, more liquidity, and a smoother price — like having more "
            "people at an auction."
        ),
        "technical": (
            "Total agent count <code>N</code>. Distributed across types by "
            "fractions. Higher N increases order flow density but CLT effects "
            "smooth aggregate demand, reducing fat tails. Use N~50-100 for "
            "pronounced stylized facts."
        ),
        "effects": "↑ smoother prices, tighter spreads · ↓ wilder swings, fatter tails",
    },
    "frac_fundamental": {
        "icon": "📊", "title": "Fundamental Fraction", "color": "blue",
        "simple": (
            "What share of traders are value investors. More fundamentalists "
            "means the price stays closer to the true value, with smaller swings."
        ),
        "technical": (
            "Fraction of agents assigned the fundamental strategy. "
            "Increases mean reversion in the price process. Fraction of noise "
            "traders = <code>1 - frac_fundamental - frac_trend</code>."
        ),
        "effects": "↑ price tracks fundamental, less volatility · ↓ price wanders, more noise",
    },
    "frac_trend": {
        "icon": "📈", "title": "Trend Fraction", "color": "blue",
        "simple": (
            "What share of traders chase momentum. More trend followers "
            "means bigger price swings and stronger momentum effects."
        ),
        "technical": (
            "Fraction of agents assigned the trend strategy. "
            "Increases positive autocorrelation in absolute returns "
            "(volatility clustering) and can generate bubble/crash dynamics."
        ),
        "effects": "↑ volatility clustering, momentum · ↓ more random-walk behavior",
    },
    "mu": {
        "icon": "🎯", "title": "Long-Run Mean (mu)", "color": "green",
        "simple": (
            "The 'fair value' the fundamental drifts toward over time — "
            "like the center of a pendulum's swing. The price will orbit "
            "this value in the long run."
        ),
        "technical": (
            "Long-run mean of the O-U process: "
            "<code>F(t+1) = F(t) + κ(μ - F(t)) + σε(t)</code>. "
            "The stationary distribution is "
            "<code>N(μ, σ²/2κ)</code>."
        ),
        "effects": "Sets the level around which fundamental and price fluctuate",
    },
    "kappa": {
        "icon": "🔄", "title": "Mean Reversion Speed (kappa)", "color": "green",
        "simple": (
            "How fast the fundamental value snaps back to the long-run mean. "
            "Higher = stronger pull back (like a stiffer spring). "
            "At the default (0.01), it takes ~69 steps to close half the gap."
        ),
        "technical": (
            "Mean reversion rate in the O-U process. "
            "Half-life of deviation: <code>ln(2)/κ</code> "
            "(default κ=0.01 → ~69 steps). "
            "Stationary variance: <code>σ²/(2κ)</code>. "
            "Higher κ reduces fundamental variance and dampens price volatility."
        ),
        "effects": "↑ tighter fundamental, faster reversion · ↓ wider wandering, slower reversion",
    },
    "fundamental_initial": {
        "icon": "📍", "title": "Initial Fundamental F(0)", "color": "green",
        "simple": (
            "The starting value of the fundamental. If this is far from the "
            "long-run mean (mu), you'll see the fundamental gradually drift "
            "toward mu at the start."
        ),
        "technical": (
            "Initial condition <code>F(0)</code>. If <code>F(0) ≠ μ</code>, "
            "the transient takes ~<code>3/κ</code> steps to reach stationarity. "
            "Set <code>F(0) = μ</code> to start in equilibrium."
        ),
        "effects": "Controls the initial transient; far from mu = visible drift at start",
    },
    "fundamental_sigma": {
        "icon": "🌊", "title": "Fundamental Volatility (sigma)", "color": "green",
        "simple": (
            "How much the fundamental value bounces around at each step. "
            "Higher sigma = a bumpier road for the 'true value', which "
            "creates more trading opportunities."
        ),
        "technical": (
            "Diffusion coefficient of the O-U process. "
            "Shock term: <code>σ·ε(t)</code> where <code>ε ~ N(0,1)</code>. "
            "Stationary std dev: <code>σ/√(2κ)</code>. "
            "Controls the exogenous information flow into the market."
        ),
        "effects": "↑ more fundamental variation, more trades · ↓ calmer fundamental, thinner activity",
    },
    "fundamental_sensitivity": {
        "icon": "🔍", "title": "Fundamental Sensitivity", "color": "orange",
        "simple": (
            "How sharp-eyed the value investors are. Higher sensitivity means "
            "they react to smaller mispricings — like having very picky "
            "bargain hunters."
        ),
        "technical": (
            "Multiplier on the mispricing signal "
            "<code>(F(t)-P(t))/P(t)</code>. "
            "Higher values lower the effective threshold for fundamental "
            "traders to act, increasing their order frequency and "
            "strengthening price-fundamental coupling."
        ),
        "effects": "↑ tighter price-fundamental coupling · ↓ price wanders from fundamental",
    },
    "trend_threshold": {
        "icon": "📏", "title": "Trend Threshold", "color": "orange",
        "simple": (
            "The minimum price move needed for trend followers to act. "
            "Like a noise filter — set it high and they only react to big "
            "moves; set it to 0 and they chase every tiny wiggle."
        ),
        "technical": (
            "Minimum <code>|P(t) - P(t-1)|</code> for trend agents to "
            "submit orders. Acts as a dead zone in the momentum signal. "
            "At 0, every nonzero return triggers a trend order."
        ),
        "effects": "↑ fewer but larger momentum trades · ↓ more frequent trend trading, more noise",
    },
    "stale_order_age": {
        "icon": "⏰", "title": "Stale Order Age", "color": "purple",
        "simple": (
            "How many steps a limit order sits in the book before it's "
            "automatically cancelled. Shorter = a tidier book; longer = "
            "more depth but with old, possibly irrelevant prices."
        ),
        "technical": (
            "Orders older than this threshold (in simulation steps) are "
            "removed from the LOB. Controls book depth and staleness. "
            "Lower values reduce depth but improve price relevance."
        ),
        "effects": "↑ deeper book, wider spreads · ↓ thinner book, tighter spreads",
    },
    "steps": {
        "icon": "⏱️", "title": "Simulation Steps", "color": "blue-gray",
        "simple": (
            "How long the simulation runs. More steps gives more data and "
            "more reliable statistics, but takes longer to compute."
        ),
        "technical": (
            "Total number of discrete time steps. Stylized facts tests need "
            "at least ~500 non-zero returns for reliable statistics. "
            "Confidence bands scale as <code>1.96/√N</code>."
        ),
        "effects": "↑ better statistics, longer runtime · ↓ faster but noisier results",
    },
    "seed": {
        "icon": "🎰", "title": "Random Seed", "color": "blue-gray",
        "simple": (
            "Controls the randomness. Same seed = same results every time. "
            "Change it to see a different 'universe' with the same rules."
        ),
        "technical": (
            "NumPy RNG seed for reproducibility. "
            "Fixes all random draws (agent actions, fundamental shocks, "
            "order routing). Use different seeds to explore outcome "
            "distributions under fixed parameters."
        ),
        "effects": "Different seeds show different realizations of the same stochastic process",
    },
}

CHART_CONTENT = {
    "price_fundamental": {
        "icon": "📉", "title": "Price vs Fundamental", "color": "rose",
        "simple": (
            "The blue line is the market price; the dashed line is the "
            "'true value'. Watch how the price orbits around the fundamental."
            "<ul>"
            "<li><b>Price hugs fundamental closely</b> → fundamentalists dominate</li>"
            "<li><b>Big gaps open up</b> → trend followers or noise are winning</li>"
            "<li><b>Sharp spikes that revert</b> → classic bubble-crash pattern</li>"
            "</ul>"
        ),
        "technical": (
            "Overlays market clearing price <code>P(t)</code> and O-U "
            "fundamental <code>F(t)</code>. The gap <code>P(t)-F(t)</code> "
            "is the mispricing that fundamental agents trade on."
            "<ul>"
            "<li>Persistent deviation → weak fundamentalist force</li>"
            "<li>Mean-reverting excursions → healthy stabilization</li>"
            "<li>Growing oscillations → trend feedback loop</li>"
            "</ul>"
        ),
    },
    "spread": {
        "icon": "↔️", "title": "Bid-Ask Spread", "color": "rose",
        "simple": (
            "The gap between the best buy and best sell price. A tight "
            "spread means the market is liquid and easy to trade."
            "<ul>"
            "<li><b>Consistently narrow</b> → healthy, active market</li>"
            "<li><b>Spikes up</b> → sudden loss of liquidity (stressful moment)</li>"
            "<li><b>Widens gradually</b> → order book is thinning out</li>"
            "</ul>"
        ),
        "technical": (
            "Best ask minus best bid at each step. Proxy for market "
            "liquidity and transaction costs."
            "<ul>"
            "<li>Spread inversely related to order book depth</li>"
            "<li>Widens during fast-moving markets (trend orders deplete book)</li>"
            "<li>Stale order age affects baseline spread level</li>"
            "</ul>"
        ),
    },
    "volume": {
        "icon": "📊", "title": "Volume", "color": "rose",
        "simple": (
            "How many shares trade at each step. High volume means lots of "
            "activity and usually a well-functioning market."
            "<ul>"
            "<li><b>Steady volume</b> → balanced participation</li>"
            "<li><b>Volume bursts</b> → trend followers piling in</li>"
            "<li><b>Low/zero volume</b> → no orders are matching</li>"
            "</ul>"
        ),
        "technical": (
            "Trade count per step (number of matched orders). "
            "Driven by order submission rate and book depth."
            "<ul>"
            "<li>Positively correlated with |returns| (volume-volatility relation)</li>"
            "<li>Market orders (trend agents) contribute most to execution volume</li>"
            "<li>Limit orders contribute to depth but may not execute</li>"
            "</ul>"
        ),
    },
    "wealth": {
        "icon": "💰", "title": "Wealth Evolution", "color": "rose",
        "simple": (
            "How much money each group of traders has over time (cash + "
            "the value of their holdings)."
            "<ul>"
            "<li><b>Fundamentalists rise steadily</b> → buying cheap, selling dear works</li>"
            "<li><b>Trend followers spike then crash</b> → momentum profit is fragile</li>"
            "<li><b>Noise traders drift</b> → random trading is a slow bleed</li>"
            "</ul>"
        ),
        "technical": (
            "Mean portfolio value per strategy: "
            "<code>W(t) = cash + position × P(t)</code>."
            "<ul>"
            "<li>Fundamentalists profit from mean-reversion (contrarian)</li>"
            "<li>Trend followers profit from serial correlation (momentum)</li>"
            "<li>Relative wealth depends on market regime and parameter mix</li>"
            "</ul>"
        ),
    },
    "return_dist": {
        "icon": "📐", "title": "Return Distribution", "color": "rose",
        "simple": (
            "<b>Left: Histogram</b> — how often each size of price change "
            "occurs. Real markets have 'fat tails' (extreme events happen "
            "more often than a bell curve predicts)."
            "<br><b>Right: QQ plot</b> — dots on the line = normal; dots "
            "curving away at the ends = fat tails."
            "<ul>"
            "<li><b>Heavy tails in histogram</b> → model captures real-world extremes</li>"
            "<li><b>S-shaped QQ plot</b> → classic fat-tail signature</li>"
            "</ul>"
        ),
        "technical": (
            "Histogram of log returns <code>r(t) = ln(P(t)/P(t-1))</code> "
            "with normal overlay, plus a QQ plot against N(0,1)."
            "<ul>"
            "<li>Excess kurtosis > 0 indicates fat tails (leptokurtic)</li>"
            "<li>QQ departures at extremes show tail heaviness</li>"
            "<li>Uses only non-zero returns (steps with trades)</li>"
            "</ul>"
        ),
    },
    "acf": {
        "icon": "🔁", "title": "Autocorrelation Panel", "color": "rose",
        "simple": (
            "Three charts showing whether patterns repeat over time:"
            "<ul>"
            "<li><b>Return ACF</b> — should be near zero (no easy predictions)</li>"
            "<li><b>Absolute Return ACF</b> — should be positive (big moves "
            "cluster together = volatility clustering)</li>"
            "<li><b>Squared Return ACF</b> — similar to absolute returns, "
            "another way to detect clustering</li>"
            "</ul>"
        ),
        "technical": (
            "Sample autocorrelation functions at lags 1 to ~50:"
            "<ul>"
            "<li><code>ACF(r)</code>: Tests weak-form efficiency. Should be "
            "insignificant if no linear predictability.</li>"
            "<li><code>ACF(|r|)</code>: Tests for volatility clustering "
            "(long memory in volatility). Positive = GARCH-like dynamics.</li>"
            "<li><code>ACF(r²)</code>: Related to <code>ACF(|r|)</code> "
            "but more sensitive to extreme values.</li>"
            "<li>Blue bands = 95% confidence: <code>±1.96/√N</code></li>"
            "</ul>"
        ),
    },
    "pnl": {
        "icon": "💵", "title": "PnL by Strategy", "color": "rose",
        "simple": (
            "A bar chart and table comparing how much each group made or "
            "lost. The Sharpe ratio tells you return per unit of risk — "
            "higher is better."
            "<ul>"
            "<li><b>Positive mean PnL</b> → strategy is profitable on average</li>"
            "<li><b>High Sharpe</b> → consistent profits (not just lucky)</li>"
            "<li><b>High Std PnL</b> → big variance — some agents win, others lose</li>"
            "</ul>"
        ),
        "technical": (
            "Per-strategy portfolio metrics at terminal time:"
            "<ul>"
            "<li><code>Mean PnL</code>: Average (W_T - W_0) across agents of that type</li>"
            "<li><code>Std PnL</code>: Cross-sectional dispersion</li>"
            "<li><code>Sharpe</code>: Mean PnL / Std PnL (cross-sectional)</li>"
            "<li><code>Mean Wealth</code>: Average W_T for the group</li>"
            "</ul>"
        ),
    },
    "stylized_facts": {
        "icon": "✅", "title": "Stylized Facts Validation", "color": "rose",
        "simple": (
            "A checklist of statistical patterns found in real financial "
            "markets. A good model should reproduce these. PASS means the "
            "model matches reality for that test."
            "<ul>"
            "<li><b>Fat Tails</b> → extreme events aren't rare</li>"
            "<li><b>Volatility Clustering</b> → calm and stormy periods clump together</li>"
            "<li><b>No Return Autocorrelation</b> → can't predict future returns from past</li>"
            "<li><b>Non-Normality</b> → returns aren't bell-curve shaped</li>"
            "<li><b>Tail Index</b> → tails are heavy but not infinitely so</li>"
            "</ul>"
        ),
        "technical": (
            "Statistical tests on log returns:"
            "<ul>"
            "<li><b>Fat Tails</b>: excess kurtosis > 0 AND Jarque-Bera p &lt; 0.05</li>"
            "<li><b>Volatility Clustering</b>: mean |ACF(|r|)| at lags 1-5 "
            "> <code>1.96/√N</code></li>"
            "<li><b>No Return Autocorrelation</b>: mean |ACF(r)| at lags 1-5 "
            "&lt; <code>3 × 1.96/√N</code></li>"
            "<li><b>Non-Normality</b>: Kolmogorov-Smirnov p &lt; 0.05</li>"
            "<li><b>Tail Index</b>: Hill estimator ∈ [2, 6] "
            "(empirical equities ~3-5)</li>"
            "</ul>"
        ),
    },
}

STYLIZED_FACTS_CONTENT = {
    "fat_tails": {
        "icon": "📊", "title": "Fat Tails", "color": "blue",
        "simple": (
            "In a bell curve, extreme events are very rare. In real markets, "
            "crashes and booms happen <b>much more often</b> than a bell "
            "curve predicts. 'Fat tails' means the distribution of returns "
            "has more extreme events than expected."
        ),
        "technical": (
            "Detected when excess kurtosis > 0 AND the Jarque-Bera test "
            "rejects normality (p &lt; 0.05). Kurtosis measures the weight "
            "of tails relative to N(0,1). A normal distribution has excess "
            "kurtosis = 0; empirical stock returns typically show 5-50+."
        ),
    },
    "vol_clustering": {
        "icon": "🌀", "title": "Volatility Clustering", "color": "green",
        "simple": (
            "Markets have calm periods and stormy periods. Big price moves "
            "tend to follow big moves, and small moves follow small moves — "
            "volatility <b>clusters</b> in time. Think of it as market mood "
            "swings that last a while."
        ),
        "technical": (
            "Measured as mean |ACF(|r|)| at lags 1-5. Passes when this "
            "exceeds the 95% confidence band <code>1.96/√N</code>. "
            "Indicates long memory in the volatility process. Generated by "
            "trend-following feedback: momentum → large returns → more "
            "momentum signals → more large returns."
        ),
    },
    "no_autocorrelation": {
        "icon": "🎯", "title": "No Return Autocorrelation", "color": "orange",
        "simple": (
            "You can't predict tomorrow's return from today's. If returns "
            "were correlated, everyone would exploit the pattern until it "
            "disappears. The market is approximately <b>unpredictable</b> "
            "at the level of returns."
        ),
        "technical": (
            "Passes when mean |ACF(r)| at lags 1-5 &lt; "
            "<code>3 × 1.96/√N</code>. A relaxed threshold (3×) accounts "
            "for finite-sample noise. Indicates weak-form efficiency: linear "
            "return predictability is absent even though nonlinear "
            "dependence (volatility clustering) persists."
        ),
    },
    "non_normality": {
        "icon": "📈", "title": "Non-Normality", "color": "purple",
        "simple": (
            "Returns don't follow a perfect bell curve. They're typically "
            "<b>peaked</b> in the center (many small moves) with <b>fat "
            "tails</b> (rare but huge moves). This is one of the most "
            "robust findings in empirical finance."
        ),
        "technical": (
            "Kolmogorov-Smirnov test against N(μ, σ²) fitted to the "
            "return sample. Rejects normality at p &lt; 0.05. "
            "Complements the Jarque-Bera test (which focuses on skewness "
            "and kurtosis) with a test on the full distribution shape."
        ),
    },
    "tail_index": {
        "icon": "📏", "title": "Tail Index", "color": "blue-gray",
        "simple": (
            "A number that measures <b>how heavy</b> the tails are. "
            "Real stock markets have a tail index between 2 and 6 — "
            "heavy enough for extreme events, but not so heavy that "
            "the variance is infinite. It's the 'Goldilocks zone' of risk."
        ),
        "technical": (
            "Hill estimator applied to the top 5% of |returns|. "
            "Passes when α ∈ [2, 6]. For α ≤ 2, the variance is infinite "
            "(too heavy); for α > 6, tails decay nearly as fast as a "
            "Gaussian (too thin). Empirical equities: ~3-5."
        ),
    },
}

from market_abm.agents import AgentType
from market_abm.analytics import (
    compute_portfolio_metrics,
    compute_return_statistics,
    validate_stylized_facts,
)
from market_abm.config import DEFAULT_PARAMS
from market_abm.model import MarketModel
from market_abm.visualization import (
    plot_autocorrelation_panel,
    plot_pnl_by_strategy,
    plot_price_and_fundamental,
    plot_return_distribution,
    plot_spread_over_time,
    plot_volume_over_time,
    plot_wealth_evolution,
)

st.set_page_config(page_title="Market ABM Dashboard", page_icon="📈",
                   layout="wide")


# ── Guide Tab Renderer ───────────────────────────────────────────────────

def render_guide_tab():
    """Render the educational guide tab."""
    st.markdown(GUIDE_CSS, unsafe_allow_html=True)

    technical = st.toggle("Technical Mode", value=False,
                          key="guide_technical_mode")
    mode = "technical" if technical else "simple"
    st.caption("Showing formulas and code-level detail."
               if technical else
               "Showing plain-English explanations with analogies.")

    # ── Model Overview
    st.markdown(guide_section("🏛️", "How This Model Works"),
                unsafe_allow_html=True)
    st.markdown(guide_card("🏛️", "Model Overview",
                           OVERVIEW_CONTENT[mode], "blue", 0),
                unsafe_allow_html=True)

    # ── Three Trader Types
    st.markdown(guide_section("🧑‍💼", "The Three Trader Types"),
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (key, t) in enumerate(TRADER_CONTENT.items()):
        with cols[i]:
            st.markdown(
                guide_card(t["icon"], t["title"], t[mode], t["color"], i),
                unsafe_allow_html=True,
            )

    # ── Parameter Guide
    st.markdown(guide_section("🎛️", "Parameter Guide"),
                unsafe_allow_html=True)

    param_groups = {
        "Agents": ["n_agents", "frac_fundamental", "frac_trend"],
        "Fundamental Process": ["mu", "kappa", "fundamental_initial",
                                "fundamental_sigma"],
        "Agent Behavior": ["fundamental_sensitivity", "trend_threshold"],
        "Order Book": ["stale_order_age"],
        "Simulation": ["steps", "seed"],
    }
    group_colors = {
        "Agents": "blue",
        "Fundamental Process": "green",
        "Agent Behavior": "orange",
        "Order Book": "purple",
        "Simulation": "blue-gray",
    }

    stagger_idx = 0
    for group_name, param_keys in param_groups.items():
        color = group_colors[group_name]
        st.markdown(f'<div class="guide-group-label">{group_name}</div>',
                    unsafe_allow_html=True)
        group_cols = st.columns(min(len(param_keys), 3))
        for j, pkey in enumerate(param_keys):
            p = PARAM_CONTENT[pkey]
            body = p[mode]
            if "effects" in p:
                body += f'<div class="guide-card-fx">{p["effects"]}</div>'
            with group_cols[j % len(group_cols)]:
                st.markdown(
                    guide_card(p["icon"], p["title"], body,
                               p["color"], stagger_idx % 12),
                    unsafe_allow_html=True,
                )
            stagger_idx += 1

    # ── How to Read Each Chart
    st.markdown(guide_section("📊", "How to Read Each Chart"),
                unsafe_allow_html=True)
    chart_keys = [
        "price_fundamental", "spread", "volume", "wealth",
        "return_dist", "acf", "pnl", "stylized_facts",
    ]
    chart_cols = st.columns(2)
    for i, ckey in enumerate(chart_keys):
        c = CHART_CONTENT[ckey]
        with chart_cols[i % 2]:
            st.markdown(
                guide_card(c["icon"], c["title"], c[mode],
                           c["color"], i % 12),
                unsafe_allow_html=True,
            )

    # ── Stylized Facts Explained
    st.markdown(guide_section("🔬", "Stylized Facts Explained"),
                unsafe_allow_html=True)
    st.markdown(
        guide_card(
            "🔬", "What Are Stylized Facts?",
            ("Stylized facts are statistical patterns observed across many "
             "different markets, time periods, and asset classes. A good "
             "model should reproduce these patterns <b>endogenously</b> — "
             "meaning they emerge from the agent interactions, not because "
             "we hard-coded them."
             if not technical else
             "Empirical regularities in financial return series that are "
             "robust across assets, markets, and frequencies. They serve as "
             "moment conditions for model validation. The tests below are "
             "applied to the simulated log return series."),
            "blue-gray", 0,
        ),
        unsafe_allow_html=True,
    )
    fact_keys = ["fat_tails", "vol_clustering", "no_autocorrelation",
                 "non_normality", "tail_index"]
    for i, fkey in enumerate(fact_keys):
        f = STYLIZED_FACTS_CONTENT[fkey]
        st.markdown(
            guide_card(f["icon"], f["title"], f[mode],
                       f["color"], (i + 1) % 12),
            unsafe_allow_html=True,
        )


# ── Presets ───────────────────────────────────────────────────────────────

PRESETS = {
    "Balanced": {},
    "Mostly Noise": {
        "frac_fundamental": 0.15, "frac_trend": 0.15,
    },
    "Fund. Heavy": {
        "frac_fundamental": 0.60, "frac_trend": 0.20,
    },
    "Trend Heavy": {
        "frac_fundamental": 0.20, "frac_trend": 0.60,
    },
}

_SLIDERS = {
    "n_agents":               ("n_agents",              10,    500,   DEFAULT_PARAMS["n_agents"],               1,     "%d"),
    "frac_fundamental":       ("frac_fundamental",      0.0,   1.0,   DEFAULT_PARAMS["frac_fundamental"],       0.01,  "%.2f"),
    "frac_trend":             ("frac_trend",            0.0,   1.0,   DEFAULT_PARAMS["frac_trend"],             0.01,  "%.2f"),
    "mu":                     ("mu (long-run mean)",    50.0,  200.0, DEFAULT_PARAMS["mu"],                     1.0,   "%.0f"),
    "kappa":                  ("kappa (reversion)",     0.001, 0.2,   DEFAULT_PARAMS["kappa"],                  0.001, "%.3f"),
    "fundamental_initial":    ("F(0) initial value",    50.0,  200.0, DEFAULT_PARAMS["fundamental_initial"],    1.0,   "%.0f"),
    "fundamental_sigma":      ("sigma (fundamental)",   0.01,  2.0,   DEFAULT_PARAMS["fundamental_sigma"],      0.01,  "%.2f"),
    "fundamental_sensitivity":("fund. sensitivity",     0.1,   10.0,  DEFAULT_PARAMS["fundamental_sensitivity"],0.1,   "%.1f"),
    "trend_threshold":        ("trend threshold",       0.0,   5.0,   DEFAULT_PARAMS["trend_threshold"],        0.1,   "%.1f"),
    "stale_order_age":        ("stale order age",       1,     50,    DEFAULT_PARAMS["stale_order_age"],        1,     "%d"),
    "steps":                  ("steps",                 100,   10000, DEFAULT_PARAMS["steps"],                  100,   "%d"),
}


def _apply_preset(name):
    merged = {**DEFAULT_PARAMS, **PRESETS.get(name, {})}
    for k in _SLIDERS:
        if k in merged:
            default_type = type(_SLIDERS[k][3])
            st.session_state[k] = default_type(merged[k])
    st.session_state["preset"] = name


def _slider(key):
    label, mn, mx, default, step, fmt = _SLIDERS[key]
    kwargs = dict(min_value=mn, max_value=mx, step=step, format=fmt, key=key)
    if key not in st.session_state:
        kwargs["value"] = default
    return st.slider(label, **kwargs)


@st.cache_data(show_spinner=False)
def run_simulation(params_frozen):
    params = dict(params_frozen)
    model = MarketModel(params)
    model.run()
    data = model.output.variables.MarketModel
    returns = data["log_return"].values

    last_price = (model.order_book.last_trade_price
                  or params['fundamental_initial'])
    pnl_metrics = {}
    for atype in AgentType:
        m = compute_portfolio_metrics(list(model.traders), atype, last_price)
        pnl_metrics[atype.name.capitalize()] = m

    return data, returns, pnl_metrics


# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.title("Market ABM")

st.sidebar.markdown("**Presets**")
preset_cols = st.sidebar.columns(4)
for i, name in enumerate(PRESETS):
    if preset_cols[i].button(name, use_container_width=True):
        _apply_preset(name)

active_preset = st.session_state.get("preset", "Balanced")
st.sidebar.caption(f"Active: **{active_preset}**")

with st.sidebar.expander("Agents", expanded=True):
    n_agents = _slider("n_agents")
    frac_fundamental = _slider("frac_fundamental")
    frac_trend = _slider("frac_trend")
    frac_noise = max(0, 1.0 - frac_fundamental - frac_trend)
    st.caption(f"Noise fraction: {frac_noise:.2f}")
    if frac_fundamental + frac_trend > 1.0:
        st.error("Fundamental + Trend fractions exceed 1.0!")

with st.sidebar.expander("Fundamental Process", expanded=True):
    fundamental_initial = _slider("fundamental_initial")
    mu = _slider("mu")
    kappa = _slider("kappa")
    fundamental_sigma = _slider("fundamental_sigma")

with st.sidebar.expander("Agent Behavior", expanded=False):
    fundamental_sensitivity = _slider("fundamental_sensitivity")
    trend_threshold = _slider("trend_threshold")

with st.sidebar.expander("Order Book", expanded=False):
    stale_order_age = _slider("stale_order_age")

with st.sidebar.expander("Simulation", expanded=True):
    steps = _slider("steps")
    if "seed" not in st.session_state:
        st.session_state["seed"] = DEFAULT_PARAMS["seed"]
    seed = st.number_input("seed", step=1, key="seed")

run_clicked = st.sidebar.button("Run Simulation", type="primary",
                                use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────

if run_clicked:
    params = {
        **DEFAULT_PARAMS,
        "n_agents": n_agents,
        "frac_fundamental": frac_fundamental,
        "frac_trend": frac_trend,
        "fundamental_initial": fundamental_initial,
        "mu": mu,
        "kappa": kappa,
        "fundamental_sigma": fundamental_sigma,
        "fundamental_sensitivity": fundamental_sensitivity,
        "trend_threshold": trend_threshold,
        "stale_order_age": stale_order_age,
        "steps": steps,
        "seed": int(seed),
    }
    params_frozen = tuple(sorted(params.items()))

    with st.spinner("Running simulation..."):
        data, returns, pnl_metrics = run_simulation(params_frozen)

    st.session_state["data"] = data
    st.session_state["returns"] = returns
    st.session_state["pnl_metrics"] = pnl_metrics

# ── Main area ─────────────────────────────────────────────────────────────

tab_sim, tab_guide = st.tabs(["Simulation", "Guide"])

with tab_guide:
    render_guide_tab()

with tab_sim:
    if "data" not in st.session_state:
        st.info("Configure parameters in the sidebar, then click "
                "**Run Simulation**.")
    else:
        data = st.session_state["data"]
        returns = st.session_state["returns"]
        pnl_metrics = st.session_state["pnl_metrics"]

        # Price vs Fundamental
        st.markdown("### Price vs Fundamental")
        fig1, ax1 = plt.subplots(figsize=(12, 3.5))
        plot_price_and_fundamental(data, ax=ax1)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # Spread and Volume
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Bid-Ask Spread")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            plot_spread_over_time(data, ax=ax2)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with col_b:
            st.markdown("### Volume")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            plot_volume_over_time(data, ax=ax3)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        # Wealth evolution
        st.markdown("### Wealth Evolution by Strategy")
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        plot_wealth_evolution(data, ax=ax4)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Return distribution
        st.markdown("### Return Distribution")
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4))
        nonzero = returns[returns != 0.0]
        if len(nonzero) > 20:
            plot_return_distribution(nonzero, ax_hist=ax5a, ax_qq=ax5b)
            fig5.tight_layout()
            st.pyplot(fig5)
        else:
            st.info("Not enough trades for return distribution.")
        plt.close(fig5)

        # ACF panel
        st.markdown("### Autocorrelation Panel")
        nlags = min(50, len(nonzero) // 3) if len(nonzero) > 15 else 0
        if nlags >= 5:
            fig6, axes6 = plt.subplots(1, 3, figsize=(14, 3.5))
            plot_autocorrelation_panel(nonzero, nlags=nlags, axes=list(axes6))
            fig6.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)

        # PnL by strategy
        st.markdown("### PnL by Strategy")
        col_c, col_d = st.columns([1, 2])
        with col_c:
            fig7, ax7 = plt.subplots(figsize=(5, 4))
            plot_pnl_by_strategy(pnl_metrics, ax=ax7)
            fig7.tight_layout()
            st.pyplot(fig7)
            plt.close(fig7)

        with col_d:
            pnl_df = pd.DataFrame({
                name: {
                    "Mean PnL": f"{m['mean_pnl']:.2f}",
                    "Std PnL": f"{m['std_pnl']:.2f}",
                    "Sharpe": f"{m['sharpe']:.3f}",
                    "Mean Wealth": f"{m['mean_wealth']:.2f}",
                    "N Agents": m['n'],
                }
                for name, m in pnl_metrics.items()
            })
            st.dataframe(pnl_df, use_container_width=True)

        # Stylized facts
        with st.expander("Stylized Facts Validation"):
            if len(nonzero) > 50:
                facts = validate_stylized_facts(nonzero)
                for fact_name, result in facts.items():
                    icon = "PASS" if result["passed"] else "FAIL"
                    label = fact_name.replace("_", " ").title()
                    metrics_items = {k: v for k, v in result.items()
                                     if k not in ("passed", "criterion")}
                    metric_str = "  |  ".join(
                        f"{k}={v:.4f}" if isinstance(v, float)
                        else f"{k}={v}"
                        for k, v in metrics_items.items()
                    )
                    st.markdown(f"**[{icon}] {label}**:  {metric_str}")
                    st.caption(f"Criterion: {result['criterion']}")

            stats = compute_return_statistics(nonzero)
            st.markdown("**Summary Statistics**")
            stats_df = pd.DataFrame(
                {k: [f"{v:.6f}" if isinstance(v, float) else v]
                 for k, v in stats.items()}
            ).T
            stats_df.columns = ["Value"]
            st.dataframe(stats_df, use_container_width=True)
