"""Streamlit dashboard for the single-asset LOB market ABM."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

if "data" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click "
            "**Run Simulation**.")
    st.stop()

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
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
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
