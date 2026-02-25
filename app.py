"""Streamlit dashboard for the heterogeneous-agent financial market ABM."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_abm.config import DEFAULT_PARAMS
from market_abm.model import MarketModel
from market_abm.analytics import compute_return_statistics, validate_stylized_facts
from market_abm.visualization import (
    plot_price_and_fundamental,
    plot_strategy_fractions,
    plot_return_distribution,
    plot_autocorrelation_panel,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market ABM Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Presets ──────────────────────────────────────────────────────────────────

PRESETS = {
    "Default": {},
    "Calm Market": {
        "beta": 1.0,
        "chi": 1.0,
        "gamma": 0.0,
        "herding": 0.0,
        "noise_sigma": 0.1,
    },
    "Bubble Regime": {
        "beta": 15.0,
        "chi": 5.0,
        "gamma": 4.0,
        "herding": 3.0,
        "switch_prob": 0.12,
    },
    "Crash Mode": {
        "beta": 20.0,
        "chi": 6.0,
        "gamma": 1.0,
        "herding": 3.5,
        "phi": 0.2,
        "noise_sigma": 0.5,
    },
}

# Slider config: param_key → (label, min, max, default, step, format)
_SLIDERS = {
    # -- Agents --
    "n_agents":          ("n_agents",             10,    500,   DEFAULT_PARAMS["n_agents"],          1,     "%d"),
    "phi":               ("phi (φ)",              0.0,   3.0,   DEFAULT_PARAMS["phi"],               0.05,  "%.2f"),
    "chi":               ("chi (χ)",              0.0,   10.0,  DEFAULT_PARAMS["chi"],               0.1,   "%.1f"),
    "noise_sigma":       ("noise_sigma (σ)",      0.0,   1.0,   DEFAULT_PARAMS["noise_sigma"],       0.01,  "%.2f"),
    "chartist_memory":   ("chartist_memory",      1,     20,    DEFAULT_PARAMS["chartist_memory"],   1,     "%d"),
    # -- Market Maker --
    "lambda_mm":         ("lambda_mm (λ)",        0.001, 0.1,   DEFAULT_PARAMS["lambda_mm"],         0.001, "%.3f"),
    "gamma":             ("gamma (γ)",            0.0,   10.0,  DEFAULT_PARAMS["gamma"],             0.1,   "%.1f"),
    "mm_noise_sigma":    ("mm_noise_sigma",       0.0,   0.02,  DEFAULT_PARAMS["mm_noise_sigma"],    0.001, "%.3f"),
    # -- Strategy Switching --
    "beta":              ("beta (β)",             0.0,   30.0,  DEFAULT_PARAMS["beta"],              0.5,   "%.1f"),
    "switch_prob":       ("switch_prob",          0.01,  1.0,   DEFAULT_PARAMS["switch_prob"],       0.01,  "%.2f"),
    "herding":           ("herding",              0.0,   5.0,   DEFAULT_PARAMS["herding"],           0.1,   "%.1f"),
    "fitness_ema_alpha": ("fitness_ema_alpha (α)", 0.01,  1.0,   DEFAULT_PARAMS["fitness_ema_alpha"], 0.01,  "%.2f"),
    # -- Simulation --
    "steps":             ("steps",                100,   10000, DEFAULT_PARAMS["steps"],             100,   "%d"),
}


def _apply_preset(name: str) -> None:
    """Write preset overrides (merged with defaults) into session state slider keys."""
    merged = {**DEFAULT_PARAMS, **PRESETS.get(name, {})}
    for k in _SLIDERS:
        if k in merged:
            # Cast to same type as the slider default
            default_type = type(_SLIDERS[k][3])
            st.session_state[k] = default_type(merged[k])
    st.session_state["preset"] = name


def _slider(key: str) -> float | int:
    """Create a sidebar slider from the _SLIDERS config and return its value."""
    label, mn, mx, default, step, fmt = _SLIDERS[key]
    # Only pass value= on first render; after that, session_state owns the value.
    kwargs: dict = dict(min_value=mn, max_value=mx, step=step, format=fmt, key=key)
    if key not in st.session_state:
        kwargs["value"] = default
    return st.slider(label, **kwargs)


@st.cache_data(show_spinner=False)
def run_simulation(params_frozen: tuple) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the ABM and return (data, returns). Cached on params."""
    params = dict(params_frozen)
    model = MarketModel(params)
    model.run()
    data = model.output.variables.MarketModel
    returns = data["log_return"].values
    return data, returns


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Market ABM")

# Preset buttons
st.sidebar.markdown("**Presets**")
preset_cols = st.sidebar.columns(4)
for i, name in enumerate(PRESETS):
    if preset_cols[i].button(name, use_container_width=True):
        _apply_preset(name)

active_preset = st.session_state.get("preset", "Default")
st.sidebar.caption(f"Active: **{active_preset}**")

# Slider groups
with st.sidebar.expander("Agents", expanded=True):
    n_agents        = _slider("n_agents")
    phi             = _slider("phi")
    chi             = _slider("chi")
    noise_sigma     = _slider("noise_sigma")
    chartist_memory = _slider("chartist_memory")

with st.sidebar.expander("Market Maker", expanded=True):
    lambda_mm      = _slider("lambda_mm")
    gamma          = _slider("gamma")
    mm_noise_sigma = _slider("mm_noise_sigma")

with st.sidebar.expander("Strategy Switching", expanded=True):
    beta              = _slider("beta")
    switch_prob       = _slider("switch_prob")
    herding           = _slider("herding")
    fitness_ema_alpha = _slider("fitness_ema_alpha")

with st.sidebar.expander("Simulation", expanded=True):
    steps = _slider("steps")
    if "seed" not in st.session_state:
        st.session_state["seed"] = DEFAULT_PARAMS["seed"]
    seed  = st.number_input("seed", step=1, key="seed")

# Run button
run_clicked = st.sidebar.button("▶  Run Simulation", type="primary",
                                use_container_width=True)

# ── Build params and run ────────────────────────────────────────────────────

if run_clicked:
    params = {
        **DEFAULT_PARAMS,
        "n_agents":          n_agents,
        "phi":               phi,
        "chi":               chi,
        "noise_sigma":       noise_sigma,
        "chartist_memory":   chartist_memory,
        "lambda_mm":         lambda_mm,
        "gamma":             gamma,
        "mm_noise_sigma":    mm_noise_sigma,
        "beta":              beta,
        "switch_prob":       switch_prob,
        "herding":           herding,
        "fitness_ema_alpha": fitness_ema_alpha,
        "steps":             steps,
        "seed":              int(seed),
    }
    params_frozen = tuple(sorted(params.items()))

    with st.spinner("Running simulation..."):
        data, returns = run_simulation(params_frozen)

    st.session_state["data"] = data
    st.session_state["returns"] = returns
    st.session_state["run_params"] = params

# ── Main area ────────────────────────────────────────────────────────────────

if "data" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click **▶  Run Simulation**.")
    st.stop()

data: pd.DataFrame = st.session_state["data"]
returns: np.ndarray = st.session_state["returns"]
total_steps = len(data)

# ── Overview chart with highlighted range ────────────────────────────────────

st.markdown("### Full Timeline Overview")

# Range slider
start, end = st.slider(
    "Time range",
    min_value=0,
    max_value=total_steps - 1,
    value=(0, total_steps - 1),
    step=1,
    key="time_range",
)

# Overview figure — small, shows entire run with shaded selection
fig_overview, ax_ov = plt.subplots(figsize=(12, 2.5))
ax_ov.plot(data.index, data["price"], linewidth=0.6, alpha=0.9, label="Price")
ax_ov.plot(data.index, data["fundamental"], linewidth=0.6, alpha=0.6,
           linestyle="--", label="Fundamental")
ax_ov.axvspan(start, end, alpha=0.15, color="orange")
ax_ov.set_xlim(data.index[0], data.index[-1])
ax_ov.set_ylabel("Price")
ax_ov.legend(loc="upper left", fontsize=8)
ax_ov.tick_params(labelsize=8)
fig_overview.tight_layout()
st.pyplot(fig_overview)
plt.close(fig_overview)

# Slice data for zoomed views
data_slice = data.iloc[start : end + 1]
returns_slice = returns[start : end + 1]

if len(returns_slice) < 10:
    st.warning("Selected range too short for analysis. Widen the time range slider.")
    st.stop()

# ── Row 1: Price vs Fundamental (zoomed) ────────────────────────────────────

st.markdown("### Price vs Fundamental")
fig1, ax1 = plt.subplots(figsize=(12, 3.5))
plot_price_and_fundamental(data_slice, ax=ax1)
fig1.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ── Row 2: Strategy Fractions (zoomed) ──────────────────────────────────────

st.markdown("### Strategy Fractions")
fig2, ax2 = plt.subplots(figsize=(12, 3.5))
plot_strategy_fractions(data_slice, ax=ax2)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ── Row 3: Returns + Rolling Volatility ─────────────────────────────────────

st.markdown("### Returns & Rolling Volatility")
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

ax3a.plot(data_slice.index, returns_slice, linewidth=0.5, color="steelblue",
          alpha=0.8)
ax3a.axhline(0, color="black", linewidth=0.3)
ax3a.set_ylabel("Log Return")
ax3a.set_title("Returns")

rolling_vol = pd.Series(returns_slice, index=data_slice.index).rolling(20).std()
ax3b.plot(rolling_vol.index, rolling_vol.values, linewidth=0.8,
          color="darkorange")
ax3b.set_ylabel("Rolling σ (20-step)")
ax3b.set_xlabel("Step")
ax3b.set_title("Rolling Volatility")

fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ── Row 4: Return Distribution (hist + QQ) ──────────────────────────────────

st.markdown("### Return Distribution")
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))
plot_return_distribution(returns_slice, ax_hist=ax4a, ax_qq=ax4b)
fig4.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

# ── Row 5: ACF Panel ────────────────────────────────────────────────────────

st.markdown("### Autocorrelation Panel")
nlags = min(50, len(returns_slice) // 3)
if nlags >= 5:
    fig5, axes5 = plt.subplots(1, 3, figsize=(14, 3.5))
    plot_autocorrelation_panel(returns_slice, nlags=nlags, axes=list(axes5))
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
else:
    st.info("Range too short for meaningful ACF (need > 15 observations).")

# ── Stylized Facts Scorecard ────────────────────────────────────────────────

with st.expander("Stylized Facts Validation"):
    facts = validate_stylized_facts(returns_slice)

    for fact_name, result in facts.items():
        icon = "✅" if result["passed"] else "❌"
        label = fact_name.replace("_", " ").title()
        metrics = {k: v for k, v in result.items()
                   if k not in ("passed", "criterion")}
        metric_str = "  —  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        st.markdown(f"{icon}  **{label}**:  {metric_str}")
        st.caption(f"_Criterion: {result['criterion']}_")

    st.markdown("---")
    st.markdown("**Summary Statistics**")
    stats = compute_return_statistics(returns_slice)
    stats_df = pd.DataFrame(
        {k: [f"{v:.6f}" if isinstance(v, float) else v]
         for k, v in stats.items()}
    ).T
    stats_df.columns = ["Value"]
    st.dataframe(stats_df, use_container_width=True)
