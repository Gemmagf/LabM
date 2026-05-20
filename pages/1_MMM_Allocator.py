"""
Page 1 — Interactive MMM Budget Allocator.

Lets the visitor reallocate budget across 5 marketing channels and see
the expected revenue update live, with 90% credible intervals from a
Bayesian MMM. The ⭐ button applies the SciPy-optimised allocation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Make the MMM core helpers importable from the lab folder
ROOT = Path(__file__).resolve().parent.parent
MMM_LAB = ROOT / "01_mmm_lab"
sys.path.insert(0, str(MMM_LAB))
from mmm_core import build_lag_matrix, adstock_all_draws, channel_revenue

DATA = MMM_LAB / "data"

CHANNEL_COLORS = {
    "TV": "#d62728",
    "Out of Home": "#ff7f0e",
    "Print": "#8c564b",
    "Facebook": "#1f77b4",
    "Google Search": "#2ca02c",
}
# Map the raw column names (kept in Catalan in the underlying model) to
# nicer English labels for the deployed UI.
CHANNEL_DISPLAY = {
    "TV": "TV",
    "Publicitat exterior": "Out of Home",
    "Premsa": "Print",
    "Facebook": "Facebook",
    "Google Search": "Google Search",
}
SPEND_COLS = {
    "tv_S": "TV", "ooh_S": "Publicitat exterior", "print_S": "Premsa",
    "facebook_S": "Facebook", "search_S": "Google Search",
}
M_MIN, M_MAX = 0.3, 2.0

st.set_page_config(page_title="MMM Allocator", page_icon="📊", layout="wide")


@st.cache_data
def load_model() -> dict:
    """Load posterior samples and precompute adstock per channel."""
    post = np.load(DATA / "mmm_posterior.npz", allow_pickle=True)
    channels = [str(c) for c in post["channels"]]
    decay, sat_k, beta = post["decay"], post["sat_k"], post["beta"]
    revenue_mean = float(post["revenue_mean"])
    spend_max = post["spend_max"]
    max_lag = int(post["max_lag"])

    df = pd.read_csv(DATA / "weekly.csv").rename(columns=SPEND_COLS)
    raw_spend = {c: df[c].to_numpy() for c in channels}
    spend_total = np.array([raw_spend[c].sum() for c in channels])

    adstock_draws = {}
    for i, c in enumerate(channels):
        norm = raw_spend[c] / spend_max[i]
        lag_mat = build_lag_matrix(norm, max_lag)
        adstock_draws[c] = adstock_all_draws(lag_mat, decay[i], max_lag)

    return {
        "channels": channels, "sat_k": sat_k, "beta": beta,
        "revenue_mean": revenue_mean, "spend_total": spend_total,
        "adstock_draws": adstock_draws,
    }


@st.cache_data
def load_optimum() -> dict:
    df = pd.read_csv(DATA / "budget_optim.csv")
    return dict(zip(df["canal"], df["multiplicador"]))


def total_revenue_draws(model: dict, multipliers: dict) -> np.ndarray:
    out = np.zeros_like(model["beta"][0])
    for i, c in enumerate(model["channels"]):
        out = out + channel_revenue(
            multipliers[c], model["adstock_draws"][c],
            model["sat_k"][i], model["beta"][i], model["revenue_mean"],
        )
    return out


# ─────────────────────────────────────────────────────────────────────
model = load_model()
optimum = load_optimum()
channels = model["channels"]
spend_total = model["spend_total"]
budget_total = spend_total.sum()

st.title("📊 MMM Budget Allocator")
st.caption(
    "Move the sliders to reallocate the budget across channels and see the "
    "expected revenue update live. Based on a Bayesian MMM with 4,000 "
    "posterior samples."
)

# Initial slider state
for c in channels:
    st.session_state.setdefault(f"m_{c}", 1.0)

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("⭐ Optimise the budget for me", use_container_width=True):
    for c in channels:
        st.session_state[f"m_{c}"] = float(np.clip(optimum[c], M_MIN, M_MAX))
    st.rerun()
if b2.button("↩️ Reset to current spend", use_container_width=True):
    for c in channels:
        st.session_state[f"m_{c}"] = 1.0
    st.rerun()

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Budget per channel")
    multipliers = {}
    for i, c in enumerate(channels):
        m = st.slider(
            CHANNEL_DISPLAY[c],
            min_value=M_MIN, max_value=M_MAX, step=0.05,
            key=f"m_{c}",
            help=f"Current spend: {spend_total[i]/1e6:.2f} M€",
        )
        multipliers[c] = m
        eur = m * spend_total[i]
        st.caption(f"→ {eur/1e6:.2f} M€  ({m:.2f}× of current spend)")

rev_now = total_revenue_draws(model, {c: 1.0 for c in channels})
rev_sel = total_revenue_draws(model, multipliers)
uplift = rev_sel - rev_now
sel_budget = sum(multipliers[c] * spend_total[i] for i, c in enumerate(channels))
lo, hi = np.percentile(rev_sel, [5, 95])

with right:
    st.subheader("Expected revenue")
    m1, m2 = st.columns(2)
    m1.metric("Marketing revenue", f"{rev_sel.mean()/1e6:.1f} M€",
              f"{uplift.mean()/1e6:+.1f} M€ vs current")
    m2.metric("Total budget used", f"{sel_budget/1e6:.2f} M€",
              f"{(sel_budget-budget_total)/1e6:+.2f} M€ vs 14.53 M€",
              delta_color="off")
    st.caption(f"90% credible interval: "
               f"[{lo/1e6:.1f}, {hi/1e6:.1f}] M€")

    prob = (uplift > 0).mean() * 100
    if abs(sel_budget - budget_total) / budget_total > 0.05:
        st.warning(
            f"⚠️ Your total budget ({sel_budget/1e6:.2f} M€) drifted from "
            f"14.53 M€. The exercise is reallocation, not adding spend."
        )
    elif uplift.mean() > 0:
        st.success(
            f"✅ This allocation increases expected revenue. "
            f"Probability of positive uplift: {prob:.0f}%."
        )
    else:
        st.info("This allocation does not improve on current. Try ⭐.")

st.subheader("Budget split — current vs your choice")
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(channels))
width = 0.38
actual_pct = spend_total / budget_total * 100
sel_eur = np.array([multipliers[c] * spend_total[i] for i, c in enumerate(channels)])
sel_pct = sel_eur / sel_eur.sum() * 100
display_names = [CHANNEL_DISPLAY[c] for c in channels]
ax.bar(x - width/2, actual_pct, width, color="#bbb", label="Current")
ax.bar(x + width/2, sel_pct, width,
       color=[CHANNEL_COLORS[CHANNEL_DISPLAY[c]] for c in channels],
       label="Your choice")
ax.set_xticks(x)
ax.set_xticklabels(display_names, rotation=12, ha="right")
ax.set_ylabel("% of total budget")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
st.pyplot(fig)

with st.expander("ℹ️ How this works"):
    st.markdown(
        "- Each channel has a **saturation curve** learned by a Bayesian MMM "
        "(step 5 of the pipeline). Spending twice as much does **not** "
        "produce twice the revenue.\n"
        "- The revenue shown is the **mean of 4,000 posterior samples**; "
        "the 90% interval reflects genuine model uncertainty.\n"
        "- The ⭐ button applies the budget computed by a SciPy SLSQP "
        "optimiser (step 6), with each channel bounded to 0.3×–2.0×.\n"
        "- ⚠️ Doubling a small channel is **extrapolating** beyond observed "
        "spend — treat the recommendation as a hypothesis to validate with "
        "a controlled test."
    )
