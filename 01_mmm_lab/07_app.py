"""
Pas 7 — App Streamlit: el "budget allocator".

Mou els sliders i veu com canvia el revenue esperat (amb interval de
credibilitat). El botó "Optimitza" aplica el resultat del pas 6.

Per executar-la:
    .venv/bin/streamlit run 01_mmm_lab/07_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmm_core import build_lag_matrix, adstock_all_draws, channel_revenue

HERE = Path(__file__).parent
DATA = HERE / "data"

CHANNEL_COLORS = {
    "TV": "#d62728",
    "Publicitat exterior": "#ff7f0e",
    "Premsa": "#8c564b",
    "Facebook": "#1f77b4",
    "Google Search": "#2ca02c",
}
SPEND_COLS = {
    "tv_S": "TV", "ooh_S": "Publicitat exterior", "print_S": "Premsa",
    "facebook_S": "Facebook", "search_S": "Google Search",
}
M_MIN, M_MAX = 0.3, 2.0


@st.cache_data
def load_model() -> dict:
    """Carrega la posterior i precalcula l'adstock de cada canal (tots els draws)."""
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
st.set_page_config(page_title="MMM Budget Allocator", layout="wide")
model = load_model()
optimum = load_optimum()
channels = model["channels"]
spend_total = model["spend_total"]
budget_total = spend_total.sum()

st.title("💰 MMM Budget Allocator")
st.caption("Mou el pressupost entre canals i mira el revenue esperat. "
           "Basat en un Bayesian MMM amb 4.000 draws de la posterior.")

# Estat inicial dels sliders
for i, c in enumerate(channels):
    st.session_state.setdefault(f"m_{c}", 1.0)

# Botons d'acció (s'executen abans dels sliders)
b1, b2, _ = st.columns([1, 1, 3])
if b1.button("⭐ Optimitza'm el pressupost", use_container_width=True):
    for c in channels:
        st.session_state[f"m_{c}"] = float(np.clip(optimum[c], M_MIN, M_MAX))
    st.rerun()
if b2.button("↩️ Torna a l'actual", use_container_width=True):
    for c in channels:
        st.session_state[f"m_{c}"] = 1.0
    st.rerun()

left, right = st.columns([1, 1.3])

# ── Sliders ──
with left:
    st.subheader("Pressupost per canal")
    multipliers = {}
    for i, c in enumerate(channels):
        m = st.slider(
            f"{c}", min_value=M_MIN, max_value=M_MAX, step=0.05,
            key=f"m_{c}",
            help=f"Actual: {spend_total[i]/1e6:.2f} M€",
        )
        multipliers[c] = m
        eur = m * spend_total[i]
        st.caption(f"→ {eur/1e6:.2f} M€  ({m:.2f}× del pressupost actual)")

# ── Càlculs ──
rev_now = total_revenue_draws(model, {c: 1.0 for c in channels})
rev_sel = total_revenue_draws(model, multipliers)
uplift = rev_sel - rev_now
sel_budget = sum(multipliers[c] * spend_total[i] for i, c in enumerate(channels))
lo, hi = np.percentile(rev_sel, [5, 95])

# ── Resultats ──
with right:
    st.subheader("Revenue esperat")
    m1, m2 = st.columns(2)
    m1.metric("Revenue marketing", f"{rev_sel.mean()/1e6:.1f} M€",
              f"{uplift.mean()/1e6:+.1f} M€ vs actual")
    m2.metric("Pressupost total usat", f"{sel_budget/1e6:.2f} M€",
              f"{(sel_budget-budget_total)/1e6:+.2f} M€ vs 14.53 M€",
              delta_color="off")
    st.caption(f"Interval de credibilitat 90%: "
               f"[{lo/1e6:.1f}, {hi/1e6:.1f}] M€")

    prob = (uplift > 0).mean() * 100
    if abs(sel_budget - budget_total) / budget_total > 0.05:
        st.warning(f"⚠️ El pressupost total ({sel_budget/1e6:.2f} M€) s'allunya "
                   f"dels 14.53 M€. L'exercici és REPARTIR, no augmentar.")
    elif uplift.mean() > 0:
        st.success(f"✅ Aquest repartiment millora el revenue esperat. "
                   f"Probabilitat de guany positiu: {prob:.0f}%.")
    else:
        st.info("Aquest repartiment no millora l'actual. Prova el botó ⭐.")

# ── Gràfic: repartiment actual vs seleccionat ──
st.subheader("Repartiment del pressupost")
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(channels))
width = 0.38
actual_pct = spend_total / budget_total * 100
sel_eur = np.array([multipliers[c] * spend_total[i] for i, c in enumerate(channels)])
sel_pct = sel_eur / sel_eur.sum() * 100
ax.bar(x - width/2, actual_pct, width, color="#bbb", label="Actual")
ax.bar(x + width/2, sel_pct, width,
       color=[CHANNEL_COLORS[c] for c in channels], label="El teu repartiment")
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=12, ha="right")
ax.set_ylabel("% del pressupost")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
st.pyplot(fig)

with st.expander("ℹ️ Com funciona això"):
    st.markdown(
        "- Cada canal té una **corba de saturació** apresa amb un Bayesian MMM "
        "(pas 5). Gastar el doble **no** dóna el doble de revenue.\n"
        "- El revenue mostrat és la **mitjana de 4.000 draws** de la posterior; "
        "l'interval del 90% reflecteix la incertesa real del model.\n"
        "- El botó ⭐ aplica l'assignació òptima calculada amb `scipy` (pas 6), "
        "amb cada canal limitat a l'interval 0.3×–2.0×.\n"
        "- ⚠️ Doblar canals petits és **extrapolar** més enllà de les dades "
        "observades: tracta la recomanació com a hipòtesi a validar amb un test."
    )
