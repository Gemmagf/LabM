"""
LabM Portfolio — multi-page Streamlit app.

Entry script. Auto-discovers pages from the sibling `pages/` directory.
Run locally:   .venv/bin/streamlit run portfolio_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="LabM Portfolio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 LabM Portfolio")
st.markdown(
    "Two end-to-end labs that pair **marketing science** with "
    "**data engineering** — built to demonstrate Senior DS / ML "
    "Engineer-level work end to end."
)

st.divider()

# ── Two lab cards side by side ────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.header("🥇 Lab 01 — Causal MMM")
    st.markdown(
        "**Marketing Science with Honest Counterfactuals.** A Bayesian "
        "Marketing Mix Model with adstock, saturation and a budget "
        "optimiser that turns model output into a concrete "
        "reallocation plan."
    )
    m1, m2 = st.columns(2)
    m1.metric("Expected uplift", "+31.5 M€", help="At unchanged total budget")
    m2.metric("90% credible interval", "[+15.6, +50.8] M€")
    st.caption(
        "**Stack:** PyMC · ArviZ · SciPy · Streamlit  "
        "·  **Dataset:** Robyn `dt_simulated_weekly`"
    )
    st.page_link(
        "pages/1_MMM_Allocator.py",
        label="**Open the interactive Budget Allocator →**",
        icon="📊",
    )

with right:
    st.header("🥈 Lab 02 — Scale Lab")
    st.markdown(
        "**CTR prediction on ~196M real ad impressions, on a laptop.** "
        "A PySpark pipeline trained on a full day of the Criteo 1TB "
        "Click Logs — feature hashing, logistic regression, honest "
        "metrics."
    )
    m1, m2 = st.columns(2)
    m1.metric("AUC-ROC", "0.766", "vs 0.500 random")
    m2.metric("log-loss", "0.126", "-11.3% vs baseline", delta_color="inverse")
    st.caption(
        "**Stack:** PySpark · pyspark.ml · Hugging Face Hub  "
        "·  **Dataset:** Criteo 1TB Click Logs"
    )
    st.page_link(
        "pages/2_Scale_Lab.py",
        label="**See the Scale Lab results →**",
        icon="⚡",
    )

st.divider()

# ── About / links ─────────────────────────────────────────────────────
st.subheader("About this project")
st.markdown(
    "Each lab is fully reproducible from the repo: real datasets, "
    "every script prints a plain-language summary, and 20 pytest "
    "tests cover the maths.  \n\n"
    "**Source code:** "
    "[github.com/Gemmagf/LabM](https://github.com/Gemmagf/LabM)"
)

with st.sidebar:
    st.markdown("### LabM Portfolio")
    st.caption(
        "Built end-to-end on a 32 GB MacBook.  \n"
        "No cloud spend. No managed services."
    )
    st.divider()
    st.markdown(
        "[📂 Source on GitHub](https://github.com/Gemmagf/LabM)"
    )
