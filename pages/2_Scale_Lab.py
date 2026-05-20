"""
Page 2 — Scale Lab Results (static showcase).

Shows the outcome of training a CTR model on the full day of Criteo
1TB Click Logs (~196M real rows) using PySpark on a 32 GB laptop.
"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SCALE_LAB = ROOT / "02_scale_lab"

st.set_page_config(page_title="Scale Lab", page_icon="⚡", layout="wide")

st.title("⚡ Scale Lab — CTR prediction on 196M real rows")
st.caption(
    "A PySpark pipeline trained on a full day of the Criteo 1TB Click "
    "Logs on a 32 GB laptop. No cluster, no cloud."
)

# ── Headline metrics ──────────────────────────────────────────────────
st.subheader("Headline result (on 39.2M held-out rows)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows processed", "195,841,983", help="Full day of Criteo logs")
c2.metric("AUC-ROC", "0.766", "vs 0.500 random")
c3.metric("log-loss", "0.126", "-11.3% vs baseline", delta_color="inverse")
c4.metric("Training time", "~77 min", help="On a 32 GB MacBook")

st.markdown(
    "**Why these metrics, not accuracy:** the click-through rate is only "
    "**3.2%**, so a model that always predicts *no click* already scores "
    "96.8% accuracy. AUC-ROC, AUC-PR and log-loss are the honest metrics "
    "for an imbalanced ranking problem."
)

st.divider()

# ── ROC curve ─────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1])

with left:
    st.subheader("ROC curve")
    roc_path = SCALE_LAB / "figures" / "01_roc.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.info("Run `02_scale_lab/04_model.py` to regenerate the ROC chart.")
    st.caption(
        "Computed on 39.2M held-out rows. AUC 0.766 means the model "
        "correctly ranks a random click above a random non-click 77% of "
        "the time."
    )

with right:
    st.subheader("The pipeline")
    st.markdown(
        """
1. **Download** — Criteo Parquet files via `huggingface_hub` (script supports a 20-file dev subset or the full day).
2. **Explore** — Spark EDA: CTR 3.2%, up to 42% missing in integer features, categoricals with ~2.4M distinct values.
3. **Feature engineering** — `pyspark.ml.Pipeline`: impute nulls, log-transform, **feature hashing** (26 categoricals → 2¹⁶ buckets).
4. **Model** — logistic regression on 156.7M training rows. Evaluated with AUC-ROC, AUC-PR and log-loss.
        """
    )

st.divider()

# ── Engineering story ─────────────────────────────────────────────────
st.subheader("The engineering story (the part that matters in interviews)")

st.markdown(
    """
Training kept failing with `OutOfMemoryError` despite tuning the JVM heap
and partition size. **The diagnosis:** Spark's in-memory columnar cache
of 157M sparse 65K-dimensional vectors did not fit — and would not, on
*any* single-machine heap.

**The fix:** materialise the train/test split to disk as Parquet and let
Spark stream through it from disk instead of caching it in memory. The
same code runs unchanged on a cluster — there it would simply cache in
memory across nodes.
"""
)

with st.expander("What I had to tune"):
    st.markdown(
        """
- **Hash dimensionality** — 2¹⁸ buckets caused gradient-aggregation OOM on the laptop. Cut to 2¹⁶ (still industry-legitimate; production CTR systems use 2²⁰+).
- **Spark heap & cores** — driver memory raised to 16 GB and concurrency capped at `local[6]` to relieve GC pressure.
- **Partition size** — repartitioned to ~150K rows per partition so each task could finish without blowing its memory slice.
- **No `.persist()`** — wrote train/test to Parquet on disk instead. Reads stream; nothing OOMs.
        """
    )

st.divider()

# ── Honest limitations ────────────────────────────────────────────────
st.subheader("Honest limitations")
st.markdown(
    """
- Logistic regression is the **baseline**. Production CTR systems use gradient-boosted trees or deep models (DLRM) to push AUC toward 0.79+ — the natural next iteration.
- This runs on **1 day of the 24** available. The pipeline is day-agnostic; the full 4-billion-row dataset would need a real cluster.
- Hashed features are not interpretable — a known trade-off of the hashing trick.
"""
)

st.divider()
st.markdown(
    "**Source code:** "
    "[github.com/Gemmagf/LabM/tree/main/02_scale_lab]"
    "(https://github.com/Gemmagf/LabM/tree/main/02_scale_lab)"
)
