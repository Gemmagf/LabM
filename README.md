# LabM — Marketing Science & Data Engineering Portfolio

Two self-contained labs that demonstrate end-to-end data work: a **causal
marketing-mix model** and (planned) a **big-data engineering pipeline**.

| Lab | Question it answers | Status |
|-----|--------------------|--------|
| **01 — Causal MMM Lab** | What is the true incremental ROI of each marketing channel, and how should the budget be reallocated? | ✅ Complete |
| **02 — Scale Lab** | Can a CTR-prediction pipeline be built and validated on ~196M real ad-log rows, on a laptop? | ✅ Complete |

---

## 🥇 Lab 01 — Causal MMM Lab

> *"Marketing Science with Honest Counterfactuals"*

A Bayesian Marketing Mix Model that estimates the **incremental** revenue
contribution of five advertising channels, quantifies the **uncertainty** of
every estimate, and turns the result into a **budget recommendation**.

### The headline result

The advertiser spends **62 % of its €14.5M budget on Out-of-Home (OOH)**.
The model finds OOH's ROI credible interval is **[0.10, 1.74]** — it cannot be
confidently called profitable. Reallocating the *same* budget toward
underfunded channels yields:

```
Expected revenue uplift:  +31.5 M€   (90% CI: +15.6 to +50.8 M€)
Probability the uplift is positive:  100%
Extra money spent:  0 €  — reallocation only
```

### How it works — a 7-step pipeline

| Step | Script | What it does | Outcome |
|------|--------|--------------|---------|
| 1 | `01_download.py` | Fetch the dataset | 208 weeks of channel spend + revenue |
| 2 | `02_explorar.py` | Exploratory charts | Budget is 83% offline, 12% digital |
| 3 | `03_baseline.py` | Naïve OLS regression | R² = 0.44, implausible ROIs (confounding) |
| 4 | `04_adstock_saturacio.py` | Adstock + saturation transforms | R² = 0.87, Durbin-Watson 1.1 → 2.1 |
| 5 | `05_bayesian_mmm.py` | Bayesian MMM with PyMC | ROI per channel with 90% credible intervals |
| 6 | `06_optimitzador.py` | Budget optimisation (SLSQP) | Reallocation plan, +31.5 M€ expected |
| 7 | `07_app.py` | Streamlit budget allocator | Interactive what-if tool |

### Methodology notes

- **Adstock** (carryover): geometric decay, learned per channel. The data
  corrected hand-picked priors — e.g. TV's effective memory was shorter than
  assumed, paid search's longer.
- **Saturation** (diminishing returns): exponential response curve
  `1 − exp(−x/k)`. Chosen over the Hill `xᵅ` form because the latter produced
  >1600 sampler divergences (exploding gradients near zero spend); the
  exponential form converged cleanly (1 divergence, R-hat ≈ 1.006).
- **Confounding control**: Fourier seasonality terms + linear trend +
  competitor sales, so seasonal demand is not mis-attributed to digital ads.
- **Optimisation**: the saturation curve makes this a *nonlinear* convex
  problem — `scipy` SLSQP, not linear programming. Each channel is bounded to
  0.3×–2.0× of current spend (realistic rebalancing).

### Honest limitations

This lab is named for honesty about counterfactuals — so:

- The dataset (Meta's Robyn `dt_simulated_weekly`) is **simulated** from
  realistic patterns. No truly-real MMM dataset is public; companies do not
  release marketing spend. The *methodology* is what transfers.
- Facebook and Search ROI intervals are wide ([5–46], [2–41]) — small-budget
  channels carry little information. Tightening them needs a geo-experiment.
- The optimiser's recommendation pushes most channels to their constraint
  bounds, i.e. the model *wants* an even more aggressive rebalance. Doubling
  small channels extrapolates beyond observed spend — treat the plan as a
  hypothesis to validate with a staged test, not a guarantee.

### Key figures (`01_mmm_lab/figures/`)

- `09_descomposicio.png` — weekly revenue decomposed into base + each channel
- `11_roi_bayesia.png` — ROI per channel with 90% credible intervals
- `16_corbes_resposta.png` — saturation curves, current vs recommended spend
- `17_uplift.png` — posterior distribution of the revenue uplift

---

## 🥈 Lab 02 — Scale Lab

> *"CTR prediction on ~196M real ad impressions — on a laptop"*

A click-through-rate (CTR) prediction pipeline built with **PySpark**, trained
and evaluated on a full day of the **Criteo 1TB Click Logs** —
**195,841,983 real ad-impression rows**.

### The headline result

| Metric | Model | Trivial baseline |
|--------|-------|------------------|
| AUC-ROC | **0.766** | 0.500 |
| AUC-PR | 0.128 | 0.032 |
| log-loss | **0.126** | 0.142 |

The logistic-regression model cuts log-loss **11.3 %** below the constant-rate
baseline. Trained on 156.7M rows in ~77 minutes on a 32 GB laptop — no cloud,
no cluster.

### How it works — a 4-step pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_download.py` | Download Criteo Parquet files from Hugging Face (configurable subset → full day) |
| 2 | `02_explorar.py` | Spark exploration: 3.2 % CTR, up to 42 % missing values, categoricals with millions of distinct values |
| 3 | `03_features.py` | `pyspark.ml` pipeline: impute, log-transform, **feature hashing** (26 categoricals → 2¹⁶ buckets) |
| 4 | `04_model.py` | Train/test split, logistic regression, evaluate with AUC / log-loss |

### Methodology notes

- **Why not accuracy:** with only 3.2 % positives, a model that always predicts
  "no click" scores 96.8 % accuracy. AUC-ROC, AUC-PR and log-loss are the
  honest metrics.
- **Feature hashing:** one categorical column has ~2.4M distinct values —
  one-hot encoding would mean millions of columns. The hashing trick maps them
  to a fixed 2¹⁶ buckets. The hash space is sized for a laptop; a cluster would
  use 2²⁰+.
- **Memory engineering:** training repeatedly hit Spark `OutOfMemoryError`. The
  fix: **materialise the train/test split to disk as Parquet** instead of
  caching 157M sparse vectors in memory, plus tuned executor cores and
  partition size. The same code runs unchanged on a cluster.

### Honest limitations

- Logistic regression is the *baseline*. Production CTR systems use
  gradient-boosted trees or deep models (DLRM) to push AUC toward 0.79+ — the
  natural next iteration.
- This uses **1 day of the 24** available. The pipeline is day-agnostic; the
  full 4-billion-row dataset would need a real cluster.
- Hashed features are not interpretable — a known trade-off of the hashing
  trick versus explicit encodings.

### Key figure (`02_scale_lab/figures/`)

- `01_roc.png` — ROC curve of the CTR model on 39M held-out rows

---

## Tech stack

**Marketing science (Lab 01):** `PyMC` · `ArviZ` · `statsmodels` · `SciPy` · `Streamlit`
**Data engineering (Lab 02):** `PySpark` · `Hugging Face Hub` · `pyarrow`
**Shared:** `Python` · `NumPy` / `pandas` · `Matplotlib` · `pytest`

## Running it

```bash
# from the LabM/ directory
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# run the MMM pipeline in order
.venv/bin/python 01_mmm_lab/01_download.py
.venv/bin/python 01_mmm_lab/02_explorar.py
.venv/bin/python 01_mmm_lab/03_baseline.py
.venv/bin/python 01_mmm_lab/04_adstock_saturacio.py
.venv/bin/python 01_mmm_lab/05_bayesian_mmm.py     # ~25s of sampling
.venv/bin/python 01_mmm_lab/06_optimitzador.py

# launch the interactive allocator
.venv/bin/streamlit run 01_mmm_lab/07_app.py
```

Each script prints a plain-language summary and writes its charts to
`01_mmm_lab/figures/`.

### Scale Lab (PySpark — requires Java 17)

```bash
brew install openjdk@17        # PySpark needs a JDK

.venv/bin/python 02_scale_lab/01_download.py      # 20-file dev subset (~1 GB)
.venv/bin/python 02_scale_lab/02_explorar.py
.venv/bin/python 02_scale_lab/03_features.py
.venv/bin/python 02_scale_lab/04_model.py

.venv/bin/python 02_scale_lab/01_download.py all  # full day (~196M rows, ~12 GB)
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

20 tests in total. `01_mmm_lab/test_mmm_core.py` covers the adstock, saturation
and optimiser maths; `02_scale_lab/test_scale_core.py` covers the ROC
computation and runs the Spark feature-cleaning logic on a live SparkSession.

## Data sources

- **MMM Lab** — Robyn `dt_simulated_weekly` dataset, Meta, open-source
  ([facebookexperimental/Robyn](https://github.com/facebookexperimental/Robyn)).
- **Scale Lab** — Criteo 1TB Click Logs, CC-BY-NC-SA
  ([criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs)).
