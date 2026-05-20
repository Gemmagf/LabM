"""
Pas 5 — Bayesian MMM amb PyMC.

Diferència respecte al pas 4:
  · Al pas 4 vam FIXAR a mà els hiperparàmetres (decay=0.65, etc.).
  · Aquí els APRENEM de les dades. Cada paràmetre té una distribució
    a posteriori → cada ROI ve amb un interval de credibilitat del 90%.

Estructura del model (per cada canal):
    spend → adstock(decay) → saturació(sat_k) → coeficient → contribució

    revenue = base + Σ contribucions + estacionalitat + tendència + competidor + soroll

Saturació: corba de rendiments decreixents  sat(x) = 1 - exp(-x / k).
És suau i estable per al sampler (la Hill x^α dóna divergències perquè
la derivada de x^α prop de zero es dispara).

Sortides:
  · data/mmm_posterior.nc   → la posterior desada (la fa servir el pas 6)
  · data/roi_bayesian.csv   → taula de ROI amb intervals
  · figures/11..14          → gràfics
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from mmm_core import build_lag_matrix, fourier_terms

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).parent
DATA = HERE / "data" / "weekly.csv"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

SPEND_COLS = {
    "tv_S": "TV",
    "ooh_S": "Publicitat exterior",
    "print_S": "Premsa",
    "facebook_S": "Facebook",
    "search_S": "Google Search",
}
CHANNELS = list(SPEND_COLS.values())
CHANNEL_COLORS = {
    "TV": "#d62728",
    "Publicitat exterior": "#ff7f0e",
    "Premsa": "#8c564b",
    "Facebook": "#1f77b4",
    "Google Search": "#2ca02c",
}
MAX_LAG = 8  # setmanes de memòria màxima per l'adstock


def group_dataset(idata, name: str):
    """Retorna un grup de la InferenceData/DataTree com a xarray.Dataset."""
    g = idata[name] if name in idata.children else getattr(idata, name)
    return g.to_dataset() if hasattr(g, "to_dataset") else g


def main() -> None:
    print("=" * 60)
    print("PAS 5 — Bayesian MMM (PyMC)")
    print("=" * 60)

    df = pd.read_csv(DATA, parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
    df = df.rename(columns=SPEND_COLS)
    T = len(df)

    # --- Escalat (estabilitat numèrica del sampler) ---
    revenue_mean = df["revenue"].mean()
    y = (df["revenue"] / revenue_mean).to_numpy()

    raw_spend = {c: df[c].to_numpy() for c in CHANNELS}
    spend_max = {c: max(raw_spend[c].max(), 1.0) for c in CHANNELS}
    norm_spend = {c: raw_spend[c] / spend_max[c] for c in CHANNELS}

    lag_mats = {c: build_lag_matrix(norm_spend[c], MAX_LAG) for c in CHANNELS}
    season = fourier_terms(T)
    trend = np.arange(T) / T
    competitor = (df["competitor_sales_B"] / df["competitor_sales_B"].max()).to_numpy()

    print(f"\n{T} setmanes · {len(CHANNELS)} canals · adstock fins a {MAX_LAG} setmanes")
    print("Construint el model...")

    coords = {"channel": CHANNELS}
    with pm.Model(coords=coords) as model:
        # Priors dels hiperparàmetres (un per canal)
        decay = pm.Beta("decay", alpha=2.0, beta=2.0, dims="channel")
        sat_k = pm.Gamma("sat_k", alpha=3.0, beta=6.0, dims="channel")   # mitjana 0.5
        beta = pm.HalfNormal("beta", sigma=0.5, dims="channel")

        # Controls
        intercept = pm.Normal("intercept", mu=0.9, sigma=0.3)
        season_beta = pm.Normal("season_beta", mu=0.0, sigma=0.25, shape=season.shape[1])
        trend_beta = pm.Normal("trend_beta", mu=0.0, sigma=0.5)
        comp_beta = pm.Normal("comp_beta", mu=0.0, sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        # Contribució de cada canal: adstock → saturació → coeficient
        contributions = []
        contrib_eur = []
        lags = pt.arange(MAX_LAG)
        for i, c in enumerate(CHANNELS):
            w = decay[i] ** lags
            w = w / pt.sum(w)
            adstocked = pt.dot(lag_mats[c], w)               # (T,)
            sat = 1.0 - pt.exp(-adstocked / sat_k[i])        # rendiments decreixents (0–1)
            contrib = beta[i] * sat                          # (T,) en unitats escalades
            contributions.append(contrib)
            contrib_eur.append(pt.sum(contrib) * revenue_mean)

        pm.Deterministic("contrib_eur", pt.stack(contrib_eur), dims="channel")

        mu = (
            intercept
            + sum(contributions)
            + pt.dot(season, season_beta)
            + trend_beta * trend
            + comp_beta * competitor
        )
        pm.Deterministic("mu", mu)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        print("Mostrejant la posterior (NUTS, 4 cadenes)... pot trigar uns minuts.\n")
        idata = pm.sample(
            draws=1000, tune=1500, chains=4, cores=4,
            target_accept=0.95, random_seed=42, progressbar=True,
        )

    posterior = group_dataset(idata, "posterior")
    sample_stats = group_dataset(idata, "sample_stats")

    # --- Diagnòstic de convergència ---
    print("\n" + "=" * 60)
    print("CONVERGÈNCIA")
    print("=" * 60)
    n_div = int(sample_stats["diverging"].values.sum())
    print(f"Divergències: {n_div}  (com menys, millor; idealment 0)")
    rhat = az.rhat(posterior, var_names=["decay", "sat_k", "beta", "intercept", "sigma"])
    max_rhat = float(max(np.nanmax(rhat[v].values) for v in rhat.data_vars))
    print(f"R-hat màxim:  {max_rhat:.4f}   (< 1.01 = convergència correcta)")
    if max_rhat > 1.01 or n_div > 20:
        print("  ⚠ convergència imperfecta; resultats orientatius.")
    else:
        print("  ✓ cadenes convergides.")

    # --- ROI per canal amb interval de credibilitat ---
    contrib_post = posterior["contrib_eur"].stack(s=("chain", "draw")).values  # (channel, samples)
    print("\n" + "=" * 60)
    print("ROI PER CANAL  (€ revenue incremental per € gastat)")
    print("=" * 60)
    print(f"\n{'Canal':<22} {'ROI mitjà':>10} {'IC 90%':>22} {'quota €':>10}")
    total_spend = sum(raw_spend[c].sum() for c in CHANNELS)
    rows = []
    for i, c in enumerate(CHANNELS):
        roi_samples = contrib_post[i] / raw_spend[c].sum()
        roi_mean = float(roi_samples.mean())
        lo, hi = (float(v) for v in np.percentile(roi_samples, [5, 95]))
        share = raw_spend[c].sum() / total_spend * 100
        rows.append({"canal": c, "roi_mean": roi_mean, "roi_lo": lo, "roi_hi": hi,
                     "budget_share_pct": share, "spend_total": raw_spend[c].sum()})
        print(f"{c:<22} {roi_mean:>10.2f}   [{lo:>6.2f}, {hi:>6.2f}]      {share:>6.1f} %")

    roi_df = pd.DataFrame(rows)
    roi_csv = HERE / "data" / "roi_bayesian.csv"
    roi_df.to_csv(roi_csv, index=False)
    print(f"\nTaula guardada: {roi_csv.relative_to(HERE.parent)}")

    # Posterior dels decay (adstock après)
    decay_post = posterior["decay"].stack(s=("chain", "draw")).values
    print("\nADSTOCK après (decay)  ·  comparat amb el valor fixat al pas 4")
    print("-" * 60)
    fixed = {"TV": 0.65, "Publicitat exterior": 0.55, "Premsa": 0.45, "Facebook": 0.20, "Google Search": 0.05}
    for i, c in enumerate(CHANNELS):
        d_mean = decay_post[i].mean()
        d_lo, d_hi = np.percentile(decay_post[i], [5, 95])
        print(f"  {c:<22} après {d_mean:.2f}  [{d_lo:.2f}, {d_hi:.2f}]   ·  fixat pas 4: {fixed[c]:.2f}")

    # --- Gràfics ---
    _plot_roi_intervals(roi_df)
    _plot_decay_posterior(decay_post)
    _plot_fit(df, posterior, revenue_mean)
    _plot_roi_vs_step4(roi_df)
    print(f"\nGràfics guardats a {OUT.relative_to(HERE.parent)}/  (11–14)")

    # --- Desar la posterior per al pas 6 (format numpy, sense dependències) ---
    flat = posterior.stack(s=("chain", "draw"))
    npz_path = HERE / "data" / "mmm_posterior.npz"
    np.savez(
        npz_path,
        decay=flat["decay"].values,            # (channel, samples)
        sat_k=flat["sat_k"].values,            # (channel, samples)
        beta=flat["beta"].values,              # (channel, samples)
        intercept=flat["intercept"].values,    # (samples,)
        season_beta=flat["season_beta"].values,
        trend_beta=flat["trend_beta"].values,
        comp_beta=flat["comp_beta"].values,
        sigma=flat["sigma"].values,
        channels=np.array(CHANNELS),
        revenue_mean=np.array(revenue_mean),
        spend_max=np.array([spend_max[c] for c in CHANNELS]),
        max_lag=np.array(MAX_LAG),
    )
    print(f"Posterior desada: {npz_path.relative_to(HERE.parent)}")

    print("\n" + "=" * 60)
    print("→ PROPER PAS 6: optimitzador de pressupost (PuLP) que mou els")
    print("  € entre canals per maximitzar el revenue esperat.")
    print("=" * 60)


def _plot_roi_intervals(roi_df: pd.DataFrame) -> None:
    d = roi_df.sort_values("roi_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(d))
    ax.errorbar(d["roi_mean"], y,
                xerr=[d["roi_mean"] - d["roi_lo"], d["roi_hi"] - d["roi_mean"]],
                fmt="none", color="#222", capsize=5, linewidth=1.5)
    for i, r in d.iterrows():
        ax.scatter([r["roi_mean"]], [i], color=CHANNEL_COLORS[r["canal"]], s=90, zorder=5)
    ax.axvline(1, color="#888", linestyle=":", linewidth=1)
    ax.text(1.05, len(d) - 0.4, "break-even", fontsize=8, color="#666")
    ax.set_yticks(y)
    ax.set_yticklabels(d["canal"])
    ax.set_xlabel("ROI  (€ revenue incremental per € gastat)")
    ax.set_title("ROI Bayesià per canal · interval de credibilitat 90%", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "11_roi_bayesia.png", dpi=140)
    plt.close(fig)


def _plot_decay_posterior(decay_post: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, c in enumerate(CHANNELS):
        ax.hist(decay_post[i], bins=40, alpha=0.55, color=CHANNEL_COLORS[c],
                label=c, density=True)
    ax.set_title("Posterior de l'adstock (decay) après per canal", fontsize=12, fontweight="bold")
    ax.set_xlabel("decay  (0 = efecte immediat · 1 = memòria llarga)")
    ax.set_ylabel("densitat")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "12_adstock_posterior.png", dpi=140)
    plt.close(fig)


def _plot_fit(df: pd.DataFrame, posterior, revenue_mean: float) -> None:
    mu = posterior["mu"].stack(s=("chain", "draw")).values  # (T, samples)
    pred_mean = mu.mean(axis=1) * revenue_mean
    lo = np.percentile(mu, 5, axis=1) * revenue_mean
    hi = np.percentile(mu, 95, axis=1) * revenue_mean
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(df["DATE"], lo / 1e6, hi / 1e6, color="#2ca02c", alpha=0.2, label="Interval 90%")
    ax.plot(df["DATE"], pred_mean / 1e6, color="#2ca02c", linewidth=1.4, label="Revenue esperat (model)")
    ax.plot(df["DATE"], df["revenue"] / 1e6, color="#222", linewidth=1.2, label="Revenue real")
    ax.set_title("Bayesian MMM · ajustament amb incertesa", fontsize=12, fontweight="bold")
    ax.set_ylabel("Revenue (milions €)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "13_bayesian_fit.png", dpi=140)
    plt.close(fig)


def _plot_roi_vs_step4(roi_df: pd.DataFrame) -> None:
    step4 = {"TV": 5.54, "Publicitat exterior": 0.47, "Premsa": 8.34,
             "Facebook": 15.81, "Google Search": 0.29}
    x = np.arange(len(CHANNELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width/2, [step4[c] for c in CHANNELS], width, color="#bbb",
           label="Pas 4 (hiperparàmetres fixats)")
    roi_map = dict(zip(roi_df["canal"], roi_df["roi_mean"]))
    lo_map = dict(zip(roi_df["canal"], roi_df["roi_lo"]))
    hi_map = dict(zip(roi_df["canal"], roi_df["roi_hi"]))
    means = [roi_map[c] for c in CHANNELS]
    errs = [[roi_map[c] - lo_map[c] for c in CHANNELS],
            [hi_map[c] - roi_map[c] for c in CHANNELS]]
    ax.bar(x + width/2, means, width, color=[CHANNEL_COLORS[c] for c in CHANNELS],
           label="Pas 5 (Bayesià, après)")
    ax.errorbar(x + width/2, means, yerr=errs, fmt="none", color="#222", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=15, ha="right")
    ax.set_ylabel("ROI (€ per €)")
    ax.set_title("ROI · pas 4 (fixat) vs pas 5 (Bayesià amb incertesa)", fontsize=12, fontweight="bold")
    ax.axhline(1, color="#888", linestyle=":", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "14_roi_final.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
