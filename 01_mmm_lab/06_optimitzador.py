"""
Pas 6 — Optimitzador de pressupost.

Pregunta: amb el MMM del pas 5, com repartiríem els mateixos 14.5 M€
entre els 5 canals per maximitzar el revenue esperat?

NOTA TÈCNICA — per què scipy i no PuLP:
  PuLP resol programació LINEAL (objectius i restriccions de línia recta).
  El nostre MMM té SATURACIÓ: sat(x) = 1 - exp(-x/k), que és una CORBA.
  Maximitzar suma de corbes còncaves amb restriccions lineals és
  optimització convexa → scipy.optimize (SLSQP) ho fa directament i bé.

Decisió d'optimitzar un MULTIPLICADOR per canal:
  m_c ∈ [0.3, 2.0]  → cap canal pot baixar de 0.3× ni pujar de 2× el que
  rep ara (rebalanceig realista, no canvis impossibles d'un dia per l'altre).
  Restricció: el pressupost TOTAL es manté igual (reassignació, no augment).

Sortides:
  · data/budget_optim.csv   → taula actual vs recomanat
  · figures/15..17          → gràfics
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
M_MIN, M_MAX = 0.3, 2.0  # límits realistes del multiplicador per canal


def main() -> None:
    print("=" * 60)
    print("PAS 6 — Optimitzador de pressupost")
    print("=" * 60)

    post = np.load(DATA / "mmm_posterior.npz", allow_pickle=True)
    channels = [str(c) for c in post["channels"]]
    decay = post["decay"]            # (channel, S)
    sat_k = post["sat_k"]            # (channel, S)
    beta = post["beta"]              # (channel, S)
    revenue_mean = float(post["revenue_mean"])
    spend_max = post["spend_max"]    # (channel,)
    max_lag = int(post["max_lag"])
    n_draws = decay.shape[1]

    df = pd.read_csv(DATA / "weekly.csv").rename(columns=SPEND_COLS)
    raw_spend = {c: df[c].to_numpy() for c in channels}
    spend_total = np.array([raw_spend[c].sum() for c in channels])
    budget_total = spend_total.sum()

    # Adstock per canal: (T, S) amb tots els draws, i (T,) amb la mitjana
    norm_spend = {c: raw_spend[c] / spend_max[i] for i, c in enumerate(channels)}
    lag_mats = {c: build_lag_matrix(norm_spend[c], max_lag) for c in channels}
    adstock_draws = {c: adstock_all_draws(lag_mats[c], decay[i], max_lag)
                     for i, c in enumerate(channels)}
    adstock_mean = {c: adstock_draws[c].mean(axis=1) for c in channels}
    sat_k_mean = sat_k.mean(axis=1)
    beta_mean = beta.mean(axis=1)

    print(f"\nPressupost total a repartir: {budget_total/1e6:.2f} M€")
    print(f"Posterior: {n_draws} draws · multiplicador per canal ∈ [{M_MIN}, {M_MAX}]")

    # --- Optimització (paràmetres = mitjana de la posterior) ---
    def total_revenue(m: np.ndarray) -> float:
        return sum(
            channel_revenue(m[i], adstock_mean[c], sat_k_mean[i], beta_mean[i], revenue_mean)
            for i, c in enumerate(channels)
        )

    # Escalat per condicionar bé el solver:
    #  · objectiu en M€ (ordre ~67, no ~67.000.000)
    #  · restricció en fraccions (quota de pressupost que suma 1)
    share = spend_total / budget_total

    def neg_revenue(m: np.ndarray) -> float:
        return -total_revenue(m) / 1e6

    budget_constraint = {
        "type": "eq",
        "fun": lambda m: float(np.dot(m, share) - 1.0),
    }
    bounds = [(M_MIN, M_MAX)] * len(channels)
    x0 = np.ones(len(channels))

    result = minimize(neg_revenue, x0, method="SLSQP", bounds=bounds,
                      constraints=[budget_constraint],
                      options={"maxiter": 500, "ftol": 1e-9})
    m_opt = result.x
    if not result.success:
        print(f"  ⚠ l'optimitzador avisa: {result.message}")

    # --- Distribucions de revenue (propagant la incertesa de la posterior) ---
    def total_revenue_draws(m: np.ndarray) -> np.ndarray:
        out = np.zeros(n_draws)
        for i, c in enumerate(channels):
            out += channel_revenue(m[i], adstock_draws[c], sat_k[i], beta[i], revenue_mean)
        return out

    rev_current = total_revenue_draws(np.ones(len(channels)))
    rev_optim = total_revenue_draws(m_opt)
    uplift = rev_optim - rev_current

    # --- Taula de resultats ---
    print("\n" + "=" * 60)
    print("PRESSUPOST RECOMANAT")
    print("=" * 60)
    print(f"\n{'Canal':<22} {'Actual':>12} {'Recomanat':>12} {'Canvi':>10}")
    rows = []
    for i, c in enumerate(channels):
        cur = spend_total[i]
        rec = m_opt[i] * spend_total[i]
        rows.append({
            "canal": c,
            "actual_eur": cur, "actual_pct": cur / budget_total * 100,
            "recomanat_eur": rec, "recomanat_pct": rec / budget_total * 100,
            "multiplicador": m_opt[i],
        })
        fletxa = "↑" if m_opt[i] > 1.02 else ("↓" if m_opt[i] < 0.98 else "=")
        print(f"{c:<22} {cur/1e6:>9.2f} M€ {rec/1e6:>9.2f} M€ {fletxa} {m_opt[i]:>6.2f}×")

    opt_df = pd.DataFrame(rows)
    opt_df.to_csv(DATA / "budget_optim.csv", index=False)

    print(f"\n{'':22} {budget_total/1e6:>9.2f} M€ {(m_opt*spend_total).sum()/1e6:>9.2f} M€  (total igual ✓)")

    print("\n" + "=" * 60)
    print("REVENUE INCREMENTAL ESPERAT")
    print("=" * 60)
    lo, hi = np.percentile(uplift, [5, 95])
    print(f"\nRevenue marketing actual    : {rev_current.mean()/1e6:>8.2f} M€")
    print(f"Revenue marketing optimitzat: {rev_optim.mean()/1e6:>8.2f} M€")
    print(f"\nGUANY ESPERAT  : +{uplift.mean()/1e6:.2f} M€")
    print(f"Interval 90%   : [{lo/1e6:+.2f}, {hi/1e6:+.2f}] M€")
    prob_pos = (uplift > 0).mean() * 100
    print(f"Probabilitat que el guany sigui positiu: {prob_pos:.0f}%")
    print(f"\n(El mateix pressupost, només millor repartit. Cap € extra.)")

    # --- Gràfics ---
    _plot_allocation(opt_df, budget_total)
    _plot_response_curves(channels, adstock_mean, sat_k_mean, beta_mean,
                          revenue_mean, spend_total, m_opt)
    _plot_uplift(uplift)
    print(f"\nGràfics guardats: figures/15–17 · taula: data/budget_optim.csv")

    print("\n" + "=" * 60)
    print("→ PROPER PAS 7: app Streamlit on mous els € amb sliders i veus")
    print("  el revenue esperat actualitzar-se en directe.")
    print("=" * 60)


def _plot_allocation(opt_df: pd.DataFrame, budget_total: float) -> None:
    x = np.arange(len(opt_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width/2, opt_df["actual_pct"], width, color="#bbb", label="Actual")
    ax.bar(x + width/2, opt_df["recomanat_pct"], width,
           color=[CHANNEL_COLORS[c] for c in opt_df["canal"]], label="Recomanat")
    for i, r in opt_df.iterrows():
        ax.text(i - width/2, r["actual_pct"] + 1, f"{r['actual_pct']:.0f}%", ha="center", fontsize=8)
        ax.text(i + width/2, r["recomanat_pct"] + 1, f"{r['recomanat_pct']:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(opt_df["canal"], rotation=15, ha="right")
    ax.set_ylabel("% del pressupost total")
    ax.set_title("Repartiment del pressupost · actual vs recomanat", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "15_allocacio.png", dpi=140)
    plt.close(fig)


def _plot_response_curves(channels, adstock_mean, sat_k_mean, beta_mean,
                          revenue_mean, spend_total, m_opt) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ms = np.linspace(0, M_MAX, 120)
    for i, c in enumerate(channels):
        rev = np.array([channel_revenue(m, adstock_mean[c], sat_k_mean[i],
                                        beta_mean[i], revenue_mean) for m in ms])
        spend_axis = ms * spend_total[i] / 1e6
        ax.plot(spend_axis, rev / 1e6, color=CHANNEL_COLORS[c], linewidth=2, label=c)
        # punt actual
        cur_rev = channel_revenue(1.0, adstock_mean[c], sat_k_mean[i], beta_mean[i], revenue_mean)
        ax.scatter([spend_total[i]/1e6], [cur_rev/1e6], color=CHANNEL_COLORS[c],
                   marker="o", s=55, edgecolor="white", zorder=5)
        # punt optimitzat
        opt_rev = channel_revenue(m_opt[i], adstock_mean[c], sat_k_mean[i], beta_mean[i], revenue_mean)
        ax.scatter([m_opt[i]*spend_total[i]/1e6], [opt_rev/1e6], color=CHANNEL_COLORS[c],
                   marker="*", s=180, edgecolor="black", linewidth=0.5, zorder=6)
    ax.set_xlabel("Spend en el canal (M€)")
    ax.set_ylabel("Revenue incremental del canal (M€)")
    ax.set_title("Corbes de resposta · ● = actual   ★ = recomanat", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "16_corbes_resposta.png", dpi=140)
    plt.close(fig)


def _plot_uplift(uplift: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(uplift / 1e6, bins=50, color="#2ca02c", alpha=0.75)
    ax.axvline(0, color="#888", linestyle=":", linewidth=1)
    ax.axvline(uplift.mean() / 1e6, color="#d62728", linewidth=1.8,
               label=f"Guany esperat +{uplift.mean()/1e6:.2f} M€")
    ax.set_xlabel("Revenue incremental amb el pressupost optimitzat (M€)")
    ax.set_ylabel("nombre de draws de la posterior")
    ax.set_title("Distribució del guany · honest sobre la incertesa", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "17_uplift.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
