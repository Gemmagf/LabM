"""
Pas 4 — Adstock + Saturació + Estacionalitat.

Dues idees clau del MMM, més una tercera per corregir el confounding:

  1. ADSTOCK (efecte arrossegat)
     L'spend en publicitat d'avui afecta vendes durant N setmanes.
     Fórmula geomètrica:  a_t = x_t + λ · a_{t-1}     (0 < λ < 1)
     TV i OOH tenen λ alt (memoria llarga).
     Search té λ baix (efecte immediat).

  2. SATURACIÓ (Hill / S-curve)
     El segon milió gastat ven menys que el primer (rendiments decreixents).
     Fórmula Hill:  y = x^α / (x^α + K^α)
     K = punt on s'arriba al 50% del màxim.

  3. CONTROLS (estacionalitat + competidor)
     Sense ells, el model atribueix les pujades de Nadal a Search/Facebook.
     Afegim 2 termes de Fourier (52 setmanes) + competitor_sales_B.

Aquí utilitzem HIPERPARÀMETRES FIXOS raonables. Al pas 5 (Bayesian)
els aprendrem dels dades.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

from mmm_core import adstock_geometric, hill_saturation, fourier_terms

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
CHANNEL_COLORS = {
    "TV": "#d62728",
    "Publicitat exterior": "#ff7f0e",
    "Premsa": "#8c564b",
    "Facebook": "#1f77b4",
    "Google Search": "#2ca02c",
}

# Hiperparàmetres per canal (basats en literatura Robyn / LightweightMMM).
# decay = quant arrossega l'efecte setmana rere setmana (0 = res, 1 = etern).
# hill_alpha = curvatura de la corba de saturació (>1 = forma S).
# hill_k_pct = % del màxim spend on s'arriba al 50% de la resposta màxima.
HYPERPARAMS = {
    "TV":                  {"decay": 0.65, "alpha": 2.5, "k_pct": 0.50},
    "Publicitat exterior": {"decay": 0.55, "alpha": 2.0, "k_pct": 0.50},
    "Premsa":              {"decay": 0.45, "alpha": 1.8, "k_pct": 0.40},
    "Facebook":            {"decay": 0.20, "alpha": 1.5, "k_pct": 0.30},
    "Google Search":       {"decay": 0.05, "alpha": 1.3, "k_pct": 0.30},
}


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica adstock + saturació + estacionalitat. Retorna X i info per ROI."""
    X = pd.DataFrame(index=df.index)
    info = {}

    for canal in SPEND_COLS.values():
        spend = df[canal].to_numpy()
        h = HYPERPARAMS[canal]

        adstocked = adstock_geometric(spend, h["decay"])
        k_value = h["k_pct"] * adstocked.max()
        saturated = hill_saturation(adstocked, h["alpha"], k_value)

        X[canal] = saturated
        info[canal] = {"raw_spend_sum": spend.sum(), "transformed": saturated, "raw": spend}

    # Estacionalitat anual (4 termes)
    fourier = fourier_terms(len(df))
    for i in range(fourier.shape[1]):
        X[f"season_{i}"] = fourier[:, i]

    # Control: tendència lineal + competidor
    X["trend"] = np.arange(len(df)) / len(df)
    X["competitor"] = df["competitor_sales_B"] / df["competitor_sales_B"].max()

    return X, info


def compute_contributions(df: pd.DataFrame, X: pd.DataFrame, model) -> pd.DataFrame:
    """Contribució en € per canal i setmana (coef × variable transformada)."""
    contrib = pd.DataFrame(index=df.index)
    for canal in SPEND_COLS.values():
        contrib[canal] = model.params[canal] * X[canal]
    # base = constant + estacionalitat + tendència + competidor
    base = model.params["const"] * np.ones(len(df))
    for col in X.columns:
        if col not in SPEND_COLS.values():
            base = base + model.params[col] * X[col].to_numpy()
    contrib["Base + estacionalitat"] = base
    return contrib


def plot_transformations(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Adstock: exemple TV
    tv_raw = df["TV"].to_numpy()
    tv_ads = adstock_geometric(tv_raw, HYPERPARAMS["TV"]["decay"])
    axes[0].plot(df["DATE"], tv_raw / 1e3, color="#aaa", linewidth=1, label="TV spend setmanal")
    axes[0].plot(df["DATE"], tv_ads / 1e3, color="#d62728", linewidth=1.5, label=f"TV adstocked (λ={HYPERPARAMS['TV']['decay']})")
    axes[0].set_title("Adstock — l'efecte de la TV arrossega setmanes", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("milers €")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].grid(True, alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)

    # Saturació: corba teòrica
    x = np.linspace(0, 1, 200)
    for canal in ["TV", "Facebook", "Google Search"]:
        h = HYPERPARAMS[canal]
        y = hill_saturation(x, h["alpha"], h["k_pct"])
        axes[1].plot(x, y, color=CHANNEL_COLORS[canal], linewidth=2, label=f"{canal}  (α={h['alpha']}, K={h['k_pct']})")
    axes[1].set_title("Saturació Hill — el segon milió ven menys que el primer", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Spend (normalitzat 0–1)")
    axes[1].set_ylabel("Resposta (0–1)")
    axes[1].grid(True, alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)

    path = OUT / "07_transformacions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_fit_comparison(df: pd.DataFrame, model, r2_baseline: float) -> Path:
    y_pred = model.predict(sm.add_constant(model.model.exog[:, 1:]))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["DATE"], df["revenue"] / 1e6, color="#222", linewidth=1.5, label="Real")
    ax.plot(df["DATE"], y_pred / 1e6, color="#2ca02c", linewidth=1.5, linestyle="--",
            label=f"Predit (MMM amb adstock + saturació)")
    ax.set_title(f"Adstock+Saturació · R² = {model.rsquared:.3f}   (baseline OLS era {r2_baseline:.3f})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (milions €)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    path = OUT / "08_mmm_fit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_contributions_stacked(df: pd.DataFrame, contrib: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    order = ["Base + estacionalitat"] + list(SPEND_COLS.values())
    colors_map = {**CHANNEL_COLORS, "Base + estacionalitat": "#cccccc"}
    bottoms = np.zeros(len(df))
    for name in order:
        values = contrib[name].to_numpy() / 1e6
        ax.fill_between(df["DATE"], bottoms, bottoms + values, label=name,
                        color=colors_map[name], alpha=0.85, linewidth=0)
        bottoms = bottoms + values
    ax.plot(df["DATE"], df["revenue"] / 1e6, color="#000", linewidth=1, label="Revenue real")
    ax.set_title("Descomposició setmanal del revenue · contribució per canal", fontsize=13, fontweight="bold")
    ax.set_ylabel("milions €")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    path = OUT / "09_descomposicio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_roi_comparison(roi_naive: dict, roi_new: dict) -> Path:
    canals = list(SPEND_COLS.values())
    x = np.arange(len(canals))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width/2, [roi_naive[c] for c in canals], width, color="#bbb", label="Baseline OLS")
    ax.bar(x + width/2, [roi_new[c] for c in canals], width,
           color=[CHANNEL_COLORS[c] for c in canals], label="MMM (adstock + saturació)")
    ax.set_xticks(x)
    ax.set_xticklabels(canals, rotation=15, ha="right")
    ax.set_ylabel("ROI (€ revenue per € spend)")
    ax.set_title("Comparació ROI · abans i després de les transformacions", fontsize=13, fontweight="bold")
    ax.axhline(1, color="#888", linewidth=0.8, linestyle=":")
    ax.text(len(canals) - 0.5, 1.05, "ROI = 1 (break-even)", fontsize=9, color="#666")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    path = OUT / "10_roi_comparacio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    print("=" * 60)
    print("PAS 4 — MMM amb adstock + saturació + estacionalitat")
    print("=" * 60)
    df = pd.read_csv(DATA, parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
    df = df.rename(columns=SPEND_COLS)

    # Construïm la matriu de disseny amb transformacions
    X, info = build_design_matrix(df)
    X_with_const = sm.add_constant(X)
    model = sm.OLS(df["revenue"], X_with_const).fit()

    # Baseline naïf per comparar
    X_naive = sm.add_constant(df[list(SPEND_COLS.values())])
    model_naive = sm.OLS(df["revenue"], X_naive).fit()

    print(f"\nR² baseline OLS         : {model_naive.rsquared:.4f}")
    print(f"R² MMM (adstock+sat)    : {model.rsquared:.4f}   ⬆ +{(model.rsquared - model_naive.rsquared)*100:.1f} pp")
    print(f"Durbin-Watson baseline  : {sm.stats.stattools.durbin_watson(model_naive.resid):.2f}")
    print(f"Durbin-Watson MMM       : {sm.stats.stattools.durbin_watson(model.resid):.2f}   (millor com més proper a 2)")

    # Càlcul de ROI per canal: contribució_total / spend_total
    contrib = compute_contributions(df, X, model)
    print("\nROI per canal (contribució total / spend total)")
    print("-" * 60)
    print(f"{'Canal':<22} {'ROI naïf':>10} {'ROI MMM':>10} {'Δ':>10}")
    roi_new = {}
    roi_naive_dict = {c: model_naive.params[c] for c in SPEND_COLS.values()}
    for canal in SPEND_COLS.values():
        contribution_total = contrib[canal].sum()
        spend_total = info[canal]["raw_spend_sum"]
        roi_mmm = contribution_total / spend_total if spend_total > 0 else 0.0
        roi_new[canal] = roi_mmm
        delta = roi_mmm - roi_naive_dict[canal]
        print(f"{canal:<22} {roi_naive_dict[canal]:>10.2f} {roi_mmm:>10.2f} {delta:>+10.2f}")

    base_contrib = contrib["Base + estacionalitat"].sum()
    base_pct = base_contrib / df["revenue"].sum() * 100
    print(f"\nBase + estacionalitat   : {base_pct:.1f} % del revenue total")
    print("(això és vendes que tindries fins i tot SENSE publicitat — \"base demand\")")

    # Gràfics
    p1 = plot_transformations(df)
    p2 = plot_fit_comparison(df, model, model_naive.rsquared)
    p3 = plot_contributions_stacked(df, contrib)
    p4 = plot_roi_comparison(roi_naive_dict, roi_new)
    print(f"\nGràfics guardats:")
    for p in [p1, p2, p3, p4]:
        print(f"  · {p.relative_to(HERE.parent)}")

    print("\n" + "=" * 60)
    print("LECTURA DELS NOUS ROIs")
    print("=" * 60)
    sorted_roi = sorted(roi_new.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nCanals ordenats per ROI (eficiència):")
    for canal, roi in sorted_roi:
        share_budget = info[canal]["raw_spend_sum"] / sum(v["raw_spend_sum"] for v in info.values()) * 100
        print(f"  {canal:<22} ROI {roi:>6.2f}×   ·   quota pressupost actual {share_budget:>5.1f} %")

    print("\n→  La pregunta clau pel pas 6 (budget optimizer):")
    print("   Si el canal X té ROI alt però poca quota, hauríem de moure-li € ?")
    print("\n→  Però primer: pas 5 = Bayesian MMM, que aprèn els hiperparàmetres")
    print("   en lloc de fixar-los a mà, i ens dóna intervals de credibilitat.")


if __name__ == "__main__":
    main()
