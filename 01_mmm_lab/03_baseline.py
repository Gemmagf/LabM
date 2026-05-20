"""
Pas 3 — Baseline ingenu: regressió linial.

L'objectiu d'aquest pas NO es obtenir el millor model. Es obtenir el PITJOR
model raonable per tenir un "abans" amb el que comparar el Bayesian MMM.

Modelem:
    revenue ~ tv_S + ooh_S + print_S + facebook_S + search_S

Ignorem deliberadament:
  - adstock (la publicitat te efecte arrossegat varies setmanes)
  - saturacio (gastar 2x no genera 2x; rendiments decreixents)
  - estacionalitat (Nadal, estiu, etc.)
  - competidor
  - base demand (vendes que tindries sense fer publicitat)

A l'entrevista: "vaig comencar pel baseline standard. Aqui veus els problemes."
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

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


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
    return df.rename(columns=SPEND_COLS)


def fit_baseline(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = df[list(SPEND_COLS.values())]
    X = sm.add_constant(X)
    y = df["revenue"]
    return sm.OLS(y, X).fit()


def plot_fit(df: pd.DataFrame, model) -> Path:
    y_pred = model.predict(sm.add_constant(df[list(SPEND_COLS.values())]))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["DATE"], df["revenue"] / 1e6, color="#222", linewidth=1.5, label="Real")
    ax.plot(df["DATE"], y_pred / 1e6, color="#d62728", linewidth=1.5, linestyle="--", label="Predit (baseline)")
    ax.set_title(f"Baseline OLS · ajustament  (R² = {model.rsquared:.3f})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (milions €)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    path = OUT / "04_baseline_fit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_residuals(df: pd.DataFrame, model) -> Path:
    resid = model.resid
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.plot(df["DATE"], resid / 1e6, color="#1f77b4", linewidth=1.2)
    ax.fill_between(df["DATE"], resid / 1e6, 0, alpha=0.15, color="#1f77b4")
    ax.set_title("Residus del baseline (real - predit)  →  si el model fos bo, serien soroll aleatori", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residu (M€)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    path = OUT / "05_baseline_residuals.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_naive_roi(model) -> tuple[Path, pd.DataFrame]:
    rows = []
    for canal in SPEND_COLS.values():
        coef = model.params[canal]
        ci_low, ci_high = model.conf_int().loc[canal]
        rows.append({"canal": canal, "roi": coef, "ci_low": ci_low, "ci_high": ci_high})
    roi_df = pd.DataFrame(rows).sort_values("roi")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [CHANNEL_COLORS[c] for c in roi_df["canal"]]
    y = np.arange(len(roi_df))
    ax.barh(y, roi_df["roi"], color=colors, alpha=0.85)
    ax.errorbar(roi_df["roi"], y, xerr=[roi_df["roi"] - roi_df["ci_low"], roi_df["ci_high"] - roi_df["roi"]],
                fmt="none", color="#222", capsize=4, linewidth=1.2)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(roi_df["canal"])
    ax.set_xlabel("ROI naïf  (€ revenue per € gastat)")
    ax.set_title("ROI per canal segons baseline OLS  (amb intervals de confiança 95%)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25)
    path = OUT / "06_baseline_roi.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, roi_df


def main() -> None:
    print("=" * 60)
    print("PAS 3 — Baseline naïf (OLS)")
    print("=" * 60)
    df = load()
    model = fit_baseline(df)

    print("\nRESUM DEL MODEL")
    print("-" * 60)
    print(f"R²              : {model.rsquared:.4f}")
    print(f"R² ajustat      : {model.rsquared_adj:.4f}")
    print(f"AIC             : {model.aic:,.0f}")
    print(f"BIC             : {model.bic:,.0f}")
    print(f"Durbin-Watson   : {sm.stats.stattools.durbin_watson(model.resid):.3f}  (≈2 = sense autocorrelació)")

    print("\nCOEFICIENTS  (€ revenue per € spend)")
    print("-" * 60)
    print(f"{'Canal':<22} {'Coef':>10} {'p-value':>10} {'IC 95%':>20}")
    for canal in SPEND_COLS.values():
        coef = model.params[canal]
        pval = model.pvalues[canal]
        ci_low, ci_high = model.conf_int().loc[canal]
        sig = " *" if pval < 0.05 else "  "
        print(f"{canal:<22} {coef:>10.2f} {pval:>9.3f}{sig} [{ci_low:>7.1f}, {ci_high:>6.1f}]")

    p_fit = plot_fit(df, model)
    p_res = plot_residuals(df, model)
    p_roi, roi_df = plot_naive_roi(model)
    print(f"\nGràfics guardats:")
    for p in [p_fit, p_res, p_roi]:
        print(f"  · {p.relative_to(HERE.parent)}")

    print("\n" + "=" * 60)
    print("PROBLEMES VISIBLES (per defensar a l'entrevista)")
    print("=" * 60)

    resid = model.resid
    autocorr = pd.Series(resid).autocorr(lag=1)
    print(f"\n1. AUTOCORRELACIÓ DELS RESIDUS")
    print(f"   Lag-1 = {autocorr:+.3f}  →  hi ha estructura temporal que el model no captura")
    print(f"   (Durbin-Watson lluny de 2 confirma el mateix)")

    n_neg = (roi_df["roi"] < 0).sum()
    print(f"\n2. ROIs IMPLAUSIBLES")
    print(f"   {n_neg} canal(s) tenen ROI negatiu segons aquest model.")
    print(f"   Negatiu literal voldria dir: \"gastar més = vendre menys\". Improbable.")
    print(f"   El que passa: el model atribueix mal el crèdit perquè ignora adstock + saturació.")

    print(f"\n3. ZERO ADSTOCK")
    print(f"   L'OLS assumeix efecte instantani: spend setmana X → revenue setmana X.")
    print(f"   En realitat, TV de fa 4 setmanes encara està generant vendes avui.")

    print(f"\n4. ZERO SATURACIÓ")
    print(f"   L'OLS assumeix: 2x spend = 2x revenue, sense límit.")
    print(f"   En realitat: rendiments decreixents (corba en S o Hill).")

    print(f"\n→  PROPER PAS: aplicar adstock + saturació (pas 4) i refer la regressió.")
    print(f"   Aleshores: Bayesian MMM (pas 5) per tenir intervals creíbles.")


if __name__ == "__main__":
    main()
