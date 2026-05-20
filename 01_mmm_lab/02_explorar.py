"""
Pas 2 — Explorar les dades visualment.

Genera 3 graficos a `01_mmm_lab/figures/`:
  1. revenue al llarg del temps (4 anys)
  2. spend per canal al llarg del temps
  3. quota total de pressupost per canal (barres horitzontals)

L'objectiu: entendre que tenim ABANS de modelar res.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    df = df.rename(columns=SPEND_COLS)
    return df


def plot_revenue(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["DATE"], df["revenue"] / 1e6, color="#222", linewidth=1.4)
    ax.fill_between(df["DATE"], df["revenue"] / 1e6, alpha=0.08, color="#222")
    ax.set_title("Revenue setmanal · 2015-2019", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (milions €)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    path = OUT / "01_revenue.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_channels(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    for ch in SPEND_COLS.values():
        ax.plot(df["DATE"], df[ch] / 1e3, label=ch, color=CHANNEL_COLORS[ch], linewidth=1.2)
    ax.set_title("Spend setmanal per canal", fontsize=13, fontweight="bold")
    ax.set_ylabel("Spend (milers €)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    path = OUT / "02_spend_per_canal.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_budget_share(df: pd.DataFrame) -> tuple[Path, pd.Series]:
    totals = df[list(SPEND_COLS.values())].sum().sort_values()
    share = totals / totals.sum() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [CHANNEL_COLORS[c] for c in totals.index]
    ax.barh(totals.index, share.values, color=colors)
    for i, (canal, pct) in enumerate(share.items()):
        ax.text(pct + 0.7, i, f"{pct:.1f}%   ({totals[canal] / 1e6:.2f} M€)", va="center", fontsize=10)
    ax.set_title("Quota total de pressupost (4 anys) per canal", fontsize=13, fontweight="bold")
    ax.set_xlabel("% del spend total")
    ax.set_xlim(0, max(share.values) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25)
    path = OUT / "03_quota_pressupost.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, share


def main() -> None:
    print("=" * 60)
    print("PAS 2 — Explorant les dades")
    print("=" * 60)
    df = load()
    print(f"\n{len(df)} setmanes carregades · {df['DATE'].min().date()} → {df['DATE'].max().date()}")

    p1 = plot_revenue(df)
    print(f"  gravat: {p1.relative_to(HERE.parent)}")

    p2 = plot_channels(df)
    print(f"  gravat: {p2.relative_to(HERE.parent)}")

    p3, share = plot_budget_share(df)
    print(f"  gravat: {p3.relative_to(HERE.parent)}")

    print("\n" + "=" * 60)
    print("DESCOBERTES RÀPIDES")
    print("=" * 60)
    total_spend = df[list(SPEND_COLS.values())].sum().sum()
    total_revenue = df["revenue"].sum()
    print(f"\nSpend total (4 anys):    {total_spend / 1e6:>8.2f} M€")
    print(f"Revenue total (4 anys):  {total_revenue / 1e6:>8.2f} M€")
    print(f"Revenue / Spend ratio:   {total_revenue / total_spend:>8.2f}x")
    print(f"\nQuota per canal:")
    for canal, pct in share.sort_values(ascending=False).items():
        print(f"  {canal:<22} {pct:>5.1f} %")


if __name__ == "__main__":
    main()
