"""
Pas 1 — Baixa les dades del MMM Lab.

Dataset: Robyn simulated weekly (Meta, open-source)
- 208 setmanes de dades setmanals d'un anunciant
- spend per canal (TV, OOH, print, Facebook, search...)
- revenue setmanal
- variables de context (competitor sales, holidays, events)

Es "simulat" pero esta basat en patrons reals. Es el dataset standard
que utilitzen totes les empreses serioses per aprendre MMM Bayesia.
"""
from pathlib import Path
import requests
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "weekly.csv": "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/python/src/robyn/tutorials/resources/dt_simulated_weekly.csv",
    "holidays.csv": "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/python/src/robyn/tutorials/resources/dt_prophet_holidays.csv",
}


def download(name: str, url: str) -> Path:
    path = DATA_DIR / name
    if path.exists():
        print(f"  ja existeix: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
        return path
    print(f"  baixant {name}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"  guardat:    {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def main() -> None:
    print("=" * 60)
    print("PAS 1 — Baixant dades del MMM Lab")
    print("=" * 60)

    paths = {name: download(name, url) for name, url in FILES.items()}

    print("\n" + "=" * 60)
    print("RESUM DEL DATASET PRINCIPAL (weekly.csv)")
    print("=" * 60)
    df = pd.read_csv(paths["weekly.csv"])
    print(f"\nFiles:    {len(df):,}")
    print(f"Columnes: {len(df.columns)}")
    print(f"\nColumnes disponibles:")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nPrimeres 3 files:")
    print(df.head(3).to_string())
    print(f"\nRang de dates: {df['DATE'].min()} → {df['DATE'].max()}")


if __name__ == "__main__":
    main()
