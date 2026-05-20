"""
Pas 1 (Scale Lab) — Baixa un subconjunt del Criteo 1TB Click Logs.

Dataset: Criteo 1TB Click Logs (Hugging Face, format Parquet, CC-BY-NC-SA).
  · 24 dies · ~4.000 milions de files en total
  · 1 dia = 250 fitxers parquet ≈ 12 GB ≈ ~190 milions de files
  · cada fila: 1 etiqueta (click sí/no) + 13 features numèriques + 26 categòriques

ESTRATÈGIA: no baixem tot l'1 TB (impossible al portàtil). Baixem un dia.
Per desenvolupar el pipeline n'hi ha prou amb un tros petit; quan tot
funcioni, baixem el dia sencer.

S'utilitza la llibreria oficial huggingface_hub: gestiona reintents,
resume i el backend Xet automàticament (descàrregues robustes).

Ús:
    python 02_scale_lab/01_download.py            → 20 fitxers  (~1 GB, per provar)
    python 02_scale_lab/01_download.py 250        → dia sencer  (~12 GB)
    python 02_scale_lab/01_download.py all        → dia sencer
"""
import sys
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "criteo/CriteoClickLogs"
DAY = "2015-02-15"
HERE = Path(__file__).parent
# hf_hub_download recrea l'estructura del repo (data/day=.../) dins de LOCAL_DIR,
# així que apuntem a l'arrel del lab → els fitxers acaben a 02_scale_lab/data/day=...
LOCAL_DIR = HERE


def part_files_of_day() -> list[str]:
    """Llista els fitxers parquet del dia, ordenats."""
    prefix = f"data/day={DAY}/"
    files = [
        f for f in list_repo_files(REPO, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    return sorted(files)


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "20"
    n = 250 if arg == "all" else int(arg)

    print("=" * 60)
    print(f"PAS 1 (Scale Lab) — Baixant Criteo · dia {DAY}")
    print("=" * 60)

    all_files = part_files_of_day()
    selected = all_files[:n]
    print(f"\nDia {DAY}: {len(all_files)} fitxers disponibles · en baixem {len(selected)}")
    print("(huggingface_hub salta els que ja estiguin baixats i reintenta sols)\n")

    total_bytes = 0
    for i, fname in enumerate(selected, 1):
        local_path = hf_hub_download(
            repo_id=REPO, repo_type="dataset", filename=fname,
            local_dir=str(LOCAL_DIR),
        )
        size = Path(local_path).stat().st_size
        total_bytes += size
        print(f"[{i:3d}/{len(selected)}]  {Path(fname).name[:40]}  "
              f"{size/1e6:6.1f} MB   ·   acumulat {total_bytes/1e9:.2f} GB")

    print("\n" + "=" * 60)
    print(f"Fet. {len(selected)} fitxers · {total_bytes/1e9:.2f} GB")
    print(f"A: {(LOCAL_DIR / 'data' / f'day={DAY}').relative_to(HERE.parent)}/")
    print("\n→ Proper pas: inspeccionar les dades amb PySpark.")


if __name__ == "__main__":
    main()
