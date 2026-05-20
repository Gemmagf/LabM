"""
Pas 2 (Scale Lab) — Explorar el Criteo amb PySpark.

Objectiu: demostrar que llegim i processem milions de files reals amb
Spark en mode local. Calculem:
  · nombre de files
  · CTR (proporció de clicks) — la mètrica base del problema
  · valors absents per feature numèrica (les dades reals són brutes)
  · cardinalitat de features categòriques (per què no es pot fer one-hot)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_core import get_spark
from pyspark.sql import functions as F

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

INT_FEATURES = [f"integer_feature_{i}" for i in range(1, 14)]
CAT_FEATURES = [f"categorical_feature_{i}" for i in range(1, 27)]


def main() -> None:
    print("=" * 60)
    print("PAS 2 (Scale Lab) — Explorant el Criteo amb PySpark")
    print("=" * 60)

    spark = get_spark("criteo-explore")
    print(f"\nSpark {spark.version} · mode local · {spark.sparkContext.defaultParallelism} nuclis\n")

    # Llegim el directori data/: Spark detecta la partició day=... sol
    df = spark.read.parquet(str(DATA_DIR))
    df.cache()  # el reutilitzarem diverses vegades

    n_rows = df.count()
    n_files = len(list(DATA_DIR.glob("day=*/*.parquet")))
    print(f"Fitxers parquet llegits : {n_files}")
    print(f"Files totals            : {n_rows:,}")
    print(f"Columnes                : {len(df.columns)}")

    # --- CTR: la mètrica base ---
    ctr = df.select(F.avg("label").alias("ctr")).first()["ctr"]
    print("\n" + "-" * 60)
    print("CTR (CLICK-THROUGH RATE)")
    print("-" * 60)
    print(f"Proporció de clicks: {ctr:.4f}  ({ctr*100:.2f}%)")
    print(f"Un model que digués sempre 'no click' encertaria el {(1-ctr)*100:.1f}%")
    print("→ Aquest és el baseline trivial que cal superar.")

    # --- Valors absents a les features numèriques ---
    print("\n" + "-" * 60)
    print("VALORS ABSENTS · features numèriques (dades reals = brutes)")
    print("-" * 60)
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in INT_FEATURES
    ]).first()
    for c in INT_FEATURES:
        pct = null_counts[c] / n_rows * 100
        barra = "█" * int(pct / 3)
        print(f"  {c:<20} {pct:5.1f}% absents  {barra}")

    # --- Cardinalitat de features categòriques ---
    print("\n" + "-" * 60)
    print("CARDINALITAT · features categòriques (mostra de 5)")
    print("-" * 60)
    sample_cats = CAT_FEATURES[:5]
    card = df.select([
        F.approx_count_distinct(c).alias(c) for c in sample_cats
    ]).first()
    for c in sample_cats:
        print(f"  {c:<24} ≈ {card[c]:>12,} valors únics")
    print("\n→ Amb categories d'aquesta mida, el one-hot encoding és impossible")
    print("  (faria milions de columnes). Caldrà 'hashing trick' al pas 3.")

    # --- Mostra de files ---
    print("\n" + "-" * 60)
    print("MOSTRA (label + 4 primeres features numèriques)")
    print("-" * 60)
    df.select("label", *INT_FEATURES[:4]).show(5, truncate=False)

    spark.stop()
    print("=" * 60)
    print("→ Proper pas 3: preparar les features (neteja + hashing) amb Spark.")
    print("=" * 60)


if __name__ == "__main__":
    main()
