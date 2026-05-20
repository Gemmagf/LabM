"""
Pas 3 (Scale Lab) — Enginyeria de features amb Spark.

Transforma les dades crues del Criteo en un dataset llest per al model:

  1. Features NUMÈRIQUES (13):
       · valors absents → 0  (estàndard a la literatura Criteo/DLRM)
       · valors negatius → 0  (clip)
       · log(1 + x)  perquè són comptadors molt esbiaixats
  2. Features CATEGÒRIQUES (26):
       · valors absents → "__NA__"
       · HASHING TRICK: milions de valors únics → 2^18 cubells fixos
         (one-hot seria impossible: faria milions de columnes)
  3. S'ajunten en un sol vector `features` i es desa com a parquet.

Tot amb pyspark.ml.Pipeline → funciona igual amb 16M o 190M de files.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_core import get_spark, clean_features
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher, VectorAssembler

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "features"

INT_FEATURES = [f"integer_feature_{i}" for i in range(1, 14)]
CAT_FEATURES = [f"categorical_feature_{i}" for i in range(1, 27)]
N_BUCKETS = 2 ** 16   # 65.536 cubells · espai de hash dimensionat per a portàtil
                      # (a un clúster es pujaria a 2^20+; aquí prioritzem que
                      #  l'entrenament càpiga en memòria)


def main() -> None:
    print("=" * 60)
    print("PAS 3 (Scale Lab) — Enginyeria de features amb Spark")
    print("=" * 60)

    spark = get_spark("criteo-features", driver_memory="8g")
    t0 = time.time()

    df = spark.read.parquet(str(DATA_DIR))
    n_rows = df.count()
    print(f"\nFiles d'entrada: {n_rows:,}")

    # --- 1+2. Neteja i transformacions (funció compartida, testejada) ---
    clean = clean_features(df, INT_FEATURES, CAT_FEATURES)
    print("Numèriques: absents→0, negatius→0, log(1+x)")
    print(f"Categòriques: absents→'__NA__', hashing a {N_BUCKETS:,} cubells")

    # --- 3. Pipeline: hashing + assemblatge en un sol vector ---
    hasher = FeatureHasher(
        inputCols=CAT_FEATURES, outputCol="cat_hashed", numFeatures=N_BUCKETS,
    )
    assembler = VectorAssembler(
        inputCols=INT_FEATURES + ["cat_hashed"], outputCol="features",
        handleInvalid="keep",
    )
    pipeline = Pipeline(stages=[hasher, assembler])

    print("\nAjustant el pipeline de features...")
    model = pipeline.fit(clean)
    features = model.transform(clean).select("label", "features")

    # --- Desar com a parquet ---
    print(f"Desant a {OUT_DIR.relative_to(HERE.parent)}/ ...")
    features.write.mode("overwrite").parquet(str(OUT_DIR))

    elapsed = time.time() - t0
    out_files = list(OUT_DIR.glob("*.parquet"))
    out_size = sum(f.stat().st_size for f in out_files) / 1e9

    print("\n" + "=" * 60)
    print("RESULTAT")
    print("=" * 60)
    feat_dim = INT_FEATURES.__len__() + N_BUCKETS
    print(f"Files processades   : {n_rows:,}")
    print(f"Dimensió del vector : {feat_dim:,}  (13 numèriques + {N_BUCKETS:,} hash)")
    print(f"Parquet de sortida  : {len(out_files)} fitxers · {out_size:.2f} GB")
    print(f"Temps total         : {elapsed:.0f} s")

    print("\nMostra (vector dispers — només es guarden els valors no-zero):")
    features.show(3, truncate=70)

    spark.stop()
    print("=" * 60)
    print("→ Proper pas 4: entrenar un model de CTR (regressió logística) i")
    print("  avaluar-lo amb AUC — l'accuracy no serveix amb 3% de clicks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
