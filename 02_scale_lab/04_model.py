"""
Pas 4 (Scale Lab) — Model de CTR amb Spark MLlib.

Entrena una regressió logística per predir si un anunci rebrà click,
i l'avalua correctament.

Per què NO mirem l'accuracy:
  Amb un 3.2% de clicks, un model que digués sempre "no click" ja té
  un 96.8% d'accuracy. L'accuracy és inútil aquí. Les mètriques bones:
    · AUC-ROC  — capacitat d'ordenar clicks per damunt de no-clicks
    · AUC-PR   — més exigent amb classes desequilibrades
    · log-loss — la funció objectiu real del CTR (penalitza probabilitats
                 mal calibrades); es compara contra el baseline trivial.

Sortides:
  · model/            → el model entrenat, desat
  · figures/01_roc.png
"""
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_core import get_spark, roc_points
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

HERE = Path(__file__).parent
FEATURES_DIR = HERE / "features"
TRAIN_DIR = HERE / "features_train"
TEST_DIR = HERE / "features_test"
MODEL_DIR = HERE / "model"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
EPS = 1e-12


def logloss(df, prob_col: str = "p1") -> float:
    """Log-loss mitjà: -mean( y·log(p) + (1-y)·log(1-p) )."""
    p = F.least(F.greatest(F.col(prob_col), F.lit(EPS)), F.lit(1 - EPS))
    expr = -(F.col("label") * F.log(p) + (1 - F.col("label")) * F.log(1 - p))
    return df.select(F.avg(expr)).first()[0]


def main() -> None:
    print("=" * 60)
    print("PAS 4 (Scale Lab) — Model de CTR (regressió logística)")
    print("=" * 60)

    spark = get_spark("criteo-model", driver_memory="16g", cores="6")
    t0 = time.time()

    # El split es MATERIALITZA A DISC, no es cacheja en memòria.
    # Amb ~196M de vectors, la cache en memòria de Spark peta el heap;
    # escrivint el split a parquet, cada lectura posterior és en streaming
    # → impossible que peti per memòria.
    raw = spark.read.parquet(str(FEATURES_DIR))
    n_parts = max(64, raw.count() // 150_000)
    train_raw, test_raw = (
        raw.repartition(n_parts).randomSplit([0.8, 0.2], seed=42)
    )
    print("Escrivint el split train/test a disc...")
    train_raw.write.mode("overwrite").parquet(str(TRAIN_DIR))
    test_raw.write.mode("overwrite").parquet(str(TEST_DIR))

    train = spark.read.parquet(str(TRAIN_DIR))
    test = spark.read.parquet(str(TEST_DIR))
    n_train, n_test = train.count(), test.count()
    print(f"\nTrain: {n_train:,} files   ·   Test: {n_test:,} files")

    # --- Entrenament ---
    print("Entrenant la regressió logística (15 iteracions)...")
    lr = LogisticRegression(
        featuresCol="features", labelCol="label",
        maxIter=15, regParam=0.01, elasticNetParam=0.0,
    )
    model = lr.fit(train)
    train_time = time.time() - t0
    print(f"Entrenat en {train_time:.0f} s")

    # --- Prediccions sobre test ---
    pred = model.transform(test)
    pred = pred.withColumn("p1", vector_to_array("probability")[1])
    pred.persist(StorageLevel.DISK_ONLY)

    # --- Mètriques ---
    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC").evaluate(pred)
    aucpr = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderPR").evaluate(pred)
    model_logloss = logloss(pred)

    # Baseline trivial: predir sempre el CTR mitjà del train
    train_ctr = train.select(F.avg("label")).first()[0]
    test_ctr = test.select(F.avg("label")).first()[0]
    base_logloss = -(test_ctr * np.log(train_ctr)
                     + (1 - test_ctr) * np.log(1 - train_ctr))

    print("\n" + "=" * 60)
    print("RESULTATS  (sobre el test, dades no vistes)")
    print("=" * 60)
    print(f"\nCTR del train         : {train_ctr*100:.2f}%")
    print(f"\n{'Mètrica':<22}{'Model':>12}{'Baseline':>12}")
    print("-" * 46)
    print(f"{'AUC-ROC':<22}{auc:>12.4f}{0.5:>12.4f}")
    print(f"{'AUC-PR':<22}{aucpr:>12.4f}{test_ctr:>12.4f}")
    print(f"{'log-loss (↓ millor)':<22}{model_logloss:>12.4f}{base_logloss:>12.4f}")
    millora = (base_logloss - model_logloss) / base_logloss * 100
    print(f"\nEl model redueix el log-loss un {millora:.1f}% respecte al baseline.")
    print(f"AUC {auc:.3f}: ordena correctament un click i un no-click el "
          f"{auc*100:.0f}% de les vegades.")

    # --- Corba ROC (sobre una mostra del test) ---
    frac = min(1.0, 80000 / max(n_test, 1))
    sample = pred.select("label", "p1").sample(fraction=frac, seed=42).toPandas()
    fpr, tpr = roc_points(sample["label"].to_numpy(), sample["p1"].to_numpy())
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", label="Atzar (AUC = 0.5)")
    ax.set_xlabel("Falsos positius (FPR)")
    ax.set_ylabel("Veritables positius (TPR)")
    ax.set_title("Corba ROC · model de CTR Criteo", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_roc.png", dpi=140)
    plt.close(fig)

    # --- Desar el model ---
    model.write().overwrite().save(str(MODEL_DIR))
    print(f"\nModel desat a {MODEL_DIR.relative_to(HERE.parent)}/")
    print(f"Gràfic: {(FIG_DIR / '01_roc.png').relative_to(HERE.parent)}")
    print(f"Temps total: {time.time() - t0:.0f} s")

    spark.stop()
    print("=" * 60)


if __name__ == "__main__":
    main()
