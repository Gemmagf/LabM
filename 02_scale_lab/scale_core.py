"""
scale_core — utilitats compartides del Scale Lab.

  · get_spark()       → SparkSession afinada per a un portàtil
  · clean_features()  → neteja les features crues del Criteo (testejable)
  · roc_points()      → punts de la corba ROC, en numpy pur (testejable)
"""
import os
from pathlib import Path

import numpy as np
from pyspark.sql import functions as F

# Java necessari per PySpark. Si l'usuari ja té JAVA_HOME, es respecta;
# si no, es prova la ruta típica de Homebrew (openjdk@17).
_HOMEBREW_JDK = "/opt/homebrew/opt/openjdk@17"


def _ensure_java() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    if Path(_HOMEBREW_JDK).exists():
        os.environ["JAVA_HOME"] = _HOMEBREW_JDK


def get_spark(app_name: str = "LabM-ScaleLab", driver_memory: str = "4g",
              cores: str = "*"):
    """SparkSession en mode local, afinada per a un portàtil."""
    _ensure_java()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")                       # nuclis de CPU a usar
        .config("spark.driver.memory", driver_memory)    # límit de memòria
        .config("spark.sql.shuffle.partitions", "16")    # 200 (per defecte) és per clústers
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")              # menys soroll a la consola
    return spark


def clean_features(df, int_cols: list[str], cat_cols: list[str],
                   label_col: str = "label"):
    """Neteja les features crues del Criteo.

    Numèriques : valors absents → 0, negatius → 0 (clip), després log(1+x).
    Categòriques: valors absents → '__NA__'.

    Retorna un DataFrame amb label + columnes netejades.
    """
    int_exprs = [
        F.log1p(F.greatest(F.coalesce(F.col(c), F.lit(0)), F.lit(0))).alias(c)
        for c in int_cols
    ]
    cat_exprs = [
        F.coalesce(F.col(c).cast("string"), F.lit("__NA__")).alias(c)
        for c in cat_cols
    ]
    return df.select(label_col, *int_exprs, *cat_exprs)


def roc_points(labels: np.ndarray, scores: np.ndarray):
    """Punts (FPR, TPR) de la corba ROC, ordenant per score descendent."""
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])
