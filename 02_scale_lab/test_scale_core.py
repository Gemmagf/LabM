"""
Tests del mòdul scale_core.

Dos tipus de test:
  · roc_points     → numpy pur, ràpid
  · clean_features → s'executa sobre una SparkSession real amb dades
                     minúscules (test d'integració de la lògica de Spark)

Executar:   .venv/bin/python -m pytest 02_scale_lab/ -v
"""
import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_core import clean_features, roc_points


# ── Fixture: una SparkSession compartida per tots els tests ───────────
@pytest.fixture(scope="session")
def spark():
    from scale_core import get_spark
    s = get_spark("scale-core-tests", driver_memory="2g", cores="2")
    yield s
    s.stop()


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Àrea sota la corba (regla del trapezi)."""
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))


# ── roc_points (numpy pur) ────────────────────────────────────────────
def test_roc_classificador_perfecte():
    """Si els scores separen perfectament les classes, AUC = 1."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    fpr, tpr = roc_points(labels, scores)
    assert _auc(fpr, tpr) == pytest.approx(1.0, abs=1e-9)


def test_roc_classificador_invers():
    """Scores perfectament al revés → AUC = 0."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    fpr, tpr = roc_points(labels, scores)
    assert _auc(fpr, tpr) == pytest.approx(0.0, abs=1e-9)


def test_roc_comenca_a_0_acaba_a_1_i_es_monotona():
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    scores = np.array([0.2, 0.7, 0.4, 0.6, 0.9, 0.1, 0.5, 0.3])
    fpr, tpr = roc_points(labels, scores)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= 0)        # mai decreix
    assert np.all(np.diff(tpr) >= 0)


# ── clean_features (sobre Spark) ──────────────────────────────────────
INT = ["integer_feature_1", "integer_feature_2", "integer_feature_3"]
CAT = ["categorical_feature_1", "categorical_feature_2"]
SCHEMA = ("label int, integer_feature_1 int, integer_feature_2 int, "
          "integer_feature_3 int, categorical_feature_1 string, "
          "categorical_feature_2 string")


def test_clean_features_transformacions(spark):
    """Absents→0, negatius→0, log(1+x) a les numèriques; absents→'__NA__'."""
    df = spark.createDataFrame(
        [
            (1, None, -5, 10, "a", None),   # null, negatiu, positiu
            (0, 3, 0, None, None, "x"),
        ],
        SCHEMA,
    )
    rows = clean_features(df, INT, CAT).collect()
    r0, r1 = rows[0], rows[1]

    assert r0["integer_feature_1"] == pytest.approx(0.0)              # null → 0
    assert r0["integer_feature_2"] == pytest.approx(0.0)             # -5 → clip 0
    assert r0["integer_feature_3"] == pytest.approx(math.log1p(10))  # 10 → log(11)
    assert r0["categorical_feature_1"] == "a"                        # es manté
    assert r0["categorical_feature_2"] == "__NA__"                   # null → __NA__
    assert r1["integer_feature_1"] == pytest.approx(math.log1p(3))
    assert r1["categorical_feature_1"] == "__NA__"


def test_clean_features_no_deixa_cap_null(spark):
    """Després de netejar, cap columna pot tenir valors absents."""
    df = spark.createDataFrame(
        [
            (1, None, None, None, None, None),
            (0, 7, -2, 4, "z", "w"),
        ],
        SCHEMA,
    )
    clean = clean_features(df, INT, CAT)
    from pyspark.sql import functions as F
    nulls = clean.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in INT + CAT
    ]).first()
    assert all(nulls[c] == 0 for c in INT + CAT)


def test_clean_features_conserva_la_label(spark):
    """La columna label ha de passar intacta."""
    df = spark.createDataFrame(
        [(1, 5, 5, 5, "a", "b"), (0, 1, 1, 1, "c", "d")], SCHEMA,
    )
    labels = [r["label"] for r in clean_features(df, INT, CAT).collect()]
    assert sorted(labels) == [0, 1]
