"""
mmm_core — funcions matemàtiques compartides del MMM Lab.

Centralitza les transformacions clau (adstock, saturació, lag matrix...)
perquè els scripts 04–07 les importin d'aquí en lloc de duplicar-les.
És el mòdul que cobreixen els tests (test_mmm_core.py).
"""
import numpy as np


def adstock_geometric(x: np.ndarray, decay: float) -> np.ndarray:
    """Adstock geomètric recursiu (no normalitzat):  a_t = x_t + decay·a_{t-1}.

    Modela l'efecte arrossegat de la publicitat. decay ∈ [0, 1):
    0 = efecte només la mateixa setmana; proper a 1 = memòria llarga.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def hill_saturation(x: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """Saturació Hill:  y = xᵅ / (xᵅ + kᵅ).  Sortida dins [0, 1).

    alpha controla la forma (>1 = corba en S); k és el punt de mig camí
    (a x = k la sortida val exactament 0.5).
    """
    xa = np.power(np.maximum(np.asarray(x, dtype=float), 0.0), alpha)
    return xa / (xa + k**alpha)


def exponential_saturation(x: np.ndarray, k: float) -> np.ndarray:
    """Saturació exponencial (rendiments decreixents):  y = 1 - exp(-x/k).

    Sortida dins [0, 1). Suau a tot arreu — la fa servir el model Bayesià
    perquè no dóna gradients que exploten prop de zero (la Hill xᵅ sí).
    """
    return 1.0 - np.exp(-np.asarray(x, dtype=float) / k)


def build_lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Matriu (T, max_lag) on la columna l és la sèrie desplaçada l passos.

    Permet calcular l'adstock com una multiplicació de matrius en lloc
    d'un bucle recursiu (necessari per vectoritzar sobre draws).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    M = np.zeros((T, max_lag))
    for l in range(max_lag):
        M[l:, l] = x[: T - l]
    return M


def fourier_terms(T: int, period: float = 52.18, harmonics: int = 2) -> np.ndarray:
    """Termes sinus/cosinus per modelar estacionalitat. Retorna (T, 2·harmonics)."""
    idx = np.arange(T)
    cols = []
    for k in range(1, harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * idx / period))
        cols.append(np.cos(2 * np.pi * k * idx / period))
    return np.column_stack(cols)


def adstock_all_draws(lag_mat: np.ndarray, decay_draws: np.ndarray,
                      max_lag: int) -> np.ndarray:
    """Adstock amb pesos geomètrics NORMALITZATS, per molts draws alhora.

    A diferència d'adstock_geometric (recursiu, no normalitzat), aquí els
    pesos decay^l es normalitzen perquè sumin 1 → l'adstock és una mitjana
    ponderada de les últimes max_lag setmanes. Retorna (T, n_draws).
    """
    lags = np.arange(max_lag)
    w = np.asarray(decay_draws)[:, None] ** lags[None, :]   # (S, max_lag)
    w = w / w.sum(axis=1, keepdims=True)
    return lag_mat @ w.T                                     # (T, S)


def channel_revenue(m: float, adstock: np.ndarray, sat_k, beta,
                    revenue_mean: float):
    """Revenue € que aporta un canal donat el multiplicador d'spend m.

    adstock pot ser (T,) o (T, n_draws); sat_k i beta escalars o (n_draws,).
    """
    sat = exponential_saturation(m * adstock, sat_k)
    return revenue_mean * beta * sat.sum(axis=0)
