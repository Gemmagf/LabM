"""
Tests del mòdul mmm_core.

Es proven PROPIETATS (no valors exactes): que l'adstock arrossega,
que la saturació satura, que el revenue d'un canal és creixent però
amb rendiments decreixents... Així els tests segueixen valent encara
que es reajusti algun detall numèric.

Executar:   .venv/bin/python -m pytest 01_mmm_lab/ -v
"""
import numpy as np
import pytest

from mmm_core import (
    adstock_geometric,
    hill_saturation,
    exponential_saturation,
    build_lag_matrix,
    fourier_terms,
    adstock_all_draws,
    channel_revenue,
)


# ── Adstock geomètric ────────────────────────────────────────────────
def test_adstock_decay_zero_es_identitat():
    """Amb decay=0 no hi ha memòria: la sortida és igual a l'entrada."""
    x = np.array([10.0, 5.0, 0.0, 8.0])
    np.testing.assert_allclose(adstock_geometric(x, 0.0), x)


def test_adstock_impuls_decau_geometricament():
    """Un impuls únic ha de decaure com decay, decay², decay³..."""
    impuls = np.array([1.0, 0.0, 0.0, 0.0])
    out = adstock_geometric(impuls, 0.5)
    np.testing.assert_allclose(out, [1.0, 0.5, 0.25, 0.125])


def test_adstock_mai_redueix_el_total():
    """L'efecte arrossegat només pot AFEGIR: cada setmana ≥ l'spend cru."""
    x = np.array([3.0, 7.0, 2.0, 9.0, 0.0])
    out = adstock_geometric(x, 0.6)
    assert np.all(out >= x - 1e-9)


# ── Saturació Hill ───────────────────────────────────────────────────
def test_hill_surt_entre_0_i_1():
    x = np.linspace(0, 1000, 50)
    y = hill_saturation(x, alpha=2.0, k=200.0)
    assert np.all(y >= 0.0) and np.all(y < 1.0)


def test_hill_a_la_k_val_un_mig():
    """Per definició, a x = k la resposta Hill és exactament 0.5."""
    assert hill_saturation(np.array([150.0]), alpha=2.5, k=150.0)[0] == pytest.approx(0.5)


def test_hill_es_creixent():
    x = np.linspace(1, 500, 100)
    y = hill_saturation(x, alpha=1.8, k=120.0)
    assert np.all(np.diff(y) > 0)


# ── Saturació exponencial ────────────────────────────────────────────
def test_exponential_satura():
    """A x=0 val 0; per spend molt gran s'acosta a 1 sense passar-se."""
    assert exponential_saturation(np.array([0.0]), k=0.5)[0] == pytest.approx(0.0)
    gran = exponential_saturation(np.array([1e6]), k=0.5)[0]
    assert 0.999 < gran <= 1.0   # mai per sobre d'1; pot arribar a 1.0 exacte


def test_exponential_rendiments_decreixents():
    """El segon tram d'spend ha d'aportar menys resposta que el primer."""
    primer = exponential_saturation(np.array([1.0]), k=1.0)[0]
    segon = exponential_saturation(np.array([2.0]), k=1.0)[0] - primer
    assert segon < primer


# ── Lag matrix ───────────────────────────────────────────────────────
def test_lag_matrix_forma_i_columnes():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    M = build_lag_matrix(x, max_lag=3)
    assert M.shape == (4, 3)
    np.testing.assert_allclose(M[:, 0], x)              # columna 0 = sèrie original
    np.testing.assert_allclose(M[:, 1], [0, 1, 2, 3])   # desplaçada 1
    np.testing.assert_allclose(M[:, 2], [0, 0, 1, 2])   # desplaçada 2


# ── Fourier ──────────────────────────────────────────────────────────
def test_fourier_forma_i_rang():
    F = fourier_terms(T=104, harmonics=2)
    assert F.shape == (104, 4)
    assert np.all(np.abs(F) <= 1.0 + 1e-9)   # sinus i cosinus ∈ [-1, 1]


# ── Adstock vectoritzat sobre draws ──────────────────────────────────
def test_adstock_draws_forma():
    lag_mat = build_lag_matrix(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), max_lag=4)
    decay_draws = np.array([0.2, 0.5, 0.8])
    out = adstock_all_draws(lag_mat, decay_draws, max_lag=4)
    assert out.shape == (5, 3)        # (setmanes, draws)


def test_adstock_draws_decay_zero():
    """Amb decay=0 els pesos normalitzats són [1,0,0,..] → adstock = spend."""
    x = np.array([4.0, 1.0, 7.0, 2.0])
    lag_mat = build_lag_matrix(x, max_lag=4)
    out = adstock_all_draws(lag_mat, np.array([0.0]), max_lag=4)
    np.testing.assert_allclose(out[:, 0], x)


# ── Revenue per canal ────────────────────────────────────────────────
def test_channel_revenue_creix_amb_spend():
    """Gastar més no pot reduir el revenue del canal."""
    adstock = np.array([0.3, 0.5, 0.2, 0.6])
    r1 = channel_revenue(1.0, adstock, sat_k=0.4, beta=0.1, revenue_mean=2.5e6)
    r2 = channel_revenue(2.0, adstock, sat_k=0.4, beta=0.1, revenue_mean=2.5e6)
    assert r2 > r1


def test_channel_revenue_es_concau():
    """Rendiments decreixents: duplicar l'spend no duplica el revenue."""
    adstock = np.array([0.3, 0.5, 0.2, 0.6])
    kw = dict(adstock=adstock, sat_k=0.4, beta=0.1, revenue_mean=2.5e6)
    r0 = channel_revenue(0.0, **kw)
    r1 = channel_revenue(1.0, **kw)
    r2 = channel_revenue(2.0, **kw)
    guany_primer_tram = r1 - r0
    guany_segon_tram = r2 - r1
    assert guany_segon_tram < guany_primer_tram
