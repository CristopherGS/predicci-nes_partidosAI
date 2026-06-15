"""
Modelo Poisson bivariado para predicción de marcador.

Dadas tasas λ_home y λ_away (goles esperados), calcula:
  - matriz de probabilidades P(home_goals=i, away_goals=j)
  - P(local), P(empate), P(visitante)
  - P(over 2.5), P(BTTS)
  - score más probable
"""
from __future__ import annotations
import math
import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def score_matrix(lam_h: float, lam_a: float, max_goals: int = 8) -> np.ndarray:
    """Matriz (max+1) x (max+1) con P(h,a)."""
    h = np.array([poisson_pmf(i, lam_h) for i in range(max_goals + 1)])
    a = np.array([poisson_pmf(j, lam_a) for j in range(max_goals + 1)])
    m = np.outer(h, a)
    s = m.sum()
    if s > 0:
        m = m / s  # renormaliza por el corte de cola
    return m


def outcomes_from_matrix(m: np.ndarray) -> dict:
    n = m.shape[0]
    p_home = float(np.tril(m, -1).sum())
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())
    # Over 2.5
    p_over = 0.0
    p_btts = 0.0
    for i in range(n):
        for j in range(n):
            if i + j >= 3:
                p_over += m[i, j]
            if i >= 1 and j >= 1:
                p_btts += m[i, j]
    # Score más probable
    idx = np.unravel_index(np.argmax(m), m.shape)
    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over_2_5": float(p_over),
        "p_btts": float(p_btts),
        "most_likely_score": (int(idx[0]), int(idx[1])),
        "expected_home_goals": float(np.sum(np.arange(n) * m.sum(axis=1))),
        "expected_away_goals": float(np.sum(np.arange(n) * m.sum(axis=0))),
    }
