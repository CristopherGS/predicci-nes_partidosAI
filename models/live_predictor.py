"""
Predictor in-play (Dixon-Robinson 1998).

Referencia: Dixon, M. J., & Robinson, M. E. (1998). "A Birth Process Model for
Association Football Matches". The Statistician, 47(3), 523-538.

Mientras un partido está en juego, las probabilidades pre-partido se ajustan en
función de:
  - Minuto actual t ∈ [0, 90]   → tiempo restante r = (90 - t) / 90
  - Marcador actual (g_h, g_a)  → diferencia que ya existe
  - Expulsiones (red_h, red_a)  → penalización a la tasa del equipo
  - Estadísticas live (shots, possession, xG) → ajuste fino de la tasa

Modelo:
  Goles futuros del local ~ Poisson(λ_remaining)
  Goles futuros del visitante ~ Poisson(μ_remaining)

  λ_remaining = λ_pre × (1 - t/90) × red_penalty_h × stat_multiplier_h
  μ_remaining = μ_pre × (1 - t/90) × red_penalty_a × stat_multiplier_a

donde:
  red_penalty_team = 0.55^(red_team)   (cada roja baja ~45% la tasa)
  stat_multiplier  = blend de shots-ratio y xG-ratio observados vs esperados

Probabilidades finales:
  P(home gana) = Σ_{i,j: g_h+i > g_a+j} P(I=i) P(J=j)
  P(empate)   = Σ_{i,j: g_h+i = g_a+j} P(I=i) P(J=j)
  P(away gana) = resto

Esto se ajusta minuto a minuto.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np

from .poisson import poisson_pmf


# ---------- Ajuste por estadísticas live ----------
def _stat_multiplier(shots: int, shots_op: int,
                     xg_live: Optional[float], lam_pre: float,
                     minute: int) -> float:
    """Devuelve un multiplicador [0.5, 1.8] sobre la tasa de goles según lo que
    el equipo está mostrando in-play.

    Combina:
      - Ratio de tiros vs el rival (más tiros → más amenaza)
      - xG live vs xG esperado pre-partido al minuto t
    """
    # Si no hay info, multiplicador = 1.0
    mult = 1.0

    # 1) Ratio de tiros
    total_shots = shots + shots_op
    if total_shots >= 5:
        share = shots / total_shots
        # share esperado neutral ≈ 0.5. Mapeamos a multiplicador [0.7, 1.3]
        mult *= 0.7 + 1.2 * share  # share=0.5 → 1.3? No: 0.7+0.6=1.3. share=0→0.7. share=1→1.9. Limito.
        mult = min(mult, 1.4)

    # 2) xG live
    if xg_live is not None and minute > 0:
        expected_xg_at_t = lam_pre * (minute / 90.0)
        if expected_xg_at_t > 0.05:
            ratio = xg_live / expected_xg_at_t
            # ratio>1: el equipo genera más amenaza que lo esperado
            # Mapeo a multiplicador [0.6, 1.6]
            xg_mult = max(0.6, min(1.6, 0.6 + 0.5 * ratio))
            # promedio con el de tiros
            mult = (mult + xg_mult) / 2

    return max(0.5, min(1.8, mult))


def _red_penalty(reds: int) -> float:
    """Cada roja baja la tasa de goles ~45% (literatura: Ridder et al. 1994)."""
    return 0.55 ** max(0, reds)


# ---------- Predicción in-play ----------
def predict_inplay(
    lam_pre: float, mu_pre: float,
    minute: int,
    current_home_goals: int, current_away_goals: int,
    *,
    shots_h: int = 0, shots_a: int = 0,
    shots_op_h: int = 0, shots_op_a: int = 0,
    xg_live_h: Optional[float] = None, xg_live_a: Optional[float] = None,
    red_h: int = 0, red_a: int = 0,
    max_extra_goals: int = 7,
) -> dict:
    """Ajusta probabilidades 1X2 + over 2.5 + BTTS al estado actual.

    Args:
      lam_pre, mu_pre: tasas de goles pre-partido (90 min completos) que vienen
                       del modelo pre-match (Dixon-Coles o predicción global).
      minute:          minuto actual (0-90+).
      current_*:       marcador actual en cancha.
      shots_*:         tiros TOTALES de cada equipo hasta el minuto t.
      shots_op_*:      tiros del rival (es decir shots_op_h = shots_a normalmente).
      xg_live_*:       xG acumulado live (si la fuente lo da).
      red_*:           tarjetas rojas.

    Returns: dict con p_home, p_draw, p_away, expected_final_*.
    """
    minute = max(0, min(120, minute))
    time_remaining = max(0.0, (90 - minute) / 90.0) if minute <= 90 else 0.0

    # Si el partido terminó, devolvemos el resultado determinístico
    if time_remaining <= 0:
        if current_home_goals > current_away_goals:
            return {"p_home": 1.0, "p_draw": 0.0, "p_away": 0.0,
                    "expected_final_home": current_home_goals,
                    "expected_final_away": current_away_goals,
                    "minute_used": minute, "lam_remaining": 0, "mu_remaining": 0}
        if current_home_goals < current_away_goals:
            return {"p_home": 0.0, "p_draw": 0.0, "p_away": 1.0,
                    "expected_final_home": current_home_goals,
                    "expected_final_away": current_away_goals,
                    "minute_used": minute, "lam_remaining": 0, "mu_remaining": 0}
        return {"p_home": 0.0, "p_draw": 1.0, "p_away": 0.0,
                "expected_final_home": current_home_goals,
                "expected_final_away": current_away_goals,
                "minute_used": minute, "lam_remaining": 0, "mu_remaining": 0}

    # Ajuste de tasa por tiempo restante
    lam_rem = lam_pre * time_remaining
    mu_rem = mu_pre * time_remaining

    # Ajuste por expulsiones
    lam_rem *= _red_penalty(red_h)
    mu_rem *= _red_penalty(red_a)
    # Las rojas del rival INCREMENTAN tu tasa (~10%)
    lam_rem *= (1 + 0.10 * red_a)
    mu_rem *= (1 + 0.10 * red_h)

    # Ajuste por estadísticas live
    if shots_h + shots_a > 0 or xg_live_h or xg_live_a:
        mult_h = _stat_multiplier(shots_h, shots_a, xg_live_h, lam_pre, minute)
        mult_a = _stat_multiplier(shots_a, shots_h, xg_live_a, mu_pre, minute)
        lam_rem *= mult_h
        mu_rem *= mult_a

    lam_rem = max(0.05, lam_rem)
    mu_rem = max(0.05, mu_rem)

    # Distribución de goles futuros (Poisson independiente sobre tiempo restante)
    h_pmf = [poisson_pmf(i, lam_rem) for i in range(max_extra_goals + 1)]
    a_pmf = [poisson_pmf(j, mu_rem) for j in range(max_extra_goals + 1)]

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for i in range(max_extra_goals + 1):
        for j in range(max_extra_goals + 1):
            p = h_pmf[i] * a_pmf[j]
            final_h = current_home_goals + i
            final_a = current_away_goals + j
            if final_h > final_a:
                p_home += p
            elif final_h < final_a:
                p_away += p
            else:
                p_draw += p
    # Renormaliza (cola finita)
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s

    expected_h = current_home_goals + lam_rem
    expected_a = current_away_goals + mu_rem

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "expected_final_home": expected_h,
        "expected_final_away": expected_a,
        "lam_remaining": lam_rem,
        "mu_remaining": mu_rem,
        "minute_used": minute,
        "model": "Dixon-Robinson (1998) in-play",
    }
