"""
Pi-ratings (Constantinou & Fenton, 2013).

Referencia: Constantinou, A. C., & Fenton, N. E. (2013). "Determining the level
of ability of football teams by dynamic ratings based on the relative
discrepancies in scores between adjacent divisions." Journal of Quantitative
Analysis in Sports, 9(1).

A diferencia de Elo (un único rating por equipo), Pi mantiene DOS ratings por
equipo: rating como LOCAL y rating como VISITANTE. Esto captura asimetría real
(equipos que son tigres en casa pero gatitos afuera).

Algoritmo de actualización tras un partido h(R_hH) vs a(R_aA) con resultado g_h - g_a:

  1) Predicción esperada de diferencia de goles:
       gd_hat = (R_hH - R_aA) / c       con c constante (≈ 3)
       cap: gd_hat = sign × min(|gd_hat|, b)   (b ≈ 1.5 hard-cap)
  2) Error: e = (g_h - g_a) - gd_hat
  3) Actualización del rating del local:
       R_hH += λ × ψ(e)        (home rating del local, λ aprendizaje agresivo)
       R_hA += γ × ψ(e)        (away rating del local, γ "spillover" menor)
       R_aA -= λ × ψ(e)
       R_aH -= γ × ψ(e)
     ψ(e) = sign(e) × log(1 + |e|)   (escala logarítmica para outliers)

Parámetros recomendados por el paper:
  λ ≈ 0.054, γ ≈ 0.79, c ≈ 3, b ≈ 1.5

Para predecir P(home/draw/away), usamos la diferencia esperada gd_hat con una
calibración logística (parámetros aprendidos del histórico).
"""
from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from data.db import get_session, Match, Team

log = logging.getLogger("pi_ratings")

PI_PATH = Path(__file__).parent / "trained" / "pi_ratings.json"


def _psi(e: float) -> float:
    return math.copysign(math.log(1 + abs(e)), e)


@dataclass
class PiRatings:
    home: dict[str, float] = field(default_factory=dict)   # rating como local
    away: dict[str, float] = field(default_factory=dict)   # rating como visitante
    lam: float = 0.054      # learning rate "directo"
    gamma: float = 0.79     # spillover entre home/away rating
    c: float = 3.0          # divisor predicción de gd
    b: float = 1.5          # cap de gd_hat
    # Calibración logística para P(home), P(draw), P(away) a partir de gd_hat
    # Modelo: P(home) = σ(a0 + a1·gd) etc.
    calib_a: list[float] = field(default_factory=lambda: [0.0, 0.7])
    calib_h_thresh: float = 0.35   # umbral gd_hat por encima del cual P(home) sube
    trained_at: Optional[str] = None
    n_matches: int = 0

    @classmethod
    def fit(cls, session: Session, lam: float = 0.054, gamma: float = 0.79,
            c: float = 3.0, b: float = 1.5) -> "PiRatings":
        history = (
            session.query(Match)
            .filter(Match.finished == True,  # noqa
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None))
            .order_by(Match.datetime_utc.asc())
            .all()
        )
        if not history:
            raise ValueError("Sin histórico para Pi-ratings")
        home = {}
        away = {}
        gd_hats = []
        outcomes = []
        for m in history:
            r_hH = home.get(m.home, 0.0)
            r_aA = away.get(m.away, 0.0)
            gd_hat_raw = (r_hH - r_aA) / c
            gd_hat = math.copysign(min(abs(gd_hat_raw), b), gd_hat_raw)
            # Si es sede neutral, NO usar el home rating; usar el "general"
            # promedio entre home/away rating del local. Aproximación simple.
            # (En Mundial todos son neutrales salvo anfitriones.)
            gd_real = m.home_goals - m.away_goals
            # Para entrenar la calibración: guardamos gd_hat e outcome
            gd_hats.append(gd_hat)
            outcomes.append(0 if gd_real < 0 else (1 if gd_real == 0 else 2))

            err = gd_real - gd_hat
            psi_e = _psi(err)
            home[m.home] = r_hH + lam * psi_e
            away[m.home] = away.get(m.home, 0.0) + gamma * psi_e
            away[m.away] = r_aA - lam * psi_e
            home[m.away] = home.get(m.away, 0.0) - gamma * psi_e

        # Calibración: ordinal logistic light. Aproximamos con dos thresholds.
        # P(away) = σ(-(a + b·gd_hat) + t_low)
        # P(home) = 1 - σ(-(a + b·gd_hat) + t_high)
        # P(draw) = lo que queda
        gd_arr = np.array(gd_hats)
        oc_arr = np.array(outcomes)
        # Estimación simple por buckets de gd_hat
        # Ajustar [a, b] por MLE con regresión logística multinomial mínima
        from sklearn.linear_model import LogisticRegression
        X = gd_arr.reshape(-1, 1)
        lr = LogisticRegression(multi_class="multinomial", C=1.0, max_iter=500)
        try:
            lr.fit(X, oc_arr)
            # Guardamos coef y intercept para reusar sin sklearn al predecir
            calib_a = [float(lr.intercept_[k]) for k in range(3)] + \
                      [float(lr.coef_[k][0]) for k in range(3)]
        except Exception as e:
            log.warning("Calibración Pi falló (%s); uso defaults", e)
            calib_a = [0.5, 0.0, -0.5, -0.6, 0.0, 0.6]
        model = cls(
            home=home, away=away,
            lam=lam, gamma=gamma, c=c, b=b,
            calib_a=calib_a,
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_matches=len(history),
        )
        log.info("Pi-ratings ajustado sobre %d partidos (%d equipos)",
                 len(history), len(home))
        return model

    def gd_hat(self, home_team: str, away_team: str, neutral: bool) -> float:
        rh = self.home.get(home_team, 0.0)
        ra = self.away.get(away_team, 0.0)
        if neutral:
            # Sede neutral: promedio entre home/away rating de cada lado
            rh = (self.home.get(home_team, 0.0) + self.away.get(home_team, 0.0)) / 2
            ra = (self.home.get(away_team, 0.0) + self.away.get(away_team, 0.0)) / 2
        gd_raw = (rh - ra) / self.c
        return math.copysign(min(abs(gd_raw), self.b), gd_raw)

    def predict_outcomes(self, home_team: str, away_team: str,
                         neutral: bool) -> dict:
        gd = self.gd_hat(home_team, away_team, neutral)
        # calib_a layout: [int_away, int_draw, int_home, coef_away, coef_draw, coef_home]
        ints = self.calib_a[:3]
        coefs = self.calib_a[3:]
        logits = [ints[k] + coefs[k] * gd for k in range(3)]
        # softmax
        m = max(logits)
        exps = [math.exp(L - m) for L in logits]
        s = sum(exps)
        p_away, p_draw, p_home = (e / s for e in exps)
        return {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "gd_hat": gd,
        }

    def save(self, path: Path = PI_PATH):
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps({
            "home": self.home, "away": self.away,
            "lam": self.lam, "gamma": self.gamma, "c": self.c, "b": self.b,
            "calib_a": self.calib_a,
            "trained_at": self.trained_at, "n_matches": self.n_matches,
        }, indent=2))

    @classmethod
    def load(cls, path: Path = PI_PATH) -> Optional["PiRatings"]:
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return cls(**d)
        except Exception as e:
            log.warning("No pude cargar Pi-ratings: %s", e)
            return None
