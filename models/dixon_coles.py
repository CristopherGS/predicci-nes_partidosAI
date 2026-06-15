"""
Modelo Dixon-Coles (1997).

Referencia: Dixon, M. J., & Coles, S. G. (1997). "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market". Journal of the Royal
Statistical Society: Series C (Applied Statistics), 46(2), 265-280.

Es el modelo estándar usado por casas de apuestas. Modifica Poisson bivariado
con dos contribuciones clave:

  1) Parámetros de ataque (α_i) y defensa (β_i) por equipo, estimados con MLE
     sobre histórico.
  2) Función de corrección τ(x,y,λ,μ,ρ) para low-scores (0-0, 1-0, 0-1, 1-1),
     porque Poisson independiente subestima esos marcadores empíricamente.

Las tasas son:
    λ = α_h × β_a × γ        (γ = ventaja local)
    μ = α_a × β_h

Probabilidad:
    P(X=x, Y=y) = τ(x,y) × Poisson(x;λ) × Poisson(y;μ)

donde τ:
    τ(0,0) = 1 - λμρ
    τ(1,0) = 1 + μρ
    τ(0,1) = 1 + λρ
    τ(1,1) = 1 - ρ
    τ(x,y) = 1 si max(x,y) > 1

ρ típicamente cerca de -0.1 (mide la correlación negativa de low-scores).

Para entrenamiento se usa weighting temporal exponencial xi: partidos antiguos
pesan menos.

  L = Π τ(...) × Pois(x;λ) × Pois(y;μ)
  log L con weight exp(-xi × t)  donde t = días desde el partido.

Optimización: scipy.optimize.minimize (BFGS) sobre -log L.
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
from scipy.optimize import minimize
from sqlalchemy.orm import Session

from data.db import get_session, Match, Team

log = logging.getLogger("dixon_coles")

DC_PATH = Path(__file__).parent / "trained" / "dixon_coles.json"


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


@dataclass
class DixonColes:
    """Parámetros del modelo después de fittear."""
    teams: list[str] = field(default_factory=list)
    # α (ataque) y β (defensa) por equipo. Ataque promedio = 1.
    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    gamma: float = 1.3   # home advantage multiplicativo
    rho: float = -0.1    # corrección low-score
    trained_at: Optional[str] = None
    n_matches: int = 0
    xi: float = 0.0019   # decay temporal (0.0019/día ≈ medio peso a ~1 año)

    # ---------- Entrenamiento ----------
    @classmethod
    def fit(cls, session: Session, xi: float = 0.0019,
            min_matches_per_team: int = 2) -> "DixonColes":
        """Estima α, β por equipo + γ + ρ por MLE sobre el histórico.

        Args:
          xi: tasa de decaimiento temporal. 0 = sin weighting; valores típicos
              0.001-0.005 (~1-3 años de vida media).
          min_matches_per_team: descarta equipos con muy pocos partidos.
        """
        history = (
            session.query(Match)
            .filter(Match.finished == True,  # noqa
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None))
            .order_by(Match.datetime_utc.asc())
            .all()
        )
        # Contar partidos por equipo y filtrar
        counts: dict[str, int] = {}
        for m in history:
            counts[m.home] = counts.get(m.home, 0) + 1
            counts[m.away] = counts.get(m.away, 0) + 1
        teams = sorted(t for t, c in counts.items() if c >= min_matches_per_team)
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n < 4:
            raise ValueError("No hay suficientes equipos con histórico para Dixon-Coles")

        # Construir arrays para vectorizar
        rows = [m for m in history if m.home in idx and m.away in idx]
        if not rows:
            raise ValueError("Sin partidos válidos tras filtrar")
        h_idx = np.array([idx[m.home] for m in rows], dtype=np.int32)
        a_idx = np.array([idx[m.away] for m in rows], dtype=np.int32)
        hg = np.array([m.home_goals for m in rows], dtype=np.int32)
        ag = np.array([m.away_goals for m in rows], dtype=np.int32)

        # Pesos temporales: más reciente = más peso
        now = max(m.datetime_utc for m in rows)
        days = np.array([(now - m.datetime_utc).total_seconds() / 86400.0
                         for m in rows])
        w = np.exp(-xi * days)

        # Parametrización:
        #   x = [log_alpha_1,...,log_alpha_n-1, log_beta_1,...,log_beta_n, log_gamma, rho_raw]
        # alpha_n queda fijado por restricción Σ log α = 0  → producto α = 1
        # Esto evita identifiability de α y β.
        def unpack(x):
            log_a = x[:n - 1]
            # Forzar suma log_alpha = 0  ⇒ último α = -sum(otros)
            last = -np.sum(log_a)
            log_alpha = np.concatenate([log_a, [last]])
            log_beta = x[n - 1: 2 * n - 1]
            log_gamma = x[2 * n - 1]
            rho_raw = x[2 * n]
            # ρ debe ser tal que τ > 0 siempre; lo restringimos por tanh.
            rho = 0.25 * np.tanh(rho_raw)
            return np.exp(log_alpha), np.exp(log_beta), math.exp(log_gamma), rho

        def neg_log_lik(x):
            alpha, beta, gamma, rho = unpack(x)
            lam = alpha[h_idx] * beta[a_idx] * gamma
            mu = alpha[a_idx] * beta[h_idx]
            # log Poisson para los goles
            ll_pois_h = hg * np.log(lam) - lam - _gammaln_int(hg)
            ll_pois_a = ag * np.log(mu) - mu - _gammaln_int(ag)
            # corrección tau (vectorizada manual)
            tau = np.ones_like(lam)
            mask00 = (hg == 0) & (ag == 0)
            mask10 = (hg == 1) & (ag == 0)
            mask01 = (hg == 0) & (ag == 1)
            mask11 = (hg == 1) & (ag == 1)
            tau[mask00] = 1.0 - lam[mask00] * mu[mask00] * rho
            tau[mask10] = 1.0 + mu[mask10] * rho
            tau[mask01] = 1.0 + lam[mask01] * rho
            tau[mask11] = 1.0 - rho
            tau = np.clip(tau, 1e-9, None)
            ll = w * (np.log(tau) + ll_pois_h + ll_pois_a)
            return -ll.sum()

        # Init: α = β = 1 (log = 0), γ = 1.3, ρ = -0.1
        x0 = np.zeros(2 * n + 1)
        x0[2 * n - 1] = math.log(1.3)
        x0[2 * n] = -0.4  # tanh(-0.4) ≈ -0.38 → ρ ≈ -0.094
        log.info("Optimizando Dixon-Coles: %d equipos, %d partidos, %d params",
                 n, len(rows), len(x0))
        res = minimize(neg_log_lik, x0, method="L-BFGS-B",
                       options={"maxiter": 500, "disp": False})
        alpha, beta, gamma, rho = unpack(res.x)
        log.info("Dixon-Coles ajustado: γ=%.3f ρ=%.3f convergencia=%s",
                 gamma, rho, res.success)
        model = cls(
            teams=teams,
            alpha={t: float(alpha[i]) for t, i in idx.items()},
            beta={t: float(beta[i]) for t, i in idx.items()},
            gamma=float(gamma),
            rho=float(rho),
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_matches=len(rows),
            xi=xi,
        )
        return model

    # ---------- Predicción ----------
    def rates(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Devuelve (λ, μ): goles esperados local y visitante."""
        ah = self.alpha.get(home, 1.0)
        bh = self.beta.get(home, 1.0)
        aa = self.alpha.get(away, 1.0)
        ba = self.beta.get(away, 1.0)
        g = 1.0 if neutral else self.gamma
        return ah * ba * g, aa * bh

    def score_matrix(self, home: str, away: str, neutral: bool,
                     max_goals: int = 8) -> np.ndarray:
        lam, mu = self.rates(home, away, neutral)
        rho = self.rho
        h = np.array([math.exp(-lam) * lam ** k / math.factorial(k)
                      for k in range(max_goals + 1)])
        a = np.array([math.exp(-mu) * mu ** k / math.factorial(k)
                      for k in range(max_goals + 1)])
        m = np.outer(h, a)
        # Aplicar corrección tau a low-scores
        m[0, 0] *= (1.0 - lam * mu * rho)
        m[1, 0] *= (1.0 + mu * rho)
        m[0, 1] *= (1.0 + lam * rho)
        m[1, 1] *= (1.0 - rho)
        m = np.clip(m, 0, None)
        s = m.sum()
        if s > 0:
            m /= s
        return m

    def predict_outcomes(self, home: str, away: str, neutral: bool) -> dict:
        m = self.score_matrix(home, away, neutral)
        n = m.shape[0]
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        p_over = 0.0
        p_btts = 0.0
        for i in range(n):
            for j in range(n):
                if i + j >= 3:
                    p_over += m[i, j]
                if i >= 1 and j >= 1:
                    p_btts += m[i, j]
        idx = np.unravel_index(np.argmax(m), m.shape)
        lam, mu = self.rates(home, away, neutral)
        return {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "p_over_2_5": float(p_over), "p_btts": float(p_btts),
            "most_likely_score": (int(idx[0]), int(idx[1])),
            "lam": float(lam), "mu": float(mu),
        }

    # ---------- Persistencia ----------
    def save(self, path: Path = DC_PATH):
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps({
            "teams": self.teams, "alpha": self.alpha, "beta": self.beta,
            "gamma": self.gamma, "rho": self.rho,
            "trained_at": self.trained_at, "n_matches": self.n_matches,
            "xi": self.xi,
        }, indent=2))

    @classmethod
    def load(cls, path: Path = DC_PATH) -> Optional["DixonColes"]:
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return cls(**d)
        except Exception as e:
            log.warning("No pude cargar Dixon-Coles: %s", e)
            return None


# Helper: log Γ(n+1) = log(n!) para arrays de ints chicos
_GAMMALN_CACHE = [0.0]


def _gammaln_int(arr: np.ndarray) -> np.ndarray:
    """log(arr!) para enteros pequeños."""
    out = np.zeros_like(arr, dtype=np.float64)
    flat = arr.ravel()
    for i, v in enumerate(flat):
        v = int(v)
        while len(_GAMMALN_CACHE) <= v:
            _GAMMALN_CACHE.append(_GAMMALN_CACHE[-1] + math.log(len(_GAMMALN_CACHE)))
        out.ravel()[i] = _GAMMALN_CACHE[v]
    return out
