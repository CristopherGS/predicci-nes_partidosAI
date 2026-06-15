"""
Motor de predicción combinado:
  1) XGBoost clasificador 1X2 (resultado).
  2) XGBoost regresor sobre goles de cada lado (λ_home, λ_away).
  3) Poisson sobre las λ -> matriz score, BTTS, over 2.5.
  4) Blending con probabilidades Elo+Poisson como fallback robusto.
"""
from __future__ import annotations
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sqlalchemy.orm import Session

from data.db import get_session, Match, Team, Prediction
from .features import build_features, to_vector, FEATURE_ORDER
from .poisson import score_matrix, outcomes_from_matrix
from .dixon_coles import DixonColes
from .pi_ratings import PiRatings

MODEL_DIR = Path(__file__).parent / "trained"
MODEL_DIR.mkdir(exist_ok=True)
CLF_PATH = MODEL_DIR / "xgb_clf.joblib"
REG_H_PATH = MODEL_DIR / "xgb_goals_home.joblib"
REG_A_PATH = MODEL_DIR / "xgb_goals_away.joblib"
META_PATH = MODEL_DIR / "meta.json"

log = logging.getLogger("predictor")


def _elo_prob(elo_h: float, elo_a: float, neutral: bool) -> tuple[float, float, float]:
    """Probabilidad 1X2 a partir de Elo. Draw aproximado por curva empírica."""
    HOME = 100.0 if not neutral else 0.0
    diff = (elo_h + HOME) - elo_a
    # Probabilidad de no-empate del local (suma 1 con visitante "no draw")
    p_home_raw = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    # Empate aproximado por una curva campana en torno a Elo similar
    draw = 0.30 * math.exp(-(diff / 250.0) ** 2)
    p_home = (1 - draw) * p_home_raw
    p_away = (1 - draw) * (1 - p_home_raw)
    return p_home, draw, p_away


def _lambdas_from_features(f: dict) -> tuple[float, float]:
    """Estimación rápida de goles esperados sin ML (fallback)."""
    # Ataque local vs defensa visitante (y viceversa)
    avg_xg_for = (f["xg_for_h"] + f["xg_for_a"]) / 2
    avg_xg_against = (f["xg_against_h"] + f["xg_against_a"]) / 2
    league_avg = (avg_xg_for + avg_xg_against) / 2 or 1.3
    atk_h = f["xg_for_h"] / league_avg
    def_a = f["xg_against_a"] / league_avg
    atk_a = f["xg_for_a"] / league_avg
    def_h = f["xg_against_h"] / league_avg
    lam_h = league_avg * atk_h * def_a
    lam_a = league_avg * atk_a * def_h
    # Ajuste por Elo difference (sutil)
    elo_factor = 1.0 + max(min((f["elo_diff"]) / 1000.0, 0.4), -0.4)
    lam_h *= elo_factor
    lam_a /= elo_factor
    # Ventaja local
    if not f["neutral"]:
        lam_h *= 1.15
        lam_a *= 0.92
    return max(lam_h, 0.15), max(lam_a, 0.15)


class Predictor:
    """Meta-ensemble que combina:
      - Modelo ML (HGB+RF+LR calibrado)        peso 0.40
      - Dixon-Coles (1997, fórmula bookmaker)  peso 0.35
      - Pi-ratings (Constantinou 2013)         peso 0.15
      - Elo+Poisson (baseline)                 peso 0.10
    Para el marcador entero usamos la matriz Dixon-Coles si está disponible
    (suele dar low-scores más realistas), sino Poisson regular.
    """

    # Pesos del meta-ensemble (deben sumar 1)
    W_ML = 0.40
    W_DC = 0.35
    W_PI = 0.15
    W_ELO = 0.10

    def __init__(self):
        self.clf = None
        self.reg_h = None
        self.reg_a = None
        self.dc: Optional[DixonColes] = None
        self.pi: Optional[PiRatings] = None
        self.meta = {}
        self._load()

    def _load(self):
        if CLF_PATH.exists():
            try:
                self.clf = joblib.load(CLF_PATH)
                self.reg_h = joblib.load(REG_H_PATH)
                self.reg_a = joblib.load(REG_A_PATH)
                self.meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
                log.info("Modelos ML cargados.")
            except Exception as e:
                log.warning("Carga de modelos falló: %s", e)
                self.clf = None
        self.dc = DixonColes.load()
        self.pi = PiRatings.load()
        if self.dc:
            log.info("Dixon-Coles cargado (γ=%.3f ρ=%.3f, %d partidos)",
                     self.dc.gamma, self.dc.rho, self.dc.n_matches)
        if self.pi:
            log.info("Pi-ratings cargado (%d partidos)", self.pi.n_matches)

    def has_model(self) -> bool:
        return self.clf is not None

    def loaded_models(self) -> dict:
        return {
            "ml_ensemble": self.clf is not None,
            "dixon_coles": self.dc is not None,
            "pi_ratings": self.pi is not None,
            "elo_poisson": True,
        }

    def predict(self, session: Session, match: Match) -> dict:
        if not match.home or not match.away:
            return {"error": "Match aún sin equipos definidos (knockout)."}
        feats = build_features(session, match.home, match.away,
                               match.datetime_utc, bool(match.neutral),
                               match.competition in ("WC", "WC2026"))
        vec = np.array([to_vector(feats)])
        neutral = bool(match.neutral)

        # ===== 1) Modelo ML (XGBoost-style ensemble calibrado) =====
        ml_out = None
        if self.clf is not None:
            proba = self.clf.predict_proba(vec)[0]
            classes = list(self.clf.classes_)
            ml_out = {
                "p_home": float(proba[classes.index(2)]),
                "p_draw": float(proba[classes.index(1)]),
                "p_away": float(proba[classes.index(0)]),
            }

        # ===== 2) Dixon-Coles (1997) =====
        dc_out = None
        if self.dc is not None and match.home in self.dc.alpha and match.away in self.dc.alpha:
            dc_out = self.dc.predict_outcomes(match.home, match.away, neutral)

        # ===== 3) Pi-ratings (Constantinou & Fenton 2013) =====
        pi_out = None
        if self.pi is not None and match.home in self.pi.home and match.away in self.pi.away:
            pi_out = self.pi.predict_outcomes(match.home, match.away, neutral)

        # ===== 4) Baseline Elo + Poisson =====
        if self.reg_h is not None:
            lam_h = float(max(0.15, self.reg_h.predict(vec)[0]))
            lam_a = float(max(0.15, self.reg_a.predict(vec)[0]))
        else:
            lam_h, lam_a = _lambdas_from_features(feats)
        sm = score_matrix(lam_h, lam_a)
        poisson_out = outcomes_from_matrix(sm)
        elo_h, elo_d, elo_a = _elo_prob(feats["elo_h"], feats["elo_a"], neutral)
        baseline_out = {
            "p_home": 0.5 * elo_h + 0.5 * poisson_out["p_home"],
            "p_draw": 0.5 * elo_d + 0.5 * poisson_out["p_draw"],
            "p_away": 0.5 * elo_a + 0.5 * poisson_out["p_away"],
        }

        # ===== Meta-ensemble (combinación ponderada) =====
        # Reasignamos pesos si algún modelo no está disponible
        weights = []
        outs = []
        if ml_out is not None:
            weights.append(self.W_ML); outs.append(ml_out)
        if dc_out is not None:
            weights.append(self.W_DC); outs.append(dc_out)
        if pi_out is not None:
            weights.append(self.W_PI); outs.append(pi_out)
        weights.append(self.W_ELO); outs.append(baseline_out)
        ws = sum(weights)
        weights = [w / ws for w in weights]

        p_home = sum(w * o["p_home"] for w, o in zip(weights, outs))
        p_draw = sum(w * o["p_draw"] for w, o in zip(weights, outs))
        p_away = sum(w * o["p_away"] for w, o in zip(weights, outs))
        # Renormalizar por seguridad numérica
        s = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s

        # ===== Marcador entero más probable =====
        # Preferimos Dixon-Coles si está (corrige low-scores), sino Poisson
        if dc_out is not None:
            dc_matrix = self.dc.score_matrix(match.home, match.away, neutral)
            idx = np.unravel_index(np.argmax(dc_matrix), dc_matrix.shape)
            mls_h, mls_a = int(idx[0]), int(idx[1])
            # Recalcular over 2.5 y BTTS desde la matriz DC también
            n = dc_matrix.shape[0]
            p_over = sum(dc_matrix[i, j] for i in range(n) for j in range(n) if i + j >= 3)
            p_btts = sum(dc_matrix[i, j] for i in range(n) for j in range(n) if i >= 1 and j >= 1)
            score_source = "dixon-coles"
        else:
            mls_h, mls_a = poisson_out["most_likely_score"]
            p_over = poisson_out["p_over_2_5"]
            p_btts = poisson_out["p_btts"]
            score_source = "poisson"

        # Versión del modelo + nombres de los componentes activos
        components = []
        if ml_out is not None: components.append("ML-ensemble")
        if dc_out is not None: components.append("Dixon-Coles")
        if pi_out is not None: components.append("Pi-ratings")
        components.append("Elo-Poisson")
        model_v = "meta:" + "+".join(components)

        # Confianza: 1 - entropía normalizada
        ent = -sum(p * math.log(p + 1e-9) for p in (p_home, p_draw, p_away))
        confidence = 1 - ent / math.log(3)

        return {
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "lam_h": lam_h,
            "lam_a": lam_a,
            "pred_home_goals": int(mls_h),
            "pred_away_goals": int(mls_a),
            "expected_home_goals": poisson_out["expected_home_goals"],
            "expected_away_goals": poisson_out["expected_away_goals"],
            "p_over_2_5": float(p_over),
            "p_btts": float(p_btts),
            "most_likely_score": (int(mls_h), int(mls_a)),
            "confidence": confidence,
            "model_version": model_v,
            "score_source": score_source,
            "features": feats,
            # Voto desagregado por modelo (para mostrar en UI)
            "components": {
                "ml": ml_out,
                "dixon_coles": dc_out and {k: v for k, v in dc_out.items()
                                           if k in ("p_home","p_draw","p_away","lam","mu")},
                "pi_ratings": pi_out and {k: v for k, v in pi_out.items()
                                          if k in ("p_home","p_draw","p_away","gd_hat")},
                "elo_poisson": baseline_out,
            },
            "weights": dict(zip(
                ["ml", "dixon_coles", "pi_ratings", "elo_poisson"][:len(weights)],
                [round(w, 3) for w in weights],
            )),
        }

    def predict_and_store(self, session: Session, match: Match) -> dict:
        out = self.predict(session, match)
        if "error" in out:
            return out
        # Borra predicción previa para este match (mantenemos solo la última)
        session.query(Prediction).filter(Prediction.match_id == match.id).delete()
        p = Prediction(
            match_id=match.id,
            p_home=out["p_home"], p_draw=out["p_draw"], p_away=out["p_away"],
            pred_home_goals=out["pred_home_goals"],
            pred_away_goals=out["pred_away_goals"],
            over_25=out["p_over_2_5"],
            btts=out["p_btts"],
            model_version=out["model_version"],
            confidence=out["confidence"],
        )
        session.add(p)
        session.commit()
        out["prediction_id"] = p.id
        return out


_singleton: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _singleton
    if _singleton is None:
        _singleton = Predictor()
    return _singleton


def reload_predictor():
    global _singleton
    _singleton = Predictor()
