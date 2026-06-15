"""
Feature engineering para el modelo ML.

Para un partido (home, away, datetime), devuelve un dict con:
  - elo_h, elo_a, elo_diff
  - xg_for_h, xg_against_h, xg_for_a, xg_against_a
  - form_h, form_a:       puntos ponderados por recencia (últimos N)
  - form_h_w, form_a_w:   forma exponencialmente ponderada (más reciente pesa más)
  - avg_goals_h, avg_goals_a, avg_conc_h, avg_conc_a
  - rest_days_h, rest_days_a:  días desde el último partido (penaliza fatiga,
    bonifica descanso)
  - h2h_diff: diferencial de victorias en últimos 5 enfrentamientos directos
              (+ = local domina; - = visitante domina; 0 = parejo / sin historia)
  - neutral, is_wc, confed_match
"""
from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from data.db import Match, Team

LAST_N = 6
WEIGHTED_DECAY = 0.85  # peso decreciente exponencial
H2H_LAST = 5


def _last_matches(session: Session, team: str, before: datetime, n: int = LAST_N):
    return (
        session.query(Match)
        .filter(Match.finished == True,  # noqa: E712
                Match.datetime_utc < before,
                or_(Match.home == team, Match.away == team))
        .order_by(Match.datetime_utc.desc())
        .limit(n)
        .all()
    )


def _form_stats(session: Session, team: str, before: datetime):
    matches = _last_matches(session, team, before)
    if not matches:
        return {"form": 0.5, "form_w": 0.5,
                "avg_goals": 1.2, "avg_conc": 1.2,
                "last_match_dt": None, "games": 0}
    pts_total = 0
    pts_w = 0.0
    w_total = 0.0
    gf = 0
    ga = 0
    # matches están en orden descendente → más reciente primero
    for i, m in enumerate(matches):
        if m.home == team:
            mine, theirs = m.home_goals, m.away_goals
        else:
            mine, theirs = m.away_goals, m.home_goals
        gf += mine
        ga += theirs
        pts = 3 if mine > theirs else (1 if mine == theirs else 0)
        pts_total += pts
        weight = WEIGHTED_DECAY ** i
        pts_w += pts * weight
        w_total += 3 * weight  # máx pts ponderados = 3 * weight
    n = len(matches)
    return {
        "form": pts_total / (3.0 * n),
        "form_w": (pts_w / w_total) if w_total > 0 else 0.5,
        "avg_goals": gf / n,
        "avg_conc": ga / n,
        "last_match_dt": matches[0].datetime_utc,
        "games": n,
    }


def _h2h_diff(session: Session, team_a: str, team_b: str, before: datetime) -> int:
    """Diferencia de victorias en últimos H2H_LAST enfrentamientos directos.
    Positivo si team_a ganó más; negativo si team_b dominó."""
    matches = (
        session.query(Match)
        .filter(Match.finished == True,  # noqa
                Match.datetime_utc < before,
                or_(
                    and_(Match.home == team_a, Match.away == team_b),
                    and_(Match.home == team_b, Match.away == team_a),
                ))
        .order_by(Match.datetime_utc.desc())
        .limit(H2H_LAST)
        .all()
    )
    if not matches:
        return 0
    wins_a = 0
    wins_b = 0
    for m in matches:
        if m.home_goals == m.away_goals:
            continue
        winner = m.home if m.home_goals > m.away_goals else m.away
        if winner == team_a:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a - wins_b


def _rest_days(last_dt: Optional[datetime], when: datetime) -> float:
    if last_dt is None:
        return 7.0  # default razonable cuando no hay historia
    delta = (when - last_dt).total_seconds() / 86400.0
    # Cap a [0, 30] para evitar outliers (selecciones que no juegan en 2 años)
    return max(0.0, min(30.0, delta))


def build_features(session: Session, home: str, away: str, when: datetime,
                   neutral: bool, is_wc: bool) -> dict:
    th = session.get(Team, home)
    ta = session.get(Team, away)
    if th is None or ta is None:
        raise ValueError(f"Equipo no encontrado: {home} / {away}")
    fh = _form_stats(session, home, when)
    fa = _form_stats(session, away, when)
    return {
        "elo_h": th.elo,
        "elo_a": ta.elo,
        "elo_diff": th.elo - ta.elo,
        "xg_for_h": th.xg_for,
        "xg_against_h": th.xg_against,
        "xg_for_a": ta.xg_for,
        "xg_against_a": ta.xg_against,
        "form_h": fh["form"],
        "form_a": fa["form"],
        "form_h_w": fh["form_w"],
        "form_a_w": fa["form_w"],
        "avg_goals_h": fh["avg_goals"],
        "avg_goals_a": fa["avg_goals"],
        "avg_conc_h": fh["avg_conc"],
        "avg_conc_a": fa["avg_conc"],
        "rest_days_h": _rest_days(fh["last_match_dt"], when),
        "rest_days_a": _rest_days(fa["last_match_dt"], when),
        "h2h_diff": _h2h_diff(session, home, away, when),
        "neutral": int(neutral),
        "is_wc": int(is_wc),
        "confed_match": int(th.confederation != ta.confederation),
    }


FEATURE_ORDER = [
    "elo_h", "elo_a", "elo_diff",
    "xg_for_h", "xg_against_h", "xg_for_a", "xg_against_a",
    "form_h", "form_a", "form_h_w", "form_a_w",
    "avg_goals_h", "avg_goals_a", "avg_conc_h", "avg_conc_a",
    "rest_days_h", "rest_days_a",
    "h2h_diff",
    "neutral", "is_wc", "confed_match",
]


def to_vector(features: dict) -> list[float]:
    return [features[k] for k in FEATURE_ORDER]
