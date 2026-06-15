"""
Sistema Elo para selecciones (estilo eloratings.net).
- K depende de la importancia del partido (Mundial > Eliminatoria > Amistoso).
- Bonus de margen de gol.
- Ventaja de local: +100 Elo (no aplica en sede neutral).
"""
from __future__ import annotations
import math
from typing import Iterable
from sqlalchemy.orm import Session
from data.db import Team, Match

K_BY_COMPETITION = {
    "WC":       60,
    "EURO":     60,
    "COPA":     55,
    "QUAL":     40,
    "FRIENDLY": 20,
    "WC2026":   60,
}
HOME_ADV = 100.0


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def goal_diff_multiplier(gd: int) -> float:
    """Multiplica K según diferencia de goles (eloratings.net)."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def update_pair(elo_h: float, elo_a: float, hg: int, ag: int,
                comp: str, neutral: bool) -> tuple[float, float]:
    k = K_BY_COMPETITION.get(comp, 30)
    eh = elo_h + (0 if neutral else HOME_ADV)
    exp_h = expected_score(eh, elo_a)
    if hg > ag:
        s_h = 1.0
    elif hg < ag:
        s_h = 0.0
    else:
        s_h = 0.5
    g_mult = goal_diff_multiplier(hg - ag)
    delta = k * g_mult * (s_h - exp_h)
    return elo_h + delta, elo_a - delta


def recompute_all(session: Session) -> dict[str, float]:
    """Recalcula Elo de todas las selecciones cronológicamente sobre el histórico."""
    # Reset al rating semilla (mantiene el seed de seed_teams)
    teams = {t.code: t.elo for t in session.query(Team).all()}
    matches = (
        session.query(Match)
        .filter(Match.finished == True,  # noqa: E712
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None))
        .order_by(Match.datetime_utc.asc())
        .all()
    )
    for m in matches:
        if m.home not in teams or m.away not in teams:
            continue
        new_h, new_a = update_pair(
            teams[m.home], teams[m.away],
            m.home_goals, m.away_goals,
            m.competition or "FRIENDLY",
            bool(m.neutral),
        )
        teams[m.home] = new_h
        teams[m.away] = new_a
    # Persistir
    for code, elo in teams.items():
        t = session.get(Team, code)
        if t:
            t.elo = elo
    session.commit()
    return teams
