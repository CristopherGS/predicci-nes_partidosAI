"""Carga datos sembrados a la base. Idempotente: solo escribe si está vacía."""
from __future__ import annotations
from datetime import datetime
from .db import init_db, get_session, Team, Match
from .seed_teams import TEAMS
from .seed_historical import REAL_MATCHES
from .seed_wc2026 import generate_group_stage, generate_knockouts, PLAYED_RESULTS


def load_teams(session):
    """Carga selecciones del Mundial. Equipos históricos extra (que aparecen en
    REAL_MATCHES pero no clasificaron al Mundial) se insertan también con Elo
    aproximado para que el ML pueda usarlos."""
    if session.query(Team).count() > 0:
        return
    for code, name, conf, elo, xgf, xga in TEAMS:
        session.add(Team(code=code, name=name, confederation=conf,
                         elo=elo, xg_for=xgf, xg_against=xga))
    # Equipos secundarios que aparecen en histórico (no en Mundial 2026)
    extras = [
        ("WAL", "Gales",          "UEFA",     1755, 1.30, 1.25),
        ("CHI", "Chile",          "CONMEBOL", 1690, 1.20, 1.30),
        ("PER", "Perú",           "CONMEBOL", 1620, 1.05, 1.40),
        ("VEN", "Venezuela",      "CONMEBOL", 1655, 1.10, 1.35),
        ("SCO", "Escocia",        "UEFA",     1710, 1.25, 1.30),
        ("ALB", "Albania",        "UEFA",     1665, 1.15, 1.40),
        ("HUN", "Hungría",        "UEFA",     1720, 1.30, 1.30),
        ("SVK", "Eslovaquia",     "UEFA",     1685, 1.15, 1.35),
        ("SVN", "Eslovenia",      "UEFA",     1640, 1.10, 1.40),
        ("GEO", "Georgia",        "UEFA",     1660, 1.20, 1.40),
        ("BUL", "Bulgaria",       "UEFA",     1495, 0.85, 1.65),
        ("AND", "Andorra",        "UEFA",     1100, 0.40, 2.10),
        ("LAT", "Letonia",        "UEFA",     1320, 0.70, 1.85),
        ("ARM", "Armenia",        "UEFA",     1420, 0.80, 1.75),
        ("IRL", "Irlanda",        "UEFA",     1685, 1.15, 1.30),
        ("PUR", "Puerto Rico",    "CONCACAF", 1180, 0.60, 1.90),
        ("ZAM", "Zambia",         "CAF",      1545, 0.95, 1.55),
        ("NIG", "Níger",          "CAF",      1340, 0.75, 1.80),
        ("TAN", "Tanzania",       "CAF",      1430, 0.80, 1.70),
        ("CGO", "Congo",          "CAF",      1450, 0.85, 1.70),
        ("BAH", "Bahrein",        "AFC",      1380, 0.80, 1.75),
        ("IDN", "Indonesia",      "AFC",      1335, 0.75, 1.80),
    ]
    for code, name, conf, elo, xgf, xga in extras:
        session.add(Team(code=code, name=name, confederation=conf,
                         elo=elo, xg_for=xgf, xg_against=xga))
    session.commit()


def load_historical(session):
    """Inserta partidos históricos como Match con competition != WC2026."""
    if session.query(Match).filter(Match.competition != "WC2026").count() > 0:
        return
    for date_str, h, a, hg, ag, comp, neutral in REAL_MATCHES:
        dt = datetime.fromisoformat(date_str)
        # Si algún equipo no existe en Team, lo creamos con Elo medio
        _ensure_team(session, h)
        _ensure_team(session, a)
        session.add(Match(
            competition=comp,
            stage="HIST",
            datetime_utc=dt,
            home=h,
            away=a,
            home_goals=hg,
            away_goals=ag,
            neutral=neutral,
            finished=True,
        ))
    session.commit()


def _ensure_team(session, code):
    t = session.get(Team, code)
    if t is None:
        session.add(Team(code=code, name=code, confederation="?",
                         elo=1500, xg_for=1.20, xg_against=1.40))
        session.commit()


def load_wc2026_fixtures(session):
    if session.query(Match).filter(Match.competition == "WC2026").count() > 0:
        return
    fixtures = generate_group_stage() + generate_knockouts()
    for f in fixtures:
        if f["home"] is None:
            # Placeholder de knockouts: lo dejamos en la DB con NULL hosts
            session.add(Match(
                competition="WC2026",
                stage=f["stage"],
                datetime_utc=datetime.fromisoformat(f["datetime"]),
                neutral=True,
            ))
            continue
        played = PLAYED_RESULTS.get((f["home"], f["away"]))
        session.add(Match(
            competition="WC2026",
            stage=f["stage"],
            group=f["group"],
            matchday=f["matchday"],
            datetime_utc=datetime.fromisoformat(f["datetime"]),
            home=f["home"],
            away=f["away"],
            home_goals=played[0] if played else None,
            away_goals=played[1] if played else None,
            neutral=f["neutral"],
            finished=played is not None,
        ))
    session.commit()


def bootstrap():
    init_db()
    s = get_session()
    try:
        load_teams(s)
        load_historical(s)
        load_wc2026_fixtures(s)
    finally:
        s.close()


if __name__ == "__main__":
    bootstrap()
    print("DB lista.")
