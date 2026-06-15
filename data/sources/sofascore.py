"""
Source Sofascore (api.sofascore.com).

Sofascore expone una API no documentada pero estable que usan apps deportivas.
Es gratis y devuelve TODAS las stats que el tier free de football-data omite:
posesión, tiros, tiros al arco, corners, faltas, tarjetas, xG live.

Endpoints útiles:
  GET /api/v1/sport/football/events/live
       → lista de TODOS los partidos de fútbol en vivo (incluye Mundial)
  GET /api/v1/event/{id}/statistics
       → estadísticas detalladas (cuando el partido está en juego)
  GET /api/v1/event/{id}
       → detalle del partido (marcador + minuto)

Headers: hace falta un User-Agent realista o devuelve 403.
"""
from __future__ import annotations
import logging
from typing import Optional

import requests

log = logging.getLogger("source.sofascore")

BASE = "https://api.sofascore.com/api/v1"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                   "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                   "Version/17.0 Safari/605.1.15"),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}

# Mapeo de nombre Sofascore → código de DB
NAME_TO_CODE = {
    "Argentina": "ARG", "Brazil": "BRA", "Uruguay": "URU", "Colombia": "COL",
    "Ecuador": "ECU", "Paraguay": "PAR", "Bolivia": "BOL", "Chile": "CHI",
    "Peru": "PER", "Venezuela": "VEN",
    "France": "FRA", "Spain": "ESP", "England": "ENG", "Portugal": "POR",
    "Netherlands": "NED", "Germany": "GER", "Italy": "ITA", "Belgium": "BEL",
    "Croatia": "CRO", "Switzerland": "SUI", "Denmark": "DEN", "Austria": "AUT",
    "Poland": "POL", "Serbia": "SRB", "Turkey": "TUR", "Türkiye": "TUR",
    "Norway": "NOR",
    "Mexico": "MEX", "United States": "USA", "USA": "USA", "Canada": "CAN",
    "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM", "Haiti": "HAI",
    "Japan": "JPN", "Iran": "IRN", "South Korea": "KOR",
    "Australia": "AUS", "Saudi Arabia": "KSA", "Qatar": "QAT",
    "Uzbekistan": "UZB", "Jordan": "JOR",
    "Morocco": "MAR", "Senegal": "SEN", "Egypt": "EGY", "Algeria": "ALG",
    "Ivory Coast": "CIV", "Côte d'Ivoire": "CIV", "Nigeria": "NGA",
    "Cameroon": "CMR", "Tunisia": "TUN", "Ghana": "GHA",
    "New Zealand": "NZL",
}


def _fetch(path: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            log.info("sofascore %s -> %d", path, r.status_code)
            return None
        return r.json()
    except Exception as e:
        log.info("sofascore %s falló: %s", path, e)
        return None


def list_live_events() -> list[dict]:
    """Lista TODOS los partidos de fútbol en vivo. Filtramos a Mundial 2026 por
    el campo tournament.uniqueTournament.name o slug."""
    data = _fetch("/sport/football/events/live")
    if not data:
        return []
    events = data.get("events", [])
    out = []
    for ev in events:
        # Detección del Mundial: nombre o id de torneo
        ut = (ev.get("tournament") or {}).get("uniqueTournament") or {}
        name = (ut.get("name") or "").lower()
        slug = (ut.get("slug") or "").lower()
        if "world cup" in name or "fifa" in name or "world-cup" in slug or "fifa" in slug:
            out.append(ev)
    return out


def list_all_live() -> list[dict]:
    """Devuelve TODOS los partidos en vivo (sin filtrar por Mundial), por si el
    matching por torneo falla y queremos hacer el match manual por equipos."""
    data = _fetch("/sport/football/events/live")
    return (data or {}).get("events", []) if data else []


def event_stats(event_id: int) -> Optional[dict]:
    """Devuelve un dict {home: {...}, away: {...}} con las stats agregadas."""
    data = _fetch(f"/event/{event_id}/statistics")
    if not data:
        return None
    # Sofascore devuelve statistics como lista por períodos ALL/1ST/2ND
    stats = data.get("statistics", [])
    if not stats:
        return None
    # Período "ALL"
    period = next((p for p in stats if p.get("period") == "ALL"),
                  stats[0])
    groups = period.get("groups", [])
    home_stats = {}
    away_stats = {}
    for group in groups:
        for item in group.get("statisticsItems", []):
            key = item.get("key") or item.get("name", "").lower().replace(" ", "_")
            home_v = item.get("home")
            away_v = item.get("away")
            home_stats[key] = home_v
            away_stats[key] = away_v
    return {"home": home_stats, "away": away_stats}


def _to_int(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    if s.endswith("%"):
        s = s[:-1]
    try:
        return int(float(s))
    except ValueError:
        return 0


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().rstrip("%"))
    except ValueError:
        return None


def normalize_stats(raw_stats: dict) -> dict:
    """Convierte el dict crudo de Sofascore a nuestro schema LiveStats."""
    if not raw_stats:
        return {}
    h = raw_stats.get("home", {})
    a = raw_stats.get("away", {})

    def get_pair(*keys):
        for k in keys:
            if k in h or k in a:
                return h.get(k), a.get(k)
        return None, None

    poss_h, poss_a = get_pair("ballPossession", "ball_possession", "possession")
    sh_h, sh_a = get_pair("totalShotsOnGoal", "totalShots", "shots", "total_shots")
    sot_h, sot_a = get_pair("shotsOnGoal", "shots_on_goal", "shotsOnTarget")
    c_h, c_a = get_pair("cornerKicks", "corner_kicks", "corners")
    f_h, f_a = get_pair("fouls")
    y_h, y_a = get_pair("yellowCards", "yellow_cards")
    r_h, r_a = get_pair("redCards", "red_cards")
    xg_h, xg_a = get_pair("expectedGoals", "expected_goals", "xG")

    return {
        "possession_h": _to_float(poss_h),
        "possession_a": _to_float(poss_a),
        "shots_h": _to_int(sh_h), "shots_a": _to_int(sh_a),
        "shots_on_target_h": _to_int(sot_h), "shots_on_target_a": _to_int(sot_a),
        "corners_h": _to_int(c_h), "corners_a": _to_int(c_a),
        "fouls_h": _to_int(f_h), "fouls_a": _to_int(f_a),
        "yellow_h": _to_int(y_h), "yellow_a": _to_int(y_a),
        "red_h": _to_int(r_h), "red_a": _to_int(r_a),
        "xg_live_h": _to_float(xg_h), "xg_live_a": _to_float(xg_a),
    }


def parse_event_teams(ev: dict) -> tuple[Optional[str], Optional[str]]:
    home_name = (ev.get("homeTeam") or {}).get("name")
    away_name = (ev.get("awayTeam") or {}).get("name")
    return NAME_TO_CODE.get(home_name), NAME_TO_CODE.get(away_name)


def find_event_for(home_code: str, away_code: str) -> Optional[dict]:
    """Encuentra el evento Sofascore correspondiente a un partido de nuestra DB."""
    for ev in list_all_live():
        h, a = parse_event_teams(ev)
        if h == home_code and a == away_code:
            return ev
    return None
