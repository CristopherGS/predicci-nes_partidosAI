"""
Source ESPN — API pública sin autenticación, sin Cloudflare bloqueo.

Endpoints:
  GET /apis/site/v2/sports/soccer/fifa.world/scoreboard
       → lista de partidos del Mundial actual (status, marcador, minuto)
  GET /apis/site/v2/sports/soccer/fifa.world/summary?event={id}
       → boxscore detallado: tiros, posesión, corners, tarjetas, pases

ESPN devuelve campos:
  totalShots, shotsOnTarget, possessionPct, wonCorners, foulsCommitted,
  yellowCards, redCards, blockedShots, offsides, accuratePasses, passPct,
  shotPct (% de tiros que terminan al arco — proxy de "calidad de chance")

Esta es la fuente PRIMARIA de stats live para nuestro sistema.
"""
from __future__ import annotations
import logging
from typing import Optional

import requests

log = logging.getLogger("source.espn")

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
LEAGUE = "fifa.world"
HEADERS = {
    "User-Agent": "Mozilla/5.0 wc26-predictor/1.0",
    "Accept": "application/json",
}

# Mapeo de displayName de ESPN a nuestros códigos
NAME_TO_CODE = {
    "Argentina": "ARG", "Brazil": "BRA", "Uruguay": "URU", "Colombia": "COL",
    "Ecuador": "ECU", "Paraguay": "PAR", "Bolivia": "BOL", "Chile": "CHI",
    "Peru": "PER", "Venezuela": "VEN",
    "France": "FRA", "Spain": "ESP", "England": "ENG", "Portugal": "POR",
    "Netherlands": "NED", "Germany": "GER", "Italy": "ITA", "Belgium": "BEL",
    "Croatia": "CRO", "Switzerland": "SUI", "Denmark": "DEN", "Austria": "AUT",
    "Poland": "POL", "Serbia": "SRB", "Turkey": "TUR", "Türkiye": "TUR",
    "Norway": "NOR", "Sweden": "SWE", "Czechia": "CZE", "Czech Republic": "CZE",
    "Bosnia and Herzegovina": "BIH", "Bosnia-Herzegovina": "BIH",
    "Ukraine": "UKR", "Hungary": "HUN", "Slovakia": "SVK",
    "Mexico": "MEX", "United States": "USA", "USMNT": "USA", "Canada": "CAN",
    "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM", "Haiti": "HAI",
    "Honduras": "HON", "Curaçao": "CUW",
    "Japan": "JPN", "Iran": "IRN", "Korea Republic": "KOR", "South Korea": "KOR",
    "Australia": "AUS", "Saudi Arabia": "KSA", "Qatar": "QAT",
    "Uzbekistan": "UZB", "Jordan": "JOR", "Iraq": "IRQ",
    "Morocco": "MAR", "Senegal": "SEN", "Egypt": "EGY", "Algeria": "ALG",
    "Ivory Coast": "CIV", "Côte d'Ivoire": "CIV", "Nigeria": "NGA",
    "Cameroon": "CMR", "Tunisia": "TUN", "Ghana": "GHA",
    "South Africa": "RSA", "Cape Verde": "CPV", "DR Congo": "COD",
    "New Zealand": "NZL",
}


def _fetch(path: str, params: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
    url = f"{BASE}/{LEAGUE}{path}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=timeout)
        if r.status_code != 200:
            log.info("espn %s -> %d", path, r.status_code)
            return None
        return r.json()
    except Exception as e:
        log.info("espn %s falló: %s", path, e)
        return None


def list_events() -> list[dict]:
    """Devuelve los eventos de fifa.world (scheduled + live + finished)."""
    data = _fetch("/scoreboard")
    if not data:
        return []
    return data.get("events", [])


def list_live_events() -> list[dict]:
    """Filtra a los partidos EN VIVO (no scheduled, no full-time)."""
    LIVE_STATES = {"STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_FIRST_HALF",
                   "STATUS_SECOND_HALF", "STATUS_EXTRA_TIME", "STATUS_PENALTIES",
                   "STATUS_END_PERIOD"}
    out = []
    for ev in list_events():
        st = (ev.get("status") or {}).get("type") or {}
        if st.get("name") in LIVE_STATES:
            out.append(ev)
    return out


def summary(event_id: str) -> Optional[dict]:
    return _fetch("/summary", params={"event": event_id})


def parse_event_teams(ev: dict) -> tuple[Optional[str], Optional[str]]:
    comps = (ev.get("competitions") or [{}])[0]
    cs = comps.get("competitors", [])
    home_name = next((c.get("team", {}).get("displayName")
                      for c in cs if c.get("homeAway") == "home"), None)
    away_name = next((c.get("team", {}).get("displayName")
                      for c in cs if c.get("homeAway") == "away"), None)
    return NAME_TO_CODE.get(home_name), NAME_TO_CODE.get(away_name)


def parse_event_score(ev: dict) -> tuple[int, int]:
    comps = (ev.get("competitions") or [{}])[0]
    cs = comps.get("competitors", [])
    h = next((int(c.get("score", 0)) for c in cs if c.get("homeAway") == "home"), 0)
    a = next((int(c.get("score", 0)) for c in cs if c.get("homeAway") == "away"), 0)
    return h, a


def parse_event_minute(ev: dict) -> int:
    st = (ev.get("status") or {})
    # ESPN devuelve "displayClock" tipo "45'" y "period"
    clock = st.get("displayClock", "0")
    try:
        return int(str(clock).rstrip("'").rstrip("+").split(":")[0])
    except (ValueError, AttributeError):
        period = st.get("period", 1)
        return 45 if period >= 2 else 0


def _stat_value(team_stats: list, key: str, default=0):
    for s in team_stats:
        if s.get("abbreviation") == key or s.get("name") == key:
            v = s.get("displayValue", default)
            if isinstance(v, str) and v.endswith("%"):
                v = v[:-1]
            try:
                return float(v) if "." in str(v) else int(v)
            except ValueError:
                return default
    return default


def extract_stats(summary_payload: dict) -> Optional[dict]:
    """Devuelve dict con las stats normalizadas al schema LiveStats."""
    if not summary_payload:
        return None
    bs = summary_payload.get("boxscore", {})
    teams = bs.get("teams", [])
    if len(teams) < 2:
        return None
    home_team = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
    away_team = next((t for t in teams if t.get("homeAway") == "away"), teams[1])
    hs = home_team.get("statistics", [])
    as_ = away_team.get("statistics", [])

    # ESPN no devuelve xG directamente, pero shotPct * totalShots es un proxy
    # razonable de "expected goals" (calidad de chances).
    def _xg_proxy(stats):
        ts = _stat_value(stats, "totalShots", 0)
        sp = _stat_value(stats, "shotPct", 0.0)
        return round(ts * (sp / 100 if sp > 1 else sp) * 1.2, 2)

    return {
        "possession_h": _stat_value(hs, "possessionPct"),
        "possession_a": _stat_value(as_, "possessionPct"),
        "shots_h": _stat_value(hs, "totalShots"),
        "shots_a": _stat_value(as_, "totalShots"),
        "shots_on_target_h": _stat_value(hs, "shotsOnTarget"),
        "shots_on_target_a": _stat_value(as_, "shotsOnTarget"),
        "corners_h": _stat_value(hs, "wonCorners"),
        "corners_a": _stat_value(as_, "wonCorners"),
        "fouls_h": _stat_value(hs, "foulsCommitted"),
        "fouls_a": _stat_value(as_, "foulsCommitted"),
        "yellow_h": _stat_value(hs, "yellowCards"),
        "yellow_a": _stat_value(as_, "yellowCards"),
        "red_h": _stat_value(hs, "redCards"),
        "red_a": _stat_value(as_, "redCards"),
        "xg_live_h": _xg_proxy(hs),
        "xg_live_a": _xg_proxy(as_),
    }
