"""
Adapter de football-data.org v4 API.

Docs: https://www.football-data.org/documentation/api

Endpoints relevantes:
  GET /v4/competitions/WC/matches?status=IN_PLAY
  GET /v4/competitions/WC/matches?status=SCHEDULED
  GET /v4/matches/{id}                  (incluye head2head + statistics)

Plan gratis: 10 requests/min, 100/day. Por eso el poller es cada 60s.

Headers requeridos: X-Auth-Token: <token>
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional

import requests

from config import get as cfg

log = logging.getLogger("source.football_data")

BASE = "https://api.football-data.org/v4"
COMPETITION = "WC"  # FIFA World Cup

# Mapeo de nombres oficiales football-data → códigos DB.
# Incluye TODOS los equipos del Mundial 2026 real (48 selecciones).
NAME_TO_CODE = {
    # CONMEBOL
    "Argentina": "ARG", "Brazil": "BRA", "Uruguay": "URU", "Colombia": "COL",
    "Ecuador": "ECU", "Paraguay": "PAR", "Bolivia": "BOL", "Chile": "CHI",
    "Peru": "PER", "Venezuela": "VEN",
    # UEFA
    "France": "FRA", "Spain": "ESP", "England": "ENG", "Portugal": "POR",
    "Netherlands": "NED", "Germany": "GER", "Italy": "ITA", "Belgium": "BEL",
    "Croatia": "CRO", "Switzerland": "SUI", "Denmark": "DEN", "Austria": "AUT",
    "Poland": "POL", "Serbia": "SRB", "Türkiye": "TUR", "Turkey": "TUR",
    "Norway": "NOR", "Sweden": "SWE", "Czechia": "CZE", "Czech Republic": "CZE",
    "Bosnia-Herzegovina": "BIH", "Bosnia and Herzegovina": "BIH",
    "Ukraine": "UKR", "Hungary": "HUN", "Slovakia": "SVK", "Romania": "ROU",
    # CONCACAF (anfitriones + clasificados)
    "Mexico": "MEX", "United States": "USA", "USA": "USA", "Canada": "CAN",
    "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM", "Haiti": "HAI",
    "Honduras": "HON", "Curaçao": "CUW",
    # AFC
    "Japan": "JPN", "Iran": "IRN", "IR Iran": "IRN",
    "Korea Republic": "KOR", "South Korea": "KOR",
    "Australia": "AUS", "Saudi Arabia": "KSA", "Qatar": "QAT",
    "Uzbekistan": "UZB", "Jordan": "JOR", "Iraq": "IRQ",
    # CAF
    "Morocco": "MAR", "Senegal": "SEN", "Egypt": "EGY", "Algeria": "ALG",
    "Côte d'Ivoire": "CIV", "Ivory Coast": "CIV", "Nigeria": "NGA",
    "Cameroon": "CMR", "Tunisia": "TUN", "Ghana": "GHA",
    "South Africa": "RSA", "Cape Verde": "CPV", "Cape Verde Islands": "CPV",
    "DR Congo": "COD", "Democratic Republic of the Congo": "COD",
    # OFC
    "New Zealand": "NZL",
}


class FootballDataAPI:
    def __init__(self, token: Optional[str] = None):
        self.token = token or cfg("football_data_token")
        self.headers = {"X-Auth-Token": self.token} if self.token else {}

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        if not self.enabled:
            return None
        url = f"{BASE}{path}"
        try:
            r = requests.get(url, headers=self.headers, params=params or {},
                             timeout=10)
            if r.status_code == 429:
                log.warning("football-data rate limit hit")
                return None
            if r.status_code != 200:
                log.info("football-data %s -> %d", path, r.status_code)
                return None
            return r.json()
        except Exception as e:
            log.warning("football-data %s falló: %s", path, e)
            return None

    # ---------- Matches ----------
    def list_matches(self, status: Optional[str] = None) -> list[dict]:
        """status puede ser SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED, TIMED, AWARDED."""
        params = {"status": status} if status else {}
        data = self._get(f"/competitions/{COMPETITION}/matches", params)
        if not data:
            return []
        return data.get("matches", [])

    def live_matches(self) -> list[dict]:
        """Devuelve partidos en LIVE / IN_PLAY / PAUSED."""
        out = []
        for st in ("LIVE", "IN_PLAY", "PAUSED"):
            out.extend(self.list_matches(status=st))
        # De-dup por id
        seen = set()
        unique = []
        for m in out:
            if m.get("id") not in seen:
                seen.add(m.get("id"))
                unique.append(m)
        return unique

    def match_detail(self, match_id_external: int) -> Optional[dict]:
        return self._get(f"/matches/{match_id_external}")

    # ---------- Parsers ----------
    @staticmethod
    def parse_team_code(team_obj: dict) -> Optional[str]:
        if not team_obj:
            return None
        name = team_obj.get("name") or team_obj.get("shortName")
        tla = team_obj.get("tla")  # three-letter abbreviation oficial FIFA
        return NAME_TO_CODE.get(name) or tla

    @classmethod
    def normalize_match(cls, raw: dict) -> dict:
        """Convierte la respuesta API a dict plano usado por nuestra DB."""
        home = cls.parse_team_code(raw.get("homeTeam", {}))
        away = cls.parse_team_code(raw.get("awayTeam", {}))
        score = raw.get("score", {})
        full = score.get("fullTime", {})
        # En partidos en curso ft.home puede ser None; usamos halfTime + minute
        return {
            "external_id": raw.get("id"),
            "home": home,
            "away": away,
            "home_goals": full.get("home") if score.get("winner") else None,
            "away_goals": full.get("away") if score.get("winner") else None,
            "status": raw.get("status"),
            "minute": raw.get("minute"),
            "stage": raw.get("stage"),
            "group": raw.get("group", "").replace("GROUP_", "") if raw.get("group") else None,
            "matchday": raw.get("matchday"),
            "datetime": raw.get("utcDate"),
            "current_home_goals": score.get("halfTime", {}).get("home", 0) if raw.get("status") == "PAUSED" else None,
            "current_away_goals": score.get("halfTime", {}).get("away", 0) if raw.get("status") == "PAUSED" else None,
            # full live snapshot
            "live": {
                "minute": raw.get("minute"),
                "home_goals": (full.get("home") if full.get("home") is not None
                               else score.get("halfTime", {}).get("home", 0)),
                "away_goals": (full.get("away") if full.get("away") is not None
                               else score.get("halfTime", {}).get("away", 0)),
            },
        }


# Singleton
_singleton: Optional[FootballDataAPI] = None


def api() -> FootballDataAPI:
    global _singleton
    if _singleton is None:
        _singleton = FootballDataAPI()
    return _singleton
