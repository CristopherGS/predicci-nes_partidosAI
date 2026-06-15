"""
Scraper de FBref.com (Sports Reference) — datos avanzados con xG (Expected Goals)
proveniente de StatsBomb.

URLs útiles:
  https://fbref.com/en/comps/1/2026/2026-FIFA-World-Cup-Stats        (WC 2026)
  https://fbref.com/en/squads/<id>/<TeamName>-Stats                  (team stats)
  https://fbref.com/en/comps/1/schedule/2026-FIFA-World-Cup-Scores-and-Fixtures

Por la complejidad del scraping (tablas dentro de comments HTML), acá
implementamos solo el endpoint de stats por equipo (xG for/against por partido)
y devolvemos lo que se pueda parsear con un grep básico.

Si FBref bloquea o no responde, devolvemos {} y los valores xG semilla quedan.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup, Comment

log = logging.getLogger("source.fbref")

WC2026_URL = "https://fbref.com/en/comps/1/2026/2026-FIFA-World-Cup-Stats"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                   "AppleWebKit/605.1 Safari/605.1 wc26-predictor/1.0"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Mapeo de nombres FBref → códigos DB
NAME_TO_CODE = {
    "Argentina": "ARG", "Brazil": "BRA", "Uruguay": "URU", "Colombia": "COL",
    "Ecuador": "ECU", "Paraguay": "PAR",
    "France": "FRA", "Spain": "ESP", "England": "ENG", "Portugal": "POR",
    "Netherlands": "NED", "Germany": "GER", "Italy": "ITA", "Belgium": "BEL",
    "Croatia": "CRO", "Switzerland": "SUI", "Denmark": "DEN", "Austria": "AUT",
    "Poland": "POL", "Serbia": "SRB", "Turkey": "TUR", "Norway": "NOR",
    "Mexico": "MEX", "United States": "USA", "Canada": "CAN",
    "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM",
    "Japan": "JPN", "Iran": "IRN", "South Korea": "KOR", "Korea Republic": "KOR",
    "Australia": "AUS", "Saudi Arabia": "KSA", "Qatar": "QAT",
    "Uzbekistan": "UZB", "Jordan": "JOR",
    "Morocco": "MAR", "Senegal": "SEN", "Egypt": "EGY", "Algeria": "ALG",
    "Côte d'Ivoire": "CIV", "Ivory Coast": "CIV", "Nigeria": "NGA",
    "Cameroon": "CMR", "Tunisia": "TUN", "Ghana": "GHA",
    "New Zealand": "NZL", "Haiti": "HAI", "Bolivia": "BOL",
}


def fetch_team_xg(timeout: int = 12) -> Optional[dict[str, dict[str, float]]]:
    """Devuelve {team_code: {"xg_for": float, "xg_against": float}} aproximado
    desde la tabla de stats del WC2026.

    FBref oculta varias tablas dentro de comentarios HTML para evitar scrapers
    ingenuos; las recuperamos extrayendo los Comment del soup.
    """
    try:
        r = requests.get(WC2026_URL, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            log.info("FBref status %d", r.status_code)
            return None
    except Exception as e:
        log.info("FBref fetch falló: %s", e)
        return None
    soup = BeautifulSoup(r.text, "lxml")
    # Algunas tablas están en comentarios
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    for c in comments:
        if "stats_squads" in c or "Squad Standard Stats" in c:
            inner = BeautifulSoup(c, "lxml")
            return _parse_team_table(inner)
    # Si no estaban en comments, intentamos directo
    return _parse_team_table(soup)


def _parse_team_table(soup) -> dict[str, dict[str, float]]:
    out = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        # Equipo en la primera celda con role "row" generalmente
        name = cells[0].get_text(strip=True)
        if not name or name not in NAME_TO_CODE:
            continue
        code = NAME_TO_CODE[name]
        xg_for = None
        xg_against = None
        for c in cells:
            attr = c.get("data-stat", "")
            if attr in ("xg_for", "xg"):
                try:
                    xg_for = float(c.get_text(strip=True))
                except ValueError:
                    pass
            elif attr in ("xg_against", "xga"):
                try:
                    xg_against = float(c.get_text(strip=True))
                except ValueError:
                    pass
        if xg_for is not None or xg_against is not None:
            # FBref reporta xG total; lo dividimos por partidos jugados si pudimos
            mp = 1
            for c in cells:
                if c.get("data-stat") in ("games", "minutes_90s"):
                    try:
                        mp = max(1, float(c.get_text(strip=True)))
                        break
                    except ValueError:
                        pass
            out[code] = {
                "xg_for": (xg_for / mp) if xg_for else None,
                "xg_against": (xg_against / mp) if xg_against else None,
            }
    return out
