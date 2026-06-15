"""
Scraper de resultados Mundial 2026 desde Wikipedia.

Estrategia robusta:
  1. Intenta fetch a la página Wikipedia del torneo.
  2. Parsea tablas con BeautifulSoup buscando filas con marcador "X – Y".
  3. Si falla la red, devuelve lista vacía y la DB conserva sus seeds.

El parser es defensivo: cualquier excepción se reporta pero NO rompe la app.
"""
from __future__ import annotations
import re
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .db import get_session, Match, Team

log = logging.getLogger("scraper")

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
    "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_group_stage",
]

# Diccionario nombre Wikipedia -> código en nuestra DB
NAME_TO_CODE = {
    "Argentina": "ARG", "Brazil": "BRA", "Uruguay": "URU", "Colombia": "COL",
    "Ecuador": "ECU", "Paraguay": "PAR", "Bolivia": "BOL", "Chile": "CHI",
    "Peru": "PER", "Venezuela": "VEN",
    "France": "FRA", "Spain": "ESP", "England": "ENG", "Portugal": "POR",
    "Netherlands": "NED", "Germany": "GER", "Italy": "ITA", "Belgium": "BEL",
    "Croatia": "CRO", "Switzerland": "SUI", "Denmark": "DEN", "Austria": "AUT",
    "Poland": "POL", "Serbia": "SRB", "Turkey": "TUR", "Norway": "NOR",
    "Mexico": "MEX", "United States": "USA", "Canada": "CAN",
    "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM", "Haiti": "HAI",
    "Japan": "JPN", "Iran": "IRN", "South Korea": "KOR", "Australia": "AUS",
    "Saudi Arabia": "KSA", "Qatar": "QAT", "Uzbekistan": "UZB", "Jordan": "JOR",
    "Morocco": "MAR", "Senegal": "SEN", "Egypt": "EGY", "Algeria": "ALG",
    "Ivory Coast": "CIV", "Nigeria": "NGA", "Cameroon": "CMR", "Tunisia": "TUN",
    "Ghana": "GHA", "New Zealand": "NZL",
}


def _fetch(url: str, timeout: int = 8) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": "wc26-predictor/1.0"}, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        log.warning("Fetch %s falló: %s", url, e)
    return None


SCORE_RE = re.compile(r"^\s*(\d+)\s*[–\-]\s*(\d+)\s*$")


def _parse_team(td) -> Optional[str]:
    """Extrae código de equipo de una celda Wikipedia."""
    text = td.get_text(" ", strip=True)
    # Quitar marcadores tipo "(H)" para host, etc.
    text = re.sub(r"\([A-Z]\)", "", text).strip()
    return NAME_TO_CODE.get(text)


def parse_matches_from_html(html: str) -> list[dict]:
    """Devuelve una lista de dicts {home, away, home_goals, away_goals, date}."""
    soup = BeautifulSoup(html, "lxml")
    results = []
    # Heurística: filas de tablas con un span/td "score" tipo "2–1"
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        # Busca celda con marcador
        for i, c in enumerate(cells):
            m = SCORE_RE.match(c.get_text(strip=True))
            if not m:
                continue
            if i == 0 or i == len(cells) - 1:
                continue
            home = _parse_team(cells[i - 1])
            away = _parse_team(cells[i + 1])
            if not home or not away:
                continue
            results.append({
                "home": home,
                "away": away,
                "home_goals": int(m.group(1)),
                "away_goals": int(m.group(2)),
            })
            break
    return results


def refresh_from_wikipedia() -> dict:
    """Hace fetch + parse de Wikipedia + actualiza DB. Devuelve resumen."""
    updated = 0
    inserted = 0
    parsed = 0
    for url in WIKI_URLS:
        html = _fetch(url)
        if not html:
            continue
        for row in parse_matches_from_html(html):
            parsed += 1
            s = get_session()
            try:
                m = (
                    s.query(Match)
                    .filter(Match.competition == "WC2026",
                            Match.home == row["home"],
                            Match.away == row["away"])
                    .first()
                )
                if m:
                    if m.home_goals != row["home_goals"] or m.away_goals != row["away_goals"]:
                        m.home_goals = row["home_goals"]
                        m.away_goals = row["away_goals"]
                        m.finished = True
                        s.commit()
                        updated += 1
            finally:
                s.close()
    return {"parsed": parsed, "updated": updated, "inserted": inserted,
            "ok": parsed > 0, "source": "wikipedia"}


def refresh_all_sources() -> dict:
    """Refresca datos desde TODAS las fuentes (Wikipedia + eloratings + FBref).
    Cada fuente se intenta independientemente; si una falla, sigue con la otra.
    Resultado: dict con reporte por fuente."""
    from .sources.eloratings import fetch_world_ratings
    from .sources.fbref import fetch_team_xg

    report = {
        "wikipedia": {"ok": False},
        "eloratings": {"ok": False, "updated_teams": 0},
        "fbref": {"ok": False, "updated_teams": 0},
        "total_results_updated": 0,
    }

    # 1) Wikipedia → resultados de partidos
    try:
        report["wikipedia"] = refresh_from_wikipedia()
        report["total_results_updated"] += report["wikipedia"].get("updated", 0)
    except Exception as e:
        report["wikipedia"]["error"] = str(e)

    # 2) eloratings.net → Elo oficial
    try:
        ratings = fetch_world_ratings()
        if ratings:
            s = get_session()
            n_updated = 0
            try:
                for code, elo in ratings.items():
                    t = s.get(Team, code)
                    if t and abs(t.elo - elo) > 1:
                        t.elo = elo
                        n_updated += 1
                s.commit()
            finally:
                s.close()
            report["eloratings"] = {"ok": True, "updated_teams": n_updated,
                                    "total_ratings": len(ratings)}
    except Exception as e:
        report["eloratings"]["error"] = str(e)

    # 3) FBref → xG oficial por equipo
    try:
        xg = fetch_team_xg()
        if xg:
            s = get_session()
            n_updated = 0
            try:
                for code, vals in xg.items():
                    t = s.get(Team, code)
                    if not t:
                        continue
                    if vals.get("xg_for") and vals["xg_for"] > 0:
                        t.xg_for = vals["xg_for"]
                        n_updated += 1
                    if vals.get("xg_against") and vals["xg_against"] > 0:
                        t.xg_against = vals["xg_against"]
                s.commit()
            finally:
                s.close()
            report["fbref"] = {"ok": True, "updated_teams": n_updated,
                               "total_teams": len(xg)}
    except Exception as e:
        report["fbref"]["error"] = str(e)

    report["any_changes"] = (
        report["total_results_updated"] > 0
        or report["eloratings"].get("updated_teams", 0) > 0
        or report["fbref"].get("updated_teams", 0) > 0
    )
    return report
