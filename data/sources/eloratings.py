"""
Scraper de eloratings.net — el rating Elo oficial de selecciones del mundo.

Esta es la fuente que cita FIFA y los analistas. Estructura del sitio:
  https://www.eloratings.net/        → ranking mundial
  https://www.eloratings.net/<country_code>  → histórico de un país

El sitio carga datos via JavaScript en una variable global `ratingstable`,
por lo que el HTML estático no la contiene directamente. Probamos dos rutas:

  A) Endpoint JSON oficial (si lo expone): https://www.eloratings.net/World.tsv
  B) Página principal con regex sobre el JS embebido.

Si no respondieran, dejamos los Elos sembrados en la DB intactos.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

import requests

log = logging.getLogger("source.eloratings")

# Mapeo del código eloratings.net → nuestro código de DB
ELO_TO_DB = {
    "ARG": "ARG", "BRA": "BRA", "URU": "URU", "COL": "COL", "ECU": "ECU",
    "PAR": "PAR", "BOL": "BOL", "CHI": "CHI", "PER": "PER", "VEN": "VEN",
    "FRA": "FRA", "ESP": "ESP", "ENG": "ENG", "POR": "POR", "NED": "NED",
    "GER": "GER", "ITA": "ITA", "BEL": "BEL", "CRO": "CRO", "SUI": "SUI",
    "DEN": "DEN", "AUT": "AUT", "POL": "POL", "SRB": "SRB", "TUR": "TUR",
    "NOR": "NOR", "WAL": "WAL", "SCO": "SCO",
    "MEX": "MEX", "USA": "USA", "CAN": "CAN", "CRC": "CRC", "PAN": "PAN",
    "JAM": "JAM", "HAI": "HAI",
    "JPN": "JPN", "IRN": "IRN", "KOR": "KOR", "AUS": "AUS", "KSA": "KSA",
    "QAT": "QAT", "UZB": "UZB", "JOR": "JOR",
    "MAR": "MAR", "SEN": "SEN", "EGY": "EGY", "ALG": "ALG", "CIV": "CIV",
    "NGA": "NGA", "CMR": "CMR", "TUN": "TUN", "GHA": "GHA",
    "NZL": "NZL",
}

URL_TSV = "https://www.eloratings.net/World.tsv"
URL_MAIN = "https://www.eloratings.net/"


def fetch_world_ratings(timeout: int = 10) -> Optional[dict[str, float]]:
    """Devuelve {team_code: elo} o None si no se pudo obtener."""
    # Intento A: TSV directo
    try:
        r = requests.get(URL_TSV, headers={"User-Agent": "wc26-predictor/1.0"},
                         timeout=timeout)
        if r.status_code == 200 and r.text:
            return _parse_tsv(r.text)
    except Exception as e:
        log.info("TSV eloratings falló: %s", e)
    # Intento B: HTML principal con regex
    try:
        r = requests.get(URL_MAIN, headers={"User-Agent": "wc26-predictor/1.0"},
                         timeout=timeout)
        if r.status_code == 200:
            return _parse_html(r.text)
    except Exception as e:
        log.info("HTML eloratings falló: %s", e)
    return None


def _parse_tsv(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        code = parts[0].strip().upper()
        try:
            rating = float(parts[1])
        except ValueError:
            continue
        db_code = ELO_TO_DB.get(code, code)
        out[db_code] = rating
    return out


def _parse_html(html: str) -> dict[str, float]:
    """Busca la asignación de la variable ratingstable en el JS."""
    m = re.search(r"ratingstable\s*=\s*(\[\[.*?\]\]);", html, re.DOTALL)
    if not m:
        return {}
    # No es JSON estricto (puede tener comillas simples); intentamos eval-safe
    blob = m.group(1).replace("'", '"')
    try:
        import json
        rows = json.loads(blob)
    except Exception:
        return {}
    out = {}
    for row in rows:
        if len(row) < 3:
            continue
        code = str(row[1]).upper()
        try:
            rating = float(row[2])
        except (ValueError, TypeError):
            continue
        db_code = ELO_TO_DB.get(code, code)
        out[db_code] = rating
    return out
