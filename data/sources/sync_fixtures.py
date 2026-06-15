"""
Sincroniza fixtures reales del Mundial 2026 desde football-data.org y los
inserta/actualiza en nuestra DB.

Estrategia: football-data.org es la fuente de verdad. Si un Match existe con
(home, away) iguales, se actualiza. Si no existe, se crea.

Esto reemplaza el seed plausible con la realidad oficial.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional

from data.db import get_session, Match, Team
from data.sources.football_data_org import api as fd_api

log = logging.getLogger("sync_fixtures")


STAGE_MAP = {
    "GROUP_STAGE": "GROUP",
    "LAST_16": "R16",
    "QUARTER_FINALS": "QF",
    "SEMI_FINALS": "SF",
    "THIRD_PLACE": "THIRD",
    "FINAL": "FINAL",
    "PRELIMINARY_ROUND": "R32",
    "LAST_32": "R32",
}


def _ensure_team(session, code: str):
    """Crea un Team minimal si no existe."""
    if not code:
        return
    if session.get(Team, code) is None:
        session.add(Team(
            code=code, name=code, confederation="?",
            elo=1500.0, xg_for=1.2, xg_against=1.4,
        ))


def sync_all(cleanup_orphans: bool = True) -> dict:
    """Trae todos los partidos del Mundial 2026 y los sincroniza.

    Args:
      cleanup_orphans: si True, elimina partidos WC2026 que están en la DB pero
        NO existen en la fuente oficial (limpia el seed plausible viejo).
    """
    api = fd_api()
    if not api.enabled:
        return {"ok": False, "error": "Sin token configurado"}

    report = {"inserted": 0, "updated": 0, "results_updated": 0,
              "skipped_no_codes": 0, "total_fetched": 0, "deleted_orphans": 0}

    # Pedimos en bloques por status para no perder ninguno
    all_matches = []
    for status in (None, "SCHEDULED", "IN_PLAY", "PAUSED", "FINISHED", "TIMED"):
        if status:
            matches = api.list_matches(status=status)
        else:
            matches = api.list_matches()
        # De-dup por external id
        ids_seen = {m["external_id"] if "external_id" in m else m.get("id")
                    for m in all_matches}
        for m in matches:
            if m.get("id") not in ids_seen:
                all_matches.append(m)
        if not status:
            break  # El primer call sin status trae todo

    report["total_fetched"] = len(all_matches)
    log.info("Fetched %d fixtures desde football-data.org", len(all_matches))

    s = get_session()
    try:
        # IDs de match tocados durante este sync (matched or inserted)
        touched_ids: set[int] = set()
        for raw in all_matches:
            normalized = api.normalize_match(raw)
            home = normalized.get("home")
            away = normalized.get("away")
            if not home or not away:
                # Knockout TBD
                report["skipped_no_codes"] += 1
                continue
            _ensure_team(s, home)
            _ensure_team(s, away)
            # Buscar match existente por (home, away) en WC2026
            existing = (s.query(Match)
                        .filter(Match.competition == "WC2026",
                                Match.home == home,
                                Match.away == away)
                        .first())
            dt_str = normalized.get("datetime")
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")) if dt_str else None
            if dt and dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            stage = STAGE_MAP.get(normalized.get("stage"), normalized.get("stage", "GROUP"))
            group = normalized.get("group")
            matchday = normalized.get("matchday")
            status = raw.get("status")
            hg = normalized.get("home_goals")
            ag = normalized.get("away_goals")
            finished = status in ("FINISHED", "AWARDED")

            if existing:
                touched_ids.add(existing.id)
                changed = False
                if dt and existing.datetime_utc != dt:
                    existing.datetime_utc = dt; changed = True
                if stage and existing.stage != stage:
                    existing.stage = stage; changed = True
                if group and existing.group != group:
                    existing.group = group; changed = True
                if matchday and existing.matchday != matchday:
                    existing.matchday = matchday; changed = True
                if hg is not None and existing.home_goals != hg:
                    existing.home_goals = hg; existing.finished = finished
                    report["results_updated"] += 1; changed = True
                if ag is not None and existing.away_goals != ag:
                    existing.away_goals = ag; existing.finished = finished
                    changed = True
                if finished != existing.finished:
                    existing.finished = finished; changed = True
                if changed:
                    report["updated"] += 1
            else:
                new_m = Match(
                    competition="WC2026",
                    stage=stage,
                    group=group,
                    matchday=matchday,
                    datetime_utc=dt or datetime.utcnow(),
                    home=home, away=away,
                    home_goals=hg, away_goals=ag,
                    neutral=True,  # En Mundial todo es neutral salvo anfitriones
                    finished=finished,
                )
                s.add(new_m)
                s.flush()  # asigna id
                touched_ids.add(new_m.id)
                report["inserted"] += 1

        # Limpieza de fixtures huérfanos (los del seed plausible que ya no aplican)
        if cleanup_orphans and touched_ids:
            from data.db import Prediction, LiveStats
            orphans = (s.query(Match)
                       .filter(Match.competition == "WC2026",
                               ~Match.id.in_(touched_ids),
                               Match.home_goals.is_(None))
                       .all())
            orphan_ids = [m.id for m in orphans]
            if orphan_ids:
                # Borrar predictions y live_stats huérfanos antes
                (s.query(Prediction)
                 .filter(Prediction.match_id.in_(orphan_ids))
                 .delete(synchronize_session=False))
                (s.query(LiveStats)
                 .filter(LiveStats.match_id.in_(orphan_ids))
                 .delete(synchronize_session=False))
                (s.query(Match)
                 .filter(Match.id.in_(orphan_ids))
                 .delete(synchronize_session=False))
                report["deleted_orphans"] = len(orphan_ids)
                log.info("Eliminados %d fixtures huérfanos del seed", len(orphan_ids))
        s.commit()
    finally:
        s.close()
    report["ok"] = True
    return report
