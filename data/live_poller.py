"""
Poller asíncrono multi-fuente para partidos en vivo.

Estrategia:
  1) ESPN como fuente PRIMARIA — gratis, sin token, devuelve TODAS las stats
     (posesión, tiros, corners, tarjetas).
  2) football-data.org como fuente secundaria — confirma marcador y minuto
     (el plan free no devuelve statistics, solo el básico).
  3) Si ESPN matchea un partido pero football-data no, igual lo procesamos.

Para cada partido en vivo:
  - Resolvemos el Match de nuestra DB por (home, away).
  - Persistimos snapshot en LiveStats.
  - Corremos el predictor in-play (Dixon-Robinson 1998).
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from data.db import get_session, Match, LiveStats
from data.sources import espn as espn_src
from data.sources.football_data_org import api as fd_api
from models.predictor import get_predictor
from models.live_predictor import predict_inplay
from config import get as cfg

log = logging.getLogger("live_poller")

STATE = {
    "running": False,
    "last_tick": None,
    "live_count": 0,
    "last_error": None,
    "sources_used": [],
    "api_enabled": False,  # ahora indica si CUALQUIER fuente está activa
}

_stop = False


def _resolve_match(home: str, away: str) -> Optional[Match]:
    if not home or not away:
        return None
    s = get_session()
    try:
        return (s.query(Match)
                .filter(Match.competition == "WC2026",
                        Match.home == home,
                        Match.away == away)
                .order_by(Match.datetime_utc.desc())
                .first())
    finally:
        s.close()


def _persist_snapshot(match_db: Match, minute: int, status: str,
                      home_goals: int, away_goals: int,
                      stats: dict) -> Optional[LiveStats]:
    """Persiste snapshot + corre predicción in-play."""
    # Predicción in-play
    pred = get_predictor()
    sess = get_session()
    try:
        pre = pred.predict(sess, match_db) if match_db.home else None
    finally:
        sess.close()
    lam_pre = (pre or {}).get("lam_h", 1.3) if pre and "error" not in pre else 1.3
    mu_pre = (pre or {}).get("lam_a", 1.0) if pre and "error" not in pre else 1.0

    live_pred = predict_inplay(
        lam_pre, mu_pre, minute, home_goals, away_goals,
        shots_h=stats.get("shots_h", 0) or 0,
        shots_a=stats.get("shots_a", 0) or 0,
        shots_op_h=stats.get("shots_a", 0) or 0,
        shots_op_a=stats.get("shots_h", 0) or 0,
        xg_live_h=stats.get("xg_live_h"),
        xg_live_a=stats.get("xg_live_a"),
        red_h=stats.get("red_h", 0) or 0,
        red_a=stats.get("red_a", 0) or 0,
    )

    s = get_session()
    try:
        snap = LiveStats(
            match_id=match_db.id,
            updated_at=datetime.now(timezone.utc),
            minute=minute, status=status,
            home_goals_live=home_goals, away_goals_live=away_goals,
            possession_h=stats.get("possession_h"),
            possession_a=stats.get("possession_a"),
            shots_h=stats.get("shots_h", 0), shots_a=stats.get("shots_a", 0),
            shots_on_target_h=stats.get("shots_on_target_h", 0),
            shots_on_target_a=stats.get("shots_on_target_a", 0),
            corners_h=stats.get("corners_h", 0), corners_a=stats.get("corners_a", 0),
            fouls_h=stats.get("fouls_h", 0), fouls_a=stats.get("fouls_a", 0),
            yellow_h=stats.get("yellow_h", 0), yellow_a=stats.get("yellow_a", 0),
            red_h=stats.get("red_h", 0), red_a=stats.get("red_a", 0),
            xg_live_h=stats.get("xg_live_h"), xg_live_a=stats.get("xg_live_a"),
            p_home_live=live_pred["p_home"],
            p_draw_live=live_pred["p_draw"],
            p_away_live=live_pred["p_away"],
            expected_final_home=live_pred["expected_final_home"],
            expected_final_away=live_pred["expected_final_away"],
        )
        s.add(snap)
        s.commit()
        return snap
    finally:
        s.close()


def _tick_espn() -> tuple[int, list[str]]:
    """Devuelve (cantidad de partidos procesados, lista de identificadores)."""
    processed = []
    try:
        events = espn_src.list_live_events()
        for ev in events:
            home, away = espn_src.parse_event_teams(ev)
            if not home or not away:
                continue
            match_db = _resolve_match(home, away)
            if not match_db:
                log.info("ESPN: no DB match for %s vs %s", home, away)
                continue
            hg, ag = espn_src.parse_event_score(ev)
            minute = espn_src.parse_event_minute(ev)
            status = (ev.get("status") or {}).get("type", {}).get("name", "IN_PLAY")
            # Detalle con stats
            sm = espn_src.summary(str(ev["id"]))
            stats = espn_src.extract_stats(sm) or {}
            _persist_snapshot(match_db, minute, status, hg, ag, stats)
            processed.append(f"{home}-{away}")
    except Exception as e:
        log.exception("ESPN tick falló: %s", e)
    return len(processed), processed


def _tick_football_data() -> tuple[int, list[str]]:
    """Fallback: football-data sin stats detalladas, solo marcador."""
    api = fd_api()
    processed = []
    if not api.enabled:
        return 0, []
    try:
        for raw in api.live_matches():
            normalized = api.normalize_match(raw)
            home = normalized.get("home")
            away = normalized.get("away")
            if not home or not away:
                continue
            match_db = _resolve_match(home, away)
            if not match_db:
                continue
            minute = normalized.get("minute") or 0
            status = raw.get("status", "IN_PLAY")
            live = normalized.get("live", {})
            hg = live.get("home_goals", 0) or 0
            ag = live.get("away_goals", 0) or 0
            _persist_snapshot(match_db, minute, status, hg, ag, {})
            processed.append(f"{home}-{away}")
    except Exception as e:
        log.exception("football-data tick falló: %s", e)
    return len(processed), processed


def tick_once():
    """Una iteración: ESPN primero (stats completas), football-data como complemento."""
    used = []
    error = None
    try:
        n_espn, names_espn = _tick_espn()
        if n_espn > 0:
            used.append(f"ESPN ({n_espn})")
        # ESPN debería bastar; pero si no encontró nada, probamos football-data
        # (útil para partidos del Mundial que ESPN aún no muestra en LIVE)
        if n_espn == 0:
            n_fd, _ = _tick_football_data()
            if n_fd > 0:
                used.append(f"football-data ({n_fd})")
        STATE["live_count"] = n_espn  # ESPN ya es la fuente de verdad
    except Exception as e:
        log.exception("tick_once falló: %s", e)
        error = str(e)
    STATE["sources_used"] = used
    STATE["last_tick"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    STATE["last_error"] = error
    STATE["api_enabled"] = bool(used) or fd_api().enabled


async def _loop():
    global _stop
    interval = cfg("live_poll_interval_seconds", 60)
    log.info("Live poller iniciado (cada %ds, ESPN+football-data)", interval)
    await asyncio.sleep(8)
    while not _stop:
        STATE["running"] = True
        try:
            tick_once()
        except Exception as e:
            log.exception("live poller error: %s", e)
        STATE["running"] = False
        for _ in range(interval // 5):
            if _stop:
                break
            await asyncio.sleep(5)


def start():
    global _stop
    _stop = False
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_loop())
    except RuntimeError:
        log.warning("Sin event loop; live poller no arrancado")


def stop():
    global _stop
    _stop = True
