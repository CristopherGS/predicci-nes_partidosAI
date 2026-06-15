"""
FastAPI backend para el predictor del Mundial 2026.

Endpoints:
  GET  /                    -> dashboard HTML
  GET  /api/teams           -> lista de selecciones con Elo
  GET  /api/matches         -> partidos (filtros: stage, group, status)
  GET  /api/match/{id}      -> detalle + predicción cacheada
  POST /api/predict/{id}    -> genera/actualiza predicción
  POST /api/predict_all     -> predice todos los partidos no jugados
  POST /api/refresh         -> scrape Wikipedia y actualiza resultados
  GET  /api/accuracy        -> métricas de aciertos del modelo en partidos jugados
  GET  /api/groups          -> standings por grupo
  GET  /api/model_info      -> meta del modelo entrenado
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import or_

from data.db import init_db, get_session, Team, Match, Prediction, LiveStats
from data.loader import bootstrap
from data.scraper import refresh_from_wikipedia, refresh_all_sources
from data.sources.sync_fixtures import sync_all as sync_real_fixtures
from data.live_poller import (
    start as start_poller, stop as stop_poller,
    tick_once as live_tick, STATE as LIVE_STATE,
)
from models.predictor import get_predictor, META_PATH
from models.training_loop import (
    start as start_trainer, stop as stop_trainer,
    trigger_train_async, mark_dirty, STATE as TRAIN_STATE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="WC2026 Predictor", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _on_startup():
    init_db()
    bootstrap()
    # Saneamiento de datos: desmarca como "finished" partidos sin goles válidos
    s = get_session()
    try:
        fixed = (s.query(Match)
                 .filter(Match.finished == True,            # noqa
                         (Match.home_goals.is_(None) | Match.away_goals.is_(None)))
                 .update({"finished": False}, synchronize_session=False))
        if fixed:
            s.commit()
            log.info("Saneados %d partidos marcados como finished sin goles", fixed)
    finally:
        s.close()
    # Arranca el loop de auto-entrenamiento (cada 4 horas)
    start_trainer(scheduled_every_s=4 * 3600)
    # Arranca el poller de partidos en vivo (cada 60s)
    start_poller()


@app.on_event("shutdown")
def _on_shutdown():
    stop_trainer()
    stop_poller()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


# ---------- TEAMS ----------
@app.get("/api/teams")
def list_teams():
    s = get_session()
    try:
        rows = s.query(Team).order_by(Team.elo.desc()).all()
        return [{
            "code": t.code, "name": t.name, "confederation": t.confederation,
            "elo": round(t.elo, 1), "xg_for": t.xg_for, "xg_against": t.xg_against,
        } for t in rows]
    finally:
        s.close()


# ---------- MATCHES ----------
def _serialize_match(m: Match, team_names: dict, pred: Optional[Prediction] = None):
    return {
        "id": m.id,
        "competition": m.competition,
        "stage": m.stage,
        "group": m.group,
        "matchday": m.matchday,
        "datetime": m.datetime_utc.isoformat() if m.datetime_utc else None,
        "home": m.home,
        "home_name": team_names.get(m.home, m.home),
        "away": m.away,
        "away_name": team_names.get(m.away, m.away),
        "home_goals": m.home_goals,
        "away_goals": m.away_goals,
        "finished": m.finished,
        "neutral": m.neutral,
        "prediction": {
            "p_home": pred.p_home, "p_draw": pred.p_draw, "p_away": pred.p_away,
            "pred_home_goals": pred.pred_home_goals,
            "pred_away_goals": pred.pred_away_goals,
            "over_25": pred.over_25, "btts": pred.btts,
            "model_version": pred.model_version,
            "confidence": pred.confidence,
            "created_at": pred.created_at.isoformat() if pred.created_at else None,
        } if pred else None,
    }


@app.get("/api/matches")
def list_matches(
    competition: str = "WC2026",
    stage: Optional[str] = None,
    group: Optional[str] = None,
    status: Optional[str] = Query(None, regex="^(played|upcoming|all|live)$"),
):
    """
    status:
      - upcoming: el partido NO terminó Y la fecha es a futuro. Ordenado asc.
      - played:   el partido terminó O la fecha ya pasó. Ordenado desc (más reciente arriba).
      - live:     en curso (terminó=False, fecha pasó, hace <3h del kickoff).
      - all:      todos. Ordenado asc.
    """
    from sqlalchemy import or_, and_
    from datetime import timedelta
    s = get_session()
    try:
        team_names = {t.code: t.name for t in s.query(Team).all()}
        now = datetime.utcnow()
        q = s.query(Match).filter(Match.competition == competition)
        if stage:
            q = q.filter(Match.stage == stage)
        if group:
            q = q.filter(Match.group == group)

        if status == "played":
            # Jugado: finished=True O fecha ya pasó (suficiente para sacar del upcoming)
            q = q.filter(or_(Match.finished == True,   # noqa
                             Match.datetime_utc < now))
            q = q.order_by(Match.datetime_utc.desc())
        elif status == "upcoming":
            # Próximo: no terminó Y la fecha está en el futuro
            q = q.filter(and_(Match.finished == False,  # noqa
                              Match.datetime_utc >= now))
            q = q.order_by(Match.datetime_utc.asc())
        elif status == "live":
            # En vivo: no terminó, kickoff entre now-3h y now
            window = now - timedelta(hours=3)
            q = q.filter(and_(Match.finished == False,  # noqa
                              Match.datetime_utc <= now,
                              Match.datetime_utc >= window))
            q = q.order_by(Match.datetime_utc.asc())
        else:
            q = q.order_by(Match.datetime_utc.asc())

        preds = {p.match_id: p for p in s.query(Prediction).all()}
        return [_serialize_match(m, team_names, preds.get(m.id)) for m in q.all()]
    finally:
        s.close()


@app.get("/api/match/{match_id}")
def get_match(match_id: int):
    s = get_session()
    try:
        m = s.get(Match, match_id)
        if not m:
            raise HTTPException(404, "Match no encontrado")
        team_names = {t.code: t.name for t in s.query(Team).all()}
        pred = (s.query(Prediction)
                .filter(Prediction.match_id == match_id)
                .order_by(Prediction.created_at.desc())
                .first())
        return _serialize_match(m, team_names, pred)
    finally:
        s.close()


# ---------- PREDICCIONES ----------
@app.post("/api/predict/{match_id}")
def predict_match(match_id: int):
    s = get_session()
    try:
        m = s.get(Match, match_id)
        if not m:
            raise HTTPException(404, "Match no encontrado")
        if not m.home or not m.away:
            raise HTTPException(400, "Partido sin equipos definidos (knockout pendiente)")
        out = get_predictor().predict_and_store(s, m)
        return out
    finally:
        s.close()


@app.post("/api/predict_all")
def predict_all():
    s = get_session()
    try:
        pred = get_predictor()
        to_do = (s.query(Match)
                 .filter(Match.competition == "WC2026",
                         Match.home.isnot(None),
                         Match.away.isnot(None))
                 .all())
        done = 0
        for m in to_do:
            try:
                pred.predict_and_store(s, m)
                done += 1
            except Exception as e:
                log.warning("predict %s falló: %s", m.id, e)
        return {"predicted": done, "total": len(to_do),
                "model_loaded": pred.has_model()}
    finally:
        s.close()


# ---------- REFRESH ----------
@app.post("/api/refresh")
def refresh():
    """Refresca desde TODAS las fuentes (football-data.org + Wikipedia +
    eloratings.net + FBref). Si trae cambios, dispara reentrenamiento."""
    try:
        out = refresh_all_sources()
        # Sincroniza fixtures reales del Mundial desde football-data.org
        sync = sync_real_fixtures()
        out["football_data"] = sync
        if sync.get("results_updated", 0) > 0 or sync.get("inserted", 0) > 0:
            out["any_changes"] = True
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if out.get("any_changes"):
        mark_dirty("post-refresh")
    out["ok"] = True
    return out


@app.post("/api/sync_fixtures")
def sync_fixtures_endpoint():
    """Reemplaza el seed por fixtures reales del Mundial 2026."""
    try:
        out = sync_real_fixtures()
        if out.get("ok") and (out.get("inserted") or out.get("updated")):
            mark_dirty("fixture-sync")
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/reset_wc_fixtures")
def reset_wc_fixtures():
    """Borra TODOS los partidos WC2026 (incluso los del seed con resultados
    ficticios) y vuelve a sincronizar desde football-data.org. Esto deja la DB
    con SOLO los partidos reales del Mundial 2026."""
    s = get_session()
    deleted_count = 0
    try:
        wc_ids = [m.id for m in s.query(Match)
                  .filter(Match.competition == "WC2026").all()]
        if wc_ids:
            (s.query(Prediction)
             .filter(Prediction.match_id.in_(wc_ids))
             .delete(synchronize_session=False))
            (s.query(LiveStats)
             .filter(LiveStats.match_id.in_(wc_ids))
             .delete(synchronize_session=False))
            deleted_count = (s.query(Match)
                             .filter(Match.id.in_(wc_ids))
                             .delete(synchronize_session=False))
            s.commit()
    finally:
        s.close()
    # Re-sync desde football-data
    sync_out = sync_real_fixtures()
    mark_dirty("hard-reset")
    return {"deleted_wc_matches": deleted_count, "resync": sync_out}


# ---------- AUTO-TRAINING ----------
@app.post("/api/train_now")
def train_now():
    """Dispara un entrenamiento inmediato (no bloquea)."""
    started = trigger_train_async("manual")
    return {"started": started, "state": TRAIN_STATE}


@app.get("/api/train_status")
def train_status():
    return TRAIN_STATE


# ---------- ACCURACY ----------
@app.get("/api/accuracy")
def accuracy():
    """Compara predicciones vs resultados reales en partidos del WC2026 ya jugados."""
    s = get_session()
    try:
        rows = (s.query(Match, Prediction)
                .join(Prediction, Prediction.match_id == Match.id)
                .filter(Match.competition == "WC2026",
                        Match.finished == True,                # noqa
                        Match.home_goals.isnot(None),
                        Match.away_goals.isnot(None),
                        Prediction.p_home.isnot(None),
                        Prediction.p_draw.isnot(None),
                        Prediction.p_away.isnot(None))
                .all())
        if not rows:
            return {"played_with_prediction": 0, "accuracy_1x2": 0,
                    "brier_score": 0, "details": []}
        details = []
        correct = 0
        brier_sum = 0.0
        for m, p in rows:
            # Defensa adicional: si por alguna razón hay nulls, skip
            if (
                m.home_goals is None
                or m.away_goals is None
                or p.p_home is None
                or p.p_draw is None
                or p.p_away is None
            ):
                continue
            real = ("home" if m.home_goals > m.away_goals
                    else ("away" if m.home_goals < m.away_goals else "draw"))
            pred = max(("home", p.p_home), ("draw", p.p_draw), ("away", p.p_away),
                       key=lambda x: x[1])[0]
            hit = pred == real
            if hit:
                correct += 1
            y = {"home": (1, 0, 0), "draw": (0, 1, 0), "away": (0, 0, 1)}[real]
            brier_sum += sum((q - r) ** 2 for q, r in zip(
                (p.p_home, p.p_draw, p.p_away), y))
            details.append({
                "match_id": m.id, "home": m.home, "away": m.away,
                "real": f"{m.home_goals}-{m.away_goals}",
                "real_outcome": real, "pred_outcome": pred,
                "hit": hit,
                "p_home": round(p.p_home, 3), "p_draw": round(p.p_draw, 3),
                "p_away": round(p.p_away, 3),
                "pred_score": (f"{int(round(p.pred_home_goals or 0))}-"
                               f"{int(round(p.pred_away_goals or 0))}"),
            })
        n = len(details)
        if n == 0:
            return {"played_with_prediction": 0, "accuracy_1x2": 0,
                    "brier_score": 0, "details": []}
        return {
            "played_with_prediction": n,
            "accuracy_1x2": correct / n,
            "brier_score": brier_sum / n,
            "details": details,
        }
    finally:
        s.close()


# ---------- GROUPS ----------
@app.get("/api/groups")
def groups():
    """Standings de fase de grupos."""
    s = get_session()
    try:
        team_names = {t.code: t.name for t in s.query(Team).all()}
        ms = (s.query(Match)
              .filter(Match.competition == "WC2026", Match.stage == "GROUP")
              .all())
        groups_map: dict[str, dict] = {}
        for m in ms:
            if not m.group:
                continue
            g = groups_map.setdefault(m.group, {})
            for code in (m.home, m.away):
                if code and code not in g:
                    g[code] = {"code": code, "name": team_names.get(code, code),
                               "pj": 0, "g": 0, "e": 0, "p": 0, "gf": 0, "gc": 0,
                               "pts": 0}
            if m.finished and m.home_goals is not None and m.away_goals is not None:
                h = g[m.home]
                a = g[m.away]
                h["pj"] += 1; a["pj"] += 1
                h["gf"] += m.home_goals; h["gc"] += m.away_goals
                a["gf"] += m.away_goals; a["gc"] += m.home_goals
                if m.home_goals > m.away_goals:
                    h["g"] += 1; a["p"] += 1; h["pts"] += 3
                elif m.home_goals < m.away_goals:
                    a["g"] += 1; h["p"] += 1; a["pts"] += 3
                else:
                    h["e"] += 1; a["e"] += 1
                    h["pts"] += 1; a["pts"] += 1
        out = {}
        for g, table in groups_map.items():
            standings = sorted(table.values(),
                               key=lambda r: (r["pts"], r["gf"] - r["gc"], r["gf"]),
                               reverse=True)
            for t in standings:
                t["dg"] = t["gf"] - t["gc"]
            out[g] = standings
        return out
    finally:
        s.close()


# ---------- LIVE ----------
@app.get("/api/live")
def list_live():
    """Devuelve partidos en vivo con su último snapshot."""
    s = get_session()
    try:
        team_names = {t.code: t.name for t in s.query(Team).all()}
        # Última snapshot por match_id activo
        # Match "live" = tiene un snapshot reciente con status IN_PLAY o PAUSED
        latest = {}
        snaps = (s.query(LiveStats)
                 .order_by(LiveStats.updated_at.desc())
                 .limit(500).all())
        for snap in snaps:
            if snap.match_id not in latest:
                latest[snap.match_id] = snap
        out = []
        # Estados considerados "en vivo" (ESPN + football-data)
        LIVE_STATUSES = {"IN_PLAY", "LIVE", "PAUSED",
                         "STATUS_IN_PROGRESS", "STATUS_HALFTIME",
                         "STATUS_FIRST_HALF", "STATUS_SECOND_HALF",
                         "STATUS_EXTRA_TIME", "STATUS_PENALTIES",
                         "STATUS_END_PERIOD"}
        for mid, snap in latest.items():
            if snap.status not in LIVE_STATUSES:
                continue
            m = s.get(Match, mid)
            if not m:
                continue
            out.append({
                "match_id": mid,
                "home": m.home, "home_name": team_names.get(m.home, m.home),
                "away": m.away, "away_name": team_names.get(m.away, m.away),
                "minute": snap.minute,
                "status": snap.status,
                "home_goals_live": snap.home_goals_live,
                "away_goals_live": snap.away_goals_live,
                "stats": {
                    "possession_h": snap.possession_h, "possession_a": snap.possession_a,
                    "shots_h": snap.shots_h, "shots_a": snap.shots_a,
                    "shots_on_target_h": snap.shots_on_target_h,
                    "shots_on_target_a": snap.shots_on_target_a,
                    "corners_h": snap.corners_h, "corners_a": snap.corners_a,
                    "fouls_h": snap.fouls_h, "fouls_a": snap.fouls_a,
                    "yellow_h": snap.yellow_h, "yellow_a": snap.yellow_a,
                    "red_h": snap.red_h, "red_a": snap.red_a,
                    "xg_live_h": snap.xg_live_h, "xg_live_a": snap.xg_live_a,
                },
                "prediction_live": {
                    "p_home": snap.p_home_live, "p_draw": snap.p_draw_live,
                    "p_away": snap.p_away_live,
                    "expected_final_home": snap.expected_final_home,
                    "expected_final_away": snap.expected_final_away,
                },
                "updated_at": snap.updated_at.isoformat() if snap.updated_at else None,
            })
        return out
    finally:
        s.close()


@app.post("/api/live/tick")
def live_tick_now():
    """Dispara un tick manual del poller."""
    live_tick()
    return {"ok": True, "state": LIVE_STATE}


@app.get("/api/live/status")
def live_status():
    return LIVE_STATE


@app.get("/api/live/{match_id}")
def get_live(match_id: int):
    """Última snapshot + historial de las últimas 30 snapshots para este match."""
    s = get_session()
    try:
        snaps = (s.query(LiveStats)
                 .filter(LiveStats.match_id == match_id)
                 .order_by(LiveStats.updated_at.desc())
                 .limit(30).all())
        if not snaps:
            return {"history": [], "latest": None}
        latest = snaps[0]
        return {
            "latest": {
                "minute": latest.minute, "status": latest.status,
                "home_goals_live": latest.home_goals_live,
                "away_goals_live": latest.away_goals_live,
                "p_home_live": latest.p_home_live,
                "p_draw_live": latest.p_draw_live,
                "p_away_live": latest.p_away_live,
                "expected_final_home": latest.expected_final_home,
                "expected_final_away": latest.expected_final_away,
                "stats": {
                    "shots_h": latest.shots_h, "shots_a": latest.shots_a,
                    "shots_on_target_h": latest.shots_on_target_h,
                    "shots_on_target_a": latest.shots_on_target_a,
                    "possession_h": latest.possession_h,
                    "possession_a": latest.possession_a,
                    "corners_h": latest.corners_h, "corners_a": latest.corners_a,
                    "yellow_h": latest.yellow_h, "yellow_a": latest.yellow_a,
                    "red_h": latest.red_h, "red_a": latest.red_a,
                    "xg_live_h": latest.xg_live_h, "xg_live_a": latest.xg_live_a,
                },
            },
            "history": [{
                "minute": x.minute,
                "p_home": x.p_home_live, "p_draw": x.p_draw_live,
                "p_away": x.p_away_live,
                "home_goals": x.home_goals_live, "away_goals": x.away_goals_live,
                "updated_at": x.updated_at.isoformat() if x.updated_at else None,
            } for x in reversed(snaps)],
        }
    finally:
        s.close()


# ---------- MODEL INFO ----------
@app.get("/api/model_info")
def model_info():
    p = get_predictor()
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return {
        "model_loaded": p.has_model(),
        "meta": meta,
    }
