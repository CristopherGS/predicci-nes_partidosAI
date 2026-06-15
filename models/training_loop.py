"""
Loop de auto-entrenamiento en background.

Política:
  - Re-entrena periódicamente (default cada 4 horas).
  - Re-entrena tras cada refresh del scraper si trajo cambios (trigger manual).
  - Re-entrena cuando se cargan resultados nuevos (señal via mark_dirty()).
  - Nunca corre dos entrenamientos simultáneos (lock).
  - El predictor se recarga automáticamente al terminar (reload_predictor()).
"""
from __future__ import annotations
import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .predictor import reload_predictor

log = logging.getLogger("training_loop")

# Estado compartido (consultable desde el endpoint /api/train_status)
STATE = {
    "running": False,
    "last_started": None,    # ISO datetime
    "last_finished": None,
    "last_duration_s": None,
    "last_metrics": None,
    "last_error": None,
    "last_trigger": None,    # "schedule" / "manual" / "post-refresh" / "data-changed"
    "scheduled_every_s": 4 * 3600,
    "total_runs": 0,
    "dirty": False,          # flag: hay datos nuevos sin entrenar
}

_lock = threading.Lock()
_loop_task: Optional[asyncio.Task] = None
_stop_flag = False


def mark_dirty(source: str = "data-changed"):
    """Marca que hay datos nuevos. El loop entrenará en su próximo ciclo
    (o inmediato si se llama _kick())."""
    STATE["dirty"] = True
    log.info("Estado marcado como dirty (%s)", source)


def _do_training(trigger: str):
    """Bloqueante. Llama a train.train() y actualiza STATE."""
    if not _lock.acquire(blocking=False):
        log.info("Entrenamiento ya en curso; ignoro trigger %s", trigger)
        return
    try:
        from train import train as run_train
        STATE["running"] = True
        STATE["last_trigger"] = trigger
        STATE["last_started"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        STATE["last_error"] = None
        t0 = time.time()
        try:
            metrics = run_train(verbose=False)
            STATE["last_metrics"] = metrics
            reload_predictor()
            STATE["dirty"] = False
            STATE["total_runs"] += 1
            log.info("Entrenamiento OK (%s) en %.1fs", trigger, time.time() - t0)
        except Exception as e:
            STATE["last_error"] = f"{type(e).__name__}: {e}"
            log.exception("Entrenamiento falló: %s", e)
        finally:
            STATE["running"] = False
            STATE["last_finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            STATE["last_duration_s"] = round(time.time() - t0, 2)
    finally:
        _lock.release()


def trigger_train_async(trigger: str = "manual") -> bool:
    """Lanza un entrenamiento en thread separado. Devuelve False si ya hay uno."""
    if STATE["running"]:
        return False
    th = threading.Thread(target=_do_training, args=(trigger,), daemon=True)
    th.start()
    return True


async def _scheduler_loop():
    """Cada N segundos, si hay dirty=True o pasó el período, entrena."""
    global _stop_flag
    log.info("Loop de auto-entrenamiento iniciado (cada %ds)",
             STATE["scheduled_every_s"])
    elapsed = 0
    while not _stop_flag:
        try:
            # Si hay datos nuevos sin entrenar, dispara ya.
            if STATE["dirty"]:
                trigger_train_async("data-changed")
                elapsed = 0
            elif elapsed >= STATE["scheduled_every_s"]:
                trigger_train_async("schedule")
                elapsed = 0
            await asyncio.sleep(10)
            elapsed += 10
        except Exception as e:
            log.exception("Error en scheduler loop: %s", e)
            await asyncio.sleep(60)


def start(scheduled_every_s: int = 4 * 3600):
    """Arranca el loop. Llamado desde FastAPI startup."""
    global _loop_task, _stop_flag
    STATE["scheduled_every_s"] = scheduled_every_s
    _stop_flag = False
    try:
        loop = asyncio.get_event_loop()
        _loop_task = loop.create_task(_scheduler_loop())
    except RuntimeError:
        log.warning("No event loop; el scheduler no se arrancó")


def stop():
    global _stop_flag
    _stop_flag = True
