"""Carga config del proyecto (token de API, intervalos, etc).
Prioriza variables de entorno > config.json > defaults.
config.json NO se commitea (está en .gitignore)."""
from __future__ import annotations
import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"
_DEFAULTS = {
    "football_data_token": "",
    "live_poll_interval_seconds": 60,
    "training_interval_seconds": 14400,
}


def _load() -> dict:
    cfg = dict(_DEFAULTS)
    if _CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(_CONFIG_PATH.read_text()))
        except Exception:
            pass
    # env overrides
    if os.environ.get("FOOTBALL_DATA_TOKEN"):
        cfg["football_data_token"] = os.environ["FOOTBALL_DATA_TOKEN"]
    return cfg


CONFIG = _load()


def get(key: str, default=None):
    return CONFIG.get(key, default)
