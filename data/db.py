"""Capa de base de datos (SQLite + SQLAlchemy)."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = Path(__file__).parent / "wc2026.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=ENGINE, expire_on_commit=False, future=True)
Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"
    code = Column(String(4), primary_key=True)
    name = Column(String(64), nullable=False)
    confederation = Column(String(16))
    elo = Column(Float, default=1500.0)
    xg_for = Column(Float, default=1.3)
    xg_against = Column(Float, default=1.3)


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition = Column(String(16), default="WC2026")  # WC2026, WC, EURO, COPA, QUAL, FRIENDLY
    stage = Column(String(16))                          # GROUP/R32/R16/QF/SF/THIRD/FINAL
    group = Column(String(2))
    matchday = Column(Integer)
    datetime_utc = Column(DateTime, index=True)
    home = Column(String(4), ForeignKey("teams.code"))
    away = Column(String(4), ForeignKey("teams.code"))
    home_goals = Column(Integer)   # null si no jugado
    away_goals = Column(Integer)
    neutral = Column(Boolean, default=True)
    finished = Column(Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    p_home = Column(Float)
    p_draw = Column(Float)
    p_away = Column(Float)
    pred_home_goals = Column(Float)
    pred_away_goals = Column(Float)
    over_25 = Column(Float)         # probabilidad de más de 2.5 goles
    btts = Column(Float)            # both teams to score
    model_version = Column(String(64))
    confidence = Column(Float)      # confianza del modelo (entropía invertida)


class LiveStats(Base):
    """Estadísticas in-play del partido. Una fila por snapshot (cada poller tick).
    El frontend usa la última (orden por updated_at desc)."""
    __tablename__ = "live_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)
    minute = Column(Integer)                # minuto actual (0-90+)
    status = Column(String(16))             # IN_PLAY / PAUSED / FINISHED / TIMED
    home_goals_live = Column(Integer, default=0)
    away_goals_live = Column(Integer, default=0)
    possession_h = Column(Float)            # 0-100
    possession_a = Column(Float)
    shots_h = Column(Integer, default=0)
    shots_a = Column(Integer, default=0)
    shots_on_target_h = Column(Integer, default=0)
    shots_on_target_a = Column(Integer, default=0)
    corners_h = Column(Integer, default=0)
    corners_a = Column(Integer, default=0)
    fouls_h = Column(Integer, default=0)
    fouls_a = Column(Integer, default=0)
    yellow_h = Column(Integer, default=0)
    yellow_a = Column(Integer, default=0)
    red_h = Column(Integer, default=0)
    red_a = Column(Integer, default=0)
    xg_live_h = Column(Float)               # xG acumulado live (si la fuente lo da)
    xg_live_a = Column(Float)
    # Predicción in-play (re-calculada cada snapshot)
    p_home_live = Column(Float)
    p_draw_live = Column(Float)
    p_away_live = Column(Float)
    expected_final_home = Column(Float)     # marcador final esperado (puede ser decimal)
    expected_final_away = Column(Float)


def init_db():
    Base.metadata.create_all(ENGINE)


def get_session():
    return SessionLocal()
