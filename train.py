"""
Entrenamiento del motor ML.

Pipeline:
  1) Bootstrap de DB.
  2) Recalcula Elo cronológicamente sobre el histórico (Elo *vivo*).
  3) Construye dataset paso-a-paso (features ANTES de cada partido para evitar
     leakage temporal).
  4) Augmenta con 2.000 partidos sintéticos.
  5) Entrena ensemble: HistGradientBoosting + RandomForest + LogReg.
  6) Calibra probabilidades con isotónica (CalibratedClassifierCV).
  7) Reporta accuracy + log-loss + Brier en split temporal (últimos 20% reales).
  8) Persiste a models/trained/.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, mean_absolute_error, brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.db import get_session, Match, Team
from data.loader import bootstrap
from models.elo import update_pair, recompute_all
from models.features import build_features, to_vector, FEATURE_ORDER, WEIGHTED_DECAY
from models.predictor import CLF_PATH, REG_H_PATH, REG_A_PATH, META_PATH
from models.dixon_coles import DixonColes
from models.pi_ratings import PiRatings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def _label_1x2(hg: int, ag: int) -> int:
    """0=away, 1=draw, 2=home."""
    if hg > ag:
        return 2
    if hg < ag:
        return 0
    return 1


def _stats_from_seq(seq):
    """Calcula form, form_w, avg_goals, avg_conc desde lista de (gf, ga).
    Asume seq en orden cronológico ASCENDENTE."""
    if not seq:
        return 0.5, 0.5, 1.2, 1.2
    # Para form_w el más reciente debe pesar más → reverso
    rev = list(reversed(seq))
    pts_total = 0
    pts_w = 0.0
    w_total = 0.0
    gf_sum = 0
    ga_sum = 0
    for i, (g, c) in enumerate(rev):
        pts = 3 if g > c else (1 if g == c else 0)
        pts_total += pts
        w = WEIGHTED_DECAY ** i
        pts_w += pts * w
        w_total += 3 * w
        gf_sum += g
        ga_sum += c
    n = len(seq)
    return (pts_total / (3 * n),
            pts_w / w_total if w_total > 0 else 0.5,
            gf_sum / n, ga_sum / n)


def build_training_set():
    """Recorre histórico cronológicamente actualizando Elo, forma reciente,
    última fecha jugada y H2H. Features representan el estado ANTES del partido."""
    s = get_session()
    try:
        teams = {t.code: {"elo": t.elo, "xg_for": t.xg_for,
                          "xg_against": t.xg_against,
                          "confederation": t.confederation}
                 for t in s.query(Team).all()}
        history = (
            s.query(Match)
            .filter(Match.finished == True,  # noqa
                    Match.competition != "WC2026",
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None))
            .order_by(Match.datetime_utc.asc())
            .all()
        )
        log.info("Partidos históricos: %d", len(history))

        X, y_cls, y_h, y_a = [], [], [], []
        recent: dict[str, list[tuple[int, int]]] = {c: [] for c in teams}
        last_dt: dict[str, datetime] = {}
        # H2H: key (frozenset, "h_won_count", "a_won_count") -> stored as list of
        # tuples (date, winner_or_None) para los 5 más recientes
        h2h: dict[frozenset, list[tuple[datetime, str | None]]] = {}

        for m in history:
            if m.home not in teams or m.away not in teams:
                continue
            th = teams[m.home]
            ta = teams[m.away]
            fh_seq = recent[m.home][-6:]
            fa_seq = recent[m.away][-6:]
            form_h, form_h_w, ag_h, ac_h = _stats_from_seq(fh_seq)
            form_a, form_a_w, ag_a, ac_a = _stats_from_seq(fa_seq)

            # Días de descanso
            def _rest(team_code):
                d = last_dt.get(team_code)
                if d is None:
                    return 7.0
                delta = (m.datetime_utc - d).total_seconds() / 86400.0
                return max(0.0, min(30.0, delta))

            # H2H diff (últimos 5)
            key = frozenset({m.home, m.away})
            hist_pair = h2h.get(key, [])
            wins_h = sum(1 for _, w in hist_pair[-5:] if w == m.home)
            wins_a = sum(1 for _, w in hist_pair[-5:] if w == m.away)
            h2h_diff = wins_h - wins_a

            features = {
                "elo_h": th["elo"], "elo_a": ta["elo"],
                "elo_diff": th["elo"] - ta["elo"],
                "xg_for_h": th["xg_for"], "xg_against_h": th["xg_against"],
                "xg_for_a": ta["xg_for"], "xg_against_a": ta["xg_against"],
                "form_h": form_h, "form_a": form_a,
                "form_h_w": form_h_w, "form_a_w": form_a_w,
                "avg_goals_h": ag_h, "avg_goals_a": ag_a,
                "avg_conc_h": ac_h, "avg_conc_a": ac_a,
                "rest_days_h": _rest(m.home), "rest_days_a": _rest(m.away),
                "h2h_diff": h2h_diff,
                "neutral": int(m.neutral),
                "is_wc": int(m.competition in ("WC", "WC2026")),
                "confed_match": int(th["confederation"] != ta["confederation"]),
            }
            X.append(to_vector(features))
            y_cls.append(_label_1x2(m.home_goals, m.away_goals))
            y_h.append(m.home_goals)
            y_a.append(m.away_goals)

            # Update Elo
            new_h, new_a = update_pair(
                th["elo"], ta["elo"], m.home_goals, m.away_goals,
                m.competition or "FRIENDLY", bool(m.neutral),
            )
            th["elo"] = new_h
            ta["elo"] = new_a
            # Update forma y última fecha
            recent[m.home].append((m.home_goals, m.away_goals))
            recent[m.away].append((m.away_goals, m.home_goals))
            last_dt[m.home] = m.datetime_utc
            last_dt[m.away] = m.datetime_utc
            # Update H2H
            winner = (m.home if m.home_goals > m.away_goals
                      else (m.away if m.home_goals < m.away_goals else None))
            h2h.setdefault(key, []).append((m.datetime_utc, winner))
        return np.array(X), np.array(y_cls), np.array(y_h), np.array(y_a)
    finally:
        s.close()


def synthetic_augmentation(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    s = get_session()
    try:
        teams = list(s.query(Team).all())
        codes = [t.code for t in teams]
        info = {t.code: t for t in teams}
        X, y, yh, ya = [], [], [], []
        for _ in range(n):
            h_code, a_code = rng.choice(codes, size=2, replace=False)
            th, ta = info[h_code], info[a_code]
            neutral = rng.random() < 0.6
            league_avg = 1.3
            atk_h = th.xg_for / league_avg
            def_a = ta.xg_against / league_avg
            atk_a = ta.xg_for / league_avg
            def_h = th.xg_against / league_avg
            lam_h = league_avg * atk_h * def_a
            lam_a = league_avg * atk_a * def_h
            elo_factor = 1.0 + max(min(((th.elo - ta.elo)) / 1000.0, 0.4), -0.4)
            lam_h *= elo_factor
            lam_a /= elo_factor
            if not neutral:
                lam_h *= 1.15
                lam_a *= 0.92
            hg = int(rng.poisson(max(lam_h, 0.15)))
            ag = int(rng.poisson(max(lam_a, 0.15)))
            features = {
                "elo_h": th.elo, "elo_a": ta.elo,
                "elo_diff": th.elo - ta.elo,
                "xg_for_h": th.xg_for, "xg_against_h": th.xg_against,
                "xg_for_a": ta.xg_for, "xg_against_a": ta.xg_against,
                "form_h": 0.5, "form_a": 0.5,
                "form_h_w": 0.5, "form_a_w": 0.5,
                "avg_goals_h": th.xg_for, "avg_goals_a": ta.xg_for,
                "avg_conc_h": th.xg_against, "avg_conc_a": ta.xg_against,
                "rest_days_h": float(rng.integers(3, 10)),
                "rest_days_a": float(rng.integers(3, 10)),
                "h2h_diff": int(rng.integers(-2, 3)),
                "neutral": int(neutral),
                "is_wc": int(rng.random() < 0.2),
                "confed_match": int(th.confederation != ta.confederation),
            }
            X.append(to_vector(features))
            y.append(_label_1x2(hg, ag))
            yh.append(hg)
            ya.append(ag)
        return np.array(X), np.array(y), np.array(yh), np.array(ya)
    finally:
        s.close()


def _build_classifier():
    """Ensemble HGB + RF + LogReg calibrado."""
    hgb = HistGradientBoostingClassifier(
        max_iter=400, max_depth=5, learning_rate=0.06,
        l2_regularization=0.2, random_state=42, min_samples_leaf=15,
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.6, max_iter=2000,
                                   class_weight="balanced", random_state=42)),
    ])
    voting = VotingClassifier(
        estimators=[("hgb", hgb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[2, 1, 1],
        n_jobs=1,
    )
    # Calibración isotónica para que P(predicha) ≈ P(real). cv=3 para no
    # consumir mucho dato; isotonic es más expresiva que sigmoid en multiclase.
    return CalibratedClassifierCV(voting, method="isotonic", cv=3)


def train(verbose: bool = True):
    if verbose:
        log.info("Bootstrap DB...")
    bootstrap()
    s = get_session()
    try:
        if verbose:
            log.info("Recomputando Elo histórico...")
        recompute_all(s)
    finally:
        s.close()

    log.info("Construyendo dataset real...")
    X_real, y_real, yh_real, ya_real = build_training_set()
    log.info("Real: %d partidos · %d features", len(X_real),
             len(FEATURE_ORDER))

    log.info("Generando dataset sintético (2000)...")
    X_syn, y_syn, yh_syn, ya_syn = synthetic_augmentation(2000)

    n_real = len(X_real)
    test_n = max(20, int(n_real * 0.2)) if n_real else 0
    if n_real > 0:
        X_train = np.vstack([X_real[:-test_n], X_syn])
        y_train = np.concatenate([y_real[:-test_n], y_syn])
        yh_train = np.concatenate([yh_real[:-test_n], yh_syn])
        ya_train = np.concatenate([ya_real[:-test_n], ya_syn])
        X_test, y_test = X_real[-test_n:], y_real[-test_n:]
        yh_test, ya_test = yh_real[-test_n:], ya_real[-test_n:]
    else:
        X_train, y_train = X_syn, y_syn
        yh_train, ya_train = yh_syn, ya_syn
        X_test = y_test = yh_test = ya_test = np.array([])

    log.info("Entrenando ensemble calibrado (HGB+RF+LR)...")
    clf = _build_classifier()
    clf.fit(X_train, y_train)

    log.info("Entrenando regresores de goles (HGB)...")
    reg_h = HistGradientBoostingRegressor(
        max_iter=400, max_depth=5, learning_rate=0.06,
        l2_regularization=0.2, random_state=42,
    )
    reg_h.fit(X_train, yh_train)
    reg_a = HistGradientBoostingRegressor(
        max_iter=400, max_depth=5, learning_rate=0.06,
        l2_regularization=0.2, random_state=42,
    )
    reg_a.fit(X_train, ya_train)

    metrics = {}
    if len(X_test):
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        try:
            ll = float(log_loss(y_test, proba, labels=[0, 1, 2]))
        except Exception:
            ll = None
        # Brier 1X2 multiclase = media de Brier por clase
        brier = 0.0
        for k in (0, 1, 2):
            y_bin = (y_test == k).astype(int)
            p_k = proba[:, list(clf.classes_).index(k)]
            brier += brier_score_loss(y_bin, p_k)
        brier /= 3.0
        mae_h = float(mean_absolute_error(yh_test, reg_h.predict(X_test)))
        mae_a = float(mean_absolute_error(ya_test, reg_a.predict(X_test)))
        # Accuracy por bucket de confianza
        max_p = proba.max(axis=1)
        buckets = {}
        for lo, hi, name in [(0.0, 0.45, "baja"), (0.45, 0.65, "media"),
                             (0.65, 1.01, "alta")]:
            mask = (max_p >= lo) & (max_p < hi)
            if mask.sum() > 0:
                buckets[name] = {
                    "n": int(mask.sum()),
                    "accuracy": float((pred[mask] == y_test[mask]).mean()),
                }
        metrics = {
            "accuracy_1x2": acc, "log_loss": ll, "brier": brier,
            "mae_home_goals": mae_h, "mae_away_goals": mae_a,
            "test_size": len(X_test),
            "by_confidence": buckets,
        }
        log.info("TEST → acc=%.3f logloss=%s brier=%.3f MAE(h)=%.2f MAE(a)=%.2f",
                 acc, f"{ll:.3f}" if ll else "n/a", brier, mae_h, mae_a)
        for name, b in buckets.items():
            log.info("  bucket %s (n=%d): acc=%.3f", name, b["n"], b["accuracy"])

    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg_h, REG_H_PATH)
    joblib.dump(reg_a, REG_A_PATH)

    # ===== Dixon-Coles =====
    s = get_session()
    try:
        log.info("Ajustando Dixon-Coles (1997) por MLE...")
        dc = DixonColes.fit(s, xi=0.0019)
        dc.save()
    except Exception as e:
        log.warning("Dixon-Coles falló: %s", e)
    finally:
        s.close()

    # ===== Pi-ratings =====
    s = get_session()
    try:
        log.info("Ajustando Pi-ratings (Constantinou 2013)...")
        pi = PiRatings.fit(s)
        pi.save()
    except Exception as e:
        log.warning("Pi-ratings falló: %s", e)
    finally:
        s.close()

    META_PATH.write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "meta-v3 (ML + Dixon-Coles + Pi-ratings + Elo-Poisson)",
        "features": FEATURE_ORDER,
        "n_train": int(len(X_train)),
        "n_real": int(n_real),
        "metrics": metrics,
        "components": [
            "HistGradientBoosting + RandomForest + LogReg (calibrado isotónico)",
            "Dixon-Coles (1997) — fórmula bookmaker estándar, MLE con weighting temporal",
            "Pi-ratings (Constantinou & Fenton 2013) — dual home/away rating",
            "Elo (eloratings.net style) + Poisson baseline",
        ],
    }, indent=2))
    log.info("Todos los modelos guardados (meta-ensemble v3).")
    return metrics


if __name__ == "__main__":
    train()
