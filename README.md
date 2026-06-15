# Mundial 2026 — Motor de Predicciones

App web local que predice los 104 partidos del Mundial 2026 con un modelo
**Elo + Poisson + Gradient Boosting** entrenado sobre +2.000 partidos
(reales del WC22 / Euro24 / Copa América 24 / eliminatorias + sintéticos).

## Arranque rápido

```bash
cd "prediccion Python"
source venv/bin/activate
python train.py                # solo la primera vez (o si actualizás datos)
uvicorn app:app --host 127.0.0.1 --port 8765
```

Abrí http://127.0.0.1:8765

## Métricas actuales

| Métrica | Valor |
|---|---|
| Accuracy 1X2 (test temporal) | **72.4%** |
| Log-loss multiclase | 1.34 |
| MAE goles local | 0.83 |
| MAE goles visitante | 0.79 |
| Partidos reales en training | 145 |
| Partidos sintéticos en training | 2.000 |

Para referencia: bookies profesionales ~55–58% en accuracy 1X2 de fútbol
internacional. El número alto acá viene del set de test chico — esperá que baje
a ~58–65% real conforme se carguen más partidos.

## Estructura

```
app.py                   FastAPI backend (REST)
train.py                 entrena el modelo, persiste a models/trained/
data/
  seed_teams.py          48 selecciones del Mundial + 22 históricas
  seed_historical.py     145 partidos internacionales 2022-2026
  seed_wc2026.py         grupos + 72 partidos fase grupos + knockout placeholder
  db.py                  SQLAlchemy: Team, Match, Prediction
  loader.py              bootstrap idempotente de la DB
  scraper.py             pull de resultados desde Wikipedia
models/
  elo.py                 Elo dinámico (estilo eloratings.net)
  poisson.py             Poisson bivariado (matriz score, BTTS, O2.5)
  features.py            feature engineering (16 features)
  predictor.py           HistGradientBoosting + blending con Poisson
static/
  index.html, app.js, app.css   Frontend vanilla + Tailwind por CDN
```

## API REST

| Método | Endpoint | Descripción |
|---|---|---|
| GET  | `/api/teams`           | Selecciones con Elo y xG |
| GET  | `/api/matches?status=upcoming` | Partidos (filtros: stage, group, status) |
| GET  | `/api/match/{id}`      | Detalle de un partido |
| POST | `/api/predict/{id}`    | Genera/actualiza predicción |
| POST | `/api/predict_all`     | Predice todos los partidos definidos |
| POST | `/api/refresh`         | Scrape Wikipedia y actualiza resultados |
| GET  | `/api/accuracy`        | Aciertos del modelo en partidos jugados |
| GET  | `/api/groups`          | Tabla de posiciones por grupo |
| GET  | `/api/model_info`      | Metadata del modelo entrenado |

## Cómo mantenerlo afilado

1. Cuando termine cada partido, click **⟳ Scrape Wikipedia** (o `POST /api/refresh`).
2. Re-entrenar el modelo periódicamente (`python train.py`) para que el Elo y
   los HGB absorban los resultados nuevos.
3. Para cargar manualmente resultados (si Wikipedia no responde), editá
   `data/seed_wc2026.py::PLAYED_RESULTS` y reiniciá la DB borrando
   `data/wc2026.db` antes del próximo arranque.

## Algoritmo

Cada predicción combina:
- **Elo dinámico**: actualizado cronológicamente sobre el histórico, con K
  variable según importancia del partido y bonus por margen de gol.
- **Modelo HistGradientBoosting** (16 features: Elo, xG ataque/defensa, forma
  reciente, ventaja local, partido de Mundial, partido inter-confederación).
- **Poisson bivariado** sobre las λ de goles para producir matriz de marcadores,
  P(BTTS), P(over 2.5), y "marcador más probable".
- **Blending 60/40** entre las probabilidades 1X2 del clasificador y las del
  Poisson, para robustez.
