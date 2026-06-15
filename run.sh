#!/bin/bash
# Levanta el predictor del Mundial 2026 en http://127.0.0.1:8765
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "Creando venv..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi
# Entrenar si no hay modelo
if [ ! -f models/trained/xgb_clf.joblib ]; then
  echo "Entrenando modelo..."
  python train.py
fi
echo ""
echo "  → Abrí http://127.0.0.1:8765 en tu navegador"
echo ""
exec uvicorn app:app --host 127.0.0.1 --port 8765
