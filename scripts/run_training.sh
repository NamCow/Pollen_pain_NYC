#!/usr/bin/env bash
set -euo pipefail

echo "=== Pollen & Pain: Training Pipeline ==="
echo ""

echo "[1/3] Running XGBoost + Logistic Regression walk-forward CV..."
python -m src.models.train_xgboost

echo ""
echo "[2/3] Running standalone Logistic Regression training..."
python -m src.models.train_logistic_regression

echo ""
echo "[3/3] Generating final predictions (all data)..."
python -m src.models.predict

echo ""
echo "=== Training complete. Results in data/models/ ==="
