"""
Centralized configuration loader.

Reads config/settings.json once and exposes values used across all scripts.
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.json"

def _load() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)

_cfg = _load()

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(_cfg["paths"]["processed"])
MODELS_DIR = Path(_cfg["paths"]["models"])
INPUT_DIR = Path(_cfg["paths"]["input"])
GEOJSON_PATH = Path(_cfg["paths"]["geojson"])

# ── Modeling ─────────────────────────────────────────────────────────────────
TARGET = _cfg["modeling"]["target"]
MIN_AVG_MONTHLY_ED_VISITS = float(_cfg["modeling"]["min_avg_monthly_ed_visits"])
POLLEN_SEASON_MONTHS = tuple(_cfg["modeling"]["pollen_season_months"])
HOLDOUT_MONTHS = int(_cfg["modeling"]["holdout_months"])
CV_TEST_MONTHS = int(_cfg["modeling"]["cv_test_months"])
CV_MIN_TRAIN_MONTHS = int(_cfg["modeling"]["cv_min_train_months"])

# ── Model hyperparameters ────────────────────────────────────────────────────
XGBOOST_PARAMS = dict(_cfg["xgboost"])
LOGREG_PARAMS = dict(_cfg["logistic_regression"])

# ── Features ─────────────────────────────────────────────────────────────────
FEATURE_COLS = list(_cfg["features"])
REQUIRED_MODEL_FEATURES = list(_cfg["required_features"])
